// SPDX-License-Identifier: GPL-3.0-or-later
// GPU diagnostics for the crash, device-loss and freeze reports (declared in nxrt/diag.h): the first Vulkan call that failed
// anywhere in the process. Read by report code only; the path that fills it is written to be as cheap as a counter.
#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "nxrt/config.h"
#include "nxrt/diag.h"
#include "nxrt/log.h"

namespace nxrt {
namespace {

using Clock = std::chrono::steady_clock;
// Created before main: the "how far into the run" figure of a record is then measured from process start.
const Clock::time_point g_t0 = Clock::now();

uint64_t ms_since_start() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_t0).count());
}

const char* file_name(const char* path) {
    const char* slash = std::strrchr(path, '\\');
    const char* fwd = std::strrchr(path, '/');
    if (fwd > slash) slash = fwd;
    return slash ? slash + 1 : path;
}

// VK_SUCCESS is 0 and the errors are negative, so no Vulkan header is needed here (nxrt, the core library, does not link Vulkan).
const char* result_name(int r) {
    switch (r) {
        case -1: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case -2: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case -3: return "VK_ERROR_INITIALIZATION_FAILED";
        case -4: return "VK_ERROR_DEVICE_LOST";
        case -5: return "VK_ERROR_MEMORY_MAP_FAILED";
        case -6: return "VK_ERROR_LAYER_NOT_PRESENT";
        case -7: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case -8: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case -9: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case -10: return "VK_ERROR_TOO_MANY_OBJECTS";
        case -11: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case -12: return "VK_ERROR_FRAGMENTED_POOL";
        case -1000000000: return "VK_ERROR_SURFACE_LOST_KHR";
        case -1000000001: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case -1000001004: return "VK_ERROR_OUT_OF_DATE_KHR";
        case -1000003001: return "VK_ERROR_SUBOPTIMAL_KHR";
        case -1000011001: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case -1000012000: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        default: return "VK_ERROR";
    }
}

// ---- first VkResult failure -----------------------------------------------------------------------------------------------
// 0: no failure yet, 1: a thread is writing the record, 2: the record is complete and final.
enum : int { kVkNone = 0, kVkWriting = 1, kVkRecorded = 2 };
std::atomic<int> g_vk_state{kVkNone};

struct FirstFailure {
    int result = 0;
    const char* call = "";
    const char* file = "";
    int line = 0;
    DWORD tid = 0;
    char thread[48] = {};
    uint64_t ms = 0;
    uint64_t frames = 0;
    char submit[400] = {};
};
FirstFailure g_first;   // written by the one thread that wins the state exchange, read by the reports afterwards
std::atomic<std::string (*)()> g_submit_state{nullptr};

}  // namespace

void diag_vk_note(int result, const char* call, const char* file, int line) {
    int expected = kVkNone;
    if (!g_vk_state.compare_exchange_strong(expected, kVkWriting)) return;   // the first failure is the one that is kept
    FirstFailure& f = g_first;
    f.result = result;
    f.call = call ? call : "?";
    f.file = file ? file : "?";
    f.line = line;
    f.tid = GetCurrentThreadId();
    const char* tn = log_thread_name();
    std::snprintf(f.thread, sizeof f.thread, "%s", tn && tn[0] ? tn : "(unnamed)");
    f.ms = ms_since_start();
    f.frames = diag_frame_count();
    if (auto fn = g_submit_state.load(std::memory_order_acquire))
        if (const std::string s = fn(); !s.empty()) std::snprintf(f.submit, sizeof f.submit, "%s", s.c_str());
    g_vk_state.store(kVkRecorded, std::memory_order_release);
    NXLOG_E("gpu", "first Vulkan failure: %s (%d) from %s at %s:%d on thread %s (tid %lu), %.3f s in, %llu frames presented%s%s", result_name(result),
            result, f.call, file_name(f.file), f.line, f.thread, static_cast<unsigned long>(f.tid), static_cast<double>(f.ms) / 1000.0,
            static_cast<unsigned long long>(f.frames), f.submit[0] ? "; " : "", f.submit);
}

void diag_set_submit_state(std::string (*fn)()) { g_submit_state.store(fn, std::memory_order_release); }

std::string diag_vk_failure_text() {
    const int state = g_vk_state.load(std::memory_order_acquire);
    if (state == kVkNone) return "GPU first failure: (none: every Vulkan call so far returned VK_SUCCESS)\n";
    if (state == kVkWriting) return "GPU first failure: (being recorded on another thread)\n";
    char b[512];
    std::string out = "GPU first failure:\n";
    std::snprintf(b, sizeof b, "  %s (%d) from %s\n    at %s:%d on thread %s (tid %lu), %.3f s into the run, %llu frames presented\n",
                  result_name(g_first.result), g_first.result, g_first.call, file_name(g_first.file), g_first.line, g_first.thread,
                  static_cast<unsigned long>(g_first.tid), static_cast<double>(g_first.ms) / 1000.0,
                  static_cast<unsigned long long>(g_first.frames));
    out += b;
    if (g_first.submit[0]) {
        out += "    submit state when it failed: ";
        out += g_first.submit;
        out += '\n';
    }
    out += "  The first Vulkan call that failed, which every later failure in the log follows from.\n";
    return out;
}

// ---- recent dispatches (the last-N ring) -----------------------------------------------------------------------------------
// A fixed ring of the render thread's dispatches, always on ([diag] dispatch_ring picks how many a dump shows): the frame trace can
// only be aimed at a frame that presents, so a GPU hang that stops the presents had nothing to aim at. The writer is one thread, so
// each slot is a tiny seqlock: the reader retries when the writer changed the slot under it, and never blocks.
constexpr size_t kDispatchSlots = 64;

struct DispatchSlot {
    std::atomic<uint64_t> seq{0};   // even: stable, odd: being written
    DispatchNote note;
};
DispatchSlot g_dispatch[kDispatchSlots];
std::atomic<uint64_t> g_dispatch_count{0};

void diag_note_dispatch(const DispatchNote& note) {
    static const bool registered = (diag_add_report(&diag_dispatch_text), true);   // the dumps flush the ring
    const uint64_t n = g_dispatch_count.fetch_add(1, std::memory_order_relaxed) + 1;
    DispatchSlot& s = g_dispatch[static_cast<size_t>((n - 1) % kDispatchSlots)];
    DispatchNote t = note;
    t.index = n;
    const uint64_t v = s.seq.load(std::memory_order_relaxed);
    s.seq.store(v + 1, std::memory_order_relaxed);   // odd: the slot is being written
    s.note = t;
    s.seq.store(v + 2, std::memory_order_release);   // even again: stable
}

bool dispatch_slot_read(const DispatchSlot& s, DispatchNote& out) {
    for (int tries = 0; tries < 4; ++tries) {
        const uint64_t v1 = s.seq.load(std::memory_order_acquire);
        if (v1 & 1) continue;
        out = s.note;
        if (s.seq.load(std::memory_order_acquire) == v1) return true;
    }
    return false;
}

std::string diag_dispatch_text() {
    const uint64_t total = g_dispatch_count.load(std::memory_order_relaxed);
    if (total == 0) return "";   // no dispatches yet: nothing to say
    int64_t depth = config_get_int("diag", "dispatch_ring", 64);
    if (depth < 1) depth = 1;
    if (depth > static_cast<int64_t>(kDispatchSlots)) depth = static_cast<int64_t>(kDispatchSlots);
    const uint64_t first = total > static_cast<uint64_t>(depth) ? total - static_cast<uint64_t>(depth) + 1 : 1;
    char b[192];
    std::snprintf(b, sizeof b, "recent dispatches (last %llu of %llu, oldest first; [diag] dispatch_ring picks how many):\n",
                  static_cast<unsigned long long>(total - first + 1), static_cast<unsigned long long>(total));
    std::string out = b;
    for (uint64_t i = first; i <= total; ++i) {
        DispatchNote n;
        if (!dispatch_slot_read(g_dispatch[static_cast<size_t>((i - 1) % kDispatchSlots)], n)) {
            out += "  (a dispatch was being written; skipped)\n";
            continue;
        }
        std::snprintf(b, sizeof b, "  #%llu  cs %s%s%s%s  %ux%ux%u  %s  barriers %u  bindings %u%s\n",
                      static_cast<unsigned long long>(n.index),
                      n.flags & DispatchNote::kMissingCs ? "-" : n.shader[0] ? n.shader : "-",
                      n.flags & DispatchNote::kMissingCs && n.missing[0] ? " (missing " : "",
                      n.flags & DispatchNote::kMissingCs && n.missing[0] ? n.missing : "",
                      n.flags & DispatchNote::kMissingCs && n.missing[0] ? ")" : "",
                      n.x, n.y, n.z,
                      n.flags & DispatchNote::kIndirect ? "indirect" : "direct",
                      n.barriers, n.bindings,
                      n.flags & DispatchNote::kBarrierBefore ? " (a barrier before it, the compute-shadow copy-in)" : "");
        out += b;
    }
    return out;
}

std::string diag_device_lost_dump(const char* where, int result, uint64_t frames, const std::string& context) {
    char b[640];
    std::snprintf(b, sizeof b,
                  "GPU DEVICE LOST (reason: VK_ERROR_DEVICE_LOST)\n  %s returned %d after %llu presented frames: the graphics driver reset or removed the device\n"
                  "  (a driver crash or timeout, an out-of-video-memory condition, or a GPU hang). The game was stopped.\n",
                  where ? where : "?", result, static_cast<unsigned long long>(frames));
    std::string header = b;
    header += context;
    header += frames == 0
                  ? "  What to try: the first frame never finished on the GPU. Update the graphics driver; if it happens in one window mode only "
                    "(windowed, say), use another (F1 > Display) and tell us which works; [gfx] gpu in fefw.ini picks another GPU.\n"
                  : "  What to try: update the graphics driver; on a GPU that shares system memory (laptop, handheld) lower the render scale "
                    "or the texture quality. Please attach this file and fefw.log to an issue.\n";
    return diag_full_dump(header.c_str(), "device loss");
}

}  // namespace nxrt
