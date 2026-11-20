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
        case -13: return "VK_ERROR_UNKNOWN";
        case -1000000000: return "VK_ERROR_SURFACE_LOST_KHR";
        case -1000000001: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case -1000001004: return "VK_ERROR_OUT_OF_DATE_KHR";
        case -1000011001: return "VK_ERROR_VALIDATION_FAILED_EXT";
        case -1000012000: return "VK_ERROR_INVALID_SHADER_NV";
        case -1000069000: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case -1000072003: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case -1000161000: return "VK_ERROR_FRAGMENTATION";
        case -1000174001: return "VK_ERROR_NOT_PERMITTED_KHR";
        case -1000255000: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        case -1000257000: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case 1000001003: return "VK_SUBOPTIMAL_KHR";
        case 1000297000: return "VK_PIPELINE_COMPILE_REQUIRED_EXT";
        default: return "VK_ERROR";
    }
}

// Results a healthy run produces on purpose: wait_serial() retries VK_TIMEOUT in slices, the swapchain loop retries VK_NOT_READY,
// vkEnumerate/vkGet* report VK_INCOMPLETE when the output array was full, and the presentation engine hands back
// VK_SUBOPTIMAL_KHR and VK_ERROR_OUT_OF_DATE_KHR on every resize. None of them is a failure, so they must not claim the
// first-failure record: a real device loss would otherwise be reported as a timeout or a resize.
bool result_expected(int r) {
    switch (r) {
        case 1:      // VK_NOT_READY
        case 2:      // VK_TIMEOUT
        case 5:      // VK_INCOMPLETE
        case 1000001003:   // VK_SUBOPTIMAL_KHR
        case -1000001004:  // VK_ERROR_OUT_OF_DATE_KHR
            return true;
        default:
            return false;
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
    if (result_expected(result)) return;   // a healthy run produces these on purpose: not a failure
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
// only be aimed at a frame that presents, so a GPU hang that stops the presents had nothing to aim at. The writer is one thread; each
// slot is guarded by a one-byte spin flag the writer takes and releases, so the payload copy is never concurrent with a read. The
// reader (report code only) tries the flag once and skips the slot when it is held: it never blocks the render thread, and the writer
// only ever waits for a dump reader, which holds the flag for one payload copy and only while a report is being written.
constexpr size_t kDispatchSlots = 64;

struct DispatchSlot {
    std::atomic<uint8_t> held{0};   // flag the writer holds while the payload changes
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
    uint8_t expected = 0;
    while (!s.held.compare_exchange_weak(expected, 1, std::memory_order_acquire)) expected = 0;   // a dump reader holds it: one payload copy
    s.note = t;
    s.held.store(0, std::memory_order_release);
}

bool dispatch_slot_read(DispatchSlot& s, DispatchNote& out) {
    uint8_t expected = 0;
    if (!s.held.compare_exchange_strong(expected, 1, std::memory_order_acquire)) return false;   // the render thread is writing it: skip
    out = s.note;
    s.held.store(0, std::memory_order_release);
    return true;
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
        if (n.flags & DispatchNote::kIndirect) {
            std::snprintf(b, sizeof b, "  #%llu  cs %s%s%s%s  indirect args 0x%llx (dimensions unknown)  barriers %u  bindings %u%s\n",
                          static_cast<unsigned long long>(n.index),
                          n.flags & DispatchNote::kMissingCs ? "-" : n.shader[0] ? n.shader : "-",
                          n.flags & DispatchNote::kMissingCs && n.missing[0] ? " (missing " : "",
                          n.flags & DispatchNote::kMissingCs && n.missing[0] ? n.missing : "",
                          n.flags & DispatchNote::kMissingCs && n.missing[0] ? ")" : "",
                          static_cast<unsigned long long>(n.indirect),
                          n.barriers, n.bindings,
                          n.flags & DispatchNote::kBarrierBefore ? " (a barrier before it, the compute-shadow copy-in)" : "");
        } else {
            std::snprintf(b, sizeof b, "  #%llu  cs %s%s%s%s  %ux%ux%u  direct  barriers %u  bindings %u%s\n",
                          static_cast<unsigned long long>(n.index),
                          n.flags & DispatchNote::kMissingCs ? "-" : n.shader[0] ? n.shader : "-",
                          n.flags & DispatchNote::kMissingCs && n.missing[0] ? " (missing " : "",
                          n.flags & DispatchNote::kMissingCs && n.missing[0] ? n.missing : "",
                          n.flags & DispatchNote::kMissingCs && n.missing[0] ? ")" : "",
                          n.x, n.y, n.z,
                          n.barriers, n.bindings,
                          n.flags & DispatchNote::kBarrierBefore ? " (a barrier before it, the compute-shadow copy-in)" : "");
        }
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
