// SPDX-License-Identifier: GPL-3.0-or-later
// Tests for the GPU diagnostics (nxrt/diag.h): the first Vulkan failure of the process, the recent-dispatch ring and the
// sections the freeze and device-loss dumps share. The ring is process-wide, so the counts are compared as deltas; the
// synthetic freeze dump needs the watchdog, and every case stops the watchdog it starts.
#include "test_framework.h"

#include "nxrt/diag.h"

#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>

NXTEST_GROUP(gpu_diag)

using namespace std::chrono;

namespace {

std::string temp_file(const char* tag) {
    char dir[MAX_PATH];
    DWORD n = GetTempPathA(MAX_PATH, dir);
    return std::string(dir, n) + "nxrt_gd_" + tag + ".txt";
}

std::string read_file(const std::string& path) {
    std::string out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

bool has(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

bool wait_for(const std::function<bool()>& pred, int ms) {
    auto end = steady_clock::now() + milliseconds(ms);
    while (!pred()) {
        if (steady_clock::now() >= end) return false;
        std::this_thread::sleep_for(milliseconds(10));
    }
    return true;
}

// A plausible stream of dispatches: one every 3 barriers, then an indirect one that hangs. Returns the index of the
// newest note, which the ring numbers process-wide (the reports show "last 64 of <index>").
uint64_t feed_dispatches(int count) {
    for (int i = 0; i < count; ++i) {
        nxrt::DispatchNote n;
        std::snprintf(n.shader, sizeof n.shader, "%016llx", 0x9f3ac1e2ull + static_cast<unsigned long long>(i) * 0x11ull);
        n.x = 240;
        n.y = 135;
        n.z = 1;
        n.barriers = 3;
        n.bindings = 3;
        nxrt::diag_note_dispatch(n);
    }
    nxrt::DispatchNote last;
    std::snprintf(last.shader, sizeof last.shader, "7c11be0099a4d2f1");
    last.indirect = 0x1f2a9000ull;
    last.barriers = 21;
    last.bindings = 5;
    last.flags = nxrt::DispatchNote::kIndirect;
    nxrt::diag_note_dispatch(last);
    return last.index;
}

}  // namespace

// The first failure anywhere in the process wins and is never overwritten: the record made by the losing call site
// (a follow-on vkWaitForFences -4) does not displace the record made by the winning one (the batch submit).
NXTEST(first_failure_is_not_displaced) {
    nxrt::diag_vk_note(-4, "vkQueueSubmit2", __FILE__, __LINE__);
    nxrt::diag_vk_note(-3, "vkAllocateMemory", __FILE__, __LINE__);
    const std::string text = nxrt::diag_vk_failure_text();
    CHECK_MSG(has(text, "GPU first failure"), "first-failure section missing");
    CHECK_MSG(has(text, "vkQueueSubmit2"), "the winning call site missing from the record");
    CHECK_MSG(!has(text, "vkAllocateMemory"), "the losing call site displaced the record");
}

// The ring always keeps the last 64: after 70 more dispatches the dump shows the last 64 of the total, oldest first,
// with the newest (the one that hung) at the end.
NXTEST(dispatch_ring_flushes_last_64) {
    const uint64_t total = feed_dispatches(70);
    const std::string text = nxrt::diag_dispatch_text();
    char b[128];
    std::snprintf(b, sizeof b, "last 64 of %llu", static_cast<unsigned long long>(total));
    CHECK_MSG(has(text, b), "the ring did not show the last 64");
    std::snprintf(b, sizeof b, "#%llu", static_cast<unsigned long long>(total - 63));
    CHECK_MSG(has(text, b), "the oldest shown dispatch missing");
    std::snprintf(b, sizeof b, "#%llu  cs 7c11be0099a4d2f1", static_cast<unsigned long long>(total));
    CHECK_MSG(has(text, b), "the newest dispatch missing");
    const size_t newest = text.rfind(b);
    const size_t penultimate = text.rfind("#" + std::to_string(total - 1));
    CHECK_MSG(newest != std::string::npos && penultimate != std::string::npos && newest > penultimate,
              "the newest dispatch is not at the end");
    CHECK_MSG(has(text, "7c11be0099a4d2f1"), "the hung compute shader missing from the dump");
}

// The device-loss dump is the freeze dump's shape: the first failure, the memory figures, the thread dump and the
// dispatch ring (flushed before teardown), all read from the same composers the watchdog uses.
NXTEST(device_lost_dump_carries_every_section) {
    const uint64_t total = feed_dispatches(3);
    const std::string text = nxrt::diag_device_lost_dump("vkQueueSubmit2", -4, 5, "");
    CHECK_MSG(has(text, "GPU DEVICE LOST (reason: VK_ERROR_DEVICE_LOST)"), "the device-loss header missing");
    CHECK_MSG(has(text, "vkQueueSubmit2 returned -4 after 5 presented frames"), "what failed missing");
    CHECK_MSG(has(text, "GPU first failure"), "the first-failure section missing");
    CHECK_MSG(has(text, "memory at device loss"), "the memory section missing");
    CHECK_MSG(has(text, "threads:"), "the thread dump missing");
    CHECK_MSG(has(text, "recent dispatches"), "the dispatch ring missing");
    char b[64];
    std::snprintf(b, sizeof b, "#%llu", static_cast<unsigned long long>(total));
    CHECK_MSG(has(text, b), "the newest dispatch missing from the device-loss dump");
}

// The synthetic freeze dump: [diag] synthetic_dump writes the report without a real freeze, so the shape can be
// produced (and read) on a machine where nothing has failed. The env keys are cleared afterwards.
NXTEST(synthetic_freeze_dump_writes_every_section) {
    const std::string rep = temp_file("synthetic");
    std::remove(rep.c_str());
    SetEnvironmentVariableA("NXRT_DIAG_SYNTHETIC_DUMP", "freeze");
    SetEnvironmentVariableA("NXRT_DIAG_SYNTHETIC_DUMP_FRAMES", "1");
    const uint64_t total = feed_dispatches(3);
    nxrt::watchdog_start({.seconds = 5.0, .report_path = rep.c_str(), .poll_ms = 50});
    nxrt::diag_frame_tick();
    nxrt::diag_frame_tick();
    const bool ok = wait_for([&] { return read_file(rep).find("SYNTHETIC FREEZE") != std::string::npos; }, 2000);
    nxrt::watchdog_stop();
    SetEnvironmentVariableA("NXRT_DIAG_SYNTHETIC_DUMP", nullptr);
    SetEnvironmentVariableA("NXRT_DIAG_SYNTHETIC_DUMP_FRAMES", nullptr);
    CHECK_MSG(ok, "the synthetic freeze dump was not written");
    const std::string text = read_file(rep);
    CHECK_MSG(has(text, "SYNTHETIC FREEZE (asked for with [diag] synthetic_dump"), "the synthetic header missing");
    CHECK_MSG(has(text, "GPU first failure"), "the first-failure section missing");
    CHECK_MSG(has(text, "memory at freeze"), "the memory section missing");
    CHECK_MSG(has(text, "guest sync waits"), "the guest sync wait section missing");
    CHECK_MSG(has(text, "recent dispatches"), "the dispatch ring missing");
    char b[64];
    std::snprintf(b, sizeof b, "#%llu", static_cast<unsigned long long>(total));
    CHECK_MSG(has(text, b), "the newest dispatch missing from the freeze dump");
}
