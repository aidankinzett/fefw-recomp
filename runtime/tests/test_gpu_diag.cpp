// Tests for the GPU diagnostics: the first Vulkan failure record, the recent-dispatch ring and the sections the freeze and
// device-loss dumps share. The ring and the failure record are process-wide, so the counts are compared as deltas; the
// synthetic freeze dump needs the watchdog, and every case stops the watchdog it starts.
#include "test_framework.h"

#include "nxrt/diag.h"
#include "nxrt/config.h"

#include <windows.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

NXTEST_GROUP(gpu_diag)

using namespace std::chrono;

namespace {

std::string temp_file(const char* tag) {
    char dir[MAX_PATH + 1];
    const DWORD n = GetTempPathA(MAX_PATH, dir);
    std::string path(dir, n);
    path += "nxrt_gd_" + std::string(tag) + ".txt";
    return path;
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

// A plausible stream of dispatches: one every 3 barriers, then an indirect one that hangs. Returns the number fed (the
// process-wide ring counters are not readable from here, so the cases compare as deltas and look for the newest index
// relative to the count this case fed).
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
    return static_cast<uint64_t>(count) + 1;
}

}  // namespace

// Results a healthy run produces on purpose (a resize, a timeout slice, a full enumeration array) do not claim the
// record: a real device loss would otherwise be reported as a timeout or a resize. This case runs first, so the
// record it leaves behind proves the exclusion: a broken exclusion would have claimed the record with the resize.
NXTEST(expected_results_do_not_claim_the_record) {
    nxrt::diag_vk_note(-1000001004, "vkAcquireNextImageKHR", __FILE__, __LINE__);   // VK_ERROR_OUT_OF_DATE_KHR: a resize
    nxrt::diag_vk_note(2, "wait_serial", __FILE__, __LINE__);                       // VK_TIMEOUT: a retry slice
    nxrt::diag_vk_note(1000001003, "vkQueuePresentKHR", __FILE__, __LINE__);        // VK_SUBOPTIMAL_KHR
    nxrt::diag_vk_note(-4, "vkQueueSubmit2", __FILE__, __LINE__);
    const std::string text = nxrt::diag_vk_failure_text();
    CHECK_MSG(has(text, "vkQueueSubmit2"), "the real failure missing from the record");
    CHECK_MSG(has(text, "VK_ERROR_DEVICE_LOST"), "the real failure's result missing from the record");
    CHECK_MSG(!has(text, "vkAcquireNextImageKHR"), "a resize claimed the record");
    CHECK_MSG(!has(text, "wait_serial"), "a timeout slice claimed the record");
    CHECK_MSG(!has(text, "vkQueuePresentKHR"), "a suboptimal presentation claimed the record");
}

// The first failure anywhere in the process wins and is never overwritten: the record made by the losing call site
// (a follow-on vkAllocateMemory) does not displace the record made by the winning one (the batch submit). The record
// was already claimed by the case above, so these notes only test that they change nothing.
NXTEST(first_failure_is_not_displaced) {
    nxrt::diag_vk_note(-4, "vkQueueSubmit2", __FILE__, __LINE__);
    nxrt::diag_vk_note(-1000069000, "vkAllocateMemory", __FILE__, __LINE__);
    const std::string text = nxrt::diag_vk_failure_text();
    CHECK_MSG(has(text, "GPU first failure"), "first-failure section missing");
    CHECK_MSG(has(text, "vkQueueSubmit2"), "the winning call site missing from the record");
    CHECK_MSG(has(text, "VK_ERROR_DEVICE_LOST"), "the winning result missing from the record");
    CHECK_MSG(!has(text, "vkAllocateMemory"), "a later failure displaced the record");
}

// The ring always keeps the last 64: after 70 more dispatches the dump shows the last 64 of the total, oldest first,
// with the newest (the one that hung) at the end.
NXTEST(dispatch_ring_flushes_last_64) {
    const uint64_t fed = feed_dispatches(70);
    const std::string text = nxrt::diag_dispatch_text();
    CHECK_MSG(has(text, "last 64 of "), "the ring did not show the last 64");
    CHECK_MSG(has(text, "7c11be0099a4d2f1"), "the hung compute shader missing from the dump");
    CHECK_MSG(has(text, "indirect args 0x1f2a9000"), "the indirect dispatch's arguments address missing");
    CHECK_MSG(!has(text, "0x0x0"), "an indirect dispatch printed as direct dimensions");
    const size_t newest = text.rfind("7c11be0099a4d2f1");
    const size_t oldest = text.find("  #");
    CHECK_MSG(newest != std::string::npos && oldest != std::string::npos && newest > oldest, "the newest dispatch not last");
    // The depth config picks how many a dump shows: 1 shows only the newest.
    nxrt::config_set("diag", "dispatch_ring", "1");
    const std::string one = nxrt::diag_dispatch_text();
    char b[128];
    std::snprintf(b, sizeof b, "last 1 of %llu", static_cast<unsigned long long>(fed));
    CHECK_MSG(has(one, b), "the depth config did not shrink the dump");
    CHECK_MSG(has(one, "7c11be0099a4d2f1"), "the newest dispatch missing at depth 1");
    CHECK_MSG(has(one, "  #"), "the newest dispatch line missing at depth 1");
    nxrt::config_set("diag", "dispatch_ring", "64");
}

// The device-loss dump is the freeze dump's shape: the first failure, the memory figures, the thread dump and the
// dispatch ring (flushed before teardown), all read from the same composers the watchdog uses.
NXTEST(device_lost_dump_carries_every_section) {
    nxrt::diag_register_thread("gd-worker", nullptr, 1);
    feed_dispatches(3);
    const std::string text = nxrt::diag_device_lost_dump("vkQueueSubmit2", -4, 5, "");
    CHECK_MSG(has(text, "GPU DEVICE LOST"), "the device-loss header missing");
    CHECK_MSG(has(text, "GPU first failure"), "the first-failure section missing");
    CHECK_MSG(has(text, "memory at device loss"), "the memory section missing");
    CHECK_MSG(has(text, "thread gd-worker"), "the thread dump missing");
    CHECK_MSG(has(text, "kernel sync waiters"), "the guest sync wait section missing");
    CHECK_MSG(has(text, "recent dispatches"), "the dispatch ring missing");
    CHECK_MSG(has(text, "indirect args 0x1f2a9000"), "the newest indirect dispatch missing");
    nxrt::diag_unregister_thread();
}

// The synthetic freeze dump: [diag] synthetic_dump = freeze writes the report once, without a real freeze, and the run
// goes on. The env keys are set through the CRT (config reads std::getenv) and cleared afterwards.
NXTEST(synthetic_freeze_dump_writes_every_section) {
    const std::string rep = temp_file("synthetic");
    std::remove(rep.c_str());
    _putenv_s("NXRT_DIAG_SYNTHETIC_DUMP", "freeze");
    _putenv_s("NXRT_DIAG_SYNTHETIC_DUMP_FRAMES", "1");
    nxrt::diag_register_thread("gd-worker", nullptr, 1);
    feed_dispatches(3);
    nxrt::watchdog_start({.seconds = 5.0, .report_path = rep.c_str(), .poll_ms = 50});
    nxrt::diag_frame_tick();
    // The watchdog writes the dump from its own thread on the next poll: wait for the file before stopping it.
    bool wrote = false;
    for (int i = 0; i < 40 && !(wrote = GetFileAttributesA(rep.c_str()) != INVALID_FILE_ATTRIBUTES); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    nxrt::watchdog_stop();
    _putenv_s("NXRT_DIAG_SYNTHETIC_DUMP", "");
    _putenv_s("NXRT_DIAG_SYNTHETIC_DUMP_FRAMES", "");
    CHECK_MSG(wrote, "the synthetic freeze dump was not written");
    const std::string text = read_file(rep);
    CHECK_MSG(has(text, "SYNTHETIC FREEZE"), "the synthetic header missing");
    CHECK_MSG(has(text, "GPU first failure"), "the first-failure section missing");
    CHECK_MSG(has(text, "memory at freeze"), "the memory section missing");
    CHECK_MSG(has(text, "thread gd-worker"), "the thread dump missing");
    CHECK_MSG(has(text, "kernel sync waiters"), "the guest sync wait section missing");
    CHECK_MSG(has(text, "recent dispatches"), "the dispatch ring missing");
    CHECK_MSG(has(text, "indirect args 0x1f2a9000"), "the newest indirect dispatch missing");
    nxrt::diag_unregister_thread();
}
