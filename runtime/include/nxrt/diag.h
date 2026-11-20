// SPDX-License-Identifier: GPL-3.0-or-later
// Diagnostics: crash report ("HOST EXCEPTION"), thread registry, recent-file ring, freeze watchdog.
// Ideas from smo-recomp (runtime/core/hostrt.c crash_filter, docs/debugging.md), SuperMarioWonderRecomp
// (kernel_watch.cpp hang report, hle_fs.cpp recent files) and ssbu-recomp (tests/DEBUG_MODE.md).
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "nxrt/abi.h"

namespace nxrt {

// ---- thread registry ------------------------------------------------------------------------------------------------
// Every host thread that runs guest code (and the main thread) registers itself so that the crash handler can print its
// Ctx and the watchdog can dump its stack. The registry is independent of the kernel's thread objects.
struct ThreadRecord {
    std::string name;
    uint32_t host_tid = 0;        // GetCurrentThreadId
    void* host_handle = nullptr;  // a real HANDLE opened with THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION
    ::Ctx* ctx = nullptr;         // guest registers of the thread (may be null for a host-only thread)
    uint64_t guest_tid = 0;       // kernel thread id (0 = none)
};
// Registers the calling thread (duplicates the pseudo handle into a real one). Sets the thread-local "current Ctx" used by
// the crash handler. Returns the record index (handle for diag_unregister_thread).
size_t diag_register_thread(const char* name, ::Ctx* ctx, uint64_t guest_tid = 0);
void diag_unregister_thread();                 // the calling thread; closes its handle
void diag_set_thread_ctx(::Ctx* ctx);          // the Ctx now running on this thread (call_guest nesting restores the old one)
::Ctx* diag_thread_ctx();                      // thread-local current Ctx or nullptr
std::vector<ThreadRecord> diag_threads();      // snapshot of the registry (names, tids, Ctx only: host_handle is always null; use diag_threads_dup for a usable handle)
// Like diag_threads, but every host_handle is a private duplicate made under the registry lock, so a thread that unregisters meanwhile cannot
// close it under the caller. The caller must pass the vector to diag_threads_release when done.
std::vector<ThreadRecord> diag_threads_dup();
void diag_threads_release(std::vector<ThreadRecord>& recs);   // closes the duplicates and clears the vector

// ---- recent files ------------------------------------------------------------------------------------------------------
// A ring of the last 32 file/dir opens of any thread, printed by the crash handler: what the game was loading.
void diag_note_file(const char* what, const char* path);       // what: "open", "dir", "read", ...
std::vector<std::string> diag_recent_files();                  // oldest first, "[tid] what path"
void diag_clear_recent_files();

// ---- first VkResult failure ----------------------------------------------------------------------------------------------
// A crash or device-loss report otherwise names whichever call lost the race, not the first one that failed: after the GPU goes away
// every call fails, and the reporter's file ends up describing a consequence (a fence wait 17 ms after the batch submit that really
// failed, #81). The first non-success VkResult anywhere in the process is recorded here once, with the call and its site, the thread,
// how far into the run it happened, the frames presented and the submit state at that moment, and is never replaced by a later
// failure. Results a healthy run produces on purpose (VK_NOT_READY, VK_TIMEOUT, VK_INCOMPLETE, VK_SUBOPTIMAL_KHR,
// VK_ERROR_OUT_OF_DATE_KHR) are not failures and are not recorded: a real device loss would otherwise be reported as a timeout or a
// resize. Every call site that can fail with a device-level result wraps the call in NXRT_VK(), which costs one integer compare on
// the success path (no lock, no allocation, nothing logged) and records the result only when it is not VK_SUCCESS.
void diag_vk_note(int result, const char* call, const char* file, int line);
template <class R>
inline R diag_vk_check(R result, const char* call, const char* file, int line) {
    if (static_cast<int>(result) != 0) diag_vk_note(static_cast<int>(result), call, file, line);
    return result;
}
// NXRT_VK(vkSomething(...)): the call, with its result recorded if it fails. Works for any enum-shaped result (VK_SUCCESS is 0).
#define NXRT_VK(call) (::nxrt::diag_vk_check((call), #call, __FILE__, __LINE__))
// What the NVN back end reports about its submissions (last submitted serial, timeline value, submits, presents), asked once when the
// first failure is recorded and in every dump: the generic diagnostic layer does not know those counters.
void diag_set_submit_state(std::string (*fn)());
// The record as report text (a header line and its detail), "(none)" when no Vulkan call has failed: the top of a crash report.
std::string diag_vk_failure_text();

// ---- crash handler -------------------------------------------------------------------------------------------------------
struct CrashOptions {
    const char* report_path = nullptr;   // nullptr: "<exe name>_crash.txt" next to the exe; "" = no file
    bool terminate = true;               // after reporting: ExitProcess(exit_code) via TerminateProcess; false in tests
    uint32_t exit_code = 4;
    // Called with the finished report text before termination (tests). May be null.
    void (*on_report)(const char* text, void* user) = nullptr;
    void* user = nullptr;
};
// Installs SetUnhandledExceptionFilter (+ a vectored handler for stack overflow and guest traps 0xE0000001/2). Idempotent.
void crash_install(const CrashOptions& opt = {});
void crash_uninstall();
// Builds the report text for an exception without terminating (also used by the self-test).
// Layout (every section is always present, with "(none)" when it has nothing):
//   HOST EXCEPTION 0x<code> at host address 0x<addr> on thread <name> (tid N)
//     in <module> at offset 0x..        (symbolised through symbols.h: "module+0x.. (symbol+0x..)")
//     reading|writing|executing address 0x...        (access violations only)
//   last guest function entered: f_<hex>             (guest_function_of(rip), else Ctx-based when the ABI has one)
//   guest registers: x0..x30, sp, pc-ish, tpidr, tpidrro   (from the thread's registered Ctx)
//   guest frame chain (x29): lr=<hex> <- <hex> ...
//   host call stack (innermost first): module+0xOFFSET (symbol+0x..)   up to 32 frames, via RtlVirtualUnwind
//   guest call stack (innermost first): f_<hex> ...        (lifted functions found on the host stack)
//   threads: one line per registered thread (name, guest tid, lr)
//   last files opened (oldest first): ...
std::string crash_build_report(void* exception_pointers /* EXCEPTION_POINTERS* */);
// The file the crash report goes to: the path given to crash_install, else "<exe name>_crash.txt" next to the exe ("" = no file). The memory guard
// writes its report there too.
std::string crash_report_path();
// A known problem found before the crash (an incomplete game dump, #64): the report starts with "LIKELY CAUSE: <text>", which the
// launcher shows instead of the bare exit code. "" clears it.
void crash_set_hint(const char* text);
// Number of exceptions the installed handler has turned into reports (tests).
uint32_t crash_exceptions_reported();

// ---- sampling profiler (S34) -----------------------------------------------------------------------------------------
// `[prof] sample_hz` > 0 starts a thread that samples every registered thread's host stack, weighted by the CPU cycles the thread used; the
// folded stacks are written at shutdown to `[prof] sample_out`. The label (a scene name set by the test script) is the first field of each stack.
void prof_sampler_start();
void prof_sampler_stop();
void prof_sampler_label(const char* label);

// ---- freeze watchdog ------------------------------------------------------------------------------------------------------
// A thread checks every poll_ms whether diag_frame_tick() advanced; after `seconds` without a tick it dumps every registered
// thread's host stack (SuspendThread, capture RIP/stack through RtlVirtualUnwind into a fixed buffer, ResumeThread: no
// allocation while a thread is suspended), symbolised, plus guest lr / x29 chain, to the log and to the report file, then calls
// on_fire. It fires once per freeze (re-arms after the next tick).
struct WatchdogOptions {
    double seconds = 0;                  // 0 = [diag] watchdog_seconds (env NXRT_DIAG_WATCHDOG_SECONDS), default 10; < 0 = disabled
    const char* report_path = nullptr;   // nullptr: "<exe name>_freeze.txt" next to the exe; "" = no file
    void (*on_fire)(const char* dump, void* user) = nullptr;
    void* user = nullptr;
    uint32_t poll_ms = 250;
};
void watchdog_start(const WatchdogOptions& opt = {});
void watchdog_stop();
bool watchdog_running();
void diag_progress();                  // a heartbeat that is not a frame: long work that is moving (texture transcodes on a cold cache) keeps the watchdog quiet
void diag_frame_tick();                // call once per presented frame (or any heartbeat)
void diag_frame_tick_nolisten();       // the same without calling the frame listener (the test tooling's own boundary)
void diag_set_frame_listener(void (*fn)());   // called by diag_frame_tick after counting (test tooling's frame source "tick"); null removes it
uint64_t diag_frame_count();
uint32_t watchdog_fire_count();
// Extra sections of the freeze dump: each function returns text (it must never block: the process may be frozen). Registered once at start-up.
void diag_add_report(std::string (*fn)());
std::string diag_thread_dump();        // the dump text the watchdog produces, on demand (also used by the SIGUSR-style "dump_threads" file)
// The dump's own sections, for a report that is not the watchdog's: `when` labels the memory section ("freeze", "device loss", "crash").
std::string diag_memory_text(const char* when);
// Every section of a dump in one place, so the freeze report and the device-loss report have one shape: the header, the first Vulkan
// failure, the memory figures, the thread dump, the guest sync waits and every registered extra section (the NVN back end's state,
// the recent dispatches). Never blocks; the caller logs and writes the text.
std::string diag_full_dump(const char* header, const char* when);
// The device-loss report: what failed, then what the renderer knows about the GPU and the presentation (`context`, filled in by the
// gfx layer at swapchain creation), then the advice, then every section of a dump.
std::string diag_device_lost_dump(const char* where, int result, uint64_t frames, const std::string& context);


// One recent dispatch of the render thread: what a dispatch that hung the GPU looked like. Filled by the NVN back end's do_dispatch
// (nvn_replay.cpp) into a fixed ring (gpu_diag.cpp) that the freeze and device-loss dumps flush; the frame trace cannot be aimed at a
// frame that never presents, which is exactly the failure mode. The note is a plain copy, no allocation, no lock.
struct DispatchNote {
    static constexpr uint16_t kIndirect = 1, kMissingCs = 2, kBarrierBefore = 4;
    char shader[17] = {};      // the compute shader's hash (16 hex chars), "" when the shader had no translation
    char missing[17] = {};     // with kMissingCs: the cache keys SetShaders found no translation for
    uint32_t x = 0, y = 0, z = 0;
    uint64_t indirect = 0;     // the indirect arguments buffer address (0: a direct dispatch)
    uint64_t serial = 0;       // submission serial (0: unknown / test)
    uint32_t bindings = 0;     // buffer and image bindings of the dispatch's layout
    uint32_t barriers = 0;     // full barriers recorded in the command buffer since the previous dispatch
    uint16_t flags = 0;
    uint64_t index = 0;        // dispatch number since start (the dump shows "last N of M")
};
void diag_note_dispatch(const DispatchNote& note);
// The ring as report text: "recent dispatches (last N of M, oldest first)", one line per dispatch, oldest first. Registered as an
// extra report section on the first note, so the freeze and device-loss dumps flush it too.
std::string diag_dispatch_text();

}  // namespace nxrt
