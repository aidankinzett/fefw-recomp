// SPDX-License-Identifier: GPL-3.0-or-later
// FSR 3 frame generation ([dlss] frame_gen = fsr3, notes/framegen.md): the FidelityFX API 1.1.4 (amd_fidelityfx_vk.dll; found by
// ffx_files.h). Its swapchain replaces the renderer's: the swapchain functions of volk are pointed at
// wrappers when the device is made. It needs no hardware GPU scheduling (unlike DLSS-G).
#pragma once
#include <volk.h>

#include <cstdint>
#include <mutex>

namespace nxrt::gfx {
class Device;
}

namespace nxrt::extras {

bool ffx_fg_wanted();   // [dlss] frame_gen = fsr3 and amd_fidelityfx_vk.dll found (read once)
bool ffx_fg_files();    // amd_fidelityfx_vk.dll found (ffx_files.h), whatever [dlss] frame_gen says
// Device creation (vk_device.cpp): once the device and its first queue exist (before any swapchain), install with the three queues
// (present, image acquire, async compute) that follow the game's in its family.
void ffx_fg_install(gfx::Device& dev);
// Standalone mode ([dlss] frame_gen_standalone = 1, or automatic below four queues in the game's family): FFX's interpolation without its
// swapchain. At the end of the frame (backbuffer drawn, colour attachment layout) the renderer calls _generate, which records the dispatch;
// true: a generated frame is made, the submission must signal _main_semaphore and _present follows (before the real frame's present).
// graphics_queues: the queues in the game's family (0: not known yet; the answer then is only the setting). Standalone when the setting
// says so or the family has fewer than the four the FFX swapchain needs (the first answer is kept).
bool ffx_fg_standalone_wanted(uint32_t graphics_queues = 0);
void ffx_fg_install_standalone(gfx::Device& dev);
void ffx_fg_standalone_format(VkFormat format);
bool ffx_fg_standalone_generate(VkCommandBuffer cb, VkImage color, VkExtent2D extent);
VkSemaphore ffx_fg_standalone_main_semaphore();
// Copies the generated frame into another swapchain image and presents it (imgs / dones: the swapchain's images and their render-done
// semaphores); the caller holds the queue lock qm. True: it went out, and the real frame's present follows half a frame later:
//   _defer_present hands that present to a thread of its own (false: one is still pending, or not possible: then _wait_due, which lets go of
//   qm while it waits, and the caller presents itself). _take_result: a non-success result of the thread's presents since the last call.
//   _flush: before the swapchain is destroyed.
bool ffx_fg_standalone_present_generated(VkSwapchainKHR sc, const VkImage* imgs, const VkSemaphore* dones, uint32_t n, VkExtent2D ext, std::mutex& qm, VkQueue q);
bool ffx_fg_standalone_defer_present(VkSwapchainKHR sc, VkSemaphore wait, uint32_t index, VkQueue q, std::mutex& qm);
void ffx_fg_standalone_wait_due(std::mutex& qm);
int ffx_fg_standalone_take_result();
void ffx_fg_standalone_flush();

struct FfxFgFrame {
    VkImage depth = VK_NULL_HANDLE, motion = VK_NULL_HANDLE;   // render size, GENERAL, oriented as the backbuffer (framegen.cpp)
    VkExtent2D render = {};
    float jitter[2] = {}, mv_scale[2] = {};
    bool reset = false, depth_inverted = true, inputs = false, enable = false;
    VkImage hudless = VK_NULL_HANDLE;                           // backbuffer size and format, GENERAL; may be null
    VkExtent2D bb = {};
    int32_t rect[4] = {};
    float frame_ms = 16.7f;
};
// Render thread, before the present (cb: the present's command buffer, recording): configure and, with inputs, the prepare dispatch.
// False: FSR 3 frame generation is not running (no FFX swapchain).
bool ffx_fg_frame(VkCommandBuffer cb, const FfxFgFrame& f);
uint64_t ffx_fg_present_count();   // presents of the FFX swapchain so far (generated ones included)
// The window's swapchain is FFX's: the renderer hands its images to the present in SHADER_READ_ONLY_OPTIMAL, the layout FFX reads them in
// (VUID-vkCmdDraw-None-09600 with PRESENT_SRC_KHR).
bool ffx_fg_owns_swapchain();

}  // namespace nxrt::extras
