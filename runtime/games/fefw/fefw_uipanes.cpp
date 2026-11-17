// SPDX-License-Identifier: GPL-3.0-or-later
// Fortune's Weave, debug: a census of the nn::ui2d panes the game calculates ([fefw] ui_pane_log), for finding the panes of a menu
// (fefw_menumouse.cpp). Picture::Calculate (0x595740) and Window::Calculate (0x5af080) have inline sites after their LoadMtx
// (fefw_wide.cpp): x19 / x20 is the pane. Each time the number of panes calculated in a frame changes (another screen, a menu opened),
// every pane is logged: its name (+0xb0, 24 chars), its size as laid out (+0x50), its size on screen when a scale changes it, and its
// centre (the global matrix at +0x70, 3 x 4 row major: 1920 x 1080 layout units, origin at the screen's centre, y up). "W:" marks a
// window pane. The menus' selection cursor is a pane named "Cursor" in a layout of its own (L_Cursor); a pane's parent is at +0x18.
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "fefw_game.h"
#include "nxrt/config.h"
#include "nxrt/hooks.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"

namespace nxrt::games::fefw {

namespace {

constexpr const char* kTag = "fefw-ui";
constexpr uint64_t kPictureCalc = 0x595784;
constexpr uint64_t kWindowCalc = 0x5af0b8;

struct Seen {
    char name[25] = {};
    bool window = false;
    float rw = 0, rh = 0, w = 0, h = 0, x = 0, y = 0;
};

struct State {
    bool on = false;
    std::mutex mu;
    std::vector<Seen> frame;
    uint64_t frames = 0;
    size_t last_count = 0;
};

State& st() {
    static State s;
    return s;
}

void note(uint64_t pane, bool window) {
    if (!mem::is_mapped(pane, 0xd0)) return;
    Seen e;
    e.window = window;
    std::memcpy(e.name, reinterpret_cast<const void*>(pane + 0xb0), 24);
    for (char& ch : e.name)
        if (ch && !std::isprint(static_cast<unsigned char>(ch))) ch = '?';
    float size[2], m[12];
    std::memcpy(size, reinterpret_cast<const void*>(pane + 0x50), 8);
    std::memcpy(m, reinterpret_cast<const void*>(pane + 0x70), sizeof m);
    e.rw = size[0];
    e.rh = size[1];
    e.w = size[0] * std::fabs(m[0]);
    e.h = size[1] * std::fabs(m[5]);
    e.x = m[3];
    e.y = m[7];
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.frame.size() < 4096) s.frame.push_back(e);
}

void hook_picture(::Ctx* c, uint64_t, void*) { note(c->x[19], false); }
void hook_window(::Ctx* c, uint64_t, void*) { note(c->x[20], true); }

}  // namespace

void uipanes_install(uint64_t base) {
    State& s = st();
    s.on = config_get_bool("fefw", "ui_pane_log", false);
    if (!s.on) return;
    hook_add_inline(base + kPictureCalc, &hook_picture);
    hook_add_inline(base + kWindowCalc, &hook_window);
    NXLOG_I(kTag, "pane census on");
}

void uipanes_frame() {
    State& s = st();
    if (!s.on) return;
    ++s.frames;
    std::vector<Seen> f;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        f.swap(s.frame);
    }
    if (f.size() == s.last_count) return;
    s.last_count = f.size();
    std::string list;
    for (const Seen& e : f) {
        char item[128];
        if (std::fabs(e.w - e.rw) < 0.5f && std::fabs(e.h - e.rh) < 0.5f)
            std::snprintf(item, sizeof item, " %s%s %.0fx%.0f@%.0f,%.0f;", e.window ? "W:" : "", e.name, static_cast<double>(e.rw),
                          static_cast<double>(e.rh), static_cast<double>(e.x), static_cast<double>(e.y));
        else
            std::snprintf(item, sizeof item, " %s%s %.0fx%.0f(%.0fx%.0f)@%.0f,%.0f;", e.window ? "W:" : "", e.name, static_cast<double>(e.rw),
                          static_cast<double>(e.rh), static_cast<double>(e.w), static_cast<double>(e.h), static_cast<double>(e.x),
                          static_cast<double>(e.y));
        list += item;
    }
    NXLOG_I(kTag, "frame %llu: %zu panes:%s", static_cast<unsigned long long>(s.frames), f.size(), list.c_str());
}

}  // namespace nxrt::games::fefw
