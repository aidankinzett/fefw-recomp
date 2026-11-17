// SPDX-License-Identifier: GPL-3.0-or-later
// Fire Emblem: Fortune's Weave: pointing at menu entries with the mouse ([fefw] menu_pointer, on by default).
//
// The game's mouse controls move a menu's selection by the mouse's motion, one entry per flick past a threshold (the mouse system
// [0x22983a8], 0x71015860e0: the motion piles up at +0x244 / +0x248 and a step up, down, left or right is set at +0x24c once it passes
// +0x230 times the situation's scale), and a left click picks the selected entry. Nothing in the game reads where the pointer is in a
// menu. This file makes the entry under the Windows pointer the selected one.
//
// Every frame it collects the nn::ui2d panes the game calculates (Picture::Calculate 0x595740 and Window::Calculate 0x5af080 have inline
// sites after their LoadMtx, x19 / x20 = the pane; fefw_wide.cpp). Pane fields: +0x50 the size as laid out, +0x70 the global matrix (3 x 4
// row major; its translation is the pane's centre in 1920 x 1080 layout units, origin at the screen's centre, y up), +0xb0 the name.
//  - The selection cursor is the feather the menus draw at the selected entry: a pane named "Cursor" (62 x 62) in a layout of its own
//    (L_Cursor), placed by the menu's code. It points with its tip, the pane's lower right corner: level with a title menu entry's centre,
//    at the top left corner of a yes / no button.
//  - The selected entry is the pane at the tip whose kind (name and laid-out size) has siblings at other places: a list ("Select" 596 x
//    204 for the title menu, whose panes overlap; "Wdw" 352 x 74 for unit commands; "Wdw" 336 x 48 and "Wdw_Overlay" 420 x 60 in dialogs).
//    The selected entry's highlight animates its scale, so kinds compare the laid-out size. Helper panes stacked on one spot ("Bind") are
//    no list. The kind chosen before wins while it is still at the tip, so overlapping layouts do not take turns.
//  - The entry under the pointer is the sibling it is in (the nearest centre when several overlap).
// While the two differ it sends one flick of Joy-Con 2 mouse motion toward it (hid::jc2_mouse_inject) once the feather stands still, and
// waits a few frames; the PC mouse's own motion is held back meanwhile (fefw_game.cpp), so the selection never wanders off the pointer. A
// left click waits until the selection has reached the entry under the pointer (hid::jc2_mouse_defer_clicks), then picks it as the game's
// own click does. A menu with one entry needs nothing: the game's click picks it.
#include <algorithm>
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
#include "nxrt/jc2_mouse.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"

namespace nxrt::games::fefw {

namespace {

constexpr const char* kTag = "fefw-menu";
constexpr uint64_t kPictureCalc = 0x595784;
constexpr uint64_t kWindowCalc = 0x5af0b8;

struct Pane {
    char name[25] = {};
    float x = 0, y = 0, w = 0, h = 0;      // centre and size in layout units (scale applied)
    float rw = 0, rh = 0;                  // the size as laid out (the selected entry's highlight animates its scale)
    bool window = false;
    bool contains(float px, float py) const { return std::fabs(px - x) <= w * 0.5f && std::fabs(py - y) <= h * 0.5f; }
};

struct State {
    bool on = false, log = false;
    float flick = 600.0f;                   // Joy-Con 2 mouse motion of one flick ([fefw] menu_pointer_flick)
    int gap = 4;                            // frames between flicks
    std::mutex mu;
    std::vector<Pane> frame;
    int cooldown = 0;
    float last_cx = 1e9f, last_cy = 1e9f;   // the cursor pane last frame: it slides between entries, and a flick waits until it stands
    Pane kind;                              // the kind of entry chosen last (name and size): kept while it stays beside the cursor
    bool have_kind = false;
    bool pending = false;                   // the entry under the pointer is not the selected one yet
    float hov_x = 1e9f, hov_y = 1e9f;       // the entry flicked toward, where the selection was then, and flicks in a row that moved nothing
    float from_x = 1e9f, from_y = 1e9f;
    int fruitless = 0;
    uint64_t frames = 0, flicks = 0;
    bool was_active = false;
    char last_group[25] = {};
};

State& st() {
    static State s;
    return s;
}

void note(uint64_t pane, bool window) {
    if (!mem::is_mapped(pane, 0xd0)) return;
    Pane p;
    std::memcpy(p.name, reinterpret_cast<const void*>(pane + 0xb0), 24);
    p.name[24] = 0;
    float size[2], m[12];
    std::memcpy(size, reinterpret_cast<const void*>(pane + 0x50), 8);
    std::memcpy(m, reinterpret_cast<const void*>(pane + 0x70), sizeof m);
    p.x = m[3];
    p.y = m[7];
    p.rw = size[0];
    p.rh = size[1];
    p.w = size[0] * std::fabs(m[0]);
    p.h = size[1] * std::fabs(m[5]);
    p.window = window;
    if (!(p.w > 1 && p.h > 1) || !std::isfinite(p.x) || !std::isfinite(p.y)) return;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.frame.size() < 2048) s.frame.push_back(p);
}

void hook_picture(::Ctx* c, uint64_t, void*) { note(c->x[19], false); }
void hook_window(::Ctx* c, uint64_t, void*) { note(c->x[20], true); }

bool same_kind(const Pane& a, const Pane& b) {
    return std::strcmp(a.name, b.name) == 0 && std::fabs(a.rw - b.rw) < 1.0f && std::fabs(a.rh - b.rh) < 1.0f;
}

}  // namespace

bool menumouse_pending() { return st().pending; }

void menumouse_install(uint64_t base) {
    State& s = st();
    s.on = config_get_bool("fefw", "menu_pointer", true);
    s.log = config_get_bool("fefw", "map_mouse_log", false);
    s.flick = static_cast<float>(std::clamp(config_get_double("fefw", "menu_pointer_flick", 600.0), 50.0, 5000.0));
    if (!s.on) return;
    hook_add_inline(base + kPictureCalc, &hook_picture);
    hook_add_inline(base + kWindowCalc, &hook_window);
    NXLOG_I(kTag, "menu entries follow the pointer: on (flick %.0f)", static_cast<double>(s.flick));
}

// Main thread, once per presented frame. `map_busy`: the map cursor has the pointer this frame. Returns true while a menu has it (the
// caller then keeps the mouse uncaptured and holds its motion back).
bool menumouse_frame(bool map_busy) {
    State& s = st();
    if (!s.on) return false;
    ++s.frames;
    std::vector<Pane> f;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        f.swap(s.frame);
    }
    if (s.cooldown > 0) --s.cooldown;
    s.pending = false;
    const MouseInfo mi = mouse_info();
    float u = 0, v = 0;
    if (map_busy || !mi.valid || !mi.mode || wide_info().active || !mapmouse_pointer(&u, &v)) {
        s.was_active = false;
        return false;
    }
    // the selection cursor: the last "Cursor" pane calculated (dialogs are calculated over the screens beneath them)
    const Pane* cur = nullptr;
    for (const Pane& p : f)
        if (std::strcmp(p.name, "Cursor") == 0) cur = &p;
    if (!cur) {
        s.was_active = false;
        s.have_kind = false;
        return false;
    }
    const float px = (u - 0.5f) * 1920.0f, py = (0.5f - v) * 1080.0f;
    // The feather points at the selected entry with its tip, the cursor pane's lower right corner (its own layout L_Cursor, placed by the
    // menu's code: at the top left of a yes / no button, level with a title menu entry's centre).
    const float tx = cur->x + cur->w * 0.5f, ty = cur->y - cur->h * 0.5f;
    // The selected entry: a pane at the tip (in it, or within a few units) whose kind (name and laid-out size) has siblings at other
    // places. The kind chosen before wins while it is still at the tip; then the pane the tip is in (or nearest to), then the pane whose
    // centre is most nearly level with the tip (overlapping list panes: the title menu's "Select" are 204 high, 94 apart), then the longer
    // list, then the larger pane.
    constexpr float kSlack = 16.0f;
    const Pane* sel = nullptr;
    int sel_members = 0;
    float sel_out = 1e30f, sel_level = 1e30f;
    for (const Pane& p : f) {
        if (&p == cur || std::strncmp(p.name, "Cursor", 6) == 0 || p.rw >= 1500 || p.rh >= 900) continue;
        const float ex = std::max(0.0f, std::fabs(tx - p.x) - p.w * 0.5f), ey = std::max(0.0f, std::fabs(ty - p.y) - p.h * 0.5f);
        if (ex > kSlack || ey > kSlack) continue;
        // siblings at other places: a list, not panes stacked on one spot (helper panes such as "Bind" all sit at the centre)
        int members = 0;
        float x0 = p.x, x1 = p.x, y0 = p.y, y1 = p.y;
        for (const Pane& q : f) {
            if (!same_kind(p, q)) continue;
            ++members;
            x0 = std::min(x0, q.x);
            x1 = std::max(x1, q.x);
            y0 = std::min(y0, q.y);
            y1 = std::max(y1, q.y);
        }
        if (members < 2 || (x1 - x0 < 8.0f && y1 - y0 < 8.0f)) continue;
        const float out = std::sqrt(ex * ex + ey * ey), level = std::fabs(ty - p.y) + 0.1f * std::fabs(tx - p.x);
        const bool sticky = s.have_kind && same_kind(p, s.kind), sel_sticky = sel && s.have_kind && same_kind(*sel, s.kind);
        bool take = !sel;
        if (!take && sticky != sel_sticky) take = sticky;
        else if (!take && std::fabs(out - sel_out) > 2.0f) take = out < sel_out;
        else if (!take && std::fabs(level - sel_level) > 2.0f) take = level < sel_level;
        else if (!take) take = members > sel_members || (members == sel_members && p.rw * p.rh > sel->rw * sel->rh);
        if (take) {
            sel = &p;
            sel_members = members;
            sel_out = out;
            sel_level = level;
        }
    }
    if (!sel) {
        s.was_active = false;
        s.have_kind = false;
        return false;
    }
    s.kind = *sel;
    s.have_kind = true;
    // the member under the pointer: the nearest centre among those it is in (a list's panes may overlap: the title menu's "Select" are
    // 204 high, 94 apart)
    const Pane* hov = nullptr;
    float hov_d = 1e30f;
    for (const Pane& p : f) {
        if (!same_kind(p, *sel) || !p.contains(px, py)) continue;
        const float d = (p.x - px) * (p.x - px) + (p.y - py) * (p.y - py);
        if (d < hov_d) {
            hov = &p;
            hov_d = d;
        }
    }
    const bool active = true;      // a menu with entries is up: the pointer owns its selection
    if (s.log && (!s.was_active || std::strcmp(s.last_group, sel->name) != 0)) {
        NXLOG_I(kTag, "menu: %d entries of \"%s\" (%.0fx%.0f), cursor at %.0f,%.0f", sel_members, sel->name, static_cast<double>(sel->w),
                static_cast<double>(sel->h), static_cast<double>(cur->x), static_cast<double>(cur->y));
        std::strncpy(s.last_group, sel->name, 24);
    }
    s.was_active = active;
    // where the Windows pointer comes back if the mouse was captured (a camera scene before the menu): at the selected entry
    hid::jc2_mouse_set_return_point(std::clamp(sel->x / 1920.0f + 0.5f, 0.0f, 1.0f), std::clamp(0.5f - sel->y / 1080.0f, 0.0f, 1.0f));
    // a captured mouse (the request frees it next frame) leaves only where the pointer was: nothing is pointed at yet
    if (!mapmouse_pointer_live()) hov = nullptr;
    // a menu whose selection does not step with the mouse's motion (six flicks moved nothing): left to the game until another entry is
    // pointed at
    if (hov && (std::fabs(hov->x - s.hov_x) >= 1 || std::fabs(hov->y - s.hov_y) >= 1)) s.fruitless = 0;
    if (hov && s.fruitless >= 6) hov = nullptr;
    s.pending = hov && !(std::fabs(hov->x - sel->x) < 1 && std::fabs(hov->y - sel->y) < 1);   // a click now would pick the wrong entry
    const bool still = std::fabs(cur->x - s.last_cx) < 1.5f && std::fabs(cur->y - s.last_cy) < 1.5f;
    s.last_cx = cur->x;
    s.last_cy = cur->y;
    if (hov && hov != sel && !(std::fabs(hov->x - sel->x) < 1 && std::fabs(hov->y - sel->y) < 1) && s.cooldown == 0 && still) {
        const float dy = sel->y - hov->y, dx = hov->x - sel->x;      // layout y is up, the mouse's y is down
        if (std::fabs(dy) >= std::fabs(dx))
            hid::jc2_mouse_inject(0, dy > 0 ? s.flick : -s.flick);
        else
            hid::jc2_mouse_inject(dx > 0 ? s.flick : -s.flick, 0);
        s.cooldown = s.gap;
        ++s.flicks;
        const bool same = std::fabs(hov->x - s.hov_x) < 1 && std::fabs(hov->y - s.hov_y) < 1;
        s.fruitless = same && std::fabs(sel->x - s.from_x) < 1 && std::fabs(sel->y - s.from_y) < 1 ? s.fruitless + 1 : 0;
        s.hov_x = hov->x;
        s.hov_y = hov->y;
        s.from_x = sel->x;
        s.from_y = sel->y;
        if (s.fruitless >= 6 && s.log) NXLOG_I(kTag, "the selection does not follow; given up until another entry is pointed at");
        if (s.log) {
            std::string cand;
            for (const Pane& p : f) {
                if (!same_kind(p, *sel)) continue;
                char item[96];
                std::snprintf(item, sizeof item, " %s %.0fx%.0f(%.0fx%.0f)@%.0f,%.0f;", p.name, static_cast<double>(p.w), static_cast<double>(p.h),
                              static_cast<double>(p.rw), static_cast<double>(p.rh), static_cast<double>(p.x), static_cast<double>(p.y));
                cand += item;
            }
            NXLOG_I(kTag, "flick toward %.0f,%.0f from %s %.0f,%.0f (%llu flicks); cursor %.0f,%.0f %.0fx%.0f tip %.0f,%.0f; list:%s",
                    static_cast<double>(hov->x), static_cast<double>(hov->y), sel->name, static_cast<double>(sel->x), static_cast<double>(sel->y),
                    static_cast<unsigned long long>(s.flicks), static_cast<double>(cur->x), static_cast<double>(cur->y), static_cast<double>(cur->w),
                    static_cast<double>(cur->h), static_cast<double>(tx), static_cast<double>(ty), cand.c_str());
        }
    }
    return active;
}

}  // namespace nxrt::games::fefw
