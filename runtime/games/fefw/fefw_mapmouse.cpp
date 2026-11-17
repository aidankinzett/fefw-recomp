// SPDX-License-Identifier: GPL-3.0-or-later
// Fire Emblem: Fortune's Weave: the map cursor follows the Windows pointer ([fefw] map_pointer, on by default).
//
// The game's own mouse controls (docs/hid_mouse_sw2.md) move the battle and world map cursor by the mouse's motion, like a stick: the
// faster the hand, the farther it goes, small motion falls under a dead band, and the cell under the pointer means nothing. On a PC the
// cell under the pointer is what a player expects. While the map cursor runs and the game is in its mouse controls, this file finds the
// cell under the Windows pointer and steers the game's cursor there with motion of its own (hid::jc2_mouse_inject); the PC mouse's own
// motion is held back meanwhile (hid::jc2_mouse_hold_motion) and the mouse is not captured, so the Windows pointer stays visible. The
// game still moves the cursor, scrolls its camera and snaps to the cell's centre itself; clicks and the wheel are untouched (left click =
// select, right click = back, as the game's mouse controls have them).
// The target is the cell under the pointer when the pointer last moved. The camera follows the cursor once it leaves the middle of the
// screen, so a cell picked near the edge slides away from under a still pointer; chasing it would scroll to the map's end. Instead the
// cursor stops on the cell that was picked, and each further move of the mouse picks the cell under it again (moving the mouse near the
// edge scrolls on). Between moves the game owns the cursor (it puts it on a unit at a turn's start); a cell the cursor cannot reach (it
// is held to a unit's range at times) is given up after a second and a half without progress.
//
// The map cursor is game::map::Cursor::Impl, its Tick is 0x735020 (entry hook in recomp/games/fefw/main.sites; x0 = the cursor):
//   +0x08 / +0x0c   the cursor's cell: (int)((world x + 1280) * 0.2) - 256 and the same for world z (cells of 5 world units)
//   +0x10           an interpolated position (vtable at +0x10, times at +0x18 / +0x1c); +0x40 .. +0x4c the current value while it moves
//   +0xa0 / +0xa4   what is left of the mouse's movement under the 0.2 dead band; +0xb0 / +0xb4 the movement the cursor takes this frame
// When the motion stops the game snaps the cursor to the centre of its cell, (cell + 256) * 5 + 2.5 - 1280.
// The camera: 0xa7f080 turns a world point (x0) into 1920 x 1080 UI units through a view matrix (x1) and a projection (x2), both 4 x 4
// column-major floats; its inline site 0xa7f148 (fefw_wide.cpp's) sees both pointers. The map's markers and gauges call it every frame
// with the map camera, so the pair of the frame is the map's.
//
// [fefw] map_mouse_log = 1 logs the cursor, the cameras and the steering every 30 frames. [fefw] map_pointer_test = <file> (tests) takes
// the pointer from a file holding "u v" (0..1 on the picture) instead of the Windows pointer.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "fefw_game.h"
#include "nxrt/config.h"
#include "nxrt/hooks.h"
#include "nxrt/jc2_mouse.h"
#include "nxrt/log.h"
#include "nxrt/memory.h"

namespace nxrt::games::fefw {

namespace {

constexpr const char* kTag = "fefw-map";
constexpr uint64_t kCursorTick = 0x735020;          // entry hook: game::map::Cursor::Impl::Tick (x0 = the cursor)
constexpr uint64_t kWorldToScreenSite = 0xa7f148;   // inline in 0xa7f080 (x1 = view, x2 = projection)

struct Camera {
    uint64_t view_ptr = 0, proj_ptr = 0;
    float view[16] = {}, proj[16] = {};
    uint32_t calls = 0;
};

struct State {
    bool on = false, log = false;
    std::string test_file;
    float gain = 1.5f, vmin = 24.0f, vmax = 400.0f;    // motion per frame: gain * screen distance (UI px), within vmin .. vmax
    float scroll = 60.0f;                             // ... at most this while the camera scrolls ([fefw] map_pointer_scroll)
    float last_t[3] = {};                             // the view matrix's translation last frame
    bool scrolling = false;
    bool pending = false;                             // the cell picked last is not the cursor's yet
    float pu = -1, pv = -1;                           // the pointer last frame
    bool target_on = false;                           // steering to `target`, picked when the pointer last moved
    int target[2] = {};
    int32_t prog_cell[2] = {};                        // the cursor's cell when it last made progress, and frames since
    int stuck = 0;
    int off_frames = 1000;                            // frames in a row the map cursor has not been followed
    std::atomic<uint64_t> cursor{0};
    std::atomic<uint64_t> cursor_ticks{0};
    std::mutex mu;
    Camera cams[8];
    int ncams = 0;
    uint64_t frames = 0, last_ticks = 0, steered = 0, arrived = 0;
    bool was_steering = false, wide_warned = false;
};

State& st() {
    static State s;
    return s;
}

int on_cursor_tick(::Ctx* c, uint64_t, void*) {
    State& s = st();
    s.cursor.store(c->x[0], std::memory_order_relaxed);
    s.cursor_ticks.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

void on_world_to_screen(::Ctx* c, uint64_t, void*) {
    State& s = st();
    const uint64_t v = c->x[1], p = c->x[2];
    if (!mem::is_mapped(v, 64) || !mem::is_mapped(p, 64)) return;
    std::lock_guard<std::mutex> lk(s.mu);
    for (int i = 0; i < s.ncams; ++i) {
        if (s.cams[i].view_ptr == v && s.cams[i].proj_ptr == p) {
            ++s.cams[i].calls;
            return;
        }
    }
    if (s.ncams >= 8) return;
    Camera& k = s.cams[s.ncams++];
    k.view_ptr = v;
    k.proj_ptr = p;
    std::memcpy(k.view, reinterpret_cast<const void*>(v), 64);
    std::memcpy(k.proj, reinterpret_cast<const void*>(p), 64);
    k.calls = 1;
}

// 0xa7f080's arithmetic: false when the point is behind the camera.
bool project(const Camera& k, const float w[3], float* sx, float* sy) {
    const float* V = k.view;
    const float* P = k.proj;
    const float vx = V[0] * w[0] + V[4] * w[1] + V[8] * w[2] + V[12];
    const float vy = V[1] * w[0] + V[5] * w[1] + V[9] * w[2] + V[13];
    const float vz = V[2] * w[0] + V[6] * w[1] + V[10] * w[2] + V[14];
    if (!(vz < 0.0f)) return false;
    const float cw = P[3] * vx + P[7] * vy + P[11] * vz + P[15];
    if (cw == 0.0f) return false;
    const float cx = P[0] * vx + P[4] * vy + P[8] * vz + P[12];
    const float cy = P[1] * vx + P[5] * vy + P[9] * vz + P[13];
    *sx = cx / cw * 960.0f + 960.0f;
    *sy = -(cy / cw) * 540.0f + 540.0f;
    return true;
}

bool invert4(const double* m, double* inv) {
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (std::fabs(det) < 1e-18) return false;
    for (int i = 0; i < 16; ++i) inv[i] /= det;
    return true;
}

// The UI point (1920 x 1080 units, y down) on the horizontal plane y = plane_y, in world units. 0xa7f080 reads only the projection's
// x, y and w rows (its depth row may be anything), so the ray is built in view space: at two depths in front of the camera, the view-space
// x, y whose projection lands on the point solve two linear equations; the inverse view matrix takes them to the world.
bool unproject_to_plane(const Camera& k, float sx, float sy, float plane_y, float* wx, float* wz) {
    // the view matrix is affine and its last row is not stored (0xa7f080 never reads it; the game leaves zeros there)
    double view[16], vinv[16];
    for (int i = 0; i < 16; ++i) view[i] = k.view[i];
    view[3] = view[7] = view[11] = 0.0;
    view[15] = 1.0;
    if (!invert4(view, vinv)) return false;
    const float* P = k.proj;
    const double nx = (sx - 960.0) / 960.0, ny = -(sy - 540.0) / 540.0;
    // row r of P against (vx, vy, vz, 1): P[r], P[4 + r], P[8 + r], P[12 + r]; the point's equations are (row0 - nx row3) . v = 0 and
    // (row1 - ny row3) . v = 0
    const double a1 = P[0] - nx * P[3], b1 = P[4] - nx * P[7], c1 = P[8] - nx * P[11], d1 = P[12] - nx * P[15];
    const double a2 = P[1] - ny * P[3], b2 = P[5] - ny * P[7], c2 = P[9] - ny * P[11], d2 = P[13] - ny * P[15];
    const double det = a1 * b2 - a2 * b1;
    if (std::fabs(det) < 1e-12) return false;
    double pts[2][3];
    const double depth[2] = {-10.0, -100.0};    // view space looks down -z
    for (int i = 0; i < 2; ++i) {
        const double vz = depth[i];
        const double e1 = -(c1 * vz + d1), e2 = -(c2 * vz + d2);
        const double vx = (e1 * b2 - e2 * b1) / det, vy = (a1 * e2 - a2 * e1) / det;
        const double c[4] = {vx, vy, vz, 1.0};
        double o[4];
        for (int r = 0; r < 4; ++r) o[r] = vinv[r] * c[0] + vinv[4 + r] * c[1] + vinv[8 + r] * c[2] + vinv[12 + r] * c[3];
        if (o[3] == 0) return false;
        for (int r = 0; r < 3; ++r) pts[i][r] = o[r] / o[3];
    }
    const double dy = pts[1][1] - pts[0][1];
    if (std::fabs(dy) < 1e-9) return false;
    const double t = (plane_y - pts[0][1]) / dy;
    *wx = static_cast<float>(pts[0][0] + t * (pts[1][0] - pts[0][0]));
    *wz = static_cast<float>(pts[0][2] + t * (pts[1][2] - pts[0][2]));
    return true;
}

int cell_of(float w) { return static_cast<int>(std::floor((w + 1280.0f) * 0.2f)) - 256; }
float centre_of(int cell) { return static_cast<float>(cell + 256) * 5.0f + 2.5f - 1280.0f; }

bool test_pointer(const std::string& file, float* u, float* v) {
    if (file.empty()) return false;
    FILE* f = std::fopen(file.c_str(), "rb");
    if (!f) return false;
    const bool ok = std::fscanf(f, "%f %f", u, v) == 2;
    std::fclose(f);
    return ok;
}

}  // namespace

void mapmouse_install(uint64_t base) {
    State& s = st();
    s.on = config_get_bool("fefw", "map_pointer", true);
    s.log = config_get_bool("fefw", "map_mouse_log", false);
    s.test_file = config_get_str("fefw", "map_pointer_test", "");
    s.scroll = static_cast<float>(std::clamp(config_get_double("fefw", "map_pointer_scroll", 60.0), 8.0, 400.0));
    if (!s.on && !s.log) return;
    hook_add_entry(base + kCursorTick, &on_cursor_tick);
    hook_add_inline(base + kWorldToScreenSite, &on_world_to_screen);
    NXLOG_I(kTag, "map cursor follows the pointer: %s (log %d%s%s)", s.on ? "on" : "off", s.log ? 1 : 0, s.test_file.empty() ? "" : ", test pointer from ",
            s.test_file.c_str());
}

// The pointer on the picture (0..1, v down): the test file's when [fefw] map_pointer_test names one, else the Windows pointer as play mode
// last saw it over the picture.
bool mapmouse_pending() { return st().pending; }

bool mapmouse_pointer(float* u, float* v) {
    const State& s = st();
    return test_pointer(s.test_file, u, v) || hid::jc2_mouse_host_pointer(u, v);
}

// The pointer is where it is now: the test pointer, or the Windows pointer over the picture and not captured.
bool mapmouse_pointer_live() {
    const State& s = st();
    return !s.test_file.empty() || hid::jc2_mouse_pointer_free();
}

// Main thread, once per presented frame (mouse_frame), between the game's frames. Returns true while the map cursor follows the pointer
// (the caller keeps the mouse uncaptured and holds the PC mouse's motion back).
bool mapmouse_frame() {
    State& s = st();
    if (!s.on && !s.log) return false;
    ++s.frames;
    s.pending = false;
    const uint64_t ticks = s.cursor_ticks.load(std::memory_order_relaxed);
    const bool cursor_live = ticks != s.last_ticks;
    s.last_ticks = ticks;
    Camera cams[8];
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        n = s.ncams;
        std::copy(s.cams, s.cams + n, cams);
        s.ncams = 0;
    }
    const uint64_t cur = s.cursor.load(std::memory_order_relaxed);
    const MouseInfo mi = mouse_info();
    const WideInfo wi = wide_info();
    bool steering = s.on && cursor_live && mi.valid && mi.mode && n > 0 && mem::is_mapped(cur, 0xc0);
    if (steering && wi.active) {
        if (!s.wide_warned) NXLOG_W(kTag, "map pointer: not with an output wider than 16:9 yet; the map cursor keeps the game's mouse controls");
        s.wide_warned = true;
        steering = false;
    }
    float u = 0, v = 0;
    const bool have_pointer = steering && mapmouse_pointer(&u, &v);
    steering = steering && have_pointer && u >= 0 && u <= 1 && v >= 0 && v <= 1;
    if (steering != s.was_steering) {
        if (s.log) NXLOG_I(kTag, "map pointer %s", steering ? "on: the map cursor follows the Windows pointer" : "off");
        s.was_steering = steering;
    }
    // a new target when the pointer moves (not when the world moves under a still pointer, nor when the map comes back under it: the
    // game puts the cursor where it wants, on a unit, after a menu or the enemy's turn)
    if (!steering) {
        s.target_on = false;
        if (s.off_frames < 1000) ++s.off_frames;
    }
    const bool back = steering && s.off_frames > 10;
    const bool moved = steering && !back && (std::fabs(u - s.pu) > 1e-4f || std::fabs(v - s.pv) > 1e-4f);
    if (steering) {
        s.pu = u;
        s.pv = v;
        s.off_frames = 0;
    }
    if (!cursor_live || !mem::is_mapped(cur, 0xc0)) return steering;
    int32_t cell[2];
    float pos[4];
    std::memcpy(cell, reinterpret_cast<const void*>(cur + 0x08), 8);
    std::memcpy(pos, reinterpret_cast<const void*>(cur + 0x40), 16);
    const Camera& cam = cams[0];
    // The camera follows the cursor once it leaves a middle region of the screen; a cell outside that region is never reached (the world
    // slides under the pointer as the cursor goes), which reads as edge scrolling. A camera that moved this frame means exactly that, so
    // the steering slows to a speed a hand can stop by moving the pointer back.
    s.scrolling = std::fabs(cam.view[12] - s.last_t[0]) + std::fabs(cam.view[13] - s.last_t[1]) + std::fabs(cam.view[14] - s.last_t[2]) > 1e-3f;
    s.last_t[0] = cam.view[12];
    s.last_t[1] = cam.view[13];
    s.last_t[2] = cam.view[14];
    float dx = 0, dy = 0, inj_x = 0, inj_y = 0;
    float wx = 0, wz = 0;
    if (moved && unproject_to_plane(cam, u * 1920.0f, v * 1080.0f, pos[1], &wx, &wz)) {
        s.target[0] = cell_of(wx);
        s.target[1] = cell_of(wz);
        s.target_on = true;
        s.stuck = 0;
        if (s.log)
            NXLOG_I(kTag, "pointer moved to %.3f,%.3f: target cell %d,%d (cursor at %d,%d)", static_cast<double>(u), static_cast<double>(v), s.target[0],
                    s.target[1], cell[0], cell[1]);
        s.prog_cell[0] = cell[0];
        s.prog_cell[1] = cell[1];
    }
    if (s.target_on && cell[0] == s.target[0] && cell[1] == s.target[1]) {
        s.target_on = false;
        ++s.arrived;
    }
    if (s.target_on) {
        if (cell[0] != s.prog_cell[0] || cell[1] != s.prog_cell[1]) {
            s.prog_cell[0] = cell[0];
            s.prog_cell[1] = cell[1];
            s.stuck = 0;
        } else if (++s.stuck > 45) {      // held back (a unit's range, the map's edge): leave the cursor to the game
            s.target_on = false;
            if (s.log) NXLOG_I(kTag, "target cell %d,%d not reached; given up at %d,%d", s.target[0], s.target[1], cell[0], cell[1]);
        }
    }
    // where the Windows pointer comes back if the mouse was captured (a camera scene before the map): at the cursor
    if (steering) {
        float csx = 0, csy = 0;
        if (project(cam, pos, &csx, &csy)) hid::jc2_mouse_set_return_point(std::clamp(csx / 1920.0f, 0.0f, 1.0f), std::clamp(csy / 1080.0f, 0.0f, 1.0f));
    }
    const int tx = s.target_on ? s.target[0] : cell[0], tz = s.target_on ? s.target[1] : cell[1];
    if (s.target_on) {
        // steer in screen space from the cursor (its current, possibly moving, position) to the target cell's centre
        const float target[3] = {centre_of(tx), pos[1], centre_of(tz)};
        float csx = 0, csy = 0, tsx = 0, tsy = 0;
        if (project(cam, pos, &csx, &csy) && project(cam, target, &tsx, &tsy)) {
            dx = tsx - csx;
            dy = tsy - csy;
            const float d = std::sqrt(dx * dx + dy * dy);
            if (d > 0.5f) {
                const float speed = std::min(std::clamp(s.gain * d, s.vmin, s.vmax), s.scrolling ? s.scroll : s.vmax);
                inj_x = dx / d * speed;
                inj_y = dy / d * speed;
                hid::jc2_mouse_inject(inj_x, inj_y);
                ++s.steered;
            }
        }
        s.pending = true;
    }
    if (s.log && s.frames % 60 == 0) {
        for (int i = 0; i < n; ++i) {
            const float* V = cams[i].view;
            const float* P = cams[i].proj;
            float px = 0, py = 0;
            project(cams[i], pos, &px, &py);
            NXLOG_I(kTag, "camera %d (calls %u) cursor at %.1f,%.1f view %.4f %.4f %.4f | %.4f %.4f %.4f | %.4f %.4f %.4f | %.3f %.3f %.3f proj %.4f %.4f "
                    "%.4f %.4f %.4f %.4f", i, cams[i].calls, static_cast<double>(px), static_cast<double>(py), V[0], V[1], V[2], V[4], V[5], V[6], V[8], V[9],
                    V[10], V[12], V[13], V[14], P[0], P[5], P[8], P[9], P[11], P[14]);
        }
    }
    if (s.log && s.frames % 30 == 0) {
        const float* V = cam.view;
        NXLOG_I(kTag, "cursor %llx cell %d,%d pos %.2f %.2f %.2f; pointer %s %.3f,%.3f -> world %.2f,%.2f cell %d,%d; steer %.0f,%.0f (inject %.0f,%.0f); "
                "%d cameras, view t %.3f %.3f %.3f; frames steered %llu, on target %llu", static_cast<unsigned long long>(cur), cell[0], cell[1],
                static_cast<double>(pos[0]), static_cast<double>(pos[1]), static_cast<double>(pos[2]), have_pointer ? "at" : "none",
                static_cast<double>(u), static_cast<double>(v), static_cast<double>(wx), static_cast<double>(wz), tx, tz, static_cast<double>(dx),
                static_cast<double>(dy), static_cast<double>(inj_x), static_cast<double>(inj_y), n, static_cast<double>(V[12]),
                static_cast<double>(V[13]), static_cast<double>(V[14]), static_cast<unsigned long long>(s.steered),
                static_cast<unsigned long long>(s.arrived));
    }
    return steering;
}

}  // namespace nxrt::games::fefw
