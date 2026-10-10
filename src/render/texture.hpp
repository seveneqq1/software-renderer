#pragma once

#include "../math/vec_mat.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace sr {

struct Texture {
    int w = 1;
    int h = 1;
    std::vector<uint32_t> px{0xFFFFFFFFu};

    Vec3 sample_nearest(float u, float v) const {
        u -= std::floor(u);
        v -= std::floor(v);
        int x = static_cast<int>(u * static_cast<float>(w)) % w;
        int y = static_cast<int>(v * static_cast<float>(h)) % h;
        if (x < 0) x += w;
        if (y < 0) y += h;
        return unpack_rgb8(px[static_cast<size_t>(y * w + x)]);
    }

    Vec3 sample_bilinear(float u, float v) const {
        u -= std::floor(u);
        v -= std::floor(v);
        const float fx = u * static_cast<float>(w) - 0.5f;
        const float fy = v * static_cast<float>(h) - 0.5f;
        const int x0 = static_cast<int>(std::floor(fx));
        const int y0 = static_cast<int>(std::floor(fy));
        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);
        auto at = [&](int x, int y) {
            x %= w;
            if (x < 0) x += w;
            y %= h;
            if (y < 0) y += h;
            return unpack_rgb8(px[static_cast<size_t>(y * w + x)]);
        };
        const Vec3 c00 = at(x0, y0);
        const Vec3 c10 = at(x0 + 1, y0);
        const Vec3 c01 = at(x0, y0 + 1);
        const Vec3 c11 = at(x0 + 1, y0 + 1);
        return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
    }

    static Texture make_checker(int res, int tiles, Vec3 a, Vec3 b) {
        Texture t;
        t.w = t.h = std::max(res, 2);
        t.px.resize(static_cast<size_t>(t.w * t.h));
        for (int y = 0; y < t.h; ++y) {
            for (int x = 0; x < t.w; ++x) {
                const int cx = (x * tiles) / t.w;
                const int cy = (y * tiles) / t.h;
                t.px[static_cast<size_t>(y * t.w + x)] = pack_argb8(((cx + cy) & 1) ? b : a);
            }
        }
        return t;
    }

    static Texture make_brick(int res) {
        Texture t;
        t.w = t.h = std::max(res, 8);
        t.px.resize(static_cast<size_t>(t.w * t.h));
        const float bw = static_cast<float>(t.w) / 6.f;
        const float bh = static_cast<float>(t.h) / 12.f;
        for (int y = 0; y < t.h; ++y) {
            for (int x = 0; x < t.w; ++x) {
                const int row = static_cast<int>(static_cast<float>(y) / bh);
                const float xoff = (row & 1) ? bw * 0.5f : 0.f;
                const float lx = std::fmod(static_cast<float>(x) + xoff, bw);
                const float ly = std::fmod(static_cast<float>(y), bh);
                const bool mortar = lx < 2.f || ly < 2.f;
                const float n =
                    0.04f * std::sin(static_cast<float>(x) * 0.37f) * std::cos(static_cast<float>(y) * 0.21f);
                Vec3 c = mortar ? Vec3(0.18f, 0.17f, 0.16f)
                                : Vec3(0.62f + n, 0.22f + n * 0.4f, 0.16f);
                if (!mortar && ((x * 13 + y * 7) % 17 == 0)) c *= 0.85f;
                t.px[static_cast<size_t>(y * t.w + x)] = pack_argb8(c);
            }
        }
        return t;
    }

    static Texture make_grid_metal(int res) {
        Texture t;
        t.w = t.h = std::max(res, 8);
        t.px.resize(static_cast<size_t>(t.w * t.h));
        for (int y = 0; y < t.h; ++y) {
            for (int x = 0; x < t.w; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(t.w);
                const float v = static_cast<float>(y) / static_cast<float>(t.h);
                const float g = 0.18f + 0.08f * std::sin(u * 40.f) * std::sin(v * 40.f);
                const bool line = (x % (t.w / 8) < 2) || (y % (t.h / 8) < 2);
                Vec3 c = line ? Vec3(0.08f, 0.45f, 0.55f) : Vec3(0.12f + g, 0.14f + g, 0.18f + g);
                t.px[static_cast<size_t>(y * t.w + x)] = pack_argb8(c);
            }
        }
        return t;
    }

    static bool load_ppm(const std::string& path, Texture& out, std::string* err = nullptr) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            if (err) *err = "cannot open " + path;
            return false;
        }
        std::string magic;
        in >> magic;
        if (magic != "P3" && magic != "P6") {
            if (err) *err = "unsupported PPM magic";
            return false;
        }
        auto skip_ws_comments = [&]() {
            for (;;) {
                in >> std::ws;
                if (in.peek() == '#') {
                    std::string dummy;
                    std::getline(in, dummy);
                } else {
                    break;
                }
            }
        };
        int w = 0, h = 0, maxv = 0;
        skip_ws_comments();
        in >> w;
        skip_ws_comments();
        in >> h;
        skip_ws_comments();
        in >> maxv;
        if (w <= 0 || h <= 0 || maxv <= 0) return false;
        if (magic == "P6") in.get();
        out.w = w;
        out.h = h;
        out.px.resize(static_cast<size_t>(w * h));
        const float s = 1.f / static_cast<float>(maxv);
        for (int i = 0; i < w * h; ++i) {
            int r = 0, g = 0, b = 0;
            if (magic == "P3") {
                in >> r >> g >> b;
            } else {
                unsigned char rgb[3];
                in.read(reinterpret_cast<char*>(rgb), 3);
                r = rgb[0];
                g = rgb[1];
                b = rgb[2];
            }
            out.px[static_cast<size_t>(i)] =
                pack_argb8(static_cast<float>(r) * s, static_cast<float>(g) * s, static_cast<float>(b) * s);
        }
        return true;
    }
};

}  // namespace sr
