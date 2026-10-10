#pragma once

#include "../math/vec_mat.hpp"

#include <cstdint>
#include <vector>

namespace sr {

struct Framebuffer {
    int w = 0;
    int h = 0;
    std::vector<uint32_t> color;
    std::vector<float> depth;

    Framebuffer() = default;
    Framebuffer(int width, int height) { resize(width, height); }

    void resize(int width, int height) {
        w = std::max(width, 1);
        h = std::max(height, 1);
        color.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
        depth.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 1.f);
    }

    void clear(uint32_t clear_color, float clear_depth = 1.f) {
        std::fill(color.begin(), color.end(), clear_color);
        std::fill(depth.begin(), depth.end(), clear_depth);
    }

    int index(int x, int y) const { return y * w + x; }

    bool in_bounds(int x, int y) const {
        return static_cast<unsigned>(x) < static_cast<unsigned>(w) &&
               static_cast<unsigned>(y) < static_cast<unsigned>(h);
    }

    void plot(int x, int y, uint32_t rgba, float z) {
        if (!in_bounds(x, y)) return;
        const int i = index(x, y);
        if (z < depth[static_cast<size_t>(i)]) {
            depth[static_cast<size_t>(i)] = z;
            color[static_cast<size_t>(i)] = rgba;
        }
    }
};

}  // namespace sr
