#pragma once

#include "../math/vec_mat.hpp"
#include "mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace sr {

struct Uniforms {
    Mat4 model = Mat4::identity();
    Mat4 view = Mat4::identity();
    Mat4 proj = Mat4::identity();
    Mat4 mvp = Mat4::identity();
    Vec3 camera_pos{};
    Vec3 light_dir{0.f, 1.f, 0.f};
    Vec3 light_color{1.f, 0.95f, 0.88f};
    Vec3 sky_ambient{0.22f, 0.26f, 0.34f};
    Vec3 ground_ambient{0.08f, 0.07f, 0.05f};
    float shininess = 48.f;
    float spec_strength = 0.55f;
    float ambient_strength = 1.f;
    float z_near = 0.1f;
    float z_far = 80.f;
    int width = 1;
    int height = 1;
    bool backface_cull = true;
};

struct ClipVertex {
    Vec4 clip{};
    Vec3 world{};
    Vec3 n{};
    Vec2 uv{};
};

struct ScreenVertex {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;  // NDC z in [-1, 1]
    float inv_w = 1.f;
    Vec3 world{};
    Vec3 n{};
    Vec2 uv{};
};

inline ClipVertex lerp(const ClipVertex& a, const ClipVertex& b, float t) {
    return {a.clip + (b.clip - a.clip) * t, a.world + (b.world - a.world) * t,
            a.n + (b.n - a.n) * t, a.uv + (b.uv - a.uv) * t};
}

inline ClipVertex transform_vertex(const Vertex& v, const Uniforms& u) {
    ClipVertex c;
    c.world = transform_point(u.model, v.p);
    c.n = normalize(transform_vector(u.model, v.n));
    c.clip = u.mvp * Vec4(v.p, 1.f);
    c.uv = v.uv;
    return c;
}

inline float clip_plane_dist(const ClipVertex& v, int plane) {
    const Vec4& c = v.clip;
    switch (plane) {
        case 0: return c.w + c.x;
        case 1: return c.w - c.x;
        case 2: return c.w + c.y;
        case 3: return c.w - c.y;
        case 4: return c.w + c.z;
        case 5: return c.w - c.z;
        default: return 0.f;
    }
}

inline int clip_poly_plane(const ClipVertex* in, int n_in, ClipVertex* out, int plane) {
    int n_out = 0;
    if (n_in <= 0) return 0;
    for (int i = 0; i < n_in; ++i) {
        const ClipVertex& a = in[i];
        const ClipVertex& b = in[(i + 1) % n_in];
        const float da = clip_plane_dist(a, plane);
        const float db = clip_plane_dist(b, plane);
        const bool ia = da >= 0.f;
        const bool ib = db >= 0.f;
        if (ia && ib) {
            out[n_out++] = b;
        } else if (ia && !ib) {
            const float t = da / (da - db + 1e-12f);
            out[n_out++] = lerp(a, b, t);
        } else if (!ia && ib) {
            const float t = da / (da - db + 1e-12f);
            out[n_out++] = lerp(a, b, t);
            out[n_out++] = b;
        }
    }
    return n_out;
}

struct ClippedTris {
    std::array<ClipVertex, 27> verts{};
    int count = 0;  // number of output triangles
};

inline ClippedTris clip_triangle(const ClipVertex& a, const ClipVertex& b, const ClipVertex& c) {
    std::array<ClipVertex, 16> buf_a{a, b, c};
    std::array<ClipVertex, 16> buf_b{};
    int n = 3;
    ClipVertex* cur = buf_a.data();
    ClipVertex* nxt = buf_b.data();
    for (int plane = 0; plane < 6; ++plane) {
        n = clip_poly_plane(cur, n, nxt, plane);
        std::swap(cur, nxt);
        if (n < 3) break;
    }
    ClippedTris out;
    if (n < 3) return out;
    const int tri_n = n - 2;
    out.count = tri_n;
    for (int i = 0; i < tri_n; ++i) {
        out.verts[static_cast<size_t>(i * 3 + 0)] = cur[0];
        out.verts[static_cast<size_t>(i * 3 + 1)] = cur[i + 1];
        out.verts[static_cast<size_t>(i * 3 + 2)] = cur[i + 2];
    }
    return out;
}

inline ScreenVertex project_to_screen(const ClipVertex& v, int width, int height) {
    ScreenVertex s;
    const float w = v.clip.w;
    s.inv_w = (std::fabs(w) > 1e-12f) ? (1.f / w) : 0.f;
    const float ndc_x = v.clip.x * s.inv_w;
    const float ndc_y = v.clip.y * s.inv_w;
    s.z = v.clip.z * s.inv_w;
    s.x = (ndc_x * 0.5f + 0.5f) * static_cast<float>(width);
    s.y = (1.f - (ndc_y * 0.5f + 0.5f)) * static_cast<float>(height);
    s.world = v.world;
    s.n = v.n;
    s.uv = v.uv;
    return s;
}

}  // namespace sr
