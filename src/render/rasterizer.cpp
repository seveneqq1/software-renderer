#include "rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sr {
namespace {

constexpr float kAreaEps = 1e-6f;

inline float edge(float ax, float ay, float bx, float by, float px, float py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

inline void draw_line(Framebuffer& fb, const ScreenVertex& a, const ScreenVertex& b, uint32_t color) {
    float x0 = a.x, y0 = a.y, z0 = a.z * 0.5f + 0.5f;
    float x1 = b.x, y1 = b.y, z1 = b.z * 0.5f + 0.5f;
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::fabs(dx), std::fabs(dy)))));
    const float inv = 1.f / static_cast<float>(steps);
    for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) * inv;
        const int x = static_cast<int>(std::round(x0 + dx * t));
        const int y = static_cast<int>(std::round(y0 + dy * t));
        const float z = z0 + (z1 - z0) * t;
        fb.plot(x, y, color, z);
    }
}

struct TriSetup {
    ScreenVertex v0, v1, v2;
    float area = 0.f;
    int xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    float a01 = 0, a12 = 0, a20 = 0;  // dw/dx for w2, w0, w1
    float b01 = 0, b12 = 0, b20 = 0;  // dw/dy
};

inline bool setup_triangle(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c, int w,
                           int h, bool cull, TriSetup& out) {
    const float area = edge(a.x, a.y, b.x, b.y, c.x, c.y);
    if (std::fabs(area) < kAreaEps) return false;
    if (cull) {
        // CCW in NDC (y-up) is front-facing. Framebuffer y is down, so front faces have area < 0.
        if (area > 0.f) return false;
    }
    float minx = std::min({a.x, b.x, c.x});
    float maxx = std::max({a.x, b.x, c.x});
    float miny = std::min({a.y, b.y, c.y});
    float maxy = std::max({a.y, b.y, c.y});
    int xmin = std::max(0, static_cast<int>(std::floor(minx)));
    int ymin = std::max(0, static_cast<int>(std::floor(miny)));
    int xmax = std::min(w - 1, static_cast<int>(std::ceil(maxx)));
    int ymax = std::min(h - 1, static_cast<int>(std::ceil(maxy)));
    if (xmin > xmax || ymin > ymax) return false;

    out.v0 = a;
    out.v1 = b;
    out.v2 = c;
    out.area = area;
    out.xmin = xmin;
    out.xmax = xmax;
    out.ymin = ymin;
    out.ymax = ymax;
    // d/dx edge(A,B,P) = (B.y - A.y), d/dy = -(B.x - A.x)
    out.a12 = (c.y - b.y);  // w0 = edge(v1,v2,p)
    out.b12 = -(c.x - b.x);
    out.a20 = (a.y - c.y);  // w1 = edge(v2,v0,p)
    out.b20 = -(a.x - c.x);
    out.a01 = (b.y - a.y);  // w2 = edge(v0,v1,p)
    out.b01 = -(b.x - a.x);
    return true;
}

inline void interp_attributes(const TriSetup& t, float w0, float w1, float w2, float inv_area, float& z_ndc,
                              float& z_win, Vec3& world, Vec3& n, Vec2& uv) {
    const float b0 = w0 * inv_area;
    const float b1 = w1 * inv_area;
    const float b2 = w2 * inv_area;
    const float iw0 = b0 * t.v0.inv_w;
    const float iw1 = b1 * t.v1.inv_w;
    const float iw2 = b2 * t.v2.inv_w;
    const float iw = iw0 + iw1 + iw2;
    const float rcp = (std::fabs(iw) > 1e-20f) ? (1.f / iw) : 0.f;
    z_ndc = (iw0 * t.v0.z + iw1 * t.v1.z + iw2 * t.v2.z) * rcp;
    z_win = z_ndc * 0.5f + 0.5f;
    world = (t.v0.world * iw0 + t.v1.world * iw1 + t.v2.world * iw2) * rcp;
    n = (t.v0.n * iw0 + t.v1.n * iw1 + t.v2.n * iw2) * rcp;
    uv = (t.v0.uv * iw0 + t.v1.uv * iw1 + t.v2.uv * iw2) * rcp;
}

inline bool coverage(float w0, float w1, float w2, float area) {
    if (area > 0.f) return w0 >= 0.f && w1 >= 0.f && w2 >= 0.f;
    return w0 <= 0.f && w1 <= 0.f && w2 <= 0.f;
}

void fill_scalar(Framebuffer& fb, const TriSetup& t, const ShadeParams& shade, RasterMode mode,
                 DrawStats& stats) {
    const float inv_area = 1.f / t.area;
    const bool depth_only = (mode == RasterMode::Depth);
    for (int y = t.ymin; y <= t.ymax; ++y) {
        const float py = static_cast<float>(y) + 0.5f;
        const float px0 = static_cast<float>(t.xmin) + 0.5f;
        float w0 = edge(t.v1.x, t.v1.y, t.v2.x, t.v2.y, px0, py);
        float w1 = edge(t.v2.x, t.v2.y, t.v0.x, t.v0.y, px0, py);
        float w2 = edge(t.v0.x, t.v0.y, t.v1.x, t.v1.y, px0, py);
        for (int x = t.xmin; x <= t.xmax; ++x) {
            if (coverage(w0, w1, w2, t.area)) {
                float z_ndc = 0.f, z_win = 1.f;
                Vec3 world, n;
                Vec2 uv;
                interp_attributes(t, w0, w1, w2, inv_area, z_ndc, z_win, world, n, uv);
                const int idx = y * fb.w + x;
                if (z_win >= 0.f && z_win < fb.depth[static_cast<size_t>(idx)]) {
                    fb.depth[static_cast<size_t>(idx)] = z_win;
                    fb.color[static_cast<size_t>(idx)] =
                        depth_only ? shade_depth_scalar(z_ndc, shade)
                                   : shade_pixel_scalar(world, n, uv, shade);
                    ++stats.pixels;
                }
            }
            w0 += t.a12;
            w1 += t.a20;
            w2 += t.a01;
        }
    }
}

void fill_simd_impl(Framebuffer& fb, const TriSetup& t, const ShadeParams& shade, RasterMode mode,
                    DrawStats& stats) {
    const F32x inv_area = F32x::splat(1.f / t.area);
    const F32x dw0dx = F32x::splat(t.a12);
    const F32x dw1dx = F32x::splat(t.a20);
    const F32x dw2dx = F32x::splat(t.a01);
    const F32x zero = F32x::splat(0.f);
    const bool depth_only = (mode == RasterMode::Depth);
    const bool pos_area = t.area > 0.f;

    for (int y = t.ymin; y <= t.ymax; ++y) {
        const float py = static_cast<float>(y) + 0.5f;
        const float px0 = static_cast<float>(t.xmin) + 0.5f;
        const float w0_row = edge(t.v1.x, t.v1.y, t.v2.x, t.v2.y, px0, py);
        const float w1_row = edge(t.v2.x, t.v2.y, t.v0.x, t.v0.y, px0, py);
        const float w2_row = edge(t.v0.x, t.v0.y, t.v1.x, t.v1.y, px0, py);

        int x = t.xmin;
        const int simd_end = std::min(t.xmax + 1, fb.w);
        for (; x + F32x::N <= simd_end; x += F32x::N) {
            const F32x off = F32x::splat(static_cast<float>(x - t.xmin)) + F32x::lane_index();
            const F32x w0 = F32x::splat(w0_row) + off * dw0dx;
            const F32x w1 = F32x::splat(w1_row) + off * dw1dx;
            const F32x w2 = F32x::splat(w2_row) + off * dw2dx;
            Mask inside = pos_area ? (Mask::from_cmp_ge(w0, zero) & Mask::from_cmp_ge(w1, zero) &
                                      Mask::from_cmp_ge(w2, zero))
                                   : (Mask::from_cmp_le(w0, zero) & Mask::from_cmp_le(w1, zero) &
                                      Mask::from_cmp_le(w2, zero));
            if (!inside.any()) continue;

            const F32x b0 = w0 * inv_area;
            const F32x b1 = w1 * inv_area;
            const F32x b2 = w2 * inv_area;
            const F32x iw0 = b0 * F32x::splat(t.v0.inv_w);
            const F32x iw1 = b1 * F32x::splat(t.v1.inv_w);
            const F32x iw2 = b2 * F32x::splat(t.v2.inv_w);
            const F32x iw = iw0 + iw1 + iw2;
            const F32x rcp = F32x::splat(1.f) / F32x::max_(F32x::abs_(iw), F32x::splat(1e-20f));
            const F32x z_ndc =
                (iw0 * F32x::splat(t.v0.z) + iw1 * F32x::splat(t.v1.z) + iw2 * F32x::splat(t.v2.z)) * rcp;
            const F32x z_win = z_ndc * F32x::splat(0.5f) + F32x::splat(0.5f);

            float* zptr = &fb.depth[static_cast<size_t>(y * fb.w + x)];
            const F32x zbuf = F32x::loadu(zptr);
            Mask closer = Mask::from_cmp_lt(z_win, zbuf) & Mask::from_cmp_ge(z_win, zero) & inside;
            if (!closer.any()) continue;

            const F32x wx =
                (iw0 * F32x::splat(t.v0.world.x) + iw1 * F32x::splat(t.v1.world.x) + iw2 * F32x::splat(t.v2.world.x)) *
                rcp;
            const F32x wy =
                (iw0 * F32x::splat(t.v0.world.y) + iw1 * F32x::splat(t.v1.world.y) + iw2 * F32x::splat(t.v2.world.y)) *
                rcp;
            const F32x wz =
                (iw0 * F32x::splat(t.v0.world.z) + iw1 * F32x::splat(t.v1.world.z) + iw2 * F32x::splat(t.v2.world.z)) *
                rcp;
            const F32x nx =
                (iw0 * F32x::splat(t.v0.n.x) + iw1 * F32x::splat(t.v1.n.x) + iw2 * F32x::splat(t.v2.n.x)) * rcp;
            const F32x ny =
                (iw0 * F32x::splat(t.v0.n.y) + iw1 * F32x::splat(t.v1.n.y) + iw2 * F32x::splat(t.v2.n.y)) * rcp;
            const F32x nz =
                (iw0 * F32x::splat(t.v0.n.z) + iw1 * F32x::splat(t.v1.n.z) + iw2 * F32x::splat(t.v2.n.z)) * rcp;
            const F32x uu =
                (iw0 * F32x::splat(t.v0.uv.x) + iw1 * F32x::splat(t.v1.uv.x) + iw2 * F32x::splat(t.v2.uv.x)) * rcp;
            const F32x vv =
                (iw0 * F32x::splat(t.v0.uv.y) + iw1 * F32x::splat(t.v1.uv.y) + iw2 * F32x::splat(t.v2.uv.y)) * rcp;

            const ShadeBundle col = depth_only ? shade_depth_simd(z_ndc, shade)
                                               : shade_pixels_simd(wx, wy, wz, nx, ny, nz, uu, vv, closer, shade);
            store_argb_masked(&fb.color[static_cast<size_t>(y * fb.w + x)], col, closer);
            store_depth_masked(zptr, z_win, closer);
            stats.pixels += closer.count();
        }

        float w0 = w0_row + static_cast<float>(x - t.xmin) * t.a12;
        float w1 = w1_row + static_cast<float>(x - t.xmin) * t.a20;
        float w2 = w2_row + static_cast<float>(x - t.xmin) * t.a01;
        const float inva = 1.f / t.area;
        for (; x <= t.xmax; ++x) {
            if (coverage(w0, w1, w2, t.area)) {
                float z_ndc = 0.f, z_win = 1.f;
                Vec3 world, n;
                Vec2 uv;
                interp_attributes(t, w0, w1, w2, inva, z_ndc, z_win, world, n, uv);
                const int idx = y * fb.w + x;
                if (z_win >= 0.f && z_win < fb.depth[static_cast<size_t>(idx)]) {
                    fb.depth[static_cast<size_t>(idx)] = z_win;
                    fb.color[static_cast<size_t>(idx)] =
                        depth_only ? shade_depth_scalar(z_ndc, shade) : shade_pixel_scalar(world, n, uv, shade);
                    ++stats.pixels;
                }
            }
            w0 += t.a12;
            w1 += t.a20;
            w2 += t.a01;
        }
    }
}

void rasterize_triangle(Framebuffer& fb, const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c,
                        const ShadeParams& shade, RasterMode mode, ShadeMode sm, bool cull, DrawStats& stats) {
    if (mode == RasterMode::Wireframe) {
        const float area = edge(a.x, a.y, b.x, b.y, c.x, c.y);
        if (cull && area > 0.f) return;
        if (std::fabs(area) < kAreaEps) return;
        constexpr uint32_t kWire = 0xFF9BE7FF;
        draw_line(fb, a, b, kWire);
        draw_line(fb, b, c, kWire);
        draw_line(fb, c, a, kWire);
        ++stats.tris_drawn;
        return;
    }

    TriSetup setup;
    if (!setup_triangle(a, b, c, fb.w, fb.h, cull, setup)) return;
    ++stats.tris_drawn;
    if (sm == ShadeMode::SIMD) fill_simd_impl(fb, setup, shade, mode, stats);
    else fill_scalar(fb, setup, shade, mode, stats);
}

}  // namespace

ShadeParams shade_from_uniforms(const Uniforms& u, const Texture* tex) {
    ShadeParams p;
    p.light_dir = normalize(u.light_dir);
    p.light_color = u.light_color;
    p.camera_pos = u.camera_pos;
    p.sky = u.sky_ambient;
    p.ground = u.ground_ambient;
    p.shininess = u.shininess;
    p.spec_strength = u.spec_strength;
    p.ambient_strength = u.ambient_strength;
    p.z_near = u.z_near;
    p.z_far = u.z_far;
    p.texture = tex;
    return p;
}

void draw_mesh(Framebuffer& fb, const Mesh& mesh, const Uniforms& uniforms, const ShadeParams& shade,
               RasterMode raster_mode, ShadeMode shade_mode, DrawStats& stats) {
    const size_t nidx = mesh.indices.size();
    for (size_t i = 0; i + 2 < nidx; i += 3) {
        ++stats.tris_submitted;
        const Vertex& ia = mesh.vertices[mesh.indices[i]];
        const Vertex& ib = mesh.vertices[mesh.indices[i + 1]];
        const Vertex& ic = mesh.vertices[mesh.indices[i + 2]];
        const ClipVertex ca = transform_vertex(ia, uniforms);
        const ClipVertex cb = transform_vertex(ib, uniforms);
        const ClipVertex cc = transform_vertex(ic, uniforms);
        const ClippedTris clipped = clip_triangle(ca, cb, cc);
        for (int t = 0; t < clipped.count; ++t) {
            const ScreenVertex sa = project_to_screen(clipped.verts[static_cast<size_t>(t * 3 + 0)], fb.w, fb.h);
            const ScreenVertex sb = project_to_screen(clipped.verts[static_cast<size_t>(t * 3 + 1)], fb.w, fb.h);
            const ScreenVertex sc = project_to_screen(clipped.verts[static_cast<size_t>(t * 3 + 2)], fb.w, fb.h);
            rasterize_triangle(fb, sa, sb, sc, shade, raster_mode, shade_mode, uniforms.backface_cull, stats);
        }
    }
}

void apply_ssao(Framebuffer& fb, float strength) {
    if (strength <= 0.01f || fb.w < 4 || fb.h < 4) return;
    const int w = fb.w, h = fb.h;
    std::vector<float> ao(static_cast<size_t>(w * h), 1.f);
    constexpr int kOff[][2] = {{3, 0}, {-3, 0}, {0, 3}, {0, -3}, {4, 3}, {-4, 3},
                               {3, 4}, {-3, -4}, {5, 1}, {-5, -1}, {2, 5}, {-2, -5}};
    constexpr int ns = 12;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const int i = y * w + x;
            const float z = fb.depth[static_cast<size_t>(i)];
            if (z >= 0.999f) continue;
            float occ = 0.f;
            for (int s = 0; s < ns; ++s) {
                const int sx = x + kOff[s][0];
                const int sy = y + kOff[s][1];
                if (static_cast<unsigned>(sx) >= static_cast<unsigned>(w) ||
                    static_cast<unsigned>(sy) >= static_cast<unsigned>(h))
                    continue;
                const float sz = fb.depth[static_cast<size_t>(sy * w + sx)];
                if (sz < z - 0.003f) occ += 1.f;
            }
            ao[static_cast<size_t>(i)] = 1.f - (occ / static_cast<float>(ns)) * strength * 0.75f;
        }
    }
    for (int i = 0; i < w * h; ++i) {
        const float a = ao[static_cast<size_t>(i)];
        if (a >= 0.999f) continue;
        const Vec3 c = unpack_rgb8(fb.color[static_cast<size_t>(i)]);
        fb.color[static_cast<size_t>(i)] = pack_argb8(c * a);
    }
}

}  // namespace sr
