#pragma once

#include "../math/vec_mat.hpp"
#include "pipeline.hpp"
#include "texture.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#define SR_SIMD_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(SR_TARGET_NEON)
#include <arm_neon.h>
#define SR_SIMD_NEON 1
#endif

namespace sr {

inline const char* simd_isa_name() {
#if defined(SR_SIMD_AVX2)
    return "AVX2 (8-wide)";
#elif defined(SR_SIMD_NEON)
    return "ARM NEON (4-wide)";
#else
    return "Emulated SIMD (4-wide scalar)";
#endif
}

struct ShadeParams {
    Vec3 light_dir{0.f, 1.f, 0.f};
    Vec3 light_color{1.f, 0.95f, 0.88f};
    Vec3 camera_pos{};
    Vec3 sky{0.22f, 0.26f, 0.34f};
    Vec3 ground{0.08f, 0.07f, 0.05f};
    float shininess = 48.f;
    float spec_strength = 0.55f;
    float ambient_strength = 1.f;
    float z_near = 0.1f;
    float z_far = 80.f;
    const Texture* texture = nullptr;
};

inline Vec3 procedural_checker(Vec2 uv) {
    const float tiles = 8.f;
    const int cx = static_cast<int>(std::floor(uv.x * tiles));
    const int cy = static_cast<int>(std::floor(uv.y * tiles));
    return ((cx + cy) & 1) ? Vec3(0.92f, 0.90f, 0.84f) : Vec3(0.18f, 0.19f, 0.22f);
}

inline Vec3 sample_albedo(Vec2 uv, const Texture* tex) {
    if (tex) return tex->sample_bilinear(uv.x, uv.y);
    return procedural_checker(uv);
}

inline float linearize_depth(float z_ndc, float z_near, float z_far) {
    const float n = z_near, f = z_far;
    const float lin = (2.f * n * f) / (f + n - z_ndc * (f - n) + 1e-8f);
    return saturate((lin - n) / (f - n + 1e-8f));
}

inline uint32_t shade_pixel_scalar(Vec3 world, Vec3 n, Vec2 uv, const ShadeParams& p) {
    n = normalize(n);
    const Vec3 L = p.light_dir;
    const Vec3 V = normalize(p.camera_pos - world);
    const Vec3 H = normalize(L + V);
    const float ndotl = std::max(dot(n, L), 0.f);
    const float ndoth = std::max(dot(n, H), 0.f);
    const float spec = std::pow(ndoth, p.shininess) * p.spec_strength;
    const float hem = n.y * 0.5f + 0.5f;
    const float ao = 0.35f + 0.65f * hem;
    const Vec3 ambient_col = lerp(p.ground, p.sky, hem) * (p.ambient_strength * ao);
    const Vec3 albedo = sample_albedo(uv, p.texture);
    Vec3 color = albedo * ambient_col + albedo * p.light_color * ndotl + p.light_color * spec;
    color = reinhard(color);
    return pack_argb8(linear_to_srgb(color));
}

inline uint32_t shade_depth_scalar(float z_ndc, const ShadeParams& p) {
    const float vis = std::pow(linearize_depth(z_ndc, p.z_near, p.z_far), 0.35f);
    return pack_argb8(Vec3(vis, vis, vis));
}

// Portable N-wide float SIMD (AVX2 8-wide, NEON / fallback 4-wide).
struct F32x {
#if defined(SR_SIMD_AVX2)
    __m256 v{};
    static constexpr int N = 8;
    F32x() = default;
    explicit F32x(__m256 x) : v(x) {}
    static F32x splat(float x) { return F32x(_mm256_set1_ps(x)); }
    static F32x loadu(const float* p) { return F32x(_mm256_loadu_ps(p)); }
    static F32x lane_index() { return F32x(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7)); }
    void storeu(float* p) const { _mm256_storeu_ps(p, v); }
    friend F32x operator+(F32x a, F32x b) { return F32x(_mm256_add_ps(a.v, b.v)); }
    friend F32x operator-(F32x a, F32x b) { return F32x(_mm256_sub_ps(a.v, b.v)); }
    friend F32x operator*(F32x a, F32x b) { return F32x(_mm256_mul_ps(a.v, b.v)); }
    friend F32x operator/(F32x a, F32x b) { return F32x(_mm256_div_ps(a.v, b.v)); }
    F32x operator-() const { return F32x(_mm256_sub_ps(_mm256_setzero_ps(), v)); }
    static F32x min_(F32x a, F32x b) { return F32x(_mm256_min_ps(a.v, b.v)); }
    static F32x max_(F32x a, F32x b) { return F32x(_mm256_max_ps(a.v, b.v)); }
    static F32x sqrt_(F32x a) { return F32x(_mm256_sqrt_ps(a.v)); }
    static F32x abs_(F32x a) {
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        return F32x(_mm256_and_ps(a.v, mask));
    }
    static F32x floor_(F32x a) { return F32x(_mm256_floor_ps(a.v)); }
#elif defined(SR_SIMD_NEON)
    float32x4_t v{};
    static constexpr int N = 4;
    F32x() = default;
    explicit F32x(float32x4_t x) : v(x) {}
    static F32x splat(float x) { return F32x(vdupq_n_f32(x)); }
    static F32x loadu(const float* p) { return F32x(vld1q_f32(p)); }
    static F32x lane_index() {
        const float idx[4] = {0.f, 1.f, 2.f, 3.f};
        return loadu(idx);
    }
    void storeu(float* p) const { vst1q_f32(p, v); }
    friend F32x operator+(F32x a, F32x b) { return F32x(vaddq_f32(a.v, b.v)); }
    friend F32x operator-(F32x a, F32x b) { return F32x(vsubq_f32(a.v, b.v)); }
    friend F32x operator*(F32x a, F32x b) { return F32x(vmulq_f32(a.v, b.v)); }
    friend F32x operator/(F32x a, F32x b) { return F32x(vdivq_f32(a.v, b.v)); }
    F32x operator-() const { return F32x(vnegq_f32(v)); }
    static F32x min_(F32x a, F32x b) { return F32x(vminq_f32(a.v, b.v)); }
    static F32x max_(F32x a, F32x b) { return F32x(vmaxq_f32(a.v, b.v)); }
    static F32x sqrt_(F32x a) { return F32x(vsqrtq_f32(a.v)); }
    static F32x abs_(F32x a) { return F32x(vabsq_f32(a.v)); }
    static F32x floor_(F32x a) { return F32x(vrndmq_f32(a.v)); }
#else
    alignas(16) float v[4]{};
    static constexpr int N = 4;
    F32x() = default;
    static F32x splat(float x) {
        F32x r;
        r.v[0] = r.v[1] = r.v[2] = r.v[3] = x;
        return r;
    }
    static F32x loadu(const float* p) {
        F32x r;
        std::memcpy(r.v, p, sizeof(v));
        return r;
    }
    static F32x lane_index() {
        F32x r;
        r.v[0] = 0;
        r.v[1] = 1;
        r.v[2] = 2;
        r.v[3] = 3;
        return r;
    }
    void storeu(float* p) const { std::memcpy(p, v, sizeof(v)); }
    friend F32x operator+(F32x a, F32x b) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = a.v[i] + b.v[i];
        return r;
    }
    friend F32x operator-(F32x a, F32x b) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = a.v[i] - b.v[i];
        return r;
    }
    friend F32x operator*(F32x a, F32x b) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = a.v[i] * b.v[i];
        return r;
    }
    friend F32x operator/(F32x a, F32x b) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = a.v[i] / b.v[i];
        return r;
    }
    F32x operator-() const {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = -v[i];
        return r;
    }
    static F32x min_(F32x a, F32x b) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = std::min(a.v[i], b.v[i]);
        return r;
    }
    static F32x max_(F32x a, F32x b) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = std::max(a.v[i], b.v[i]);
        return r;
    }
    static F32x sqrt_(F32x a) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = std::sqrt(a.v[i]);
        return r;
    }
    static F32x abs_(F32x a) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = std::fabs(a.v[i]);
        return r;
    }
    static F32x floor_(F32x a) {
        F32x r;
        for (int i = 0; i < N; ++i) r.v[i] = std::floor(a.v[i]);
        return r;
    }
#endif

    F32x& operator+=(F32x o) { return *this = *this + o; }
    F32x& operator-=(F32x o) { return *this = *this - o; }
    F32x& operator*=(F32x o) { return *this = *this * o; }

    template <typename Fn>
    F32x map(Fn fn) const {
        alignas(32) float tmp[N];
        storeu(tmp);
        for (int i = 0; i < N; ++i) tmp[i] = fn(tmp[i]);
        return loadu(tmp);
    }
};

struct Mask {
#if defined(SR_SIMD_AVX2)
    __m256 v{};
    explicit Mask(__m256 x) : v(x) {}
    static Mask from_cmp_ge(F32x a, F32x b) { return Mask(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ)); }
    static Mask from_cmp_le(F32x a, F32x b) { return Mask(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)); }
    static Mask from_cmp_lt(F32x a, F32x b) { return Mask(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)); }
    static Mask from_cmp_gt(F32x a, F32x b) { return Mask(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)); }
    Mask operator&(Mask o) const { return Mask(_mm256_and_ps(v, o.v)); }
    Mask operator|(Mask o) const { return Mask(_mm256_or_ps(v, o.v)); }
    bool any() const { return _mm256_movemask_ps(v) != 0; }
    int count() const { return __builtin_popcount(static_cast<unsigned>(_mm256_movemask_ps(v))); }
    F32x blend(F32x a, F32x b) const { return F32x(_mm256_blendv_ps(b.v, a.v, v)); }
    void store_u32_mask(uint32_t* bits) const {
        const int m = _mm256_movemask_ps(v);
        for (int i = 0; i < F32x::N; ++i) bits[i] = (m & (1 << i)) ? 0xffffffffu : 0u;
    }
#elif defined(SR_SIMD_NEON)
    uint32x4_t v{};
    explicit Mask(uint32x4_t x) : v(x) {}
    static Mask from_cmp_ge(F32x a, F32x b) { return Mask(vcgeq_f32(a.v, b.v)); }
    static Mask from_cmp_le(F32x a, F32x b) { return Mask(vcleq_f32(a.v, b.v)); }
    static Mask from_cmp_lt(F32x a, F32x b) { return Mask(vcltq_f32(a.v, b.v)); }
    static Mask from_cmp_gt(F32x a, F32x b) { return Mask(vcgtq_f32(a.v, b.v)); }
    Mask operator&(Mask o) const { return Mask(vandq_u32(v, o.v)); }
    Mask operator|(Mask o) const { return Mask(vorrq_u32(v, o.v)); }
    bool any() const {
        const uint32x2_t s = vorr_u32(vget_low_u32(v), vget_high_u32(v));
        return (vget_lane_u32(s, 0) | vget_lane_u32(s, 1)) != 0;
    }
    int count() const {
        const uint32x4_t ones = vshrq_n_u32(v, 31);
        const uint32x2_t p = vadd_u32(vget_low_u32(ones), vget_high_u32(ones));
        return static_cast<int>(vget_lane_u32(p, 0) + vget_lane_u32(p, 1));
    }
    F32x blend(F32x a, F32x b) const { return F32x(vbslq_f32(v, a.v, b.v)); }
    void store_u32_mask(uint32_t* bits) const { vst1q_u32(bits, v); }
#else
    uint32_t v[4]{};
    static Mask from_cmp_ge(F32x a, F32x b) {
        Mask m;
        for (int i = 0; i < 4; ++i) m.v[i] = a.v[i] >= b.v[i] ? 0xffffffffu : 0u;
        return m;
    }
    static Mask from_cmp_le(F32x a, F32x b) {
        Mask m;
        for (int i = 0; i < 4; ++i) m.v[i] = a.v[i] <= b.v[i] ? 0xffffffffu : 0u;
        return m;
    }
    static Mask from_cmp_lt(F32x a, F32x b) {
        Mask m;
        for (int i = 0; i < 4; ++i) m.v[i] = a.v[i] < b.v[i] ? 0xffffffffu : 0u;
        return m;
    }
    static Mask from_cmp_gt(F32x a, F32x b) {
        Mask m;
        for (int i = 0; i < 4; ++i) m.v[i] = a.v[i] > b.v[i] ? 0xffffffffu : 0u;
        return m;
    }
    Mask operator&(Mask o) const {
        Mask m;
        for (int i = 0; i < 4; ++i) m.v[i] = v[i] & o.v[i];
        return m;
    }
    Mask operator|(Mask o) const {
        Mask m;
        for (int i = 0; i < 4; ++i) m.v[i] = v[i] | o.v[i];
        return m;
    }
    bool any() const { return (v[0] | v[1] | v[2] | v[3]) != 0; }
    int count() const {
        return (v[0] ? 1 : 0) + (v[1] ? 1 : 0) + (v[2] ? 1 : 0) + (v[3] ? 1 : 0);
    }
    F32x blend(F32x a, F32x b) const {
        F32x r;
        for (int i = 0; i < 4; ++i) r.v[i] = v[i] ? a.v[i] : b.v[i];
        return r;
    }
    void store_u32_mask(uint32_t* bits) const { std::memcpy(bits, v, sizeof(v)); }
#endif
};

inline F32x fdot(F32x ax, F32x ay, F32x az, F32x bx, F32x by, F32x bz) {
    return ax * bx + ay * by + az * bz;
}

inline void fnormalize(F32x& x, F32x& y, F32x& z) {
    F32x ls = fdot(x, y, z, x, y, z);
    ls = F32x::max_(ls, F32x::splat(1e-12f));
    const F32x inv = F32x::splat(1.f) / F32x::sqrt_(ls);
    x = x * inv;
    y = y * inv;
    z = z * inv;
}

inline void checker_albedo(F32x u, F32x v, F32x& r, F32x& g, F32x& b) {
    const F32x tiles = F32x::splat(8.f);
    const F32x iu = F32x::floor_(u * tiles);
    const F32x iv = F32x::floor_(v * tiles);
    const F32x sum = iu + iv;
    alignas(32) float s[F32x::N];
    sum.storeu(s);
    alignas(32) float mr[F32x::N], mg[F32x::N], mb[F32x::N];
    for (int i = 0; i < F32x::N; ++i) {
        const int odd = static_cast<int>(std::floor(s[i])) & 1;
        if (odd) {
            mr[i] = 0.92f;
            mg[i] = 0.90f;
            mb[i] = 0.84f;
        } else {
            mr[i] = 0.18f;
            mg[i] = 0.19f;
            mb[i] = 0.22f;
        }
    }
    r = F32x::loadu(mr);
    g = F32x::loadu(mg);
    b = F32x::loadu(mb);
}

inline void sample_albedo_lanes(F32x u, F32x v, const Texture* tex, const Mask& live, F32x& r, F32x& g,
                                F32x& b) {
    if (!tex) {
        checker_albedo(u, v, r, g, b);
        return;
    }
    alignas(32) float us[F32x::N], vs[F32x::N];
    alignas(32) uint32_t bits[F32x::N];
    u.storeu(us);
    v.storeu(vs);
    live.store_u32_mask(bits);
    alignas(32) float mr[F32x::N], mg[F32x::N], mb[F32x::N];
    for (int i = 0; i < F32x::N; ++i) {
        if (!bits[i]) {
            mr[i] = mg[i] = mb[i] = 0.f;
            continue;
        }
        const Vec3 c = tex->sample_bilinear(us[i], vs[i]);
        mr[i] = c.x;
        mg[i] = c.y;
        mb[i] = c.z;
    }
    r = F32x::loadu(mr);
    g = F32x::loadu(mg);
    b = F32x::loadu(mb);
}

struct ShadeBundle {
    F32x r, g, b;
};

inline ShadeBundle shade_pixels_simd(F32x px, F32x py, F32x pz, F32x nx, F32x ny, F32x nz, F32x u, F32x v,
                                     const Mask& live, const ShadeParams& p) {
    fnormalize(nx, ny, nz);
    const F32x lx = F32x::splat(p.light_dir.x);
    const F32x ly = F32x::splat(p.light_dir.y);
    const F32x lz = F32x::splat(p.light_dir.z);
    F32x vx = F32x::splat(p.camera_pos.x) - px;
    F32x vy = F32x::splat(p.camera_pos.y) - py;
    F32x vz = F32x::splat(p.camera_pos.z) - pz;
    fnormalize(vx, vy, vz);

    F32x hx = lx + vx, hy = ly + vy, hz = lz + vz;
    fnormalize(hx, hy, hz);

    const F32x ndotl = F32x::max_(fdot(nx, ny, nz, lx, ly, lz), F32x::splat(0.f));
    const F32x ndoth = F32x::max_(fdot(nx, ny, nz, hx, hy, hz), F32x::splat(0.f));
    F32x spec = ndoth.map([&](float x) { return std::pow(x, p.shininess); }) * F32x::splat(p.spec_strength);

    const F32x hem = ny * F32x::splat(0.5f) + F32x::splat(0.5f);
    const F32x ao = F32x::splat(0.35f) + hem * F32x::splat(0.65f);
    const F32x amb_k = ao * F32x::splat(p.ambient_strength);
    const F32x ar = (F32x::splat(p.ground.x) + (F32x::splat(p.sky.x) - F32x::splat(p.ground.x)) * hem) * amb_k;
    const F32x ag = (F32x::splat(p.ground.y) + (F32x::splat(p.sky.y) - F32x::splat(p.ground.y)) * hem) * amb_k;
    const F32x ab = (F32x::splat(p.ground.z) + (F32x::splat(p.sky.z) - F32x::splat(p.ground.z)) * hem) * amb_k;

    F32x alb_r, alb_g, alb_b;
    sample_albedo_lanes(u, v, p.texture, live, alb_r, alb_g, alb_b);

    const F32x lr = F32x::splat(p.light_color.x);
    const F32x lg = F32x::splat(p.light_color.y);
    const F32x lb = F32x::splat(p.light_color.z);

    F32x cr = alb_r * ar + alb_r * lr * ndotl + lr * spec;
    F32x cg = alb_g * ag + alb_g * lg * ndotl + lg * spec;
    F32x cb = alb_b * ab + alb_b * lb * ndotl + lb * spec;

    cr = cr / (cr + F32x::splat(1.f));
    cg = cg / (cg + F32x::splat(1.f));
    cb = cb / (cb + F32x::splat(1.f));
    cr = cr.map([](float x) { return linear_to_srgb(x); });
    cg = cg.map([](float x) { return linear_to_srgb(x); });
    cb = cb.map([](float x) { return linear_to_srgb(x); });
    return {cr, cg, cb};
}

inline ShadeBundle shade_depth_simd(F32x z_ndc, const ShadeParams& p) {
    const F32x vis = z_ndc.map([&](float z) {
        return std::pow(linearize_depth(z, p.z_near, p.z_far), 0.35f);
    });
    return {vis, vis, vis};
}

inline void store_argb_masked(uint32_t* dst, const ShadeBundle& c, const Mask& live) {
#if defined(SR_SIMD_NEON)
    const float32x4_t s = vdupq_n_f32(255.f);
    const float32x4_t z = vdupq_n_f32(0.f);
    auto quant = [&](float32x4_t ch) {
        ch = vminq_f32(vmaxq_f32(vmulq_f32(ch, s), z), s);
        return vcvtq_u32_f32(ch);
    };
    const uint32x4_t packed =
        vorrq_u32(vdupq_n_u32(0xFF000000u),
                  vorrq_u32(vshlq_n_u32(quant(c.r.v), 16),
                            vorrq_u32(vshlq_n_u32(quant(c.g.v), 8), quant(c.b.v))));
    const uint32x4_t old = vld1q_u32(dst);
    vst1q_u32(dst, vbslq_u32(live.v, packed, old));
#elif defined(SR_SIMD_AVX2)
    const __m256 s = _mm256_set1_ps(255.f);
    const __m256 z = _mm256_setzero_ps();
    auto quant = [&](__m256 ch) {
        ch = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(ch, s), z), s);
        return _mm256_cvtps_epi32(ch);
    };
    const __m256i packed = _mm256_or_si256(
        _mm256_set1_epi32(static_cast<int>(0xFF000000)),
        _mm256_or_si256(_mm256_slli_epi32(quant(c.r.v), 16),
                        _mm256_or_si256(_mm256_slli_epi32(quant(c.g.v), 8), quant(c.b.v))));
    const __m256i old = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst));
    const __m256i blended = _mm256_blendv_epi8(old, packed, _mm256_castps_si256(live.v));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), blended);
#else
    alignas(32) float r[F32x::N], g[F32x::N], b[F32x::N];
    alignas(32) uint32_t bits[F32x::N];
    c.r.storeu(r);
    c.g.storeu(g);
    c.b.storeu(b);
    live.store_u32_mask(bits);
    for (int i = 0; i < F32x::N; ++i) {
        if (bits[i]) dst[i] = pack_argb8(r[i], g[i], b[i]);
    }
#endif
}

inline void store_depth_masked(float* dst, F32x zwin, const Mask& live) {
#if defined(SR_SIMD_NEON)
    const float32x4_t old = vld1q_f32(dst);
    vst1q_f32(dst, vbslq_f32(live.v, zwin.v, old));
#elif defined(SR_SIMD_AVX2)
    const __m256 old = _mm256_loadu_ps(dst);
    _mm256_storeu_ps(dst, _mm256_blendv_ps(old, zwin.v, live.v));
#else
    alignas(32) float z[F32x::N];
    alignas(32) uint32_t bits[F32x::N];
    zwin.storeu(z);
    live.store_u32_mask(bits);
    for (int i = 0; i < F32x::N; ++i) {
        if (bits[i]) dst[i] = z[i];
    }
#endif
}

}  // namespace sr
