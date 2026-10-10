#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sr {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = 6.28318530717958647692f;
constexpr float kEps = 1e-8f;

inline float radians(float deg) { return deg * (kPi / 180.f); }
inline float degrees(float rad) { return rad * (180.f / kPi); }
inline float clamp(float v, float lo, float hi) { return std::min(hi, std::max(lo, v)); }
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline float saturate(float v) { return clamp(v, 0.f, 1.f); }

struct Vec2 {
    float x = 0.f, y = 0.f;
    constexpr Vec2() = default;
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}
};

inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, float s) { return {a.x * s, a.y * s}; }
inline Vec2 operator*(float s, Vec2 a) { return a * s; }
inline Vec2 operator*(Vec2 a, Vec2 b) { return {a.x * b.x, a.y * b.y}; }
inline Vec2 operator/(Vec2 a, float s) { return {a.x / s, a.y / s}; }
inline Vec2& operator+=(Vec2& a, Vec2 b) { a = a + b; return a; }
inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline float length(Vec2 a) { return std::sqrt(dot(a, a)); }

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;
    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    explicit constexpr Vec3(float s) : x(s), y(s), z(s) {}
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3 operator/(Vec3 a, Vec3 b) { return {a.x / b.x, a.y / b.y, a.z / b.z}; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a = a - b; return a; }
inline Vec3& operator*=(Vec3& a, float s) { a = a * s; return a; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length_sq(Vec3 a) { return dot(a, a); }
inline float length(Vec3 a) { return std::sqrt(length_sq(a)); }
inline Vec3 normalize(Vec3 a) {
    const float ls = length_sq(a);
    if (ls < kEps) return {0.f, 1.f, 0.f};
    return a * (1.f / std::sqrt(ls));
}
inline Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }
inline Vec3 min(Vec3 a, Vec3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
inline Vec3 max(Vec3 a, Vec3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
inline Vec3 abs(Vec3 a) { return {std::fabs(a.x), std::fabs(a.y), std::fabs(a.z)}; }
inline Vec3 saturate(Vec3 a) { return {saturate(a.x), saturate(a.y), saturate(a.z)}; }

struct Vec4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
    constexpr Vec4() = default;
    constexpr Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    constexpr Vec4(Vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    constexpr Vec3 xyz() const { return {x, y, z}; }
};

inline Vec4 operator+(Vec4 a, Vec4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline Vec4 operator-(Vec4 a, Vec4 b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
inline Vec4 operator*(Vec4 a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
inline Vec4 operator*(float s, Vec4 a) { return a * s; }

// Column-major 4x4, applied to column vectors: p' = M * p.
struct Mat4 {
    float m[16]{};

    static Mat4 identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f;
        return r;
    }

    float& operator()(int row, int col) { return m[col * 4 + row]; }
    float operator()(int row, int col) const { return m[col * 4 + row]; }
};

inline Vec4 operator*(const Mat4& a, const Vec4& v) {
    return {
        a(0, 0) * v.x + a(0, 1) * v.y + a(0, 2) * v.z + a(0, 3) * v.w,
        a(1, 0) * v.x + a(1, 1) * v.y + a(1, 2) * v.z + a(1, 3) * v.w,
        a(2, 0) * v.x + a(2, 1) * v.y + a(2, 2) * v.z + a(2, 3) * v.w,
        a(3, 0) * v.x + a(3, 1) * v.y + a(3, 2) * v.z + a(3, 3) * v.w};
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            r(row, c) = a(row, 0) * b(0, c) + a(row, 1) * b(1, c) +
                        a(row, 2) * b(2, c) + a(row, 3) * b(3, c);
        }
    }
    return r;
}

inline Vec3 transform_point(const Mat4& m, Vec3 p) {
    const Vec4 r = m * Vec4(p, 1.f);
    const float iw = (std::fabs(r.w) > kEps) ? (1.f / r.w) : 1.f;
    return {r.x * iw, r.y * iw, r.z * iw};
}

inline Vec3 transform_vector(const Mat4& m, Vec3 d) {
    return {m(0, 0) * d.x + m(0, 1) * d.y + m(0, 2) * d.z,
            m(1, 0) * d.x + m(1, 1) * d.y + m(1, 2) * d.z,
            m(2, 0) * d.x + m(2, 1) * d.y + m(2, 2) * d.z};
}

inline Mat4 transpose(const Mat4& a) {
    Mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) r(row, c) = a(c, row);
    return r;
}

inline Mat4 translate(Vec3 t) {
    Mat4 r = Mat4::identity();
    r(0, 3) = t.x;
    r(1, 3) = t.y;
    r(2, 3) = t.z;
    return r;
}

inline Mat4 scale(Vec3 s) {
    Mat4 r = Mat4::identity();
    r(0, 0) = s.x;
    r(1, 1) = s.y;
    r(2, 2) = s.z;
    return r;
}

inline Mat4 scale(float s) { return scale(Vec3(s, s, s)); }

inline Mat4 rotate_x(float a) {
    const float c = std::cos(a), s = std::sin(a);
    Mat4 r = Mat4::identity();
    r(1, 1) = c;
    r(2, 1) = s;
    r(1, 2) = -s;
    r(2, 2) = c;
    return r;
}

inline Mat4 rotate_y(float a) {
    const float c = std::cos(a), s = std::sin(a);
    Mat4 r = Mat4::identity();
    r(0, 0) = c;
    r(2, 0) = -s;
    r(0, 2) = s;
    r(2, 2) = c;
    return r;
}

inline Mat4 rotate_z(float a) {
    const float c = std::cos(a), s = std::sin(a);
    Mat4 r = Mat4::identity();
    r(0, 0) = c;
    r(1, 0) = s;
    r(0, 1) = -s;
    r(1, 1) = c;
    return r;
}

inline Mat4 look_at(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 f = normalize(center - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 m = Mat4::identity();
    m(0, 0) = s.x;
    m(0, 1) = s.y;
    m(0, 2) = s.z;
    m(0, 3) = -dot(s, eye);
    m(1, 0) = u.x;
    m(1, 1) = u.y;
    m(1, 2) = u.z;
    m(1, 3) = -dot(u, eye);
    m(2, 0) = -f.x;
    m(2, 1) = -f.y;
    m(2, 2) = -f.z;
    m(2, 3) = dot(f, eye);
    return m;
}

inline Mat4 perspective(float fovy_rad, float aspect, float z_near, float z_far) {
    const float f = 1.f / std::tan(fovy_rad * 0.5f);
    const float nf = 1.f / (z_near - z_far);
    Mat4 m;
    m(0, 0) = f / aspect;
    m(1, 1) = f;
    m(2, 2) = (z_far + z_near) * nf;
    m(3, 2) = -1.f;
    m(2, 3) = (2.f * z_far * z_near) * nf;
    return m;
}

inline uint32_t pack_argb8(float r, float g, float b, float a = 1.f) {
    const auto q = [](float x) -> uint32_t {
        return static_cast<uint32_t>(clamp(std::round(x * 255.f), 0.f, 255.f));
    };
    return (q(a) << 24) | (q(r) << 16) | (q(g) << 8) | q(b);
}

inline uint32_t pack_argb8(Vec3 c) { return pack_argb8(c.x, c.y, c.z); }

inline Vec3 unpack_rgb8(uint32_t p) {
    return {((p >> 16) & 255u) / 255.f, ((p >> 8) & 255u) / 255.f, (p & 255u) / 255.f};
}

inline float linear_to_srgb(float x) {
    x = saturate(x);
    return (x <= 0.0031308f) ? 12.92f * x : 1.055f * std::pow(x, 1.f / 2.2f) - 0.055f;
}

inline Vec3 linear_to_srgb(Vec3 c) {
    return {linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z)};
}

inline Vec3 reinhard(Vec3 c) { return c / (c + Vec3(1.f)); }

}  // namespace sr
