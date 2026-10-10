#pragma once

#include "math/vec_mat.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace sr {

enum class RenderPass : int { Shaded = 0, Wireframe = 1, Depth = 2 };
enum class ShadePath : int { Scalar = 0, SIMD = 1 };
enum class MeshId : int { Torus = 0, Sphere = 1, Cube = 2 };

struct AppState {
    RenderPass pass = RenderPass::Shaded;
    ShadePath path = ShadePath::SIMD;
    MeshId mesh = MeshId::Torus;

    bool dual_benchmark = true;
    bool backface_cull = true;
    bool ssao = true;
    bool auto_rotate = true;
    bool vsync = false;

    float fov_deg = 60.f;
    float cam_azimuth = 0.55f;
    float cam_elevation = 0.42f;
    float cam_distance = 7.0f;
    float z_near = 0.1f;
    float z_far = 80.f;

    float model_yaw = 0.4f;
    float model_pitch = 0.25f;
    float model_roll = 0.f;

    float light_yaw = 0.7f;
    float light_pitch = 1.05f;
    float light_intensity = 1.15f;
    float shininess = 48.f;
    float spec_strength = 0.55f;
    float ao_strength = 0.85f;

    float fps = 0.f;
    float frame_ms = 0.f;
    float scalar_ms = 0.f;
    float simd_ms = 0.f;
    int triangles_submitted = 0;
    int triangles_drawn = 0;
    int pixels_shaded = 0;
    int fb_w = 1280;
    int fb_h = 720;
    const char* simd_isa = "detecting";
    std::string status;

    static constexpr int kHist = 128;
    std::array<float, kHist> hist_scalar{};
    std::array<float, kHist> hist_simd{};
    int hist_i = 0;

    Vec3 light_dir() const {
        const float cp = std::cos(light_pitch);
        return normalize(Vec3{std::sin(light_yaw) * cp, std::sin(light_pitch), std::cos(light_yaw) * cp});
    }

    void push_timings() {
        hist_scalar[static_cast<size_t>(hist_i)] = scalar_ms;
        hist_simd[static_cast<size_t>(hist_i)] = simd_ms;
        hist_i = (hist_i + 1) % kHist;
    }
};

void draw_debug_overlay(AppState& state);

}  // namespace sr
