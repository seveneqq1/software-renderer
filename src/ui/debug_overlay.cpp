#include "debug_overlay.hpp"

#include "imgui.h"

#include <cfloat>

namespace sr {

void draw_debug_overlay(AppState& s) {
    ImGui::SetNextWindowPos(ImVec2(16.f, 16.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380.f, 640.f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("software-renderer", nullptr, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    ImGui::TextUnformatted("CPU 3D pipeline  ·  C++20");
    ImGui::Text("ISA: %s", s.simd_isa);
    ImGui::Text("Framebuffer %d × %d", s.fb_w, s.fb_h);
    ImGui::Separator();

    ImGui::TextUnformatted("Render pass");
    int pass = static_cast<int>(s.pass);
    ImGui::RadioButton("Shaded texture", &pass, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Wireframe", &pass, 1);
    ImGui::RadioButton("Depth (Z) buffer", &pass, 2);
    s.pass = static_cast<RenderPass>(pass);

    int path = static_cast<int>(s.path);
    ImGui::TextUnformatted("Pixel shader");
    ImGui::RadioButton("Scalar", &path, 0);
    ImGui::SameLine();
    ImGui::RadioButton("SIMD", &path, 1);
    s.path = static_cast<ShadePath>(path);

    ImGui::Checkbox("Dual-path benchmark", &s.dual_benchmark);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::TextUnformatted("Rasterize both scalar and SIMD every frame so the graphs stay live.");
        ImGui::EndTooltip();
    }

    ImGui::Separator();
    ImGui::Text("Frame  %.1f FPS   (%.2f ms wall)", s.fps, s.frame_ms);
    ImGui::Text("Scalar raster  %.3f ms", s.scalar_ms);
    ImGui::Text("SIMD raster    %.3f ms", s.simd_ms);
    if (s.scalar_ms > 0.05f && s.simd_ms > 0.05f) {
        ImGui::Text("Speedup        %.2fx", s.scalar_ms / s.simd_ms);
    }
    ImGui::Text("Triangles      %d submitted / %d drawn", s.triangles_submitted, s.triangles_drawn);
    ImGui::Text("Pixels shaded  %d", s.pixels_shaded);

    ImGui::PlotLines("Scalar ms", s.hist_scalar.data(), AppState::kHist, s.hist_i, nullptr, 0.f, FLT_MAX,
                     ImVec2(0.f, 64.f));
    ImGui::PlotLines("SIMD ms", s.hist_simd.data(), AppState::kHist, s.hist_i, nullptr, 0.f, FLT_MAX,
                     ImVec2(0.f, 64.f));

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("FOV (deg)", &s.fov_deg, 20.f, 110.f, "%.1f");
        ImGui::SliderFloat("Azimuth", &s.cam_azimuth, -kPi, kPi, "%.2f");
        ImGui::SliderFloat("Elevation", &s.cam_elevation, 0.05f, 1.45f, "%.2f");
        ImGui::SliderFloat("Distance", &s.cam_distance, 2.f, 18.f, "%.2f");
        ImGui::SliderFloat("Z-near", &s.z_near, 0.05f, 2.f, "%.2f");
        ImGui::SliderFloat("Z-far", &s.z_far, 10.f, 200.f, "%.1f");
    }

    if (ImGui::CollapsingHeader("Object", ImGuiTreeNodeFlags_DefaultOpen)) {
        int mesh = static_cast<int>(s.mesh);
        ImGui::Combo("Mesh", &mesh, "Torus\0Sphere\0Cube (OBJ)\0");
        s.mesh = static_cast<MeshId>(mesh);
        ImGui::Checkbox("Auto rotate", &s.auto_rotate);
        ImGui::SliderFloat("Yaw", &s.model_yaw, -kPi, kPi, "%.2f");
        ImGui::SliderFloat("Pitch", &s.model_pitch, -kPi, kPi, "%.2f");
        ImGui::SliderFloat("Roll", &s.model_roll, -kPi, kPi, "%.2f");
    }

    if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Light yaw", &s.light_yaw, -kPi, kPi, "%.2f");
        ImGui::SliderFloat("Light pitch", &s.light_pitch, 0.05f, 1.5f, "%.2f");
        ImGui::SliderFloat("Intensity", &s.light_intensity, 0.f, 3.f, "%.2f");
        ImGui::SliderFloat("Shininess", &s.shininess, 4.f, 128.f, "%.0f");
        ImGui::SliderFloat("Specular", &s.spec_strength, 0.f, 2.f, "%.2f");
        const Vec3 L = s.light_dir();
        ImGui::Text("Direction  (%.2f, %.2f, %.2f)", L.x, L.y, L.z);
    }

    if (ImGui::CollapsingHeader("Pipeline")) {
        ImGui::Checkbox("Backface cull", &s.backface_cull);
        ImGui::Checkbox("SSAO (depth AO)", &s.ssao);
        ImGui::SliderFloat("AO strength", &s.ao_strength, 0.f, 1.5f, "%.2f");
        ImGui::Checkbox("VSync", &s.vsync);
    }

    if (!s.status.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", s.status.c_str());
    }

    ImGui::Separator();
    ImGui::TextDisabled("Drag LMB orbit  ·  wheel zoom  ·  1/2/3 pass  ·  S SIMD");
    ImGui::End();
}

}  // namespace sr
