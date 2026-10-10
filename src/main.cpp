#include "app.hpp"
#include "math/vec_mat.hpp"
#include "render/framebuffer.hpp"
#include "render/mesh.hpp"
#include "render/pipeline.hpp"
#include "render/rasterizer.hpp"
#include "render/simd_shader.hpp"
#include "render/texture.hpp"
#include "ui/debug_overlay.hpp"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"

#include <SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using SteadyClock = std::chrono::steady_clock;

namespace {

fs::path find_assets() {
#ifdef SR_ASSETS_DIR
    const fs::path cmake_dir(SR_ASSETS_DIR);
    if (fs::exists(cmake_dir / "obj")) return cmake_dir;
#endif
    const fs::path cwd = fs::current_path();
    const fs::path candidates[] = {cwd / "assets", cwd / ".." / "assets", cwd / ".." / ".." / "assets"};
    for (const auto& p : candidates) {
        if (fs::exists(p / "obj")) return p;
    }
    return cwd / "assets";
}

sr::Vec3 orbit_eye(float azimuth, float elevation, float distance) {
    const float ce = std::cos(elevation);
    return {distance * ce * std::sin(azimuth), distance * std::sin(elevation),
            distance * ce * std::cos(azimuth)};
}

double ms_since(SteadyClock::time_point t0) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - t0).count();
}

void render_scene(sr::Framebuffer& fb, const sr::Mesh& ground, const sr::Mesh& object, const sr::Texture& tex_ground,
                  const sr::Texture& tex_object, const sr::AppState& app, sr::ShadeMode shade_mode,
                  sr::DrawStats& stats) {
    const sr::Vec3 eye = orbit_eye(app.cam_azimuth, app.cam_elevation, app.cam_distance);
    const sr::Vec3 target{0.f, 0.55f, 0.f};
    const float aspect = static_cast<float>(fb.w) / static_cast<float>(std::max(fb.h, 1));

    sr::Uniforms u;
    u.view = sr::look_at(eye, target, {0.f, 1.f, 0.f});
    u.proj = sr::perspective(sr::radians(app.fov_deg), aspect, app.z_near, app.z_far);
    u.camera_pos = eye;
    u.light_dir = app.light_dir();
    u.light_color = sr::Vec3{1.f, 0.96f, 0.90f} * app.light_intensity;
    u.shininess = app.shininess;
    u.spec_strength = app.spec_strength;
    u.z_near = app.z_near;
    u.z_far = app.z_far;
    u.width = fb.w;
    u.height = fb.h;
    u.backface_cull = app.backface_cull;

    const sr::RasterMode rm = (app.pass == sr::RenderPass::Wireframe) ? sr::RasterMode::Wireframe
                            : (app.pass == sr::RenderPass::Depth)     ? sr::RasterMode::Depth
                                                                     : sr::RasterMode::Shaded;

    u.model = sr::Mat4::identity();
    u.mvp = u.proj * u.view * u.model;
    auto shade_g = sr::shade_from_uniforms(u, &tex_ground);
    sr::draw_mesh(fb, ground, u, shade_g, rm, shade_mode, stats);

    u.model = sr::translate({0.f, 1.25f, 0.f}) * sr::rotate_y(app.model_yaw) * sr::rotate_x(app.model_pitch) *
              sr::rotate_z(app.model_roll);
    u.mvp = u.proj * u.view * u.model;
    auto shade_o = sr::shade_from_uniforms(u, &tex_object);
    sr::draw_mesh(fb, object, u, shade_o, rm, shade_mode, stats);
}

SDL_Texture* make_fb_texture(SDL_Renderer* renderer, int w, int h) {
    return SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);
}

bool write_ppm(const sr::Framebuffer& fb, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", fb.w, fb.h);
    for (int i = 0; i < fb.w * fb.h; ++i) {
        const uint32_t p = fb.color[static_cast<size_t>(i)];
        const unsigned char rgb[3] = {
            static_cast<unsigned char>((p >> 16) & 255u),
            static_cast<unsigned char>((p >> 8) & 255u),
            static_cast<unsigned char>(p & 255u),
        };
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
    return true;
}

int run_dump(const char* path) {
    sr::AppState app;
    app.fb_w = 960;
    app.fb_h = 540;
    app.dual_benchmark = false;
    app.path = sr::ShadePath::SIMD;
    app.ssao = true;
    app.auto_rotate = false;

    const sr::Mesh ground = sr::make_plane(14.f, 8);
    const sr::Mesh torus = sr::make_torus(1.05f, 0.38f, 48, 24);
    sr::Mesh cube_check;
    std::string cube_err;
    const bool cube_ok = sr::load_obj((find_assets() / "obj" / "cube.obj").string(), cube_check, &cube_err);
    std::printf("cube.obj: %s (%d tris)%s\n", cube_ok ? "ok" : "fail", cube_check.triangle_count(),
                cube_ok ? "" : (" " + cube_err).c_str());
    const sr::Texture tex_ground = sr::Texture::make_checker(256, 8, {0.18f, 0.19f, 0.21f}, {0.55f, 0.56f, 0.52f});
    const sr::Texture tex_brick = sr::Texture::make_brick(256);

    sr::Framebuffer fb_simd(app.fb_w, app.fb_h);
    sr::Framebuffer fb_scalar(app.fb_w, app.fb_h);
    const uint32_t clear = sr::pack_argb8(0.035f, 0.04f, 0.055f);
    fb_simd.clear(clear);
    fb_scalar.clear(clear);
    sr::DrawStats stats_simd, stats_scalar;
    const auto t_simd = SteadyClock::now();
    render_scene(fb_simd, ground, torus, tex_ground, tex_brick, app, sr::ShadeMode::SIMD, stats_simd);
    const double ms_simd = ms_since(t_simd);
    const auto t_scalar = SteadyClock::now();
    render_scene(fb_scalar, ground, torus, tex_ground, tex_brick, app, sr::ShadeMode::Scalar, stats_scalar);
    const double ms_scalar = ms_since(t_scalar);

    int mismatches = 0;
    int max_delta = 0;
    const size_t n = fb_simd.color.size();
    for (size_t i = 0; i < n; ++i) {
        const uint32_t a = fb_simd.color[i], b = fb_scalar.color[i];
        int d = 0;
        d = std::max(d, std::abs(static_cast<int>((a >> 16) & 255) - static_cast<int>((b >> 16) & 255)));
        d = std::max(d, std::abs(static_cast<int>((a >> 8) & 255) - static_cast<int>((b >> 8) & 255)));
        d = std::max(d, std::abs(static_cast<int>(a & 255) - static_cast<int>(b & 255)));
        if (d > 2) ++mismatches;
        max_delta = std::max(max_delta, d);
    }

    if (app.ssao) sr::apply_ssao(fb_simd, app.ao_strength);
    if (!write_ppm(fb_simd, path)) {
        std::fprintf(stderr, "failed to write %s\n", path);
        return 1;
    }
    std::printf("wrote %s\n", path);
    std::printf("  SIMD   %.2f ms  %d tris  %d pixels  (%s)\n", ms_simd, stats_simd.tris_drawn, stats_simd.pixels,
                sr::simd_isa_name());
    std::printf("  Scalar %.2f ms  %d tris  %d pixels\n", ms_scalar, stats_scalar.tris_drawn, stats_scalar.pixels);
    std::printf("  speedup %.2fx   color delta: %d mismatch (thr>2), max |d|=%d\n",
                (ms_simd > 0.01) ? (ms_scalar / ms_simd) : 0.0, mismatches, max_delta);

    if (std::getenv("SR_DUMP_PASSES")) {
        const sr::Texture tex_metal = sr::Texture::make_grid_metal(256);
        auto dump_pass = [&](sr::RenderPass pass, const sr::Mesh& object, const sr::Texture& tex,
                             const char* tag) {
            sr::AppState a = app;
            a.pass = pass;
            a.ssao = (pass == sr::RenderPass::Shaded);
            sr::Framebuffer f(app.fb_w, app.fb_h);
            f.clear(clear);
            sr::DrawStats st;
            render_scene(f, ground, object, tex_ground, tex, a, sr::ShadeMode::SIMD, st);
            if (a.ssao) sr::apply_ssao(f, a.ao_strength);
            std::string out = std::string("/tmp/sr-") + tag + ".ppm";
            write_ppm(f, out);
            std::printf("  extra %s (%d px)\n", out.c_str(), st.pixels);
        };
        dump_pass(sr::RenderPass::Depth, torus, tex_brick, "depth");
        dump_pass(sr::RenderPass::Wireframe, torus, tex_brick, "wire");
        if (cube_ok) dump_pass(sr::RenderPass::Shaded, cube_check, tex_metal, "cube");
        dump_pass(sr::RenderPass::Shaded, sr::make_uv_sphere(1.15f, 40, 24), tex_metal, "sphere");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--dump") == 0) {
        const char* path = (argc >= 3) ? argv[2] : "frame.ppm";
        return run_dump(path);
    }

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "software-renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1440, 900, SDL_WINDOW_RESIZABLE);
    if (!window) {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    if (!renderer) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        SDL_Log("SDL_CreateRenderer failed: %s", SDL_GetError());
        return 1;
    }
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.f;
    style.FrameRounding = 4.f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.07f, 0.08f, 0.10f, 0.92f);
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);

    sr::AppState app;
    app.simd_isa = sr::simd_isa_name();

    sr::Mesh ground = sr::make_plane(14.f, 8);
    sr::Mesh torus = sr::make_torus(1.05f, 0.38f, 48, 24);
    sr::Mesh sphere = sr::make_uv_sphere(1.15f, 40, 24);
    sr::Mesh cube;
    const fs::path assets = find_assets();
    std::string obj_err;
    if (!sr::load_obj((assets / "obj" / "cube.obj").string(), cube, &obj_err)) {
        cube = sr::make_uv_sphere(1.0f, 8, 6);
        app.status = "cube.obj missing — using generated stand-in. " + obj_err;
    } else {
        app.status = "Loaded " + (assets / "obj" / "cube.obj").string();
    }

    sr::Texture tex_ground = sr::Texture::make_checker(256, 8, {0.18f, 0.19f, 0.21f}, {0.55f, 0.56f, 0.52f});
    sr::Texture tex_brick = sr::Texture::make_brick(256);
    sr::Texture tex_metal = sr::Texture::make_grid_metal(256);
    sr::Texture tex_ppm;
    if (sr::Texture::load_ppm((assets / "obj" / "checker.ppm").string(), tex_ppm)) tex_ground = tex_ppm;

    sr::Framebuffer fb_scalar(app.fb_w, app.fb_h);
    sr::Framebuffer fb_simd(app.fb_w, app.fb_h);
    SDL_Texture* blit = make_fb_texture(renderer, app.fb_w, app.fb_h);

    bool running = true;
    auto last_title = SteadyClock::now();
    double ema_ms = 16.0;
    int last_w = 0, last_h = 0;

    while (running) {
        const auto frame_t0 = SteadyClock::now();
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_CLOSE) running = false;
            if (e.type == SDL_MOUSEWHEEL && !io.WantCaptureMouse) {
                app.cam_distance = sr::clamp(app.cam_distance - e.wheel.y * 0.45f, 2.f, 18.f);
            }
            if (e.type == SDL_MOUSEMOTION && (e.motion.state & SDL_BUTTON_LMASK) && !io.WantCaptureMouse) {
                app.cam_azimuth -= e.motion.xrel * 0.005f;
                app.cam_elevation = sr::clamp(app.cam_elevation + e.motion.yrel * 0.004f, 0.05f, 1.45f);
            }
            if (e.type == SDL_KEYDOWN && !io.WantCaptureKeyboard) {
                switch (e.key.keysym.sym) {
                    case SDLK_ESCAPE: running = false; break;
                    case SDLK_1: app.pass = sr::RenderPass::Shaded; break;
                    case SDLK_2: app.pass = sr::RenderPass::Wireframe; break;
                    case SDLK_3: app.pass = sr::RenderPass::Depth; break;
                    case SDLK_s:
                        app.path = (app.path == sr::ShadePath::SIMD) ? sr::ShadePath::Scalar : sr::ShadePath::SIMD;
                        break;
                    default: break;
                }
            }
        }

        int rw = 0, rh = 0;
        SDL_GetRendererOutputSize(renderer, &rw, &rh);
        if (rw < 64) rw = 64;
        if (rh < 64) rh = 64;
        if (rw != last_w || rh != last_h) {
            last_w = rw;
            last_h = rh;
            app.fb_w = rw;
            app.fb_h = rh;
            fb_scalar.resize(rw, rh);
            fb_simd.resize(rw, rh);
            if (blit) SDL_DestroyTexture(blit);
            blit = make_fb_texture(renderer, rw, rh);
        }

        SDL_RenderSetVSync(renderer, app.vsync ? 1 : 0);

        if (app.auto_rotate) app.model_yaw = std::fmod(app.model_yaw + 0.006f, sr::kTau);

        const sr::Mesh* object = &torus;
        const sr::Texture* tex_obj = &tex_brick;
        if (app.mesh == sr::MeshId::Sphere) {
            object = &sphere;
            tex_obj = &tex_metal;
        } else if (app.mesh == sr::MeshId::Cube) {
            object = &cube;
            tex_obj = &tex_metal;
        }

        const uint32_t clear = sr::pack_argb8(0.035f, 0.04f, 0.055f);
        sr::DrawStats stats_scalar, stats_simd;

        if (app.dual_benchmark || app.path == sr::ShadePath::Scalar) {
            fb_scalar.clear(clear);
            const auto t0 = SteadyClock::now();
            render_scene(fb_scalar, ground, *object, tex_ground, *tex_obj, app, sr::ShadeMode::Scalar, stats_scalar);
            app.scalar_ms = static_cast<float>(ms_since(t0));
        }
        if (app.dual_benchmark || app.path == sr::ShadePath::SIMD) {
            fb_simd.clear(clear);
            const auto t0 = SteadyClock::now();
            render_scene(fb_simd, ground, *object, tex_ground, *tex_obj, app, sr::ShadeMode::SIMD, stats_simd);
            app.simd_ms = static_cast<float>(ms_since(t0));
        }

        sr::Framebuffer& shown = (app.path == sr::ShadePath::SIMD) ? fb_simd : fb_scalar;
        if (app.ssao && app.pass == sr::RenderPass::Shaded) sr::apply_ssao(shown, app.ao_strength);
        const sr::DrawStats& shown_stats =
            (app.path == sr::ShadePath::SIMD) ? stats_simd : stats_scalar;
        app.triangles_submitted = shown_stats.tris_submitted;
        app.triangles_drawn = shown_stats.tris_drawn;
        app.pixels_shaded = shown_stats.pixels;
        app.push_timings();

        if (blit) {
            void* pixels = nullptr;
            int pitch = 0;
            if (SDL_LockTexture(blit, nullptr, &pixels, &pitch) == 0) {
                const auto* src = shown.color.data();
                auto* dst = static_cast<uint8_t*>(pixels);
                const int row_bytes = shown.w * static_cast<int>(sizeof(uint32_t));
                if (pitch == row_bytes) {
                    std::memcpy(dst, src, static_cast<size_t>(row_bytes) * static_cast<size_t>(shown.h));
                } else {
                    for (int y = 0; y < shown.h; ++y) {
                        std::memcpy(dst + static_cast<size_t>(y) * static_cast<size_t>(pitch),
                                    src + static_cast<size_t>(y) * static_cast<size_t>(shown.w),
                                    static_cast<size_t>(row_bytes));
                    }
                }
                SDL_UnlockTexture(blit);
            }
        }

        SDL_RenderClear(renderer);
        if (blit) SDL_RenderCopy(renderer, blit, nullptr, nullptr);

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        sr::draw_debug_overlay(app);
        ImGui::Render();
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);

        const double frame_ms = ms_since(frame_t0);
        ema_ms = ema_ms * 0.9 + frame_ms * 0.1;
        app.frame_ms = static_cast<float>(ema_ms);
        app.fps = (ema_ms > 0.01) ? static_cast<float>(1000.0 / ema_ms) : 0.f;

        if (ms_since(last_title) > 500.0) {
            char title[160];
            std::snprintf(title, sizeof(title), "software-renderer  ·  %.0f FPS  ·  %s  ·  %s", app.fps, app.simd_isa,
                          (app.path == sr::ShadePath::SIMD) ? "SIMD" : "Scalar");
            SDL_SetWindowTitle(window, title);
            last_title = SteadyClock::now();
        }
    }

    if (blit) SDL_DestroyTexture(blit);
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
