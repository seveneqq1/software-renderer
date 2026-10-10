# software-renderer

A standalone **CPU 3D software rasterizer** in modern C++20. The full graphics pipeline — model/view/perspective transforms, frustum clipping, edge-function rasterization, a high-resolution Z-buffer, perspective-correct attributes, and Blinn-Phong shading — runs on the CPU. Pixel coverage, depth testing, and shading have a scalar path and a SIMD path (**AVX2** 8-wide on x86_64, **ARM NEON** 4-wide on Apple Silicon / AArch64).

No OpenGL, Direct3D, or Vulkan is used for the 3D pipeline. SDL2 only presents the finished color buffer and hosts a Dear ImGui debug overlay.

## Pipeline

1. **Math** — `Vec2` / `Vec3` / `Vec4` and column-major `Mat4` (translate, rotate, scale, `look_at`, `perspective` with Z-near / Z-far).
2. **Vertex stage** — model, view, and projection; world-space positions and normals.
3. **Clip** — Sutherland–Hodgman against the six homogeneous frustum planes, then fan triangulation.
4. **Rasterize** — bounding-box edge functions, backface culling, 1/w attribute interpolation (perspective-correct UVs, normals, positions).
5. **Fragment** — Blinn-Phong directional lighting, hemispheric ambient occlusion, optional screen-space AO from the depth buffer.
6. **SIMD** — 4 or 8 adjacent pixels: coverage mask, Z-test, shade, masked stores.

## Build

**Dependencies:** a C++20 compiler, CMake ≥ 3.16, and SDL2.

```bash
# macOS
brew install sdl2 cmake

# Debian / Ubuntu
sudo apt install build-essential cmake libsdl2-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/software-renderer

# Headless one-frame dump (no window) plus scalar vs SIMD timing
./build/software-renderer --dump frame.ppm
```

CMake auto-detects the ISA: `-mavx2 -mfma` on x86_64, NEON (always available) on arm64, and `-mfpu=neon` on 32-bit ARM.

## Controls

| Input | Action |
| --- | --- |
| LMB drag | Orbit camera |
| Mouse wheel | Dolly |
| `1` / `2` / `3` | Shaded / wireframe / depth buffer |
| `S` | Toggle scalar vs SIMD shading |
| Overlay sliders | FOV, light direction, object rotation, SSAO, dual-path benchmark |

The overlay plots **scalar vs SIMD raster time** (milliseconds) when dual-path benchmark is enabled.

## Layout

```
software-renderer/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── math/vec_mat.hpp
│   ├── render/pipeline.hpp
│   ├── render/rasterizer.cpp
│   ├── render/simd_shader.hpp
│   └── ui/debug_overlay.cpp
├── assets/obj/          # cube.obj, plane.obj, checker.ppm
└── third_party/imgui/   # vendored Dear ImGui + SDL2 renderer backend
```

## Publish to GitHub

```bash
gh repo create software-renderer --public --source=. --remote=origin --push
```

## License

MIT. Dear ImGui is also MIT (see `third_party/imgui/LICENSE.txt`).
