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

CMake auto-detects the ISA: `-mavx2 -mfma` on x86_64, NEON (always available) on arm64, and `-mfpu=neon` on 32-bit ARM.

---

## Launch on macOS (Terminal)

Install Homebrew if you do not have it: https://brew.sh

```bash
brew install cmake sdl2
git clone https://github.com/seveneqq1/software-renderer.git
cd software-renderer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/software-renderer
```

Headless one-frame dump (no window) plus scalar vs SIMD timing:

```bash
./build/software-renderer --dump frame.ppm
```

Apple Silicon uses the NEON shader path automatically. Intel Macs use AVX2.

If CMake cannot find SDL2:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(brew --prefix sdl2)"
```

---

## Launch on Linux

### Debian / Ubuntu / Linux Mint

```bash
sudo apt update
sudo apt install build-essential cmake libsdl2-dev git
git clone https://github.com/seveneqq1/software-renderer.git
cd software-renderer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/software-renderer
```

### Fedora

```bash
sudo dnf install gcc-c++ cmake SDL2-devel git
git clone https://github.com/seveneqq1/software-renderer.git
cd software-renderer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/software-renderer
```

### Arch / Manjaro

```bash
sudo pacman -S --needed base-devel cmake sdl2 git
git clone https://github.com/seveneqq1/software-renderer.git
cd software-renderer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/software-renderer
```

Headless dump:

```bash
./build/software-renderer --dump frame.ppm
```

---

## Launch on Windows

You need **Git**, **CMake**, a **C++20 compiler**, and **SDL2**.

### 1. Install tools

- Git: https://git-scm.com/download/win
- CMake: https://cmake.org/download/ (add CMake to `PATH`)
- Visual Studio 2022 with the **Desktop development with C++** workload: https://visualstudio.microsoft.com/

### 2. Install SDL2 with vcpkg (recommended)

In **PowerShell** or **Command Prompt**:

```bat
git clone https://github.com/microsoft/vcpkg.git %USERPROFILE%\vcpkg
%USERPROFILE%\vcpkg\bootstrap-vcpkg.bat
%USERPROFILE%\vcpkg\vcpkg install sdl2:x64-windows
```

### 3. Clone, build, and run

```bat
git clone https://github.com/seveneqq1/software-renderer.git
cd software-renderer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
build\Release\software-renderer.exe
```

Headless dump:

```bat
build\Release\software-renderer.exe --dump frame.ppm
```

If you prefer a Visual Studio solution instead of an immediate binary:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
```

Then open `build\software-renderer.sln`, or run `build\Release\software-renderer.exe` from a terminal.

x64 Windows builds enable AVX2 automatically.

---

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

## License

MIT. Dear ImGui is also MIT (see `third_party/imgui/LICENSE.txt`).
