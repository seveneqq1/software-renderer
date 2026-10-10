cmake_minimum_required(VERSION 3.16)
project(software-renderer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(APPLE AND EXISTS "/opt/homebrew")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()
if(APPLE AND EXISTS "/usr/local")
    list(APPEND CMAKE_PREFIX_PATH "/usr/local")
endif()

find_package(SDL2 REQUIRED)

if(TARGET SDL2::SDL2)
    set(SR_SDL_TARGET SDL2::SDL2)
elseif(TARGET SDL2::SDL2-static)
    set(SR_SDL_TARGET SDL2::SDL2-static)
else()
    add_library(sr_sdl_unknown INTERFACE)
    target_include_directories(sr_sdl_unknown INTERFACE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(sr_sdl_unknown INTERFACE ${SDL2_LIBRARIES})
    set(SR_SDL_TARGET sr_sdl_unknown)
endif()

add_library(imgui STATIC
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_sdl2.cpp
    third_party/imgui/backends/imgui_impl_sdlrenderer2.cpp
)
target_include_directories(imgui PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/imgui
    ${CMAKE_SOURCE_DIR}/third_party/imgui/backends
)
target_link_libraries(imgui PUBLIC ${SR_SDL_TARGET})
target_compile_features(imgui PUBLIC cxx_std_17)

add_executable(software-renderer
    src/main.cpp
    src/render/rasterizer.cpp
    src/ui/debug_overlay.cpp
)

target_include_directories(software-renderer PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(software-renderer PRIVATE imgui ${SR_SDL_TARGET})
if(TARGET SDL2::SDL2main)
    target_link_libraries(software-renderer PRIVATE SDL2::SDL2main)
endif()

target_compile_definitions(software-renderer PRIVATE
    SR_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets"
)

if(MSVC)
    target_compile_options(software-renderer PRIVATE /W4 /O2)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        target_compile_options(software-renderer PRIVATE /arch:AVX2)
        target_compile_definitions(software-renderer PRIVATE SR_TARGET_AVX2=1)
    endif()
else()
    target_compile_options(software-renderer PRIVATE -Wall -Wextra -Wpedantic -O3)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|x86")
        target_compile_options(software-renderer PRIVATE -mavx2 -mfma)
        target_compile_definitions(software-renderer PRIVATE SR_TARGET_AVX2=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        target_compile_definitions(software-renderer PRIVATE SR_TARGET_NEON=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        target_compile_options(software-renderer PRIVATE -mfpu=neon)
        target_compile_definitions(software-renderer PRIVATE SR_TARGET_NEON=1)
    endif()
endif()

source_group(TREE ${CMAKE_SOURCE_DIR}/src PREFIX src FILES
    src/main.cpp
    src/render/rasterizer.cpp
    src/ui/debug_overlay.cpp
)
