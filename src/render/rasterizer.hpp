#pragma once

#include "framebuffer.hpp"
#include "mesh.hpp"
#include "pipeline.hpp"
#include "simd_shader.hpp"

namespace sr {

enum class RasterMode { Shaded, Wireframe, Depth };
enum class ShadeMode { Scalar, SIMD };

struct DrawStats {
    int tris_submitted = 0;
    int tris_drawn = 0;
    int pixels = 0;
};

void draw_mesh(Framebuffer& fb, const Mesh& mesh, const Uniforms& uniforms, const ShadeParams& shade,
               RasterMode raster_mode, ShadeMode shade_mode, DrawStats& stats);

void apply_ssao(Framebuffer& fb, float strength);

ShadeParams shade_from_uniforms(const Uniforms& u, const Texture* tex);

}  // namespace sr
