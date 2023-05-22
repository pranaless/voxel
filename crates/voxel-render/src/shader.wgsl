struct VertexOutput {
    @builtin(position)
    pos: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
}

struct Camera {
    viewport: mat4x4<f32>,
    transform: mat4x4<f32>,
}

struct ChunkMesh {
    transform: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var t_tex: texture_2d_array<f32>;

@group(1) @binding(1)
var s_tex: sampler;

@group(2) @binding(0)
var<uniform> mesh: ChunkMesh;

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.pos = camera.viewport
               * camera.transform
               * mesh.transform
               * vec4<f32>(pos, 1.0);
    output.uv = uv;
    return output;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_tex, s_tex, uv, 0);
}