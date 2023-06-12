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

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var t_tex: texture_2d_array<f32>;

@group(1) @binding(1)
var s_tex: sampler;

@vertex
fn vs_main(
    @location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tr_x: vec4<f32>,
    @location(3) tr_y: vec4<f32>,
    @location(4) tr_z: vec4<f32>,
    @location(5) tr_w: vec4<f32>,
) -> VertexOutput {
    let transform = mat4x4<f32>(tr_x, tr_y, tr_z, tr_w);
    var output: VertexOutput;
    output.pos = camera.viewport
               * camera.transform
               * transform
               * pos;
    output.uv = uv;
    return output;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_tex, s_tex, uv, 0);
}