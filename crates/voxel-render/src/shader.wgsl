struct VertexOutput {
    @builtin(position)
    pos: vec4<f32>,
    // @location(0)
    // uv: vec2<f32>,
}

struct InstanceTransform {
    @location(2)
    x: vec4<f32>,
    @location(3)
    y: vec4<f32>,
    @location(4)
    z: vec4<f32>,
    @location(5)
    w: vec4<f32>,
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
    // @location(1) uv: vec2<f32>,
    tr: InstanceTransform,
) -> VertexOutput {
    let transform = mat4x4<f32>(tr.x, tr.y, tr.z, tr.w);
    var output: VertexOutput;
    output.pos = camera.viewport
               * camera.transform
               * transform
               * pos;
    // output.uv = uv;
    return output;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.8, 0.6, 0.4, 1.0);
}