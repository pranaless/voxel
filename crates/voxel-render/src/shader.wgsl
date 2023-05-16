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

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.pos = camera.viewport
               * camera.transform
               * vec4<f32>(pos, 1.0);
    output.uv = uv;
    return output;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(uv, 0.0, 1.0);
}