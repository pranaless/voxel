struct VertexOutput {
    @builtin(position)
    pos: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.pos = vec4<f32>(pos, 1.0);
    output.uv = uv;
    return output;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(uv, 0.0, 1.0);
}