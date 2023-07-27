use bytemuck::{Pod, Zeroable};
use cgmath::Matrix4;
use wgpu::{VertexBufferLayout, VertexStepMode};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct InstanceTransformData {
    transform: [f32; 16],
}
impl InstanceTransformData {
    pub fn new(matrix: Matrix4<f64>) -> Self {
        Self {
            transform: *matrix.cast::<f32>().unwrap().as_ref(),
        }
    }

    pub const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: 64,
        step_mode: VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            2 => Float32x4,
            3 => Float32x4,
            4 => Float32x4,
            5 => Float32x4,
        ],
    };
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct VertexData {
    position: [f32; 4],
}
impl VertexData {
    pub const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: 16,
        step_mode: VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x4,
        ],
    };
}
