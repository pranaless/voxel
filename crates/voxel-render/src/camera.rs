use bytemuck::{Pod, Zeroable};
use cgmath::{
    InnerSpace, Matrix, Matrix3, Matrix4, PerspectiveFov, SquareMatrix, Vector2, Vector3,
};
use wgpu::{util::DeviceExt, BufferUsages};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
/// Internal type representing the contents of the camera uniform.
struct CameraUniform {
    viewport: [f32; 16],
    transform: [f32; 16],
}
impl CameraUniform {
    pub fn new(viewport: PerspectiveFov<f64>, transform: Matrix4<f64>) -> Self {
        let viewport = {
            #[rustfmt::skip]
            const OPENGL_TO_WGPU_MATRIX: Matrix4<f64> = Matrix4::new(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.5, 1.0,
            );

            let viewport: Matrix4<f64> = viewport.into();
            let viewport = OPENGL_TO_WGPU_MATRIX * viewport;
            let viewport: Matrix4<f32> = viewport.cast().unwrap(); // f64 -> f32 cast, guaranteed not to fail
            *viewport.as_ref()
        };
        let transform = {
            let transform = transform.invert().unwrap();
            let transform: Matrix4<f32> = transform.cast().unwrap(); // f64 -> f32 cast, guaranteed not to fail
            *transform.as_ref()
        };
        Self {
            viewport,
            transform,
        }
    }
}

pub struct Camera {
    buffer: wgpu::Buffer,
    pub viewport: PerspectiveFov<f64>,
    pub transform: Matrix4<f64>,
}
impl Camera {
    pub const fn layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn entry(&self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: self.buffer.as_entire_binding(),
        }
    }

    pub fn new(
        device: &wgpu::Device,
        viewport: PerspectiveFov<f64>,
        transform: Matrix4<f64>,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(&CameraUniform::new(viewport, transform)),
        });
        Camera {
            buffer,
            viewport,
            transform,
        }
    }

    #[inline]
    pub fn apply_transform(&mut self, delta: Matrix4<f64>) {
        self.transform = self.transform * delta;
    }

    #[inline]
    pub fn apply_rotation(&mut self, delta: Vector2<f64>) {
        let z = delta.extend(1.0).normalize();
        let x = Vector3::unit_y().cross(z).normalize();
        let y = z.cross(x);
        let matrix = Matrix4::from(Matrix3::from_cols(x, y, z).transpose());
        self.apply_transform(matrix);
    }

    pub fn commit(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::new(self.viewport, self.transform)),
        );
    }
}
