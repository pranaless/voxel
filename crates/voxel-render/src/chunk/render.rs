use bytemuck::{Pod, Zeroable};
use cgmath::Matrix4;
use itertools::Itertools;
use std::{fmt, vec::IntoIter};
use wgpu::{util::DeviceExt, BufferUsages};

use super::Sided;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
}
impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: 20,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 12,
                shader_location: 1,
            },
        ],
    };
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct Face(pub bool);

#[derive(Clone, Copy, Eq, PartialEq)]
struct Extent(u8);
impl Extent {
    pub fn new(start: usize, extent: usize) -> Extent {
        let start = (start as u8) << 4;
        let extent = (extent as u8) & 0b1111;
        Extent(start | extent)
    }

    #[inline]
    pub fn start(self) -> usize {
        (self.0 >> 4) as usize
    }

    #[inline]
    pub fn extent(self) -> usize {
        (self.0 & 0b1111) as usize
    }

    #[inline]
    pub fn end(self) -> usize {
        self.start() + self.extent()
    }
}
impl fmt::Debug for Extent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..={}", self.start(), self.end())
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct Quad {
    row: Extent,
    col: Extent,
    face: Face,
}
impl Quad {
    fn merge_row(self, other: Quad) -> Result<Self, (Self, Self)> {
        if other.col == self.col
            && other.row.start() == self.row.end() + 1
            && self.face == other.face
        {
            Ok(Quad {
                row: Extent(self.row.0 + 1),
                ..self
            })
        } else {
            Err((self, other))
        }
    }

    fn merge_col(self, other: Quad) -> Result<Self, (Self, Self)> {
        if other.row == self.row
            && other.col.start() == self.col.end() + 1
            && other.face == self.face
        {
            Ok(Quad {
                col: Extent(self.col.0 + 1),
                ..self
            })
        } else {
            Err((self, other))
        }
    }
}

fn chunk_pane(faces: impl Iterator<Item = Face>) -> IntoIter<Quad> {
    faces
        .enumerate()
        .map(|(i, f)| Quad {
            row: Extent::new(i & 0b1111, 0),
            col: Extent::new((i >> 4) & 0b1111, 0),
            face: f,
        })
        .filter(|q| q.face.0)
        .coalesce(|o, t| o.merge_row(t))
        .sorted_unstable_by_key(|v| v.row.start() << 4 | v.col.start())
        .coalesce(|o, t| o.merge_col(t))
        .sorted_unstable_by_key(|v| v.col.start() << 4 | v.row.start())
}

pub struct ChunkMesh {
    bind_group: wgpu::BindGroup,
    state: wgpu::Buffer,

    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    length: u32,
}
impl ChunkMesh {
    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let state = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: state.as_entire_binding(),
            }],
        });

        let vertex = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 0,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let index = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 0,
            usage: BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        Self {
            state,
            bind_group,
            vertex,
            index,
            length: 0,
        }
    }

    pub fn update_transform(&self, queue: &wgpu::Queue, transform: Matrix4<f64>) {
        let transform: Matrix4<f32> = transform.cast().unwrap();
        let transform: &[f32; 16] = transform.as_ref();
        queue.write_buffer(&self.state, 0, bytemuck::cast_slice(transform));
    }

    pub fn update_mesh(&mut self, device: &wgpu::Device, faces: &[[[Sided<Face>; 16]; 16]; 16]) {
        fn orient(idx: u8, j: usize, k: usize) -> [usize; 3] {
            let ori = idx >> 5;
            let i = (idx & 0b1111) as usize;
            match ori {
                0 => [i, j, k],
                1 => [j, i, k],
                2 => [k, j, i],
                _ => unreachable!(),
            }
        }

        fn build_quad(idx: u8, quad: Quad) -> [Vertex; 4] {
            let (j, h) = (quad.col.start(), quad.col.extent() + 1);
            let (k, w) = (quad.row.start(), quad.row.extent() + 1);
            let [x, y, z] = orient(idx, j, k).map(|v| v as f32);
            let (w, h) = (w as f32, h as f32);

            let o = match idx >> 4 {
                0 => [[0.0, 0.0, 0.0], [0.0, 0.0, w], [0.0, h, 0.0], [0.0, h, w]],
                1 => [[1.0, 0.0, w], [1.0, 0.0, 0.0], [1.0, h, w], [1.0, h, 0.0]],
                2 => [[0.0, 0.0, w], [0.0, 0.0, 0.0], [h, 0.0, w], [h, 0.0, 0.0]],
                3 => [[0.0, 1.0, 0.0], [0.0, 1.0, w], [h, 1.0, 0.0], [h, 1.0, w]],
                4 => [[w, 0.0, 0.0], [0.0, 0.0, 0.0], [w, h, 0.0], [0.0, h, 0.0]],
                5 => [[0.0, 0.0, 1.0], [w, 0.0, 1.0], [0.0, h, 1.0], [w, h, 1.0]],
                _ => unreachable!(),
            };
            [
                Vertex {
                    pos: [x + o[0][0], y + o[0][1], z + o[0][2]],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    pos: [x + o[1][0], y + o[1][1], z + o[1][2]],
                    uv: [w, 0.0],
                },
                Vertex {
                    pos: [x + o[2][0], y + o[2][1], z + o[2][2]],
                    uv: [0.0, h],
                },
                Vertex {
                    pos: [x + o[3][0], y + o[3][1], z + o[3][2]],
                    uv: [w, h],
                },
            ]
        }

        let mut vertex = Vec::<Vertex>::new();
        let mut index = Vec::<u32>::new();

        for idx in 0..96 {
            let faces = (0..16)
                .cartesian_product(0..16)
                .map(move |(j, k)| orient(idx, j, k))
                .map(|[x, y, z]| &faces[x][y][z])
                .map(move |s| match idx >> 4 {
                    0 => s.neg_x,
                    1 => s.pos_x,
                    2 => s.neg_y,
                    3 => s.pos_y,
                    4 => s.neg_z,
                    5 => s.pos_z,
                    _ => unreachable!(),
                });
            let pane = chunk_pane(faces).collect::<Vec<_>>();
            vertex.reserve(4 * pane.len());
            index.reserve(6 * pane.len());
            pane.into_iter()
                .map(move |q| build_quad(idx, q))
                .for_each(|q| {
                    let i = vertex.len() as u32;
                    vertex.extend_from_slice(&q);
                    index.extend_from_slice(&[i, i + 1, i + 2, i + 2, i + 1, i + 3]);
                });
        }

        self.vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage: wgpu::BufferUsages::VERTEX,
            contents: bytemuck::cast_slice(&vertex),
        });
        self.index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage: wgpu::BufferUsages::INDEX,
            contents: bytemuck::cast_slice(&index),
        });
        self.length = index.len() as u32;
    }

    pub fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, bind_group_index: u32) {
        rpass.set_bind_group(bind_group_index, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex.slice(..));
        rpass.set_index_buffer(self.index.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..self.length, 0, 0..1);
    }
}
