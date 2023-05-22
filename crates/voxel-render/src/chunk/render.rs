use bytemuck::{Pod, Zeroable};
use itertools::Itertools;
use std::{fmt, vec::IntoIter};

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

pub fn chunk_mesh(faces: &[[[Sided<Face>; 16]; 16]; 16]) -> impl Iterator<Item = [Vertex; 4]> + '_ {
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

    (0..96).flat_map(|idx| {
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
        chunk_pane(faces).map(move |q| build_quad(idx, q))
    })
}
