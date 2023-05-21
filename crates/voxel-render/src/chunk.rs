use itertools::{chain, Itertools};
use std::fmt;

use crate::Vertex;

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

#[derive(Debug, Clone, Default)]
struct ChunkPane(Vec<Quad>);
impl ChunkPane {
    pub fn new(faces: impl Iterator<Item = Face>) -> Self {
        let mesh = faces
            .enumerate()
            .map(|(i, f)| Quad {
                row: Extent::new(i & 0b1111, 0),
                col: Extent::new((i >> 4) & 0b1111, 0),
                face: f,
            })
            .coalesce(|o, t| o.merge_row(t))
            .sorted_unstable_by_key(|v| v.row.start() << 4 | v.col.start())
            .coalesce(|o, t| o.merge_col(t))
            .sorted_unstable_by_key(|v| v.col.start() << 4 | v.row.start())
            .collect::<Vec<_>>();
        Self(mesh)
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Sided<S> {
    pub neg_x: S,
    pub pos_x: S,
    pub neg_y: S,
    pub pos_y: S,
    pub neg_z: S,
    pub pos_z: S,
}
impl<S: Clone> Sided<S> {
    pub fn new(s: S) -> Self {
        Sided {
            neg_x: s.clone(),
            pos_x: s.clone(),
            neg_y: s.clone(),
            pos_y: s.clone(),
            neg_z: s.clone(),
            pos_z: s,
        }
    }
}

pub struct ChunkMesh {
    panes: Vec<Sided<ChunkPane>>,
}
impl ChunkMesh {
    pub fn new(faces: &[[[Sided<Face>; 16]; 16]; 16]) -> Self {
        fn pane(mut p: impl FnMut(usize, usize) -> Face) -> ChunkPane {
            ChunkPane::new((0..16).cartesian_product(0..16).map(|(i, j)| p(i, j)))
        }

        let panes = (0..16)
            .map(|i| Sided {
                neg_x: pane(|y, z| faces[i][y][z].neg_x),
                pos_x: pane(|y, z| faces[i][y][z].pos_x),
                neg_y: pane(|x, z| faces[x][i][z].neg_y),
                pos_y: pane(|x, z| faces[x][i][z].pos_y),
                neg_z: pane(|y, x| faces[x][y][i].neg_z),
                pos_z: pane(|y, x| faces[x][y][i].pos_z),
            })
            .collect::<Vec<_>>();

        ChunkMesh { panes }
    }

    pub fn quads(&self) -> impl Iterator<Item = ([Vertex; 4], Face)> + '_ {
        self.panes.iter().enumerate().flat_map(|(i, s)| {
            chain![
                s.neg_x.0.iter().map(move |q| {
                    let x = i as f32;
                    let (z, w) = (q.row.start() as f32, q.row.extent() as f32 + 1.0);
                    let (y, h) = (q.col.start() as f32, q.col.extent() as f32 + 1.0);
                    #[rustfmt::skip]
                    let quad = [
                        Vertex { pos: [x, y,     z],     uv: [0.0, 0.0] },
                        Vertex { pos: [x, y,     z + w], uv: [  w, 0.0] },
                        Vertex { pos: [x, y + h, z],     uv: [0.0,   h] },
                        Vertex { pos: [x, y + h, z + w], uv: [  w,   h] },
                    ];
                    (quad, q.face)
                }),
                s.pos_x.0.iter().map(move |q| {
                    let x = i as f32 + 1.0;
                    let (z, w) = (q.row.start() as f32, q.row.extent() as f32 + 1.0);
                    let (y, h) = (q.col.start() as f32, q.col.extent() as f32 + 1.0);
                    #[rustfmt::skip]
                    let quad = [
                        Vertex { pos: [x, y,     z + w], uv: [0.0, 0.0] },
                        Vertex { pos: [x, y,     z],     uv: [  w, 0.0] },
                        Vertex { pos: [x, y + h, z + w], uv: [0.0,   h] },
                        Vertex { pos: [x, y + h, z],     uv: [  w,   h] },
                    ];
                    (quad, q.face)
                }),
                s.neg_y.0.iter().map(move |q| {
                    let y = i as f32;
                    let (z, w) = (q.row.start() as f32, q.row.extent() as f32 + 1.0);
                    let (x, h) = (q.col.start() as f32, q.col.extent() as f32 + 1.0);
                    #[rustfmt::skip]
                    let quad = [
                        Vertex { pos: [x,     y, z + w], uv: [0.0, 0.0] },
                        Vertex { pos: [x,     y, z],     uv: [  w, 0.0] },
                        Vertex { pos: [x + h, y, z + w], uv: [0.0,   h] },
                        Vertex { pos: [x + h, y, z],     uv: [  w,   h] },
                    ];
                    (quad, q.face)
                }),
                s.pos_y.0.iter().map(move |q| {
                    let y = i as f32 + 1.0;
                    let (z, w) = (q.row.start() as f32, q.row.extent() as f32 + 1.0);
                    let (x, h) = (q.col.start() as f32, q.col.extent() as f32 + 1.0);
                    #[rustfmt::skip]
                    let quad = [
                        Vertex { pos: [x,     y, z],     uv: [0.0, 0.0] },
                        Vertex { pos: [x,     y, z + w], uv: [  w, 0.0] },
                        Vertex { pos: [x + h, y, z],     uv: [0.0,   h] },
                        Vertex { pos: [x + h, y, z + w], uv: [  w,   h] },
                    ];
                    (quad, q.face)
                }),
                s.neg_z.0.iter().map(move |q| {
                    let z = i as f32;
                    let (x, w) = (q.row.start() as f32, q.row.extent() as f32 + 1.0);
                    let (y, h) = (q.col.start() as f32, q.col.extent() as f32 + 1.0);
                    #[rustfmt::skip]
                    let quad = [
                        Vertex { pos: [x + w, y,     z], uv: [0.0, 0.0] },
                        Vertex { pos: [x,     y,     z], uv: [  w, 0.0] },
                        Vertex { pos: [x + w, y + h, z], uv: [0.0,   h] },
                        Vertex { pos: [x,     y + h, z], uv: [  w,   h] },
                    ];
                    (quad, q.face)
                }),
                s.pos_z.0.iter().map(move |q| {
                    let z = i as f32 + 1.0;
                    let (x, w) = (q.row.start() as f32, q.row.extent() as f32 + 1.0);
                    let (y, h) = (q.col.start() as f32, q.col.extent() as f32 + 1.0);
                    #[rustfmt::skip]
                    let quad = [
                        Vertex { pos: [x,     y,     z], uv: [0.0, 0.0] },
                        Vertex { pos: [x + w, y,     z], uv: [  w, 0.0] },
                        Vertex { pos: [x,     y + h, z], uv: [0.0,   h] },
                        Vertex { pos: [x + w, y + h, z], uv: [  w,   h] },
                    ];
                    (quad, q.face)
                }),
            ]
        })
    }
}
