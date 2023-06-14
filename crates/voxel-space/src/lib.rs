use cgmath::{InnerSpace, Matrix4, Vector3, Vector4};
use slotmap::{new_key_type, SlotMap};
use std::ops::{Index, IndexMut};

pub fn translation(v: Vector3<f64>) -> Matrix4<f64> {
    let w = (1.0 + v.magnitude2()).sqrt();
    let c = (v / (w + 1.0)).extend(1.0);
    Matrix4::from_cols(
        c * v.x + Vector4::unit_x(),
        c * v.y + Vector4::unit_y(),
        c * v.z + Vector4::unit_z(),
        v.extend(w),
    )
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Side {
    NegX,
    PosX,
    NegY,
    PosY,
    NegZ,
    PosZ,
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
impl<S> Index<Side> for Sided<S> {
    type Output = S;

    fn index(&self, side: Side) -> &Self::Output {
        match side {
            Side::NegX => &self.neg_x,
            Side::PosX => &self.pos_x,
            Side::NegY => &self.neg_y,
            Side::PosY => &self.pos_y,
            Side::NegZ => &self.neg_z,
            Side::PosZ => &self.pos_z,
        }
    }
}
impl<S> IndexMut<Side> for Sided<S> {
    fn index_mut(&mut self, side: Side) -> &mut Self::Output {
        match side {
            Side::NegX => &mut self.neg_x,
            Side::PosX => &mut self.pos_x,
            Side::NegY => &mut self.neg_y,
            Side::PosY => &mut self.pos_y,
            Side::NegZ => &mut self.neg_z,
            Side::PosZ => &mut self.pos_z,
        }
    }
}

type State = u16;

fn generate_inner<T, F: FnMut(&T, Side, State) -> T>(
    parent: &T,
    state: State,
    radius: usize,
    insert: &mut F,
) {
    let side = if state != ORIGIN {
        match state & 0o7 {
            0 => Side::NegX,
            1 => Side::PosX,
            2 => Side::NegY,
            3 => Side::PosY,
            4 => Side::NegZ,
            5 => Side::PosZ,
            _ => return,
        }
    } else {
        return;
    };

    let this = insert(parent, side, state);
    if radius > 0 {
        branch(state).for_each(|state| generate_inner(&this, state, radius - 1, insert));
    }
}

fn generate_origin<T, F: FnMut(&T, Side, State) -> T>(origin: &T, radius: usize, mut insert: F) {
    if radius > 0 {
        branch(ORIGIN).for_each(|state| generate_inner(origin, state, radius - 1, &mut insert));
    }
}

pub const ORIGIN: State = 0o177777;

pub fn branch(state: State) -> impl Iterator<Item = State> {
    let child = if state == ORIGIN {
        [
            Some(0o000),
            Some(0o111),
            Some(0o222),
            Some(0o333),
            Some(0o444),
            Some(0o555),
        ]
    } else {
        let mut branch = [true; 6];
        branch[(state as usize & 0o7) ^ 1] = false;
        branch[(state as usize >> 3 & 0o7) ^ 1] = false;
        if state >> 6 & 0o7 < state >> 3 & 0o7 {
            branch[(state as usize >> 6 & 0o7) ^ 1] = false;
        }

        let state = state << 3 & 0o777;
        let mut child = [None; 6];
        for (i, child) in child.iter_mut().enumerate() {
            if branch[i] {
                *child = Some(state | i as State);
            }
        }
        child
    };
    child.into_iter().flatten()
}

new_key_type! {
    pub struct Cell;
}

struct CellData {
    links: [Cell; 6],
    state: u16,
    is_leaf: bool,
}

pub struct Space {
    cells: SlotMap<Cell, CellData>,
}
impl Space {
    pub fn new() -> (Self, Cell) {
        let mut cells = SlotMap::with_key();
        let origin = cells.insert(CellData {
            links: [Cell::default(); 6],
            state: ORIGIN,
            is_leaf: true,
        });
        (Space { cells }, origin)
    }

    pub fn branch(&mut self, cell: Cell) -> impl Iterator<Item = (Side, Cell)> {
        let branches: Vec<_> = if self.cells[cell].is_leaf {
            self.cells[cell].is_leaf = false;
            branch(self.cells[cell].state)
                .map(move |state| {
                    let mut links = [Cell::default(); 6];
                    links[(state as usize & 0o7) ^ 1] = cell;
                    let link = self.cells.insert(CellData {
                        links,
                        state,
                        is_leaf: true,
                    });
                    self.cells[cell].links[state as usize & 0o7] = link;
                    let side = match state & 0o7 {
                        0 => Side::NegX,
                        1 => Side::PosX,
                        2 => Side::NegY,
                        3 => Side::PosY,
                        4 => Side::NegZ,
                        5 => Side::PosZ,
                        _ => unreachable!(),
                    };
                    (side, link)
                })
                .collect()
        } else {
            branch(self.cells[cell].state)
                .map(|state| {
                    let link = self.cells[cell].links[state as usize & 0o7];
                    let side = match state & 0o7 {
                        0 => Side::NegX,
                        1 => Side::PosX,
                        2 => Side::NegY,
                        3 => Side::PosY,
                        4 => Side::NegZ,
                        5 => Side::PosZ,
                        _ => unreachable!(),
                    };
                    (side, link)
                })
                .collect()
        };
        branches.into_iter()
    }

    pub fn is_leaf(&self, cell: Cell) -> bool {
        self.cells[cell].is_leaf
    }

    pub fn parent(&self, cell: Cell) -> Cell {
        let cell = &self.cells[cell];
        if cell.state == ORIGIN {
            Cell::default()
        } else {
            cell.links[(cell.state as usize & 0o7) ^ 1]
        }
    }

    pub fn generate<T, I, N>(&mut self, cell: Cell, value: T, mut insert: I, mut next: N)
    where
        I: FnMut(Cell, &T) -> bool,
        N: FnMut(Side, Cell, &T) -> T,
    {
        fn inner<T, I, N>(space: &mut Space, cell: Cell, value: T, insert: &mut I, next: &mut N)
        where
            I: FnMut(Cell, &T) -> bool,
            N: FnMut(Side, Cell, &T) -> T,
        {
            if insert(cell, &value) {
                space.branch(cell).for_each(|(side, cell)| {
                    inner(space, cell, next(side, cell, &value), insert, next);
                });
            }
        }
        inner(self, cell, value, &mut insert, &mut next);
    }
}

#[cfg(test)]
mod tests {
    use cgmath::One;

    use super::*;

    #[test]
    fn generate_no_overlap() {
        let mut cells = Vec::with_capacity(4096);

        const STEP: f64 = 1.272_019_649_514_069;
        let trs = Sided {
            neg_x: translation(Vector3::new(-STEP, 0.0, 0.0)),
            pos_x: translation(Vector3::new(STEP, 0.0, 0.0)),
            neg_y: translation(Vector3::new(0.0, -STEP, 0.0)),
            pos_y: translation(Vector3::new(0.0, STEP, 0.0)),
            neg_z: translation(Vector3::new(0.0, 0.0, -STEP)),
            pos_z: translation(Vector3::new(0.0, 0.0, STEP)),
        };

        cells.push((Vector4::new(0.0, 0.0, 0.0, 1.0), ORIGIN));
        generate_origin(&Matrix4::one(), 6, |tr, sd, st| {
            let tr = tr * trs[sd];
            let v = tr.w;
            for (o, so) in cells.iter() {
                let dot = v.truncate().dot(o.truncate()) - v.w * o.w;
                let dist = (dot * dot - 1.0).abs().sqrt();
                assert!(
                    dist + 1e-5 >= STEP,
                    "{st:04o} overlaps {so:04o}: {v:?}, {o:?}"
                );
            }
            cells.push((tr.w, st));
            tr
        });
    }
}
