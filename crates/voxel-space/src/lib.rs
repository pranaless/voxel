use cgmath::{InnerSpace, Matrix4, Vector3, Vector4};
use parking_lot::Mutex;
use slotmap::{new_key_type, SlotMap};
use std::{
    ops::{Index, IndexMut},
    sync::Arc,
};

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

pub type State = u16;

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
        if state >> 6 & 0o7 > state >> 3 & 0o7 {
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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct CellData {
    links: [Cell; 6],
    state: State,
    is_leaf: bool,
}

#[derive(Clone, Copy)]
struct WalkerState {
    orient: u8,
    cell: Cell,
}
impl WalkerState {
    fn new(cell: Cell) -> Self {
        WalkerState { orient: 0o20, cell }
    }

    fn get(&self, cells: &mut SlotMap<Cell, CellData>) -> CellData {
        let cell = self.cell;
        if cells[cell].is_leaf {
            cells[cell].is_leaf = false;
            branch(cells[cell].state).for_each(|state| {
                let mut links = [Cell::default(); 6];
                links[(state as usize & 0o7) ^ 1] = cell;
                let link = cells.insert(CellData {
                    links,
                    state,
                    is_leaf: true,
                });
                cells[cell].links[state as usize & 0o7] = link;
            });
        }
        cells[cell]
    }

    fn walk(&mut self, cells: &mut SlotMap<Cell, CellData>, side: u16) {
        #[rustfmt::skip]
        const CROSS: [u16; 64] = [
            7, 7, 5, 4, 2, 3, 7, 7,
            7, 7, 4, 5, 3, 2, 7, 7,
            4, 5, 7, 7, 1, 0, 7, 7,
            5, 4, 7, 7, 0, 1, 7, 7,
            3, 2, 0, 1, 7, 7, 7, 7,
            2, 3, 1, 0, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
        ];

        let orient = {
            let x = self.orient as u16 & 0o7;
            let y = self.orient as u16 >> 3;
            let z = CROSS[self.orient as usize];
            [x, x ^ 1, y, y ^ 1, z, z ^ 1]
        };

        let cell = self.get(cells);
        let norm = orient[side as usize];
        let pnorm = cell.state & 0o7;

        if cell.state == ORIGIN
            || pnorm ^ 1 == norm
            || branch(cell.state).any(|state| state & 0o7 == norm)
        {
            self.cell = cell.links[norm as usize];
        } else {
            #[rustfmt::skip]
            const ROTATION: [u8; 64] = [
                0, 1, 5, 4, 2, 3, 7, 7,
                0, 1, 4, 5, 3, 2, 7, 7,
                4, 5, 2, 3, 1, 0, 7, 7,
                5, 4, 2, 3, 0, 1, 7, 7,
                3, 3, 0, 1, 4, 5, 7, 7,
                2, 2, 1, 0, 4, 5, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7,
            ];

            let pside = orient.into_iter().position(|v| v == pnorm).unwrap() as u16;
            self.cell = cell.links[pnorm as usize ^ 1];
            self.walk(cells, side);
            self.walk(cells, pside);
            self.walk(cells, side ^ 1);

            let rot = (CROSS[(norm as usize) << 3 | (pnorm as usize)] as usize) << 3;
            let x = ROTATION[rot | (self.orient as usize & 0o7)];
            let y = ROTATION[rot | (self.orient as usize >> 3)];
            self.orient = y << 3 | x;
        }
    }
}

#[derive(Clone)]
pub struct Walker {
    cells: Arc<Mutex<SlotMap<Cell, CellData>>>,
    state: WalkerState,
}
impl Walker {
    pub fn new() -> Walker {
        let mut cells = SlotMap::with_key();
        let origin = cells.insert(CellData {
            links: [Cell::default(); 6],
            state: ORIGIN,
            is_leaf: true,
        });
        Walker {
            cells: Arc::new(Mutex::new(cells)),
            state: WalkerState::new(origin),
        }
    }

    pub fn walk(&mut self, side: Side) {
        let side = match side {
            Side::NegX => 0,
            Side::PosX => 1,
            Side::NegY => 2,
            Side::PosY => 3,
            Side::NegZ => 4,
            Side::PosZ => 5,
        };
        self.state.walk(&mut self.cells.lock(), side);
    }

    pub fn cell(&self) -> Cell {
        self.state.cell
    }

    pub fn generate<T, I, N>(&self, value: T, mut insert: I, mut next: N)
    where
        I: FnMut(Cell, &T) -> bool,
        N: FnMut(Side, &T) -> T,
    {
        fn inner<T, I, N>(
            cells: &Mutex<SlotMap<Cell, CellData>>,
            walker: WalkerState,
            state: State,
            value: T,
            insert: &mut I,
            next: &mut N,
        ) where
            I: FnMut(Cell, &T) -> bool,
            N: FnMut(Side, &T) -> T,
        {
            if insert(walker.cell, &value) {
                branch(state).for_each(|state| {
                    let mut walker = walker;
                    walker.walk(&mut cells.lock(), state & 0o7);
                    let side = match state & 0o7 {
                        0 => Side::NegX,
                        1 => Side::PosX,
                        2 => Side::NegY,
                        3 => Side::PosY,
                        4 => Side::NegZ,
                        5 => Side::PosZ,
                        _ => unreachable!(),
                    };
                    inner(cells, walker, state, next(side, &value), insert, next);
                });
            }
        }
        inner(
            &self.cells,
            self.state,
            ORIGIN,
            value,
            &mut insert,
            &mut next,
        );
    }
}

#[cfg(test)]
mod tests {
    use cgmath::One;

    use super::*;

    #[test]
    fn generate_no_overlap() {
        let walker = Walker::new();
        let mut cells = Vec::<(Vector4<f64>, Vec<Side>)>::new();

        const STEP: f64 = 1.272_019_649_514_069;
        let trs = Sided {
            neg_x: translation(Vector3::new(-STEP, 0.0, 0.0)),
            pos_x: translation(Vector3::new(STEP, 0.0, 0.0)),
            neg_y: translation(Vector3::new(0.0, -STEP, 0.0)),
            pos_y: translation(Vector3::new(0.0, STEP, 0.0)),
            neg_z: translation(Vector3::new(0.0, 0.0, -STEP)),
            pos_z: translation(Vector3::new(0.0, 0.0, STEP)),
        };

        walker.generate(
            (Matrix4::one(), Vec::new()),
            |_cell, (tr, path)| {
                let v = tr.w;
                for (o, po) in cells.iter() {
                    let dot = v.truncate().dot(o.truncate()) - v.w * o.w;
                    let dist = (dot * dot - 1.0).abs().sqrt();
                    assert!(
                        dist + 1e-5 >= STEP,
                        "{path:?} overlaps {po:?}: {v:?}, {o:?}"
                    );
                }
                cells.push((v, path.clone()));
                path.len() < 7
            },
            |side, (tr, path)| {
                (tr * trs[side], {
                    let mut path = path.clone();
                    path.push(side);
                    path
                })
            },
        );
    }

    #[test]
    fn walker() {
        let mut walker = Walker::new();
        let origin = walker.cell();
        walker.generate(3, |_cell, &radius| radius > 0, |_side, &radius| radius - 1);
        walker.walk(Side::NegX);
        walker.walk(Side::PosX);
        assert_eq!(walker.cell(), origin);

        walker.walk(Side::PosY);
        walker.walk(Side::NegX);
        walker.walk(Side::NegY);
        walker.walk(Side::PosX);
        walker.walk(Side::PosY);
        assert_eq!(walker.cell(), origin);

        walker.walk(Side::PosZ);
        walker.walk(Side::NegX);
        walker.walk(Side::NegZ);
        walker.walk(Side::PosX);
        walker.walk(Side::PosZ);
        assert_eq!(walker.cell(), origin);

        walker.walk(Side::PosZ);
        walker.walk(Side::NegY);
        walker.walk(Side::NegZ);
        walker.walk(Side::PosY);
        walker.walk(Side::PosZ);
        assert_eq!(walker.cell(), origin);
    }
}
