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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct State(u32);
impl State {
    pub const ORIGIN: Self = Self(7);

    fn get(self, i: usize) -> u8 {
        (self.0 >> (3 * (i - 1)) & 0o7) as u8
    }

    fn with(self, step: u8) -> Self {
        if self == Self::ORIGIN {
            Self(0o111_111 * step as u32)
        } else {
            Self((self.0 << 3 & 0o777_777) | step as u32)
        }
    }

    pub fn last(self) -> u8 {
        (self.0 & 0o7) as u8
    }

    pub fn branch(self) -> impl Iterator<Item = Self> {
        let links = if self == Self::ORIGIN {
            [true; 6]
        } else {
            let mut links = [true; 6];
            if self.get(6) ^ self.get(3) == 1 && self.get(5) ^ self.get(2) == 1 {
                links[self.get(4) as usize ^ 1] = false;
            }
            if self.get(6) ^ self.get(2) == 1 {
                links[self.get(3) as usize ^ 1] = false;
            }
            if self.get(5) ^ self.get(2) == 1 {
                links[self.get(3) as usize ^ 1] = false;
            }
            if self.get(3) > self.get(2) {
                links[self.get(3) as usize ^ 1] = false;
            }
            links[self.get(2) as usize ^ 1] = false;
            links[self.get(1) as usize ^ 1] = false;
            links
        };
        links
            .into_iter()
            .enumerate()
            .filter_map(move |(i, link)| if link { Some(self.with(i as u8)) } else { None })
    }
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
            cells[cell].state.branch().for_each(|state| {
                let mut links = [Cell::default(); 6];
                links[state.last() as usize ^ 1] = cell;
                let link = cells.insert(CellData {
                    links,
                    state,
                    is_leaf: true,
                });
                cells[cell].links[state.last() as usize] = link;
            });
        }
        cells[cell]
    }

    // FIXME: Rotation doesn't work right
    fn walk(&mut self, cells: &mut SlotMap<Cell, CellData>, side: u8) {
        #[rustfmt::skip]
        const CROSS: [u8; 64] = [
            7, 7, 5, 4, 2, 3, 7, 7,
            7, 7, 4, 5, 3, 2, 7, 7,
            4, 5, 7, 7, 1, 0, 7, 7,
            5, 4, 7, 7, 0, 1, 7, 7,
            3, 2, 0, 1, 7, 7, 7, 7,
            2, 3, 1, 0, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
        ];

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

        enum Step {
            Side(u8),
            Rot(u8),
        }

        let mut steps = vec![Step::Side(side)];

        while let Some(step) = steps.pop() {
            match step {
                Step::Side(side) => {
                    let orient = {
                        let x = self.orient & 0o7;
                        let y = self.orient >> 3;
                        let z = CROSS[self.orient as usize];
                        [x, x ^ 1, y, y ^ 1, z, z ^ 1]
                    };

                    let cell = self.get(cells);
                    let norm = orient[side as usize];
                    let pnorm = (cell.state.0 & 0o7) as u8;

                    if cell.state == State::ORIGIN
                        || pnorm ^ 1 == norm
                        || cell
                            .state
                            .branch()
                            .any(|state| (state.0 & 0o7) as u8 == norm)
                    {
                        self.cell = cell.links[norm as usize];
                    } else {
                        let pside = orient.into_iter().position(|v| v == pnorm).unwrap() as u8;
                        let rot = CROSS[(norm as usize) << 3 | (pnorm as usize)];
                        self.cell = cell.links[pnorm as usize ^ 1];
                        steps.push(Step::Rot(rot));
                        steps.push(Step::Side(side ^ 1));
                        steps.push(Step::Side(pside));
                        steps.push(Step::Side(side));
                    }
                }
                Step::Rot(rot) => {
                    let rot = (rot as usize) << 3;
                    let x = ROTATION[rot | (self.orient as usize & 0o7)];
                    let y = ROTATION[rot | (self.orient as usize >> 3)];
                    self.orient = y << 3 | x;
                }
            }
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
            state: State::ORIGIN,
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
                state.branch().for_each(|state| {
                    let mut walker = walker;
                    walker.walk(&mut cells.lock(), state.last());
                    let side = match state.last() {
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
            State::ORIGIN,
            value,
            &mut insert,
            &mut next,
        );
    }
}

#[cfg(test)]
mod tests {
    use cgmath::One;
    use slotmap::SecondaryMap;

    use super::*;

    fn walk_path(walker: &Walker, path: Vec<Side>) -> Cell {
        path.into_iter()
            .fold(walker.clone(), |mut w, s| {
                w.walk(s);
                w
            })
            .cell()
    }

    macro_rules! assert_cell_eq {
        ($walker:expr, [$($a:ident),* $(,)?], [$($b:ident),* $(,)?] $(,)?) => {
            assert_eq!(
                walk_path($walker, vec![$(Side::$a),*]),
                walk_path($walker, vec![$(Side::$b),*]),
            )
        };
    }

    #[test]
    fn generate_no_overlap() {
        let walker = Walker::new();
        let mut cells = SecondaryMap::<Cell, (Vector4<f64>, Vec<Side>)>::new();

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
            |cell, (tr, path)| {
                let v = tr.w;
                for (_c, (o, po)) in cells.iter() {
                    let dot = v.truncate().dot(o.truncate()) - v.w * o.w;
                    let dist = (dot * dot - 1.0).abs().sqrt();
                    assert!(
                        dist + 1e-5 >= STEP,
                        "{path:?} overlaps {po:?}: {v:?}, {o:?}",
                    );
                }
                cells.insert(cell, (v, path.clone()));
                path.len() < 5
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
        let walker = Walker::new();
        assert_cell_eq!(&walker, [NegX, PosX], []);
        assert_cell_eq!(&walker, [PosY, NegX, NegY], [NegX, PosY]);
        assert_cell_eq!(&walker, [NegY, NegX, NegX, PosY], [NegX, NegY, NegY, PosX]);
        assert_cell_eq!(&walker, [NegY, NegX, NegX, PosY, NegY], [NegY, NegX, NegX]);
        assert_cell_eq!(
            &walker,
            [NegY, NegZ, NegX, PosY, PosY, PosX],
            [NegZ, NegY, NegX, NegX, PosZ],
        );
    }
}
