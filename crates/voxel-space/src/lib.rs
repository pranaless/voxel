use cgmath::{InnerSpace, Matrix4, Vector3, Vector4};
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
    let side = if state != 0o100000 {
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
        generate_child(state, |state| {
            generate_inner(&this, state, radius - 1, insert)
        });
    }
}

pub fn generate_child<F: FnMut(State)>(state: State, mut insert: F) {
    if state & 0o100000 != 0 {
        match state & 0o77777 {
            0 => {
                insert(0o00000);
                insert(0o11111);
                insert(0o22222);
                insert(0o33333);
                insert(0o44444);
                insert(0o55555);
            }
            _ => {}
        }
    } else {
        let mut child = [true; 6];
        child[(state as usize & 0o7) ^ 1] = false;
        child[(state as usize >> 3 & 0o7) ^ 1] = false;
        child[(state as usize >> 6 & 0o7) ^ 1] = false;

        let state = state << 3 & 0o77777;
        for (i, child) in child.into_iter().enumerate() {
            if child {
                insert(state | i as State);
            }
        }
    }
}

pub fn generate<T, F: FnMut(&T, Side, State) -> T>(
    parent: &T,
    state: State,
    radius: usize,
    mut insert: F,
) {
    generate_inner(parent, state, radius, &mut insert);
}

pub fn generate_origin<T, F: FnMut(&T, Side, State) -> T>(
    origin: &T,
    radius: usize,
    mut insert: F,
) {
    if radius > 0 {
        generate_child(0o100000, |state| {
            generate_inner(origin, state, radius - 1, &mut insert)
        });
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

        cells.push((Vector4::new(0.0, 0.0, 0.0, 1.0), 0o100000));
        generate_origin(&Matrix4::one(), 6, |tr, sd, st| {
            let tr = tr * trs[sd];
            cells.push((tr.w, st));
            tr
        });

        for (i, (a, sa)) in cells.iter().enumerate() {
            for (b, sb) in cells.iter().skip(i + 1) {
                let dot = a.truncate().dot(b.truncate()) - a.w * b.w;
                let dist = (dot * dot - 1.0).abs().sqrt();
                assert!(
                    dist + 1e-5 >= STEP,
                    "{sa:04o} overlaps {sb:04o}: {a:?}, {b:?}"
                );
            }
        }
    }
}
