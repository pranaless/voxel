pub mod render;

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
