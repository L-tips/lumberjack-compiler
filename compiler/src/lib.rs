use half::bf16;
pub use lumberjack_model;

pub mod compiled_model;
pub mod compiler;
pub mod csv_source;
pub mod feature_vectors;
pub mod problem;

pub use compiler::PlacementStrategy;

pub trait Feature: PartialOrd<Self> + Clone {
    const ZERO: Self;

    fn into_bf16(self) -> bf16;
}

impl Feature for f32 {
    const ZERO: f32 = 0.0_f32;

    fn into_bf16(self) -> bf16 {
        bf16::from_f32(self)
    }
}

impl Feature for bf16 {
    const ZERO: Self = Self::ZERO;

    fn into_bf16(self) -> bf16 {
        self
    }
}
