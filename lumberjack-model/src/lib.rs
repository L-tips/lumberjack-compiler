#![no_std]

pub use phf;

pub mod model;
pub mod storage;

pub use model::Model;
pub use storage::BackingStorage;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    WrongProblemType,
    MalformedForest,
    MisalignedData,
    NoTargets,
    NoFeatures,
}
