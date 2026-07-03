#![no_std]

pub mod forest;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    WrongProblemType,
    MalformedForest,
    MisalignedData,
    NoTargets,
    NoFeatures,
}
