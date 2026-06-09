#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

pub mod forest;
pub mod ptr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    WrongProblemType,
    MalformedForest,
    NoTargets,
    NoFeatures,
}
