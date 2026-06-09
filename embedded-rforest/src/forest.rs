use core::{
    fmt::{self, Debug},
    marker::PhantomData,
    num::NonZeroU8,
};

use half::bf16;
use zerocopy::{
    FromBytes, Immutable, IntoBytes, KnownLayout, TryFromBytes,
    byteorder::little_endian::{U16, U32},
};

use crate::{Error, ptr::NodePointer};

pub mod deserialize;

#[cfg(feature = "std")]
pub mod serialize;

pub trait ProblemType {
    type Output: Copy;
    const HAS_TARGETS: bool;
}

pub trait Predict {
    type ProblemType: ProblemType;

    /// Make a prediction based on input values (features)
    fn predict(&self, features: &[bf16]) -> <Self::ProblemType as ProblemType>::Output;
}

pub struct Classification {
    num_targets: NonZeroU8,
}

impl Classification {
    pub fn new(num_targets: u8) -> Result<Self, Error> {
        let num_targets = NonZeroU8::new(num_targets).ok_or(Error::MalformedForest)?;
        Ok(Self { num_targets })
    }
}

impl ProblemType for Classification {
    type Output = u16;
    const HAS_TARGETS: bool = true;
}

#[repr(transparent)]
#[derive(IntoBytes, Clone, Copy, KnownLayout, Immutable, FromBytes)]
pub struct Flags(U16);

impl Flags {
    fn new(split_var_idx: u16, left_is_prediction: bool, right_is_prediction: bool) -> Self {
        assert!(split_var_idx <= u16::MAX >> 2);

        let val = split_var_idx
            | ((left_is_prediction as u16) << (16 - 1))
            | ((right_is_prediction as u16) << (16 - 2));
        Self(U16::new(val))
    }

    pub fn left_prediction(&self) -> bool {
        (self.0 >> (16 - 1)) & 1 != 0
    }

    pub fn right_prediction(&self) -> bool {
        (self.0 >> (16 - 2)) & 1 != 0
    }

    pub fn split_var_idx(&self) -> u16 {
        (self.0 & (u16::MAX >> 2)).get()
    }
}

impl Debug for Flags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Flags {{ left is leaf: {}, right is leaf: {}, split var: {} }}",
            self.left_prediction(),
            self.right_prediction(),
            self.split_var_idx()
        )
    }
}

#[derive(Debug, Clone, IntoBytes, KnownLayout, Immutable, FromBytes)]
#[repr(C, align(8))]
pub struct Branch {
    left: NodePointer,
    right: NodePointer,
    split_at: bf16,
    flags: Flags,
}

impl Branch {
    #[inline]
    pub fn new(
        split_with: u16,
        split_at: bf16,
        left: NodePointer,
        right: NodePointer,
        left_leaf: bool,
        right_leaf: bool,
    ) -> Self {
        let flags = Flags::new(split_with, left_leaf, right_leaf);
        Self {
            flags,
            split_at,
            left,
            right,
        }
    }

    #[inline]
    pub fn split_with(&self) -> u16 {
        self.flags.split_var_idx()
    }

    #[inline]
    pub fn split_at(&self) -> bf16 {
        self.split_at
    }

    #[inline]
    pub fn left_ptr(&self) -> NodePointer {
        self.left
    }

    #[inline]
    pub fn right_ptr(&self) -> NodePointer {
        self.right
    }

    pub fn flags(&self) -> Flags {
        self.flags
    }
}

impl fmt::Display for Branch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Branch | split var: {}, split: {}, left: {}, right: {}",
            self.flags.split_var_idx(),
            self.split_at,
            self.left,
            self.right
        )
    }
}

/// An array-backed, optimized random forest model
#[repr(C, align(4))]
#[derive(TryFromBytes, KnownLayout, Immutable)]
pub struct OptimizedForest<'data, P: ProblemType> {
    num_trees: U32,
    num_features: u8,
    /// If num_targets is Some, we have a classification problem.
    /// Otherwise, we have a regression problem.
    num_targets: Option<NonZeroU8>,
    _padding: [u8; 2],
    nodes: &'data [Branch],
    _problem: PhantomData<P>,
}

impl<P: ProblemType> OptimizedForest<'_, P> {
    pub fn nodes(&self) -> &[Branch] {
        self.nodes
    }

    pub fn num_trees(&self) -> u32 {
        self.num_trees.get()
    }

    pub fn num_features(&self) -> u8 {
        self.num_features
    }

    pub fn verify(&self) -> Result<(), Error> {
        let nodes_len = self.nodes().len();

        for (i, branch) in self.nodes().iter().enumerate() {
            let is_left_prediction = branch.flags().left_prediction();
            let is_right_prediction = branch.flags().right_prediction();

            let left_ptr = branch.left_ptr().as_ptr() as usize;
            let right_ptr = branch.right_ptr().as_ptr() as usize;

            if (!is_left_prediction && (left_ptr <= i || left_ptr >= nodes_len))
                || (!is_right_prediction && (right_ptr <= i || right_ptr >= nodes_len))
            {
                #[cfg(feature = "std")]
                println!(
                    "Malformed forest: idx: {i}, nodes_len: {nodes_len}, is_left_prediction: {is_left_prediction}, is_right_prediction: {is_right_prediction}, left_ptr: {left_ptr}, right_ptr: {right_ptr}"
                );
                return Err(Error::MalformedForest);
            }
        }

        Ok(())
    }

    fn next_left(&self, branch: &Branch) -> &Branch {
        &self.nodes[branch.left_ptr().as_ptr() as usize]
    }

    fn next_right(&self, branch: &Branch) -> &Branch {
        &self.nodes[branch.right_ptr().as_ptr() as usize]
    }
}

impl<'data> OptimizedForest<'data, Classification> {
    pub fn new(
        num_trees: u32,
        nodes: &'data [Branch],
        num_features: u8,
        problem: Classification,
    ) -> Result<Self, Error> {
        Ok(Self {
            num_trees: U32::new(num_trees),
            nodes,
            num_features,
            num_targets: Some(problem.num_targets),
            _padding: [0; 2],
            _problem: PhantomData,
        })
    }

    pub fn num_targets(&self) -> Option<NonZeroU8> {
        self.num_targets
    }
}

impl Predict for OptimizedForest<'_, Classification> {
    type ProblemType = Classification;

    #[inline(never)]
    fn predict(&self, features: &[bf16]) -> <Self::ProblemType as ProblemType>::Output {
        let mut votes = [0; 255];

        for tree_id in 0..self.num_trees.get() {
            let mut node = &self.nodes[tree_id as usize];

            let prediction = loop {
                let test = features[node.split_with() as usize] <= node.split_at();

                if test {
                    if node.flags.left_prediction() {
                        break node.left_ptr().as_ptr();
                    } else {
                        node = self.next_left(node);
                    }
                } else if node.flags.right_prediction() {
                    break node.right_ptr().as_ptr();
                } else {
                    node = self.next_right(node);
                }
            };

            // Register the vote for this tree's prediction
            let vote = votes
                .get_mut(prediction as usize)
                .expect("Not enough space for this class");
            *vote += 1;
        }

        votes
            .into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .unwrap()
            .0
            .try_into()
            .unwrap()
    }
}

impl<P: ProblemType> fmt::Display for OptimizedForest<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tgts) = self.num_targets {
            writeln!(
                f,
                "OPTIMIZED CLASSIFICATION Forest: {} trees, size {}, {} features, {} targets\n------------",
                self.num_trees,
                self.nodes.len(),
                self.num_features,
                tgts
            )?;
        } else {
            writeln!(
                f,
                "OPTIMIZED REGRESSION Forest: {} trees, size {}, {} features\n------------",
                self.num_trees,
                self.nodes.len(),
                self.num_features,
            )?;
        }

        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f, "\t{i}: {node}")?;
        }
        writeln!(f, "------------")?;
        Ok(())
    }
}
