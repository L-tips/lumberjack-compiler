use core::fmt::{self, Debug};
use std::{mem::ManuallyDrop, num::NonZeroU16};

use half::bf16;
use zerocopy::{
    FromBytes, Immutable, IntoBytes, KnownLayout, TryFromBytes,
    byteorder::little_endian::{U16, U32},
};

use crate::Error;

pub mod deserialize;

#[cfg(feature = "std")]
pub mod serialize;

pub const ALIGNMENT: usize = 16;
pub(crate) type NodePointer = zerocopy::little_endian::U16;

pub type PredictionOutput = u16;

pub struct Classification {
    num_targets: NonZeroU16,
}

impl Classification {
    pub fn new(num_targets: u16) -> Result<Self, Error> {
        let num_targets = NonZeroU16::new(num_targets).ok_or(Error::MalformedForest)?;
        Ok(Self { num_targets })
    }
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
pub struct TreeHeader {
    tree_len: u32,
    first_node_idx: u32,
}

impl TreeHeader {
    pub fn new(tree_len: u32, first_node_idx: u32) -> Self {
        Self {
            tree_len,
            first_node_idx,
        }
    }

    pub fn set_tree_len(&mut self, tree_len: u32) {
        self.tree_len = tree_len;
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
        left: u16,
        right: u16,
        left_leaf: bool,
        right_leaf: bool,
    ) -> Self {
        let flags = Flags::new(split_with, left_leaf, right_leaf);
        Self {
            flags,
            split_at,
            left: U16::new(left),
            right: U16::new(right),
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

    pub fn is_padding(&self) -> bool {
        self.as_bytes().iter().all(|b| *b == 0)
    }
}

impl fmt::Display for Branch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Branch | split var: {}, split: {}, left: {}/{}, right: {}/{}",
            self.flags.split_var_idx(),
            self.split_at,
            self.flags.left_prediction(),
            self.left,
            self.flags.right_prediction(),
            self.right,
        )
    }
}

pub union Node {
    pub header: ManuallyDrop<TreeHeader>,
    pub padding: u64,
    pub branch: ManuallyDrop<Branch>,
}

/// An array-backed, optimized random forest model
#[repr(C, align(16))]
#[derive(TryFromBytes, KnownLayout, Immutable)]
pub struct OptimizedForest<'data> {
    num_trees: U32,
    num_features: U16,
    /// If num_targets is Some, we have a classification problem.
    /// Otherwise, we have a regression problem.
    num_targets: U16,
    _padding: u64,
    nodes: &'data [Node],
}

impl<'data> OptimizedForest<'data> {
    pub fn new(
        num_trees: u32,
        nodes: &'data [Node],
        num_features: NonZeroU16,
        problem: Classification,
    ) -> Result<Self, Error> {
        Ok(Self {
            num_trees: U32::new(num_trees),
            num_features: U16::new(num_features.get()),
            num_targets: U16::new(problem.num_targets.get()),
            _padding: 0,
            nodes,
        })
    }

    pub fn nodes(&self) -> &[Node] {
        self.nodes
    }

    pub fn num_trees(&self) -> u32 {
        self.num_trees.get()
    }

    pub fn num_targets(&self) -> U16 {
        self.num_targets
    }

    pub fn num_features(&self) -> U16 {
        self.num_features
    }

    pub fn verify(&self) -> Result<(), Error> {
        let nodes_len = self.nodes().len();

        for header_idx in self.tree_headers() {
            let header = unsafe { &self.nodes[header_idx].header };

            if !(header as *const _ as usize).is_multiple_of(ALIGNMENT) {
                return Err(Error::MisalignedData);
            }
            let last_node_idx = header_idx + header.tree_len as usize - 1;
            let tree_nodes = &self.nodes[header_idx..=last_node_idx];

            for (i, branch) in tree_nodes
                .iter()
                .enumerate()
                .skip(header.first_node_idx as _)
            {
                let branch = unsafe { &branch.branch };

                // Skip padding
                if branch.is_padding() {
                    continue;
                }

                let is_left_prediction = branch.flags().left_prediction();
                let is_right_prediction = branch.flags().right_prediction();

                let left_ptr = branch.left_ptr().get() as usize;
                let right_ptr = branch.right_ptr().get() as usize;

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
        }

        Ok(())
    }

    pub fn next_left<'a>(tree_nodes: &'a [Node], branch: &ManuallyDrop<Branch>) -> &'a Node {
        &tree_nodes[branch.left_ptr().get() as usize]
    }

    pub fn next_right<'a>(tree_nodes: &'a [Node], branch: &ManuallyDrop<Branch>) -> &'a Node {
        &tree_nodes[branch.right_ptr().get() as usize]
    }

    /// Return an iterator which yields the indices of all tree headers in this
    /// forest
    pub fn tree_headers(&self) -> HeadersIterator<'_> {
        HeadersIterator::new(self.nodes, self.num_trees.get() as _)
    }

    #[inline(never)]
    pub fn predict(&self, features: &[bf16]) -> PredictionOutput {
        const MAX_NUM_TREES: usize = 255;
        let mut votes = [0; MAX_NUM_TREES];

        for header_idx in self.tree_headers() {
            let header = unsafe { &self.nodes[header_idx].header };
            println!("idx: {header_idx}, header: {header:?}");
            let last_node_idx = header_idx + header.tree_len as usize - 1;
            let tree_nodes = &self.nodes[header_idx..=last_node_idx];

            let mut node = unsafe { &tree_nodes[header.first_node_idx as usize].branch };

            let prediction = loop {
                let test = features[node.split_with() as usize] <= node.split_at();

                if test {
                    if node.flags.left_prediction() {
                        break node.left_ptr().get();
                    } else {
                        node = unsafe { &Self::next_left(tree_nodes, node).branch };
                    }
                } else if node.flags.right_prediction() {
                    break node.right_ptr().get();
                } else {
                    node = unsafe { &Self::next_right(tree_nodes, node).branch };
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

pub struct HeadersIterator<'a> {
    nodes: &'a [Node],
    current_idx: usize,
    tree_idx: usize,
    num_trees: usize,
    first_pass: bool,
}

impl<'a> HeadersIterator<'a> {
    fn new(nodes: &'a [Node], num_trees: usize) -> Self {
        Self {
            nodes,
            current_idx: 0,
            tree_idx: 0,
            first_pass: true,
            num_trees,
        }
    }

    pub fn len(&self) -> usize {
        self.num_trees
    }

    pub fn is_empty(&self) -> bool {
        self.num_trees == 0
    }
}

impl<'a> Iterator for HeadersIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.first_pass {
            self.first_pass = false;
            return Some(0);
        }
        self.tree_idx += 1;
        if self.tree_idx < self.num_trees {
            self.current_idx += unsafe { self.nodes[self.current_idx].header.tree_len as usize };
            Some(self.current_idx)
        } else {
            None
        }
    }
}

impl fmt::Display for OptimizedForest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "OPTIMIZED CLASSIFICATION Forest: {} trees, size {}, {} features,
{} targets\n------------",
            self.num_trees,
            self.nodes.len(),
            self.num_features,
            self.num_targets
        )?;

        writeln!(f, "TREE #0")?;

        let mut tree_idx = 0;
        let mut node_idx = 0;
        let mut tree_len = 0;
        let mut first_node_idx = 0;

        for (abs_id, node) in self.nodes.iter().enumerate() {
            write!(f, "[{abs_id}/{node_idx}]\t")?;
            if node_idx == 0 {
                let header = unsafe { &node.header };
                writeln!(f, "{header:?}")?;

                first_node_idx = header.first_node_idx;
                tree_len = header.tree_len;
            } else if node_idx < first_node_idx || unsafe { node.branch.is_padding() } {
                let padding = unsafe { node.padding };
                writeln!(f, "Padding | {padding}")?;
            } else {
                let n = unsafe { &node.branch };
                writeln!(f, "{}", ManuallyDrop::into_inner(n.clone()))?;
            }

            node_idx += 1;

            if node_idx >= tree_len {
                tree_idx += 1;
                node_idx = 0;
                writeln!(f, "TREE #{tree_idx}")?;
            }
        }
        writeln!(f, "------------")?;
        Ok(())
    }
}
