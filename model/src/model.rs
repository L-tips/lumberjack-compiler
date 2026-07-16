use core::fmt::{self, Debug};
use core::num::NonZeroU16;

use half::bf16;
use heapless::Vec;
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

/// A 3-byte integer
///
/// # Alignment
///
/// Caution: this struct is 1-byte aligned!
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Debug, Clone, Copy, TryFromBytes, Immutable)]
struct u24([u8; 3]);

impl u24 {
    pub fn as_bytes_le(&self) -> &[u8; 3] {
        &self.0
    }
}

impl TryFrom<u32> for u24 {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value >= u32::from_le_bytes([0xff, 0xff, 0xff, 0x0]) {
            return Err(());
        }
        Ok(Self(value.to_le_bytes()[0..3].try_into().unwrap()))
    }
}

impl From<u24> for u32 {
    fn from(val: u24) -> Self {
        u32::from_le_bytes([val.0[0], val.0[1], val.0[2], 0])
    }
}

#[repr(transparent)]
#[derive(IntoBytes, Clone, Copy, KnownLayout, Immutable, FromBytes)]
pub struct Flags(U16);

impl Flags {
    const fn new(split_var_idx: u16, left_is_prediction: bool, right_is_prediction: bool) -> Self {
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

    pub fn both_predictions(&self) -> bool {
        self.left_prediction() && self.right_prediction()
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
#[repr(transparent)]
pub struct CacheMetadata(U16);

impl CacheMetadata {
    pub fn new_cell_header(num_trees: NonZeroU16) -> Result<Self, Error> {
        Ok(Self(num_trees.get().into()))
    }

    pub fn new_empty() -> Self {
        Self(0.into())
    }

    /// Returns `true` if the cache header flag is set
    pub fn is_cell_header(&self) -> bool {
        self.0.get() > 0
    }

    /// Return the number of trees in the cache if this is a cache header.
    ///
    /// Returns [`None`]` otherwise.
    pub fn get_num_trees(&self) -> Option<u16> {
        if self.is_cell_header() {
            Some(self.0.get())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, IntoBytes, KnownLayout, Immutable, FromBytes)]
#[repr(C, align(8))]
pub struct TreeHeader {
    tree_len: U32,
    first_node_idx: U16,
    cache_metadata: CacheMetadata,
}

impl TreeHeader {
    pub fn new(
        tree_len: u32,
        first_node_idx: u16,
        cache_metadata: CacheMetadata,
    ) -> Result<Self, Error> {
        Ok(Self {
            tree_len: tree_len.into(),
            first_node_idx: first_node_idx.into(),
            cache_metadata,
        })
    }

    pub fn set_tree_len(&mut self, tree_len: u32) {
        self.tree_len = tree_len.into();
    }

    pub fn set_cache_metadata(&mut self, cache_metadata: CacheMetadata) {
        self.cache_metadata = cache_metadata
    }

    pub fn cache_metadata(&self) -> &CacheMetadata {
        &self.cache_metadata
    }

    pub fn tree_len(&self) -> u32 {
        self.tree_len.get()
    }

    pub fn first_node_idx(&self) -> u16 {
        self.first_node_idx.get()
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

#[repr(C, align(8))]
#[derive(Clone, IntoBytes, KnownLayout, Immutable, FromBytes, Debug)]
pub struct Node([u8; 8]);

pub const PADDING: Node = Node([0; 8]);

impl Node {
    #[inline]
    pub fn as_header(&self) -> &TreeHeader {
        // Infallible: any 8 bytes are a valid TreeHeader (a u32, a u24 and a u8).
        TreeHeader::ref_from_bytes(&self.0).unwrap()
    }

    #[inline]
    pub fn as_header_mut(&mut self) -> &mut TreeHeader {
        // Infallible: any 8 bytes are a valid TreeHeader (a u32, a u24 and a u8).
        TreeHeader::mut_from_bytes(&mut self.0).unwrap()
    }

    pub fn from_header(header: TreeHeader) -> Self {
        Self(header.as_bytes().try_into().unwrap())
    }

    pub fn as_branch(&self) -> &Branch {
        // Infaillible for the same reason
        Branch::ref_from_bytes(&self.0).unwrap()
    }

    pub fn from_branch(branch: Branch) -> Self {
        Self(branch.as_bytes().try_into().unwrap())
    }

    pub fn is_padding(&self) -> bool {
        self.0.iter().all(|b| *b == 0)
    }

    /// View a slice of Nodes as a byte slice.
    pub fn slice_as_bytes(nodes: &[Node]) -> &[u8] {
        let byte_len = core::mem::size_of_val(nodes);
        unsafe { core::slice::from_raw_parts(nodes.as_ptr().cast::<u8>(), byte_len) }
    }
}

/// An array-backed, optimized random forest model
#[repr(C, align(16))]
#[derive(TryFromBytes, KnownLayout, Immutable)]
pub struct Model<'data> {
    num_trees: u24,
    num_cells: u8,
    num_features: U16,
    num_targets: U16,
    _padding: u64,
    nodes: &'data [Node],
}

impl<'data> Model<'data> {
    pub fn new(
        num_trees: u32,
        num_cells: u8,
        nodes: &'data [Node],
        num_features: NonZeroU16,
        num_targets: NonZeroU16,
    ) -> Result<Self, Error> {
        let num_trees = num_trees
            .try_into()
            .expect("num_trees must fit into 3 bytes");

        Ok(Self {
            num_trees,
            num_cells,
            num_features: U16::new(num_features.get()),
            num_targets: U16::new(num_targets.get()),
            _padding: 0,
            nodes,
        })
    }

    pub fn nodes(&self) -> &[Node] {
        self.nodes
    }

    pub fn num_trees(&self) -> u32 {
        self.num_trees.into()
    }
    pub fn num_trees_to_bytes(&self) -> &[u8] {
        self.num_trees.as_bytes_le()
    }

    pub fn num_cells(&self) -> u8 {
        self.num_cells
    }

    pub fn num_targets(&self) -> U16 {
        self.num_targets
    }

    pub fn num_features(&self) -> U16 {
        self.num_features
    }

    /// Verify that the forest contains no circular references
    #[cfg(feature = "std")]
    pub fn verify_acyclic(&self) -> Result<(), Error> {
        use std::collections::{HashMap, HashSet};
        use std::vec::Vec;

        let nodes_len = self.nodes().len();
        let mut num_trees = 0;

        for header_idx in self.tree_headers() {
            num_trees += 1;
            let (header, tree_nodes) = self.get_tree(header_idx);

            if !(header as *const _ as usize).is_multiple_of(ALIGNMENT) {
                return Err(Error::MisalignedData);
            }

            // DFS cycle detection using three-color marking:
            // White (absent) = unvisited, Gray (false) = in current path, Black (true) =
            // done
            let mut state: HashMap<usize, bool> = HashMap::new();

            let root_idx = header.first_node_idx() as usize;

            // Find the root: the one node not referenced as a child by any other node
            let mut referenced = HashSet::new();
            for node in tree_nodes.iter().skip(root_idx) {
                if node.is_padding() {
                    continue;
                }
                let branch = node.as_branch();
                if !branch.flags().left_prediction() {
                    let ptr = branch.left_ptr().get() as usize;
                    if ptr < nodes_len {
                        referenced.insert(ptr);
                    }
                }
                if !branch.flags().right_prediction() {
                    let ptr = branch.right_ptr().get() as usize;
                    if ptr < nodes_len {
                        referenced.insert(ptr);
                    }
                }
            }
            let root = tree_nodes
                .iter()
                .enumerate()
                .skip(root_idx)
                .find(|(i, n)| !n.is_padding() && !referenced.contains(i))
                .map(|(i, _)| i)
                .ok_or(Error::MalformedForest)?;

            // Iterative DFS with explicit stack to avoid stack overflow on deep trees.
            // Each stack entry is (node_idx, already_pushed_children).
            let mut stack: Vec<(usize, bool)> = vec![(root, false)];

            while let Some((idx, children_pushed)) = stack.last_mut() {
                let idx = *idx;

                if *children_pushed {
                    // All descendants processed — mark black (done)
                    stack.pop();
                    state.insert(idx, true);
                    continue;
                }

                *children_pushed = true;

                match state.get(&idx) {
                    Some(true) => {
                        // Already fully processed, skip
                        stack.pop();
                        continue;
                    }
                    Some(false) => {
                        // Encountered a gray node — cycle detected
                        return Err(Error::CyclicTree);
                    }
                    None => {}
                }

                // Validate index is in bounds
                if idx >= nodes_len {
                    return Err(Error::MalformedForest);
                }

                let node = &tree_nodes[idx];
                if node.is_padding() {
                    return Err(Error::MalformedForest);
                }

                // Mark gray (in current path)
                state.insert(idx, false);

                let branch = node.as_branch();
                if !branch.flags().left_prediction() {
                    stack.push((branch.left_ptr().get() as usize, false));
                }
                if !branch.flags().right_prediction() {
                    stack.push((branch.right_ptr().get() as usize, false));
                }
            }
        }

        assert_eq!(num_trees, self.num_trees());
        Ok(())
    }

    pub fn next_left<'a>(tree_nodes: &'a [Node], branch: &Branch) -> &'a Node {
        &tree_nodes[branch.left_ptr().get() as usize]
    }

    pub fn next_right<'a>(tree_nodes: &'a [Node], branch: &Branch) -> &'a Node {
        &tree_nodes[branch.right_ptr().get() as usize]
    }

    /// Return an iterator which yields the indices of all tree headers in this
    /// forest
    pub fn tree_headers(&self) -> HeadersIterator<'_> {
        iter_trees(self.nodes())
    }

    /// Return an iterator which yields the indices of all cell headers in this
    /// forest
    pub fn cache_headers(&self) -> CellsIterator<'_> {
        iter_cells(self.nodes())
    }

    /// Perform a prediction with the given features vector.
    ///
    /// In case of a tie, the class with the lowest index wins.
    #[inline(never)]
    pub fn predict(&self, features: &[bf16]) -> PredictionOutput {
        const MAX_NUM_TREES: usize = 255;
        let mut votes = [0; MAX_NUM_TREES];

        for header_idx in self.tree_headers() {
            let (header, tree_nodes) = self.get_tree(header_idx);

            let mut node = tree_nodes[header.first_node_idx() as usize].as_branch();

            let prediction = loop {
                let test = features[node.split_with() as usize] <= node.split_at();

                if test {
                    if node.flags.left_prediction() {
                        break node.left_ptr().get();
                    } else {
                        node = Self::next_left(tree_nodes, node).as_branch();
                    }
                } else if node.flags.right_prediction() {
                    break node.right_ptr().get();
                } else {
                    node = Self::next_right(tree_nodes, node).as_branch();
                }
            };

            // Register the vote for this tree's prediction
            let vote = votes
                .get_mut(prediction as usize)
                .expect("Not enough space for this class");
            *vote += 1;
        }

        // Select the class with the highest votes, preferring the lowest class index in
        // case of a tie
        votes
            .into_iter()
            .enumerate()
            .max_by(|(idx_a, votes_a), (idx_b, votes_b)| {
                votes_a.cmp(votes_b).then_with(|| idx_b.cmp(idx_a))
            })
            .unwrap()
            .0
            .try_into()
            .unwrap()
    }

    /// Get the header and node slice of the tree that starts at the provided
    /// index
    pub fn get_tree(&self, header_idx: usize) -> (&TreeHeader, &[Node]) {
        let header = self.nodes[header_idx].as_header();

        let last_node_idx = header_idx + header.tree_len() as usize - 1;
        (header, &self.nodes[header_idx..=last_node_idx])
    }

    /// Takes a compiled [`Model`], and returns a [`Vec`] of `&[Node]`, each
    /// `Vec`` element representing the nodes which should be written to a
    /// single cell's cache.
    pub fn split_cache_data<const MAX_CELLS: usize>(&self) -> Vec<&[Node], MAX_CELLS> {
        let mut result = heapless::Vec::new();

        for cache_head_idx in self.cache_headers() {
            let num_trees_in_cache = self.nodes()[cache_head_idx]
                .as_header()
                .cache_metadata()
                .get_num_trees()
                .expect("Node is a cell header");

            let last_tree_node_idx = iter_trees(&self.nodes()[cache_head_idx..])
                .take(num_trees_in_cache as usize)
                .last()
                .unwrap()
                + cache_head_idx;

            let last_tree_len = self.nodes()[last_tree_node_idx].as_header().tree_len();

            let end_idx = last_tree_node_idx + last_tree_len as usize;
            let slice = &self.nodes()[cache_head_idx..end_idx];

            result
                .push(slice)
                .unwrap_or_else(|_| panic!("Model uses more than {MAX_CELLS} caches"));
        }

        result
    }
}

/// Iterate over the tree headers in the provided [`Node`] slice.
///
/// The first node in the slice *must* be a tree header.
pub fn iter_trees(nodes: &[Node]) -> HeadersIterator<'_> {
    HeadersIterator::new(nodes)
}

pub struct HeadersIterator<'a> {
    nodes: &'a [Node],
    current_idx: usize,
    first_pass: bool,
}

impl<'a> HeadersIterator<'a> {
    fn new(nodes: &'a [Node]) -> Self {
        Self {
            nodes,
            current_idx: 0,
            first_pass: true,
        }
    }
}

impl<'a> Iterator for HeadersIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.first_pass {
            self.first_pass = false;
            return Some(0);
        }
        self.current_idx += self.nodes[self.current_idx].as_header().tree_len() as usize;
        (self.current_idx < self.nodes.len()).then_some(self.current_idx)
    }
}

/// Iterate over the cell headers in the provided [`Node`] slice.
///
/// The first node in the slice *must* be a cell header.
pub fn iter_cells(nodes: &[Node]) -> CellsIterator<'_> {
    CellsIterator::new(nodes)
}

pub struct CellsIterator<'a> {
    iter: HeadersIterator<'a>,
}

impl<'a> CellsIterator<'a> {
    fn new(nodes: &'a [Node]) -> Self {
        Self {
            iter: HeadersIterator::new(nodes),
        }
    }
}

impl<'a> Iterator for CellsIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(idx) = self.iter.next() {
            if self.iter.nodes[idx]
                .as_header()
                .cache_metadata
                .is_cell_header()
            {
                return Some(idx);
            }
        }

        None
    }
}

impl fmt::Display for Model<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "OPTIMIZED CLASSIFICATION Forest: {} trees, size {}, {} features,
{} targets\n------------",
            self.num_trees(),
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
            let branch = node.as_branch();

            write!(f, "[{abs_id}/{node_idx}]\t")?;
            if node_idx == 0 {
                let header = node.as_header();
                writeln!(f, "{header:?}")?;

                first_node_idx = header.first_node_idx();
                tree_len = header.tree_len();
            } else if node_idx < first_node_idx || node.is_padding() {
                writeln!(f, "Padding | {:?}", node.as_bytes())?;
            } else {
                writeln!(f, "{branch}")?;
            }

            node_idx += 1;

            if node_idx as u32 >= tree_len {
                tree_idx += 1;
                node_idx = 0;
                writeln!(f, "TREE #{tree_idx}")?;
            }
        }
        writeln!(f, "------------")?;
        Ok(())
    }
}
