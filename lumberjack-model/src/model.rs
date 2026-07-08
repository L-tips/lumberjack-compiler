use core::fmt::{self, Debug};
use core::num::NonZeroU16;

use half::bf16;
use zerocopy::{
    FromBytes, Immutable, IntoBytes, KnownLayout, TryFromBytes,
    byteorder::little_endian::{U16, U32},
};

use crate::Error;

pub mod deserialize;

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

    pub fn tree_len(&self) -> u32 {
        self.tree_len
    }

    pub fn first_node_idx(&self) -> u32 {
        self.first_node_idx
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
#[derive(Clone, IntoBytes, KnownLayout, Immutable, FromBytes)]
pub struct Node([u8; 8]);

pub const PADDING: Node = Node([0; 8]);

impl Node {
    #[inline]
    pub fn as_header(&self) -> &TreeHeader {
        // Infallible: any 8 bytes are a valid TreeHeader (two u32s).
        TreeHeader::ref_from_bytes(&self.0).unwrap()
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
}

/// An array-backed, optimized random forest model
#[repr(C, align(16))]
#[derive(TryFromBytes, KnownLayout, Immutable)]
pub struct Model<'data> {
    num_trees: U32,
    num_features: U16,
    /// If num_targets is Some, we have a classification problem.
    /// Otherwise, we have a regression problem.
    num_targets: U16,
    _padding: u64,
    nodes: &'data [Node],
}

impl<'data> Model<'data> {
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

    pub fn num_trees(&self) -> U32 {
        self.num_trees
    }

    pub fn num_targets(&self) -> U16 {
        self.num_targets
    }

    pub fn num_features(&self) -> U16 {
        self.num_features
    }

    pub fn verify(&self) -> Result<(), Error> {
        let nodes_len = self.nodes().len();

        let mut num_trees = 0;
        for header_idx in self.tree_headers() {
            num_trees += 1;
            let (header, tree_nodes) = self.get_tree(header_idx);

            if !(header as *const _ as usize).is_multiple_of(ALIGNMENT) {
                return Err(Error::MisalignedData);
            }

            for (i, node) in tree_nodes
                .iter()
                .enumerate()
                .skip(header.first_node_idx as _)
            {
                // Skip padding
                if node.is_padding() {
                    continue;
                }

                let branch = node.as_branch();

                let is_left_prediction = branch.flags().left_prediction();
                let is_right_prediction = branch.flags().right_prediction();

                let left_ptr = branch.left_ptr().get() as usize;
                let right_ptr = branch.right_ptr().get() as usize;

                if (!is_left_prediction && (left_ptr <= i || left_ptr >= nodes_len))
                    || (!is_right_prediction && (right_ptr <= i || right_ptr >= nodes_len))
                {
                    return Err(Error::MalformedForest);
                }
            }
        }

        assert_eq!(num_trees, self.num_trees().get());

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
        HeadersIterator::new(self.nodes, self.num_trees.get() as _)
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

            let mut node = tree_nodes[header.first_node_idx as usize].as_branch();

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

        #[cfg(feature = "std")]
        println!("votes: {votes:?}");

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

    /// Get the tree's nodes and header
    fn get_tree(&self, header_idx: usize) -> (&TreeHeader, &[Node]) {
        let header = self.nodes[header_idx].as_header();

        let last_node_idx = header_idx + header.tree_len as usize - 1;
        (header, &self.nodes[header_idx..=last_node_idx])
    }

    /// Perform an analysis of the model, and return some useful metrics.
    ///
    /// The function takes a number of cells as an optional parameter. If set,
    /// will also info about the cell utilization.
    #[cfg(feature = "std")]
    pub fn analyze(&self, num_cells: Option<usize>) {
        use bytesize::ByteSize;
        println!("--- Lumberjack model analysis ---");
        self.verify()
            .unwrap_or_else(|e| panic!("Could not verify forest: {e:?}"));

        println!(
            "Random forest model with:\n\t- {} trees\n\t- {} features\n\t- {} targets",
            self.num_trees(),
            self.num_features(),
            self.num_targets()
        );

        let size_bytes = ByteSize::b(size_of_val(self.nodes()) as u64);
        println!("Total size: {} nodes ({size_bytes})", self.nodes().len(),);

        struct TreeMetadata {
            size: usize,
            depth: usize,
        }

        let mut tree_metadata = Vec::new();

        for (i, header_idx) in self.tree_headers().enumerate() {
            let (header, nodes) = self.get_tree(header_idx);
            let max_depth = tree_max_depth(nodes, header.first_node_idx as usize);

            let size_bytes = bytesize::ByteSize::b(size_of_val(nodes) as u64);
            println!(
                "/ Tree {i} / Size: {} nodes ({size_bytes}) / Max depth: {max_depth}",
                nodes.len(),
            );

            tree_metadata.push(TreeMetadata {
                size: nodes.len(),
                depth: max_depth,
            });
        }

        let max_size = tree_metadata.iter().map(|m| m.size).max().unwrap();
        let max_depth = tree_metadata.iter().map(|m| m.depth).max().unwrap();

        let size_bytes = ByteSize::b((max_size * size_of::<Node>()) as u64);
        println!();
        println!("Max tree size: {max_size} nodes ({size_bytes})",);
        println!("Max tree depth: {max_depth}\n");

        if let Some(c) = num_cells {
            println!("/ Cell analysis /");

            let num_trees = self.num_trees().get() as usize;

            let base = num_trees / c;
            let extra = num_trees % c;

            let mut trees_iter = self.tree_headers().enumerate();

            for cell_idx in 0..c {
                // Lowest cache indices get the extra trees.
                let num_trees = base + usize::from(cell_idx < extra);
                let headers = trees_iter.by_ref().take(num_trees);

                let mut required_capacity = 0;
                let mut max_depth = 0;
                for (tree_idx, header_idx) in headers {
                    let header = self.nodes[header_idx].as_header();
                    required_capacity += header.tree_len() as usize;
                    max_depth += tree_metadata[tree_idx].depth;
                }

                let size_bytes = ByteSize((required_capacity * size_of::<Node>()) as u64);
                println!(
                    "  Cell {cell_idx}:\n\t{num_trees} trees\n\t{required_capacity} nodes ({size_bytes})\n\tMax depth: {max_depth}"
                );
            }
        }

        println!("------");
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
            self.current_idx += self.nodes[self.current_idx].as_header().tree_len as usize;
            Some(self.current_idx)
        } else {
            None
        }
    }
}

impl fmt::Display for Model<'_> {
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
            let branch = node.as_branch();

            write!(f, "[{abs_id}/{node_idx}]\t")?;
            if node_idx == 0 {
                let header = node.as_header();
                writeln!(f, "{header:?}")?;

                first_node_idx = header.first_node_idx;
                tree_len = header.tree_len;
            } else if node_idx < first_node_idx || node.is_padding() {
                writeln!(f, "Padding | {:?}", node.as_bytes())?;
            } else {
                writeln!(f, "{branch}")?;
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

/// Maximum depth of a single tree.
#[allow(dead_code)]
fn tree_max_depth(tree_nodes: &[Node], root_idx: usize) -> usize {
    let branch = tree_nodes[root_idx].as_branch();

    let left_depth = if branch.flags().left_prediction() {
        1
    } else {
        1 + tree_max_depth(tree_nodes, branch.left_ptr().get() as usize)
    };

    let right_depth = if branch.flags().right_prediction() {
        1
    } else {
        1 + tree_max_depth(tree_nodes, branch.right_ptr().get() as usize)
    };

    left_depth.max(right_depth)
}
