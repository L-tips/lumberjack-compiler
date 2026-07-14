use std::collections::HashMap;
use std::fmt::{self, Display};
use std::num::NonZeroU16;

use color_eyre::Result;
use color_eyre::eyre::{Context, ContextCompat, eyre};
use half::bf16;
use lumberjack_model::model::CacheMetadata;

use crate::Feature;
use crate::problem::{Map, ProblemDefinition};

mod tree_compiler;
pub use tree_compiler::PlacementStrategy;

mod tree_partition;
pub use tree_partition::PartitionStrategy;
pub(crate) use tree_partition::tree_max_depth;

#[derive(Debug, Clone)]
pub struct BranchNode<F: Feature> {
    pub(super) split_with: u16,
    pub(super) split_at: F,
    pub(super) left: usize,
    pub(super) right: usize,
}

impl<F: Feature + Display> Display for BranchNode<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Branch | split_with: {}, split_at: {}, left: {}, right: {}",
            self.split_with, self.split_at, self.left, self.right
        )
    }
}

pub type PredictionOutput = u16;

#[derive(Debug, PartialEq, Clone)]
pub struct LeafNode {
    pub(super) prediction: PredictionOutput,
}

#[derive(Debug, Clone)]
pub enum Node<F: Feature> {
    Leaf(LeafNode),
    Branch(BranchNode<F>),
}

impl<F: Feature> Node<F> {
    pub fn is_branch(&self) -> bool {
        matches!(self, Self::Branch(_))
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    pub fn as_branch(&self) -> Option<&BranchNode<F>> {
        match self {
            Node::Branch(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_leaf(&self) -> Option<&LeafNode> {
        match self {
            Node::Leaf(l) => Some(l),
            _ => None,
        }
    }
}

impl<F: Feature + Display> Display for Node<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Leaf(leaf) => write!(f, "Leaf   | prediction: {}", leaf.prediction),
            Node::Branch(b) => write!(f, "{b}"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Tree<F: Feature> {
    nodes: Vec<Node<F>>,
}

impl<F: Feature> Tree<F> {
    pub fn new(nodes: Vec<Node<F>>) -> Self {
        Self { nodes }
    }

    fn next_left(&self, branch: &BranchNode<F>) -> &Node<F> {
        &self.nodes[branch.left]
    }

    fn next_right(&self, branch: &BranchNode<F>) -> &Node<F> {
        &self.nodes[branch.right]
    }
}

impl<F: Feature + Display> Display for Tree<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, n) in self.nodes.iter().enumerate() {
            writeln!(f, "\tNode {i}:\t{n}")?;
        }

        Ok(())
    }
}

/// Intermediate representation (IR) of a tree ensemble model
#[derive(Debug)]
pub struct InterRep<F: Feature> {
    trees: Vec<Tree<F>>,
    problem: ProblemDefinition,
}

impl<F: Feature> InterRep<F> {
    pub(crate) fn new(trees: Vec<Tree<F>>, problem: ProblemDefinition) -> Self {
        Self { trees, problem }
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn num_features(&self) -> usize {
        self.problem.features().len()
    }

    pub fn num_targets(&self) -> usize {
        self.problem.targets().len()
    }

    pub fn targets(&self) -> &Map {
        self.problem.targets()
    }

    pub fn features(&self) -> &Map {
        self.problem.features()
    }

    pub fn problem(&self) -> &ProblemDefinition {
        &self.problem
    }

    /// Make a prediction based on input feature vector. Returns a tuple
    /// containing the prediction, and its number of votes.
    pub fn predict<'a, 's: 'a>(&'s self, features: &[F]) -> (&'a str, usize) {
        let votes = self.prediction_votes(features);

        let (best_result, num_votes) = best_result(&votes);

        let prediction = self
            .targets()
            .iter()
            .find(|(_, t)| **t == best_result)
            .unwrap()
            .0
            .as_ref();

        (prediction, num_votes)
    }

    /// Make a prediction, and return a [`HashMap`] of `(C, V)` where `C` is the
    /// class ID and `V` its number of votes
    pub fn prediction_votes(&self, features: &[F]) -> HashMap<u16, usize> {
        let mut votes = HashMap::new();

        // Descend into each tree to make a prediction
        for tree in &self.trees {
            // Tree root
            let mut node = &tree.nodes[0];

            let prediction = loop {
                match node {
                    Node::Branch(b) => {
                        let test = features[b.split_with as usize] <= b.split_at;
                        if test {
                            node = tree.next_left(b)
                        } else {
                            node = tree.next_right(b)
                        }
                    }
                    Node::Leaf(l) => {
                        break l.prediction;
                    }
                }
            };

            *votes.entry(prediction).or_insert(0) += 1;
        }

        votes
    }

    /// Turn this [`ForestModel`] into a
    /// [`Model`](lumberjack_model::model::Model).
    pub fn compile(
        &self,
        num_cells: u8,
        placement_strategy: PlacementStrategy,
        partition_strategy: PartitionStrategy,
    ) -> Result<Vec<lumberjack_model::model::Node>> {
        // Step 1: Compile each tree individually
        let compiled_trees = self
            .trees
            .iter()
            .map(|t| tree_compiler::compile(t, placement_strategy))
            .collect::<Result<Vec<_>, _>>()?;

        // Step 2: Partition trees into cell chunks
        let mut cell_chunks = partition_strategy.partition(&compiled_trees, num_cells);

        // Step 2.1: Set cache metadata for cache headers
        for cell_trees in cell_chunks.iter_mut() {
            let num_trees = cell_trees.len();

            // Skip empty cells
            if num_trees < 1 {
                continue;
            }

            cell_trees[0][0].as_header_mut().set_cache_metadata(
                CacheMetadata::new_cell_header(
                    NonZeroU16::new(
                        num_trees
                            .try_into()
                            .context("num_trees must fit inside a u16")?,
                    )
                    .context("Number of caches must not be zero")?,
                )
                .map_err(|e| eyre!("Invalid cache metadata: {e:?}"))?,
            );
        }

        // Step 3: Merge all trees into a single Vec
        let compiled_nodes = cell_chunks.into_iter().flatten().flatten().collect();
        Ok(compiled_nodes)
    }
}

impl InterRep<f32> {
    /// Turn an `IntermediateRepresentation<f32>` into an
    /// `IntermediateRepresentation<bf16>` by truncating the splits
    pub fn quantize_splits(self) -> InterRep<bf16> {
        let trees = self
            .trees
            .into_iter()
            .map(|tree| {
                let nodes = tree
                    .nodes
                    .into_iter()
                    .map(|node| match node {
                        Node::Leaf(leaf) => Node::Leaf(leaf),
                        Node::Branch(branch) => Node::Branch(BranchNode {
                            split_with: branch.split_with,
                            split_at: bf16::from_f32(branch.split_at),
                            left: branch.left,
                            right: branch.right,
                        }),
                    })
                    .collect();

                Tree::new(nodes)
            })
            .collect();

        InterRep {
            trees,
            problem: self.problem,
        }
    }
}

impl<F: Feature + Display> Display for InterRep<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Classification Forest: {} trees, size {}, {} features, {} targets\n------------",
            self.num_trees(),
            self.trees.iter().map(|t| t.nodes.len()).sum::<usize>(),
            self.problem.features().len(),
            self.problem.targets().len(),
        )?;
        for (i, tree) in self.trees.iter().enumerate() {
            writeln!(f, "Tree {i}:")?;
            for (j, node) in tree.nodes.iter().enumerate() {
                writeln!(f, "\t{j}: {node}")?;
            }
        }

        writeln!(f, "------------")?;

        let mut features_ordered = self.problem.features().iter().collect::<Vec<_>>();
        features_ordered.sort_by(|a, b| a.1.cmp(b.1));

        writeln!(f, "Features: ")?;
        for feat in features_ordered.iter() {
            writeln!(f, "\t{}: {}", feat.1, feat.0)?;
        }

        let mut targets_ordered = self.problem.targets().iter().collect::<Vec<_>>();
        targets_ordered.sort_by(|a, b| a.1.cmp(b.1));

        writeln!(f, "Targets: ")?;
        for t in targets_ordered.iter() {
            writeln!(f, "\t{}: {}", t.1, t.0)?;
        }

        writeln!(f, "------------")?;

        Ok(())
    }
}

/// Given a map of class votes, return the best class ID and its number of
/// votes.
///
/// A tie breaker selects the lowest class index in case of a tie.
pub fn best_result(votes: &HashMap<u16, usize>) -> (u16, usize) {
    let (id, votes) = votes
        .iter()
        .max_by(|(idx_a, votes_a), (idx_b, votes_b)| {
            votes_a.cmp(votes_b).then_with(|| idx_b.cmp(idx_a))
        })
        .unwrap();
    (*id, *votes)
}
