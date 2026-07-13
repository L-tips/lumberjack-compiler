use std::cmp::PartialOrd;
use std::collections::{HashMap, VecDeque};
use std::fmt::{self, Display};
use std::num::NonZeroU16;

use color_eyre::Result;
use color_eyre::eyre::{Context, ContextCompat, eyre};
use half::bf16;
use lumberjack_model::model::{CacheMetadata, PADDING, TreeHeader, iter_trees};
use tap::Tap;

use crate::problem::{Map, ProblemDefinition};

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

/// IR of a tree ensemble model
#[derive(Debug)]
pub struct IntermediateRepresentation<F: Feature> {
    trees: Vec<Tree<F>>,
    problem: ProblemDefinition,
}

impl<F: Feature> IntermediateRepresentation<F> {
    pub(crate) fn new(trees: Vec<Tree<F>>, problem: ProblemDefinition) -> Self {
        Self { trees, problem }
    }

    /// Turn this [`ForestModel`] into a
    /// [`Model`](lumberjack_model::model::Model).
    pub fn compile(&self, num_cells: u8) -> Result<Vec<lumberjack_model::model::Node>> {
        let max_forest_len = self.trees.iter().map(|t| t.nodes.len()).sum();
        let mut forest_nodes = Vec::with_capacity(max_forest_len);

        // Sort trees by length so that the longest ones end up at the end. If HW is
        // using multiple tree cells, the longer trees have a higher likelihood to end
        // up in a cell that will have fewer trees than its counterparts, improving the
        // average case prediction time.
        let trees = self
            .trees
            .clone()
            .tap_mut(|v| v.sort_by_key(|t| t.nodes.len()));

        for tree in trees.iter() {
            // The vec containing the nodes which have already been assigned
            let mut placed_nodes = Vec::with_capacity(tree.nodes.len());

            // Nodes that haven't been placed yet
            let mut waiting_nodes: VecDeque<_> = tree.nodes.iter().enumerate().collect();

            // Keep a list of nodes that will be placed later for better optimization
            let mut deferred_nodes: VecDeque<LinkedNode> = VecDeque::new();

            let mut parent_id = None;
            let (mut id, mut extracted_node) = waiting_nodes.pop_front().unwrap();
            loop {
                if let Some(n) = extracted_node.as_branch() {
                    // println!("extracted node with ID {id}: {extracted_node}");
                    // println!("unplaced_nodes: {unplaced_nodes:?}");

                    // Go hunting for left and right children
                    let left_child = extract_branch_if_leaf(&mut waiting_nodes, n.left);
                    let right_child = extract_branch_if_leaf(&mut waiting_nodes, n.right);

                    // println!(
                    //     "Node ID {id}. left: {left_child:?}, right: {right_child:?}, parent:
                    // {parent_id:?}" );

                    let node_to_place = LinkedNode {
                        id,
                        _parent_id: parent_id,
                        left_child,
                        right_child,
                        split_at: n.split_at.clone().into_bf16(),
                        split_with: n.split_with,
                    };

                    // If node is a double prediction, avoid placing it at a 128b boundary where it
                    // would be more optimal to place a node with at least one branch
                    if matches!(left_child, Branch::Prediction(_))
                        && matches!(right_child, Branch::Prediction(_))
                        && placed_nodes.len().is_multiple_of(2)
                    {
                        deferred_nodes.push_back(node_to_place);
                    } else {
                        placed_nodes.push(node_to_place);
                    }

                    parent_id = Some(id);

                    // println!("placed nodes: {placed_nodes:?}");

                    // Previous node was 128-bit aligned. Add one of its children right next to it
                    // to take advantage of the superscalar arch.
                    if placed_nodes.len() % 2 == 1 {
                        if let Branch::Ptr(l) = left_child {
                            (id, extracted_node) = extract_branch(&mut waiting_nodes, l);
                            continue;
                        } else if let Branch::Ptr(r) = right_child {
                            (id, extracted_node) = extract_branch(&mut waiting_nodes, r);
                            continue;
                        }
                    }

                    let Some((new_id, new_node)) = waiting_nodes.pop_front() else {
                        break;
                    };

                    id = new_id;
                    extracted_node = new_node;
                } else {
                    unreachable!("Leaf node should have been extracted already");
                }
            }

            for d in deferred_nodes {
                placed_nodes.push(d);
            }

            assert_eq!(waiting_nodes.len(), 0);

            // Check that every node at an even index has one of its children right after to
            // maximize superscalar utilization.
            for chunk in placed_nodes.chunks(2) {
                // The acceptable case is if both nodes in the
                // cache line are double predictions (ie, have no pointers).
                if chunk.len() == 2 {
                    if chunk[0].left_child.is_prediction()
                        && chunk[0].right_child.is_prediction()
                        && chunk[1].left_child.is_prediction()
                        && chunk[1].right_child.is_prediction()
                    {
                        continue;
                    }

                    let mut utilization_is_maximized = false;
                    if let Branch::Ptr(ptr) = chunk[0].left_child {
                        utilization_is_maximized = ptr == chunk[1].id;
                    }

                    if let Branch::Ptr(ptr) = chunk[0].right_child
                        && !utilization_is_maximized
                    {
                        utilization_is_maximized = ptr == chunk[1].id;
                    }

                    assert!(utilization_is_maximized);
                }
            }

            // Add extra padding at the end to make the tree length even
            let tail_padding = if placed_nodes.len().is_multiple_of(2) {
                0
            } else {
                1
            };

            // Add header + padding for full tree length
            let tree_len: u32 = (placed_nodes.len() + tail_padding + 2)
                .try_into()
                .context("Tree length should fit into u32")?;

            let mut optimized_tree = Vec::with_capacity(tree_len as usize);

            // Add header + padding at beginning
            let cache_metadata = CacheMetadata::new_empty();
            optimized_tree.extend([
                lumberjack_model::model::Node::from_header(
                    TreeHeader::new(tree_len, 2, cache_metadata)
                        .map_err(|e| eyre!("Cannot compile model: {e:?}"))?,
                ),
                PADDING,
            ]);

            for node in &placed_nodes {
                let left = match node.left_child {
                    Branch::Ptr(id) => position_of(&placed_nodes, id) + 2,
                    Branch::Prediction(p) => p,
                };

                let right = match node.right_child {
                    Branch::Ptr(id) => position_of(&placed_nodes, id) + 2,
                    Branch::Prediction(p) => p,
                };

                optimized_tree.push(lumberjack_model::model::Node::from_branch(
                    lumberjack_model::model::Branch::new(
                        node.split_with,
                        node.split_at,
                        left,
                        right,
                        node.left_child.is_prediction(),
                        node.right_child.is_prediction(),
                    ),
                ));
            }

            optimized_tree.extend(std::iter::repeat_n(PADDING, tail_padding));

            assert_eq!(optimized_tree.len(), tree_len as usize);

            forest_nodes.extend(optimized_tree);
        }

        split_cells(&mut forest_nodes, num_cells, trees.len())?;

        Ok(forest_nodes)
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

        println!("votes: {votes:?}");

        votes
    }
}

impl IntermediateRepresentation<f32> {
    /// Turn an `IntermediateRepresentation<f32>` into an
    /// `IntermediateRepresentation<bf16>` by truncating the splits
    pub fn quantize_splits(self) -> IntermediateRepresentation<bf16> {
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

        IntermediateRepresentation {
            trees,
            problem: self.problem,
        }
    }
}

impl<F: Feature + Display> Display for IntermediateRepresentation<F> {
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

#[derive(Debug, Clone, Copy)]
enum Branch {
    Ptr(usize),
    Prediction(u16),
}

impl Branch {
    fn is_prediction(&self) -> bool {
        matches!(self, Branch::Prediction(_))
    }
}

/// Transitive data structure keeping track of a node's parent
#[derive(Debug)]
struct LinkedNode {
    id: usize,
    split_with: u16,
    split_at: bf16,
    left_child: Branch,
    right_child: Branch,
    _parent_id: Option<usize>,
}

impl Display for LinkedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ID {}\t| split_with: {}, split_at: {}, left: {:?}, right: {:?}",
            self.id, self.split_with, self.split_at, self.left_child, self.right_child
        )
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

/// Extract the node from the queue if it's a leaf, otherwise leave it in.
fn extract_branch_if_leaf<F: Feature>(
    unplaced_nodes: &mut VecDeque<(usize, &Node<F>)>,
    id: usize,
) -> Branch {
    let node = unplaced_nodes
        .iter()
        .find(|(i, _)| *i == id)
        .map(|(_, n)| *n)
        .unwrap_or_else(|| panic!("Node ID {id} could not be found"));

    match node {
        Node::Leaf(l) => {
            let pos = unplaced_nodes.iter().position(|(i, _)| *i == id).unwrap();
            unplaced_nodes.remove(pos).unwrap();
            Branch::Prediction(l.prediction)
        }
        Node::Branch(_) => Branch::Ptr(id),
    }
}

fn extract_branch<'a, F: Feature>(
    unplaced_nodes: &mut VecDeque<(usize, &'a Node<F>)>,
    id: usize,
) -> (usize, &'a Node<F>) {
    let pos = unplaced_nodes.iter().position(|(i, _)| *i == id).unwrap();
    unplaced_nodes.remove(pos).unwrap()
}

fn position_of(nodes: &[LinkedNode], id: usize) -> u16 {
    nodes
        .iter()
        .position(|n| n.id == id)
        .unwrap_or_else(|| panic!("Could not find node ID {id}"))
        .try_into()
        .expect("Node index does not fit into u16")
}

/// Split the model to distribute trees as evenly as possible between the
/// provided number of cells
pub fn split_cells(
    nodes: &mut [lumberjack_model::model::Node],
    num_cells: u8,
    total_trees: usize,
) -> Result<()> {
    // Avoid division by zero: first cache holds all the trees
    if num_cells <= 1 {
        nodes[0].as_header_mut().set_cache_metadata(
            CacheMetadata::new_cell_header(
                NonZeroU16::new(
                    total_trees
                        .try_into()
                        .context("num_trees must fit inside a u16")?,
                )
                .context("Number of caches must not be zero")?,
            )
            .map_err(|e| eyre!("Invalid cache metadata: {e:?}"))?,
        );
        return Ok(());
    }

    let base: u16 = (total_trees / num_cells as usize)
        .try_into()
        .context("num_trees must fit inside a u16")?;
    let extra = total_trees % num_cells as usize;

    let header_indices = iter_trees(nodes).collect::<Vec<_>>();

    let mut tree_pos = 0;
    for cell_idx in 0..num_cells {
        let num_trees = base + u16::from((cell_idx as usize) < extra);

        // Skip cells that would have no trees
        if tree_pos >= header_indices.len() {
            continue;
        }

        let header_idx = header_indices[tree_pos];
        let metadata = CacheMetadata::new_cell_header(
            NonZeroU16::new(num_trees).context("Number of caches must not be zero")?,
        )
        .unwrap();
        nodes[header_idx]
            .as_header_mut()
            .set_cache_metadata(metadata.clone());

        tree_pos += num_trees as usize
    }

    Ok(())
}
