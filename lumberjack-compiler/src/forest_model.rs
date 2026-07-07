use std::collections::{HashMap, VecDeque};
use std::fmt;

use half::bf16;
use lumberjack_model::model::{PADDING, TreeHeader};
use tap::Tap;

use crate::problem::{Map, Problem};

#[derive(Debug, Clone)]
pub struct BranchNode {
    pub(super) split_with: u16,
    pub(super) split_at: bf16,
    pub(super) left: usize,
    pub(super) right: usize,
}

impl fmt::Display for BranchNode {
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
pub enum Node {
    Leaf(LeafNode),
    Branch(BranchNode),
}

impl Node {
    pub fn is_branch(&self) -> bool {
        matches!(self, Self::Branch(_))
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    pub fn as_branch(&self) -> Option<&BranchNode> {
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

impl fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Leaf(leaf) => write!(f, "Leaf   | prediction: {}", leaf.prediction),
            Node::Branch(b) => write!(f, "{b}"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }

    fn next_left(&self, branch: &BranchNode) -> &Node {
        &self.nodes[branch.left]
    }

    fn next_right(&self, branch: &BranchNode) -> &Node {
        &self.nodes[branch.right]
    }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, n) in self.nodes.iter().enumerate() {
            writeln!(f, "\tNode {i}:\t{n}")?;
        }

        Ok(())
    }
}

/// An array-backed, non-optimized random forest model
#[derive(Debug)]
pub struct ForestModel {
    trees: Vec<Tree>,
    problem: Problem,
}

impl ForestModel {
    pub(crate) fn new(trees: Vec<Tree>, problem: Problem) -> Self {
        Self { trees, problem }
    }

    /// Turn this [`ForestModel`] into a
    /// [`Model`](lumberjack_model::model::Model).
    pub fn compile(&self) -> Vec<lumberjack_model::model::Node> {
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
                        split_at: n.split_at,
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
                .expect("Tree length should fit into u32");

            let mut optimized_tree = Vec::with_capacity(tree_len as usize);

            // Add header + padding at beginning
            optimized_tree.extend([
                lumberjack_model::model::Node::from_header(TreeHeader::new(tree_len, 2)),
                PADDING,
            ]);

            for node in &placed_nodes {
                let left = match node.left_child {
                    Branch::Ptr(id) => id_position(&placed_nodes, id) + 2,
                    Branch::Prediction(p) => p,
                };

                let right = match node.right_child {
                    Branch::Ptr(id) => id_position(&placed_nodes, id) + 2,
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

        forest_nodes
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

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Make a prediction based on input feature vector
    pub fn predict(&self, features: &[bf16]) -> String {
        // Reserve space to store each tree's prediction
        let mut results = Vec::with_capacity(self.num_trees());

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

            results.push(prediction);
        }

        // Count the number of votes for each category
        let mut votes = HashMap::new();
        for &target in results.iter() {
            *votes.entry(target).or_insert(0) += 1;
        }

        let best_result = votes
            .iter()
            .max_by(|(idx_a, votes_a), (idx_b, votes_b)| {
                votes_a.cmp(votes_b).then_with(|| idx_b.cmp(idx_a))
            })
            .map(|(num, _)| num)
            .unwrap();

        println!("votes: {votes:?}");

        self.targets()
            .iter()
            .find(|(_, t)| *t == best_result)
            .unwrap()
            .0
            .clone()
    }
}

impl fmt::Display for ForestModel {
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

impl fmt::Display for LinkedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ID {}\t| split_with: {}, split_at: {}, left: {:?}, right: {:?}",
            self.id, self.split_with, self.split_at, self.left_child, self.right_child
        )
    }
}

/// Extract the node from the queue if it's a leaf, otherwise leave it in.
fn extract_branch_if_leaf(unplaced_nodes: &mut VecDeque<(usize, &Node)>, id: usize) -> Branch {
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

fn extract_branch<'a>(
    unplaced_nodes: &mut VecDeque<(usize, &'a Node)>,
    id: usize,
) -> (usize, &'a Node) {
    let pos = unplaced_nodes.iter().position(|(i, _)| *i == id).unwrap();
    unplaced_nodes.remove(pos).unwrap()
}

fn id_position(nodes: &[LinkedNode], id: usize) -> u16 {
    nodes
        .iter()
        .position(|n| n.id == id)
        .unwrap_or_else(|| panic!("Could not find node ID {id}"))
        .try_into()
        .expect("Node index does not fit into u16")
}
