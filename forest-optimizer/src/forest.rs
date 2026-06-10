use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::{cell::RefCell, mem::ManuallyDrop};

use color_eyre::Result;
use embedded_rforest::{forest::TreeHeader, ptr::NodePointer};
use half::bf16;
use tap::Tap;

use crate::{
    csv_forest::CsvForest,
    problem_type::{Classification, Map},
};

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

    // /// Calculate by how much we need to offset a branch's left and right
    // /// pointers, given that the trees are getting disjoined from their root,
    // /// which is stored at the front of the forest.
    // pub fn offset(self, tree_sizes: &[usize], tree_index: usize) -> Self {
    //     // The offset is the sum of the size of all preceding trees, up to the
    // current     // one, plus the total number of trees in the forest (to make
    // space for all root     // nodes to be in front)
    //     let offset =
    //         tree_sizes[..tree_index].iter().sum::<usize>() + tree_sizes.len() -
    // (tree_index + 1);

    //     if let Node::Branch(mut branch) = self {
    //         branch.left += offset;
    //         branch.right += offset;
    //         Node::Branch(branch)
    //     } else {
    //         self
    //     }
    // }
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
struct Tree {
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
pub struct Forest {
    trees: Vec<Tree>,
    problem: Classification,
}

impl Forest {
    /// Convert a [`SerializedForest`] into a [`Forest`].
    ///
    /// In practice, this method flattens the nodes, putting all tree roots in
    /// front of the array.
    pub fn from_serialized(serialized: CsvForest) -> Result<Self> {
        let problem = serialized.problem();

        // Find all nodes which have an index of 1. These are our tree roots.
        let mut tree_roots: Vec<_> = serialized
            .nodes()
            .iter()
            .filter_map(|n| {
                if n.node_idx() == 1 {
                    Some(n.tree_idx())
                } else {
                    None
                }
            })
            .collect();
        tree_roots.sort();

        // Check that all tree roots are numbered sequentially
        assert!(
            tree_roots.iter().enumerate().all(|(i, &v)| v == i + 1),
            "Mismatch within tree indices"
        );

        // Create an array with enough space for all our trees
        let mut trees = Vec::with_capacity(tree_roots.len());

        // Descend into each tree and create the array structure
        for i in 0..tree_roots.len() {
            let tree_idx = i + 1;

            // Collect just the nodes belonging to this tree, and place them in order
            let tree_nodes = {
                let mut nodes = serialized
                    .nodes()
                    .iter()
                    .filter_map(|n| {
                        if n.tree_idx() == tree_idx {
                            Some((n.node_idx(), n.clone().normalize(problem)))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                nodes.sort_by_key(|(a, _)| *a);
                nodes
                    .into_iter()
                    .map(|(_, n)| n)
                    .collect::<Result<Vec<_>, _>>()?
            };

            trees.push(Tree::new(tree_nodes));
        }

        Ok(Self {
            trees,
            problem: serialized.problem().clone(),
        })
    }

    /// Turn this [`Forest`] into an
    /// [`OptimizedForest`](embedded_rforest::forest::OptimizedForest).
    pub fn optimize_nodes(&self) -> Vec<embedded_rforest::forest::Node> {
        // Sort trees by length so that the longest ones end up at the end. If HW is
        // using multiple tree cells, the longer trees have a higher likelihood to end
        // up in a cell that will have fewer trees than its counterparts, improving the
        // average case prediction time.
        let trees = self
            .trees
            .clone()
            .tap_mut(|v| v.sort_by_key(|t| t.nodes.len()));

        for (tree_id, tree) in trees.iter().enumerate() {
            // The vec containing the nodes which have already been assigned
            let mut placed_nodes = Vec::with_capacity(tree.nodes.len());

            // Nodes that haven't been placed yet
            let mut waiting_nodes: VecDeque<_> = tree.nodes.iter().enumerate().collect();

            // Keep a list of nodes that will be placed later for better optimization
            let mut deferred_nodes: VecDeque<LinkedNode> = VecDeque::new();

            // Each tree starts with a header. We don't know the tree's length yet, so leave
            // at 0 for now. We also always leave a node's worth of padding so
            // that the first "real" node ends up on a 128-byte boundary. This is to
            // optimize the model if the hardware is using a superscalar architecture.
            placed_nodes.extend([LinkedNode::Header { child_id: 0 }, LinkedNode::Padding]);

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

                    let node_to_place = LinkedNode::Branch {
                        id,
                        parent_id,
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
                } else if let Some(n) = extracted_node.as_leaf() {
                    panic!("Leaf node should have been extracted already");
                } else {
                    unreachable!()
                }
            }

            // TODO: place padding where convenient
            // TODO: place deferred nodes
            // TODO: translate node IDs into indices and convert to
            // embedded_rforest::forest::Node

            println!("Tree {tree_id}:");
            println!("{tree}");
            println!("Optimized:");
            for (idx, n) in placed_nodes.iter().enumerate() {
                println!("({idx}) {n:?}");
            }
        }

        // TODO: edit tree header with tree length

        todo!()
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn num_features(&self) -> usize {
        self.problem.features().len()
    }
}

impl Forest {
    pub fn num_targets(&self) -> usize {
        self.problem.targets().len()
    }

    pub fn targets(&self) -> &Map {
        self.problem.targets()
    }

    pub fn features(&self) -> &Map {
        self.problem.features()
    }

    /// Make a prediction based on input values (features)
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
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(num, _)| num)
            .unwrap();

        self.targets()
            .iter()
            .find(|(_, t)| **t == best_result)
            .unwrap()
            .0
            .clone()
    }
}

impl fmt::Display for Forest {
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
            println!("Tree {i}:");
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

/// Transitive data structure keeping track of a node's parent
enum LinkedNode {
    Header {
        child_id: usize,
    },
    Padding,
    Branch {
        // node: Node
        id: usize,
        split_with: u16,
        split_at: bf16,
        parent_id: Option<usize>,
        left_child: Branch,
        right_child: Branch,
    },
}

impl std::fmt::Debug for LinkedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkedNode::Header { child_id: child } => write!(f, "Header, child: {child}"),
            LinkedNode::Padding => write!(f, "Padding"),
            LinkedNode::Branch {
                id,
                split_with,
                split_at,
                parent_id: parent,
                left_child,
                right_child,
            } => f
                .debug_struct("LinkedNode Branch")
                .field("id", id)
                .field("parent", parent)
                .field("left_child", left_child)
                .field("right_child", right_child)
                .field("split_with", split_with)
                .field("split_at", split_at)
                .finish(),
        }
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
