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
        #[derive(Debug)]
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
                parent: usize,
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
                        parent,
                        left_child,
                        right_child,
                    } => f
                        .debug_struct("LinkedNode Branch")
                        .field("id", id)
                        .field("parent", parent)
                        .field("left_child", left_child)
                        .field("right_child", right_child)
                        .finish(),
                }
            }
        }

        // Sort trees by length so that the longest ones end up at the end. If HW is
        // using multiple tree cells, the longer trees have a higher likelihood to end
        // up in a cell that will have fewer trees than its counterparts, improving the
        // average case prediction time.
        let trees = self
            .trees
            .clone()
            .tap_mut(|v| v.sort_by_key(|t| t.nodes.len()));

        for (tree_id, tree) in trees.iter().enumerate() {
            // The vec containing the nodes which have been assigned
            let mut placed_nodes = Vec::with_capacity(tree.nodes.len());

            // Nodes that haven't been placed yet
            let mut unplaced_nodes: HashMap<_, _> = tree.nodes.iter().enumerate().collect();

            // Each tree starts with a header. We don't know the tree's length yet, so leave
            // at 0 for now. We also always leave a node's worth of padding so
            // that the first "real" node ends up on a 128-byte boundary. This is to
            // optimize the model if the hardware is using a superscalar architecture.
            placed_nodes.extend([LinkedNode::Header { child_id: 0 }, LinkedNode::Padding]);

            while let Some((id, extracted_node)) = unplaced_nodes.pop_front() {
                if let Some(n) = extracted_node.as_branch() {
                    // If node is a branch, go hunting for their left and right
                    // branches
                    let Some(left) = unplaced_nodes.get(n.left) else {
                        panic!("Node ID _ could not be found in tree {tree_id}");
                    };

                    let Some(right) = unplaced_nodes.get(n.right) else {
                        panic!("Node ID _ could not be found in tree {tree_id}");
                    };

                    // TODO: shouldn't be pop_front_if, rather some extraction from the middle.
                    // Maybe using a HashMap instead?
                    let (left_child, right_child) = match (left, right) {
                        (Node::Branch(_), Node::Branch(_)) => {
                            (Branch::Ptr(n.left), Branch::Ptr(n.right))
                        }
                        (Node::Branch(_), Node::Leaf(_)) => {
                            let right_leaf = unplaced_nodes.remove(n.right).unwrap();
                            let right_leaf = right_leaf.as_leaf().unwrap();
                            (
                                Branch::Ptr(n.left),
                                Branch::Prediction(right_leaf.prediction),
                            )
                        }
                        (Node::Leaf(_), Node::Branch(_)) => {
                            let left_leaf = unplaced_nodes.remove(n.left).unwrap();
                            let left_leaf = left_leaf.as_leaf().unwrap();
                            (
                                Branch::Prediction(left_leaf.prediction),
                                Branch::Ptr(n.right),
                            )
                        }
                        (Node::Leaf(_), Node::Leaf(_)) => {
                            let left_leaf = unplaced_nodes.remove(n.left).unwrap();
                            let left_leaf = left_leaf.as_leaf().unwrap();
                            let right_leaf = unplaced_nodes.remove(n.right).unwrap();
                            let right_leaf = right_leaf.as_leaf().unwrap();

                            (
                                Branch::Prediction(left_leaf.prediction),
                                Branch::Prediction(right_leaf.prediction),
                            )
                        }
                    };

                    let node_to_place = LinkedNode::Branch {
                        id,
                        parent: todo!(),
                        left_child,
                        right_child,
                    };

                    // TODO: place node
                    // here
                } else if let Some(n) = extracted_node.as_leaf() {
                    panic!("Leaf node should have been extracted already");
                } else {
                    unreachable!()
                }
            }
        }

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
