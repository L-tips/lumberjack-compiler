use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

use color_eyre::Result;
use embedded_rforest::ptr::NodePointer;

use crate::{
    problem_type::{Classification, Map, ProblemType, Regression},
    serialized_forest::{SerializedForest, SerializedNode},
};

#[derive(Debug, Clone)]
pub struct BranchNode {
    pub(super) split_with: u32,
    pub(super) split_at: f32,
    pub(super) left: u32,
    pub(super) right: u32,
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

#[derive(Debug, PartialEq, Clone)]
pub struct LeafNode<P: ProblemType> {
    pub(super) prediction: P::Output,
}

#[derive(Debug, Clone)]
pub enum Node<P: ProblemType> {
    Leaf(LeafNode<P>),
    Branch(BranchNode),
}

impl<P: ProblemType> Node<P> {
    pub fn is_branch(&self) -> bool {
        matches!(self, Self::Branch(_))
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    pub fn take_leaf(&self) -> Option<&LeafNode<P>> {
        match self {
            Node::Leaf(l) => Some(l),
            _ => None,
        }
    }

    /// Calculate by how much we need to offset a branch's left and right
    /// pointers, given that the trees are getting disjoined from their root,
    /// which is stored at the front of the forest.
    pub fn offset(self, tree_sizes: &[usize], tree_index: usize) -> Self {
        // The offset is the sum of the size of all preceding trees, up to the current
        // one, plus the total number of trees in the forest (to make space for all root
        // nodes to be in front)
        let offset =
            tree_sizes[..tree_index].iter().sum::<usize>() + tree_sizes.len() - (tree_index + 1);
        let offset: u32 = offset.try_into().expect("Offset overflow");

        if let Node::Branch(mut branch) = self {
            branch.left += offset;
            branch.right += offset;
            Node::Branch(branch)
        } else {
            self
        }
    }
}

impl<P: ProblemType> fmt::Display for Node<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Leaf(leaf) => write!(f, "Leaf   | prediction: {}", leaf.prediction),
            Node::Branch(b) => write!(f, "{b}"),
        }
    }
}

#[derive(Debug)]
struct Tree<P: ProblemType> {
    nodes: Vec<Node<P>>,
}

impl<P: ProblemType> Tree<P> {
    pub fn new(nodes: Vec<Node<P>>) -> Self {
        Self { nodes }
    }
}

/// An array-backed, non-optimized random forest model
#[derive(Debug)]
pub struct Forest<P: ProblemType> {
    num_trees: usize,
    nodes: Vec<Node<P>>,
    problem: P,
}

impl<P> Forest<P>
where
    P: ProblemType,
{
    /// Convert a [`SerializedForest`] into a [`Forest`].
    ///
    /// In practice, this method flattens the nodes, putting all tree roots in
    /// front of the array.
    pub fn from_serialized<N: SerializedNode<ProblemType = P>>(
        serialized: SerializedForest<N>,
    ) -> Result<Self> {
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
                nodes.sort_by(|(a, _), (b, _)| a.cmp(b));
                nodes
                    .into_iter()
                    .map(|(_, n)| n)
                    .collect::<Result<Vec<_>, _>>()?
            };

            trees.push(Tree::new(tree_nodes));
        }

        // Collect the size of each tree in a vector
        let tree_sizes = trees.iter().map(|t| t.nodes.len()).collect::<Vec<_>>();

        // forest_nodes will store the flattened collection of all nodes in this forest
        let mut forest_nodes = Vec::with_capacity(tree_sizes.iter().sum());

        // Combine all trees into a flat forest structure
        // Start by adding the root of each tree to the beginning of the array
        for (i, tree) in trees.iter().enumerate() {
            let node = tree.nodes[0].clone().offset(&tree_sizes, i);
            forest_nodes.push(node);
        }

        // Then add the rest of the nodes
        for (i, tree) in trees.into_iter().enumerate() {
            // Skipping the root node, as it is already inserted at the start of the forest
            for node in tree.nodes.into_iter().skip(1) {
                forest_nodes.push(node.offset(&tree_sizes, i));
            }
        }

        for (i, node) in forest_nodes.iter().enumerate() {
            // Verify that our forest size fits in an u32
            let i: u32 = i.try_into().expect("Index overflow");

            // Ensure that every node only ever branches to another node further down the
            // vec
            if let Node::Branch(b) = node {
                assert!(b.left > i && b.right > i);
            }
        }

        Ok(Self {
            num_trees: tree_sizes.len(),
            nodes: forest_nodes,
            problem: serialized.problem().clone(),
        })
    }

    /// Turn this [`Forest`] into an [`OptimizedForest`].
    #[expect(private_bounds)]
    pub fn optimize_nodes(&self) -> Vec<embedded_rforest::forest::Branch>
    where
        P: UpdatePointers,
    {
        // Start by collecing branch indices, incrementing the branch index only if the
        // node is a branch.
        let mut branch_idx = 0;
        let nodes = self
            .nodes
            .clone()
            .into_iter()
            .map(|n| {
                if n.is_branch() {
                    branch_idx += 1;
                }
                RefCell::new(TransitionBranch::from_node(&self.nodes, n, branch_idx - 1))
            })
            .collect::<Vec<_>>();

        // Descend the tree, replacing each decision with an optimized node pointer.
        let nodes = nodes
            .iter()
            .map(|n| P::update_pointers(&nodes, n))
            .filter_map(|mut n| n.take())
            .collect::<Vec<_>>();

        // Sanity check: since each tree can be represented as a DAG, we check that no
        // node points backwards in the tree, leading to potential circular structures.
        for (i, n) in nodes.iter().enumerate() {
            if !n.flags().left_prediction() {
                assert!(n.left_ptr().as_ptr() as usize > i);
            }
            if !n.flags().right_prediction() {
                assert!(n.right_ptr().as_ptr() as usize > i);
            }
        }

        nodes
    }

    pub fn nodes(&self) -> &[Node<P>] {
        &self.nodes
    }

    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    pub fn num_features(&self) -> usize {
        self.problem.features().len()
    }

    pub fn features(&self) -> &Map {
        self.problem.features()
    }

    fn next_left(&self, branch: &BranchNode) -> &Node<P> {
        &self.nodes[branch.left as usize]
    }

    fn next_right(&self, branch: &BranchNode) -> &Node<P> {
        &self.nodes[branch.right as usize]
    }
}

struct TransitionBranch<P: ProblemType> {
    id: u32,
    split_with: u32,
    split_at: f32,
    left: TransitionNode<P>,
    right: TransitionNode<P>,
}

enum TransitionNode<P: ProblemType> {
    Leaf(P::Output),
    Branch(u32),
}

impl<P: ProblemType> TransitionBranch<P> {
    fn from_node(nodes: &[Node<P>], node: Node<P>, id: u32) -> Option<Self> {
        // Only transform branch nodes by looking ahead to find out if the next
        // left/right nodes contain a prediction
        let Node::Branch(branch) = node else {
            return None;
        };

        let left = match &nodes[branch.left as usize] {
            Node::Leaf(leaf) => TransitionNode::Leaf(leaf.prediction),
            Node::Branch(_) => TransitionNode::Branch(branch.left),
        };

        let right = match &nodes[branch.right as usize] {
            Node::Leaf(leaf) => TransitionNode::Leaf(leaf.prediction),
            Node::Branch(_) => TransitionNode::Branch(branch.right),
        };

        Some(TransitionBranch {
            id,
            split_with: branch.split_with,
            split_at: branch.split_at,
            left,
            right,
        })
    }
}

impl Forest<Classification> {
    pub fn num_targets(&self) -> usize {
        self.problem.targets().len()
    }

    pub fn targets(&self) -> &Map {
        self.problem.targets()
    }

    /// Make a prediction based on input values (features)
    pub fn predict(&self, features: &[f32]) -> String {
        // Reserve space to store each tree's prediction
        let mut results = Vec::with_capacity(self.num_trees);

        // Descend into each tree to make a prediction
        for tree_id in 0..self.num_trees {
            // The tree root is stored at the tree index
            let mut node = &self.nodes[tree_id];

            let prediction = loop {
                match node {
                    Node::Branch(b) => {
                        let test = features[b.split_with as usize] <= b.split_at;
                        if test {
                            node = self.next_left(b)
                        } else {
                            node = self.next_right(b)
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

impl Forest<Regression> {
    /// Make a prediction based on input values (features)
    pub fn predict(&self, features: &[f32]) -> f32 {
        // Reserve space to store each tree's prediction
        let mut result = 0.0;

        // Descend into each tree to make a prediction
        for tree_id in 0..self.num_trees {
            // The tree root is stored at the tree index
            let mut node = &self.nodes[tree_id];

            let prediction = loop {
                match node {
                    Node::Branch(b) => {
                        let test = features[b.split_with as usize] <= b.split_at;
                        if test {
                            node = self.next_left(b)
                        } else {
                            node = self.next_right(b)
                        }
                    }
                    Node::Leaf(l) => {
                        break l.prediction;
                    }
                }
            };

            result += prediction;
        }

        result / self.num_trees as f32
    }
}

impl fmt::Display for Forest<Classification> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Classification Forest: {} trees, size {}, {} features, {} targets\n------------",
            self.num_trees,
            self.nodes.len(),
            self.problem.features().len(),
            self.problem.targets().len(),
        )?;
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f, "\t{i}: {node}")?;
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

impl fmt::Display for Forest<Regression> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Regression Forest: {} trees, size {}, {} features\n------------",
            self.num_trees,
            self.nodes.len(),
            self.problem.features().len(),
        )?;
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f, "\t{i}: {node}")?;
        }
        writeln!(f, "------------")?;

        let mut features_ordered = self.problem.features().iter().collect::<Vec<_>>();
        features_ordered.sort_by(|a, b| a.1.cmp(b.1));

        writeln!(f, "Features: ")?;
        for feat in features_ordered.iter() {
            writeln!(f, "\t{}: {}", feat.1, feat.0)?;
        }

        writeln!(f, "------------")?;

        Ok(())
    }
}

trait UpdatePointers: ProblemType {
    fn update_pointers(
        nodes: &[RefCell<Option<TransitionBranch<Self>>>],
        branch: &RefCell<Option<TransitionBranch<Self>>>,
    ) -> Option<embedded_rforest::forest::Branch>;
}

impl UpdatePointers for Classification {
    fn update_pointers(
        nodes: &[RefCell<Option<TransitionBranch<Self>>>],
        branch: &RefCell<Option<TransitionBranch<Self>>>,
    ) -> Option<embedded_rforest::forest::Branch> {
        let branch = branch.borrow();
        let branch = branch.as_ref()?;

        let (left_pred, left_val) = match branch.left {
            TransitionNode::Leaf(l) => (true, l),
            TransitionNode::Branch(b) => {
                let next = nodes[b as usize].borrow().as_ref()?.id;
                (false, next)
            }
        };

        let (right_pred, right_val) = match branch.right {
            TransitionNode::Leaf(l) => (true, l),
            TransitionNode::Branch(b) => {
                let next = nodes[b as usize].borrow().as_ref()?.id;
                (false, next)
            }
        };

        Some(embedded_rforest::forest::Branch::new(
            branch.split_with,
            branch.split_at,
            NodePointer::new_ptr(left_val),
            NodePointer::new_ptr(right_val),
            left_pred,
            right_pred,
        ))
    }
}

impl UpdatePointers for Regression {
    fn update_pointers(
        nodes: &[RefCell<Option<TransitionBranch<Self>>>],
        branch: &RefCell<Option<TransitionBranch<Self>>>,
    ) -> Option<embedded_rforest::forest::Branch> {
        let branch = branch.borrow();
        let branch = branch.as_ref()?;

        let (left_pred, left_ptr) = match branch.left {
            TransitionNode::Leaf(l) => (true, NodePointer::new_f32(l)),
            TransitionNode::Branch(b) => {
                let next = nodes[b as usize].borrow().as_ref()?.id;
                (false, NodePointer::new_ptr(next))
            }
        };

        let (right_pred, right_ptr) = match branch.right {
            TransitionNode::Leaf(l) => (true, NodePointer::new_f32(l)),
            TransitionNode::Branch(b) => {
                let next = nodes[b as usize].borrow().as_ref()?.id;
                (false, NodePointer::new_ptr(next))
            }
        };

        Some(embedded_rforest::forest::Branch::new(
            branch.split_with,
            branch.split_at,
            left_ptr,
            right_ptr,
            left_pred,
            right_pred,
        ))
    }
}
