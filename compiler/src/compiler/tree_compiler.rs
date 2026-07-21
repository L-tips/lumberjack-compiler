use std::{
    collections::{HashSet, VecDeque},
    fmt,
};

use color_eyre::eyre::{Context, eyre};
use fastrand::Rng;
use half::bf16;
use lumberjack_model::model::{self, CacheMetadata, PADDING, TreeHeader};

use crate::{
    Feature,
    compiler::{Node, Tree},
};

/// Node placement strategy for tree compilation
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    clap::ValueEnum,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum PlacementStrategy {
    #[default]
    ExecutionAware,
    Random,
    BreadthFirst,
    DepthFirst,
}

impl PlacementStrategy {
    fn placement_fn<F: Feature>(&self) -> impl Fn(&Tree<F>) -> Vec<LinkedNode> {
        match self {
            Self::ExecutionAware => execution_aware_placement,
            Self::Random => random_placement,
            Self::BreadthFirst => breadth_first_placement,
            Self::DepthFirst => depth_first_placement,
        }
    }

    fn place_nodes<F: Feature>(&self, tree: &Tree<F>) -> Vec<LinkedNode> {
        self.placement_fn()(tree)
    }
}

impl fmt::Display for PlacementStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::ExecutionAware => "execution-aware",
            Self::Random => "random",
            Self::BreadthFirst => "breadth-first",
            Self::DepthFirst => "depth-first",
        };
        f.write_str(s)
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

#[derive(Debug)]
enum LinkedNode {
    Branch(LinkedBranchNode),
    Padding,
}

impl fmt::Display for LinkedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkedNode::Branch(b) => write!(f, "{b}"),
            LinkedNode::Padding => write!(f, "[PADDING]"),
        }
    }
}

#[derive(Debug)]
struct LinkedBranchNode {
    id: usize,
    split_with: u16,
    split_at: bf16,
    left_child: Branch,
    right_child: Branch,
    _parent_id: Option<usize>,
}

impl fmt::Display for LinkedBranchNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ID {}\t| split_with: {}, split_at: {}, left: {:?}, right: {:?}",
            self.id, self.split_with, self.split_at, self.left_child, self.right_child
        )
    }
}

fn position_of(nodes: &[LinkedNode], id: usize) -> u16 {
    nodes
        .iter()
        .position(|n| matches!(n, LinkedNode::Branch(b) if b.id == id))
        .unwrap_or_else(|| panic!("Could not find node ID {id}"))
        .try_into()
        .expect("Node index does not fit into u16")
}

/// Compile a single [`Tree`] using the execution-aware node placement algorithm
pub(super) fn compile<F: Feature>(
    tree: &Tree<F>,
    strategy: PlacementStrategy,
) -> color_eyre::Result<Vec<model::Node>> {
    let placed_nodes = strategy.place_nodes(tree);
    build_compiled_nodes(&placed_nodes)
}

// fn execution_aware_placement_v2<F: Feature>(tree: &Tree<F>) ->
// Vec<LinkedNode> {     let mut placed_nodes =
// Vec::with_capacity(tree.nodes.len());     let mut nodes_to_place: VecDeque<_>
// = tree.nodes.iter().enumerate().collect();     let mut deferred_nodes:
// Vec<LinkedNode> = Vec::new();

//     let mut parent_id = None;
//     let (mut id, mut extracted_node) = nodes_to_place.pop_front().unwrap();

//     loop {
//         if let Some(n) = extracted_node.as_branch() {
//             let left_child = extract_branch_if_leaf(&mut nodes_to_place,
// n.left);             let right_child = extract_branch_if_leaf(&mut
// nodes_to_place, n.right);

//             let node_to_place = LinkedNode {
//                 id,
//                 _parent_id: parent_id,
//                 left_child,
//                 right_child,
//                 split_at: n.split_at.clone().into_bf16(),
//                 split_with: n.split_with,
//             };

//             // Double-prediction nodes have no pointer children, so they
// cannot             // enable superscalar execution for the node paired with
// them.             // Defer them to avoid occupying an even slot that a node
// with             // pointers could use.
//             if matches!(left_child, Branch::Prediction(_))
//                 && matches!(right_child, Branch::Prediction(_))
//                 && placed_nodes.len().is_multiple_of(2)
//             {
//                 deferred_nodes.push(node_to_place);
//             } else {
//                 placed_nodes.push(node_to_place);
//             }

//             parent_id = Some(id);

//             // We just placed a node at an even index. Fill the odd slot with
//             // whichever pointer child will yield the best superscalar
// pairing             // on the *next* even slot: prefer children that are
// themselves             // double-predictions (freeing the next even slot for
// a node with             // pointers), then one-prediction, then two-pointer
// nodes.             if !placed_nodes.len().is_multiple_of(2)
//                 && let Some(next_id) = pick_next_child(tree, left_child,
// right_child)             {
//                 (id, extracted_node) = extract_branch(&mut nodes_to_place,
// next_id);                 continue;
//             }

//             let Some((new_id, new_node)) = nodes_to_place.pop_front() else {
//                 for d in deferred_nodes {
//                     placed_nodes.push(d);
//                 }
//                 assert_eq!(nodes_to_place.len(), 0);
//                 assert_utilization(&placed_nodes);
//                 return placed_nodes;
//             };

//             id = new_id;
//             extracted_node = new_node;
//         } else {
//             unreachable!("Leaf node should have been extracted already");
//         }
//     }

//     for d in deferred_nodes {
//         placed_nodes.push(d);
//     }

//     assert_eq!(nodes_to_place.len(), 0);
//     assert_utilization(&placed_nodes);
//     placed_nodes
// }

// fn execution_aware_placement_v3<F: Feature>(tree: &Tree<F>) ->
// Vec<LinkedNode> {     let mut placed_nodes =
// Vec::with_capacity(tree.nodes.len());     let mut nodes_to_place: VecDeque<_>
// = tree.nodes.iter().enumerate().collect();     let mut deferred_nodes:
// Vec<LinkedNode> = Vec::new();

//     // Nodes that are children of already-placed nodes, waiting to be placed.
//     // Stored as (prediction_count, id) so we can prioritize.
//     let mut pending: Vec<(u8, usize)> = Vec::new();

//     let mut parent_id = None;
//     let (mut id, mut extracted_node) = nodes_to_place.pop_front().unwrap();

//     loop {
//         if let Some(n) = extracted_node.as_branch() {
//             let left_child = extract_branch_if_leaf(&mut nodes_to_place,
// n.left);             let right_child = extract_branch_if_leaf(&mut
// nodes_to_place, n.right);

//             let node_to_place = LinkedNode {
//                 id,
//                 _parent_id: parent_id,
//                 left_child,
//                 right_child,
//                 split_at: n.split_at.clone().into_bf16(),
//                 split_with: n.split_with,
//             };

//             let is_double_prediction = matches!(left_child,
// Branch::Prediction(_))                 && matches!(right_child,
// Branch::Prediction(_));

//             if is_double_prediction && placed_nodes.len().is_multiple_of(2) {
//                 deferred_nodes.push(node_to_place);
//             } else {
//                 placed_nodes.push(node_to_place);

//                 // Enqueue pointer children into pending, tagged with their
//                 // prediction count so we can make priority decisions later.
//                 for child in [left_child, right_child] {
//                     if let Branch::Ptr(child_id) = child {
//                         let branch =
// tree.nodes[child_id].as_branch().unwrap();                         let l =
// match &tree.nodes[branch.left] {                             Node::Leaf(_) =>
// Branch::Prediction(0),                             Node::Branch(_) =>
// Branch::Ptr(branch.left),                         };
//                         let r = match &tree.nodes[branch.right] {
//                             Node::Leaf(_) => Branch::Prediction(0),
//                             Node::Branch(_) => Branch::Ptr(branch.right),
//                         };
//                         let preds = count_predictions(l, r);
//                         pending.push((preds, child_id));
//                     }
//                 }
//             }

//             parent_id = Some(id);

//             // Just placed at an even index — fill the odd slot with
// whichever             // pending child maximizes the *next* even slot's
// options.             // Priority: 2 predictions > 1 > 0 (same as
// pick_next_child).             if placed_nodes.len() % 2 == 1 {
//                 if let Some(next_id) = pick_next_child(tree, left_child,
// right_child) {                     // Remove from pending since we're placing
// it now                     pending.retain(|&(_, pid)| pid != next_id);
//                     (id, extracted_node) = extract_branch(&mut
// nodes_to_place, next_id);                     continue;
//                 }
//             }

//             // Odd slot filled, or current node had no pointer children.
//             // Next even slot: prefer a pending node (child of something
// already             // placed) over an arbitrary unvisited node, to keep
// related nodes             // close. Among pending, prefer nodes with MORE
// pointer children so             // that double-predictions don't burn even
// slots.             let next = if let Some(best) = pending
//                 .iter()
//                 .enumerate()
//                 .filter(|(_, (preds, _))| *preds < 2) // prefer nodes that
// can still pair                 .max_by_key(|(_, (preds, _))| *preds)
//                 .map(|(i, _)| i)
//             {
//                 let (_, next_id) = pending.remove(best);
//                 extract_branch(&mut nodes_to_place, next_id)
//             } else {
//                 // No useful pending nodes; fall back to queue order,
//                 // skipping any already extracted as children.
//                 let Some(next) = nodes_to_place.pop_front() else {
//                     break;
//                 };
//                 next
//             };

//             id = next.0;
//             extracted_node = next.1;
//         } else {
//             unreachable!("Leaf node should have been extracted already");
//         }
//     }

//     for d in deferred_nodes {
//         placed_nodes.push(d);
//     }

//     assert_eq!(nodes_to_place.len(), 0);
//     assert_utilization(&placed_nodes);
//     placed_nodes
// }

fn make_linked_node<F: Feature>(tree: &Tree<F>, id: usize, parent_id: Option<usize>) -> LinkedNode {
    let branch = tree.nodes[id].as_branch().unwrap();
    let left_child = match &tree.nodes[branch.left] {
        Node::Leaf(l) => Branch::Prediction(l.prediction),
        Node::Branch(_) => Branch::Ptr(branch.left),
    };
    let right_child = match &tree.nodes[branch.right] {
        Node::Leaf(l) => Branch::Prediction(l.prediction),
        Node::Branch(_) => Branch::Ptr(branch.right),
    };
    LinkedNode::Branch(LinkedBranchNode {
        id,
        _parent_id: parent_id,
        left_child,
        right_child,
        split_at: branch.split_at.clone().into_bf16(),
        split_with: branch.split_with,
    })
}

fn prediction_count<F: Feature>(tree: &Tree<F>, id: usize) -> u8 {
    let branch = tree.nodes[id].as_branch().unwrap();
    u8::from(tree.nodes[branch.left].is_leaf()) + u8::from(tree.nodes[branch.right].is_leaf())
}

fn execution_aware_placement<F: Feature>(tree: &Tree<F>) -> Vec<LinkedNode> {
    // Special case, root is always paired with the header
    let root_id = 0;

    let mut double_prediction: Vec<usize> = Vec::new();
    let mut single_leaf: Vec<usize> = Vec::new();
    let mut double_ptr: Vec<usize> = Vec::new();

    for (id, node) in tree.nodes.iter().enumerate() {
        if id == root_id {
            continue;
        }
        let Some(branch) = node.as_branch() else {
            continue;
        };
        let left_is_leaf = tree.nodes[branch.left].is_leaf();
        let right_is_leaf = tree.nodes[branch.right].is_leaf();
        match (left_is_leaf, right_is_leaf) {
            (true, true) => double_prediction.push(id),
            (true, false) | (false, true) => single_leaf.push(id),
            (false, false) => double_ptr.push(id),
        }
    }

    let mut placed: HashSet<usize> = HashSet::new();
    placed.insert(root_id);
    let mut placed_nodes: Vec<LinkedNode> = Vec::new();
    placed_nodes.push(make_linked_node(tree, root_id, None));

    // Pass 1: single-leaf nodes paired with their branch child
    for &id in &single_leaf {
        if placed.contains(&id) {
            continue;
        }
        let node = make_linked_node(tree, id, None);
        let branch_child = match node {
            LinkedNode::Branch(ref b) => match (b.left_child, b.right_child) {
                (Branch::Ptr(l), _) => l,
                (_, Branch::Ptr(r)) => r,
                _ => unreachable!(),
            },
            LinkedNode::Padding => unreachable!(),
        };
        placed.insert(id);
        placed_nodes.push(node);
        placed.insert(branch_child);
        placed_nodes.push(make_linked_node(tree, branch_child, Some(id)));
    }

    // Pass 2: double-ptr nodes paired with best unplaced child (2leaf > 1leaf >
    // 2ptr)
    for &id in &double_ptr {
        if placed.contains(&id) {
            continue;
        }
        let node = make_linked_node(tree, id, None);
        let (l, r) = match node {
            LinkedNode::Branch(ref b) => match (b.left_child, b.right_child) {
                (Branch::Ptr(l), Branch::Ptr(r)) => (l, r),
                _ => unreachable!(),
            },
            LinkedNode::Padding => unreachable!(),
        };

        placed.insert(id);
        placed_nodes.push(node);

        let best_child = match (placed.contains(&l), placed.contains(&r)) {
            (false, false) => {
                // Pick by priority: higher prediction_count wins
                if prediction_count(tree, l) >= prediction_count(tree, r) {
                    l
                } else {
                    r
                }
            }
            (false, true) => l,
            (true, false) => r,
            (true, true) => {
                // Both already placed; pad the odd slot
                placed_nodes.push(LinkedNode::Padding);
                continue;
            }
        };

        placed.insert(best_child);
        placed_nodes.push(make_linked_node(tree, best_child, Some(id)));
    }

    // Pass 3: double-prediction nodes
    for &id in &double_prediction {
        if placed.contains(&id) {
            continue;
        }
        placed_nodes.push(make_linked_node(tree, id, None));
    }

    placed_nodes
}

/// Place nodes in a random, but reproducible manner.
fn random_placement<F: Feature>(tree: &Tree<F>) -> Vec<LinkedNode> {
    const SEED: u64 = 0xAEF3210F_2311DAFF;

    let mut rng = Rng::with_seed(SEED);

    // Collect only branch nodes (leaves are inlined as predictions)
    let mut branch_indices: Vec<usize> = tree
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.is_branch())
        .map(|(i, _)| i)
        .collect();

    rng.shuffle(&mut branch_indices);

    // Ensure root is the first node in the
    if let Some(root_pos) = branch_indices.iter().position(|&i| i == 0) {
        branch_indices.swap(0, root_pos);
    }

    branch_indices
        .iter()
        .map(|&id| {
            let branch = tree.nodes[id]
                .as_branch()
                .unwrap_or_else(|| panic!("Node at index {id} is a branch"));

            let left_child = match &tree.nodes[branch.left] {
                Node::Leaf(l) => Branch::Prediction(l.prediction),
                Node::Branch(_) => Branch::Ptr(branch.left),
            };

            let right_child = match &tree.nodes[branch.right] {
                Node::Leaf(l) => Branch::Prediction(l.prediction),
                Node::Branch(_) => Branch::Ptr(branch.right),
            };

            LinkedNode::Branch(LinkedBranchNode {
                id,
                _parent_id: None,
                left_child,
                right_child,
                split_at: branch.split_at.clone().into_bf16(),
                split_with: branch.split_with,
            })
        })
        .collect()
}

fn breadth_first_placement<F: Feature>(tree: &Tree<F>) -> Vec<LinkedNode> {
    let mut placed_nodes = Vec::with_capacity(tree.nodes.len());
    let mut queue = VecDeque::new();

    // Find root (index 0) if it's a branch
    if tree.nodes[0].is_branch() {
        queue.push_back(0usize);
    } else {
        panic!("Tree root is not a branch node");
    }

    while let Some(id) = queue.pop_front() {
        let branch = tree.nodes[id].as_branch().unwrap();

        let left_child = match &tree.nodes[branch.left] {
            Node::Leaf(l) => Branch::Prediction(l.prediction),
            Node::Branch(_) => {
                queue.push_back(branch.left);
                Branch::Ptr(branch.left)
            }
        };

        let right_child = match &tree.nodes[branch.right] {
            Node::Leaf(l) => Branch::Prediction(l.prediction),
            Node::Branch(_) => {
                queue.push_back(branch.right);
                Branch::Ptr(branch.right)
            }
        };

        placed_nodes.push(LinkedNode::Branch(LinkedBranchNode {
            id,
            _parent_id: None,
            left_child,
            right_child,
            split_at: branch.split_at.clone().into_bf16(),
            split_with: branch.split_with,
        }));
    }

    placed_nodes
}

fn depth_first_placement<F: Feature>(tree: &Tree<F>) -> Vec<LinkedNode> {
    let mut placed_nodes = Vec::with_capacity(tree.nodes.len());
    let mut stack = Vec::new();

    if tree.nodes[0].is_branch() {
        stack.push((0usize, None));
    } else {
        panic!("Tree root is not a branch node");
    }

    while let Some((id, parent_id)) = stack.pop() {
        let branch = tree.nodes[id].as_branch().unwrap();

        let left_child = match &tree.nodes[branch.left] {
            Node::Leaf(l) => Branch::Prediction(l.prediction),
            Node::Branch(_) => Branch::Ptr(branch.left),
        };

        let right_child = match &tree.nodes[branch.right] {
            Node::Leaf(l) => Branch::Prediction(l.prediction),
            Node::Branch(_) => Branch::Ptr(branch.right),
        };

        placed_nodes.push(LinkedNode::Branch(LinkedBranchNode {
            id,
            _parent_id: parent_id,
            left_child,
            right_child,
            split_at: branch.split_at.clone().into_bf16(),
            split_with: branch.split_with,
        }));

        // Push right before left so left is popped first (pre-order traversal)
        if let Branch::Ptr(r) = right_child {
            stack.push((r, Some(id)));
        }
        if let Branch::Ptr(l) = left_child {
            stack.push((l, Some(id)));
        }
    }

    placed_nodes
}

fn build_compiled_nodes(placed_nodes: &[LinkedNode]) -> color_eyre::Result<Vec<model::Node>> {
    const HEADER_SIZE: u16 = 1;

    // Optionally, add extra padding at the end to make the tree length even
    let tail_padding = usize::from(placed_nodes.len().is_multiple_of(2));

    // Add header + padding for full tree length
    let tree_len: u32 = (placed_nodes.len() + tail_padding + HEADER_SIZE as usize)
        .try_into()
        .context("Tree length should fit into u32")?;

    // We now have a Vec of LinkedNodes in the correct order. Me must now convert it
    // into a Vec of model::Node, and adjut the child pointers accordingly.
    let mut compiled_nodes = Vec::with_capacity(tree_len as usize);

    // Add header at beginning. Placing the first node at index 1 takes advantage of
    // direct execution is header is aligned.
    // Also leave cache metadata empty. It will be written as necessary at the cache
    // split step.
    let empty = CacheMetadata::new_empty();
    let header = [lumberjack_model::model::Node::from_header(
        TreeHeader::new(tree_len, HEADER_SIZE, empty)
            .map_err(|e| eyre!("Cannot compile model: {e:?}"))?,
    )];
    let header_len: u16 = header.len().try_into().unwrap();
    compiled_nodes.extend(header);

    for node in placed_nodes {
        let new = match node {
            LinkedNode::Padding => PADDING,
            LinkedNode::Branch(b) => compile_branch(b, placed_nodes, header_len),
        };

        compiled_nodes.push(new);
    }

    compiled_nodes.extend(std::iter::repeat_n(PADDING, tail_padding));

    assert_eq!(compiled_nodes.len(), tree_len as usize);
    assert!(compiled_nodes.len().is_multiple_of(2));

    Ok(compiled_nodes)
}

fn compile_branch(
    n: &LinkedBranchNode,
    placed_nodes: &[LinkedNode],
    header_len: u16,
) -> model::Node {
    let left = match n.left_child {
        Branch::Ptr(id) => position_of(placed_nodes, id) + header_len,
        Branch::Prediction(p) => p,
    };
    let right = match n.right_child {
        Branch::Ptr(id) => position_of(placed_nodes, id) + header_len,
        Branch::Prediction(p) => p,
    };
    lumberjack_model::model::Node::from_branch(lumberjack_model::model::Branch::new(
        n.split_with,
        n.split_at,
        left,
        right,
        n.left_child.is_prediction(),
        n.right_child.is_prediction(),
    ))
}
