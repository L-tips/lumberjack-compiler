use aligned_vec::AVec;
use zerocopy::IntoBytes;

use lumberjack_model::model::{ALIGNMENT, Model};

/// Write the provided model to an aligned byte vector
pub fn to_bytes(forest: &Model) -> AVec<u8> {
    let mut bytes = AVec::<u8>::with_capacity(ALIGNMENT, 8);

    // Number of trees (4 bytes)
    bytes.extend_from_slice(forest.num_trees().to_bytes().as_slice());

    // Number of features (2 bytes)
    bytes.extend_from_slice(&forest.num_features().to_bytes());

    // Number of targets (2 bytes)
    bytes.extend_from_slice(&forest.num_targets().to_bytes());

    // Padding
    bytes.extend_from_slice(&0u64.to_le_bytes());

    // Performance: reserve some extra space in the vec for all our nodes
    bytes.reserve(size_of_val(forest.nodes()));

    // Insert all the nodes
    for node in forest.nodes() {
        // Just for byte-copying purposes, we assume all nodes are branches.
        bytes.extend_from_slice(node.as_bytes());
    }

    bytes
}
