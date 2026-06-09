use aligned_vec::AVec;
use zerocopy::IntoBytes;

use crate::forest::LEN_PADDING;

use super::OptimizedForest;

impl OptimizedForest<'_> {
    pub fn to_bytes(&self) -> AVec<u8> {
        let mut bytes = AVec::<u8>::with_capacity(4, 8);

        // Number of trees (4 bytes)
        bytes.extend_from_slice(self.num_trees.to_bytes().as_slice());

        // Number of features (1 byte)
        bytes.push(self.num_features().get());

        // Number of targets (1 byte)
        bytes.push(self.num_targets().get());

        // Padding (2 bytes)
        bytes.extend_from_slice(&[0; LEN_PADDING]);

        // Performance: reserve some extra space in the vec for all our nodes
        bytes.reserve(size_of_val(self.nodes));

        // Insert all the nodes
        for node in self.nodes {
            // Just for byte-copying purposes, we assume all nodes are branches.
            bytes.extend_from_slice(unsafe { node.branch.clone() }.as_bytes());
        }

        bytes
    }
}
