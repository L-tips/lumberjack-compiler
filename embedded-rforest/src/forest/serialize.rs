use aligned_vec::AVec;
use zerocopy::IntoBytes;

use super::{OptimizedForest, ProblemType};

impl<P: ProblemType> OptimizedForest<'_, P> {
    pub fn to_bytes(&self) -> AVec<u8> {
        let mut bytes = AVec::<u8>::with_capacity(4, 8);

        // Number of trees (4 bytes)
        bytes.extend_from_slice(self.num_trees.to_bytes().as_slice());

        // Number of features (1 byte)
        bytes.push(self.num_features);

        // Number of targets (1 byte)
        if let Some(b) = self.num_targets {
            bytes.push(b.get());
        } else {
            bytes.push(0);
        }

        // Padding (2 bytes)
        bytes.extend_from_slice(&[0; 2]);

        // Performance: reserve some extra space in the vec for all our nodes
        bytes.reserve(size_of_val(self.nodes));

        // Insert all the nodes
        for node in self.nodes {
            bytes.extend_from_slice(node.as_bytes());
        }

        bytes
    }
}
