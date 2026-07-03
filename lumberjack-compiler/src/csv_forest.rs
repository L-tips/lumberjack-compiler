use crate::forest::{BranchNode, LeafNode, Node};
use crate::problem_type::{Classification, Map};
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::path::Path;
use std::{fs, io};

use color_eyre::Result;
use color_eyre::eyre::{OptionExt, eyre};
use half::bf16;
use serde::{Deserialize, Deserializer};

pub trait NodeType {}

/// A single node of a [`SerializedForest`] in classification mode
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CsvNode {
    /// Tree index. 1-indexed.
    pub tree_idx: usize,
    /// Node index. 1-indexed.
    pub node_idx: usize,
    /// Pointer to left branch node
    #[serde(rename = "left daughter")]
    pub left: usize,
    /// Pointer to right branch node
    #[serde(rename = "right daughter")]
    pub right: usize,
    /// The variable on which to split
    #[serde(rename = "split var", deserialize_with = "string_or_na")]
    pub split_on: Option<String>,
    /// The split point
    #[serde(rename = "split point")]
    pub split_at: f32,
    /// The node status. A value of 1 represents a branch, and -1 represents a
    /// prediction
    pub status: i8,
    /// The predicted variable
    #[serde(deserialize_with = "string_or_na")]
    pub prediction: Option<String>,
}

impl CsvNode {
    /// Find the feature ID of this node's split variable
    pub fn feature_id(&self, features_map: &Map) -> Option<u16> {
        features_map.get(self.split_on.as_ref()?).copied()
    }

    /// Find the target ID of this node's prediction
    pub fn target_id(&self, targets_map: &Map) -> Option<u16> {
        targets_map.get(self.prediction.as_ref()?).copied()
    }

    fn deserialize<R: io::Read>(
        problem: &mut Classification,
        rdr: &mut csv::Reader<R>,
    ) -> Result<Vec<Self>> {
        let mut feat_count = 0;
        let mut target_count = 0;

        let mut nodes = Vec::new();

        for result in rdr.deserialize() {
            let record: CsvNode = result?;

            if let Some(feat) = &record.split_on {
                assert_ne!(record.left, 0, "Node doesn't have a left daughter");
                assert_ne!(record.right, 0, "Node doesn't have a right daughter");

                // Map all available features and assign an index to each
                if let Entry::Vacant(e) = problem.features_mut().entry(feat.clone()) {
                    e.insert(feat_count);
                    feat_count += 1;
                }
            }

            // Map all available targets and assign an index to each
            if let Some(target) = &record.prediction {
                assert_eq!(record.status, -1, "Node is not a classification prediction");

                if let Entry::Vacant(e) = problem.targets_mut().entry(target.clone()) {
                    e.insert(target_count);
                    target_count += 1;
                }
            }

            nodes.push(record);
        }

        Ok(nodes)
    }

    pub fn normalize(self, problem: &Classification) -> Result<Node> {
        if self.split_on.is_some() {
            let branch = BranchNode {
                split_with: self
                    .feature_id(problem.features())
                    .ok_or_eyre("Feature ID missing")?,
                split_at: bf16::from_f32(self.split_at),
                left: self.left - 1,
                right: self.right - 1,
            };

            return Ok(Node::Branch(branch));
        } else if self.prediction.is_some() {
            let leaf = LeafNode {
                prediction: self
                    .target_id(problem.targets())
                    .ok_or_eyre("Target ID missing")?,
            };

            return Ok(Node::Leaf(leaf));
        }
        Err(eyre!("Node is not a branch nor a leaf"))
    }

    pub fn node_idx(&self) -> usize {
        self.node_idx
    }

    pub fn tree_idx(&self) -> usize {
        self.tree_idx
    }
}

#[derive(Debug)]
pub struct CsvForest {
    nodes: Vec<CsvNode>,
    problem: Classification,
}

impl CsvForest {
    pub fn problem(&self) -> &Classification {
        &self.problem
    }

    /// Get the features of this forest
    pub fn features(&self) -> &Map {
        self.problem.features()
    }

    pub fn nodes(&self) -> &[CsvNode] {
        &self.nodes
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let rdr = fs::File::open(path.as_ref())?;
        let mut rdr = csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .from_reader(rdr);

        let mut problem = Classification::default();

        let nodes = CsvNode::deserialize(&mut problem, &mut rdr)?;

        Ok(CsvForest { nodes, problem })
    }
}

impl CsvForest {
    /// Get the targets of this forest
    pub fn targets(&self) -> &Map {
        self.problem.targets()
    }
}

/// Deserialize a string into an `Option<String>`, returning `None` if the
/// string is empty or the literal "NA".
fn string_or_na<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    // Deserialize as a string
    let s: String = String::deserialize(deserializer)?;

    // Check if the string is "NA" (without quotes)
    if s == "NA" || s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}
