use crate::forest::{BranchNode, LeafNode, Node};
use crate::problem_type::{Classification, Map, PredictionType, ProblemType, Regression};
use crate::typelevel::private::Sealed;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::{fs, io};

use color_eyre::Result;
use color_eyre::eyre::{Context, ContextCompat, OptionExt, eyre};
use half::bf16;
use serde::{Deserialize, Deserializer};

pub trait NodeType {}

pub trait SerializedNode: Sealed + Clone {
    type ProblemType: ProblemType;

    fn deserialize<R: io::Read>(
        problem: &mut Self::ProblemType,
        rdr: &mut csv::Reader<R>,
    ) -> Result<Vec<Self>>;

    /// Turn a serialized node into a [`Node`]. This function also
    /// renormalizes indices to use 0-indexing, and converts feature and target
    /// names to their indices.
    fn normalize(self, problem: &Self::ProblemType) -> Result<Node<Self::ProblemType>>;

    fn node_idx(&self) -> usize;
    fn tree_idx(&self) -> usize;
}

/// A single node of a [`SerializedForest`] in classification mode
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SerializedClassificationNode {
    /// Tree index. 1-indexed.
    pub tree_idx: usize,
    /// Node index. 1-indexed.
    pub node_idx: usize,
    /// Pointer to left branch node
    #[serde(rename = "left daughter")]
    pub left: u32,
    /// Pointer to right branch node
    #[serde(rename = "right daughter")]
    pub right: u32,
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

impl SerializedClassificationNode {
    /// Find the feature ID of this node's split variable
    pub fn feature_id(&self, features_map: &Map) -> Option<u32> {
        features_map.get(self.split_on.as_ref()?).copied()
    }

    /// Find the target ID of this node's prediction
    pub fn target_id(&self, targets_map: &Map) -> Option<u32> {
        targets_map.get(self.prediction.as_ref()?).copied()
    }
}

impl Sealed for SerializedClassificationNode {}

impl SerializedNode for SerializedClassificationNode {
    type ProblemType = Classification;

    fn deserialize<R: io::Read>(
        problem: &mut Self::ProblemType,
        rdr: &mut csv::Reader<R>,
    ) -> Result<Vec<Self>> {
        let mut feat_count = 0;
        let mut target_count = 0;

        let mut nodes = Vec::new();

        for result in rdr.deserialize() {
            let record: SerializedClassificationNode = result?;

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

    fn normalize(self, problem: &Self::ProblemType) -> Result<Node<Self::ProblemType>> {
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

    fn node_idx(&self) -> usize {
        self.node_idx
    }

    fn tree_idx(&self) -> usize {
        self.tree_idx
    }
}

/// A single node of a [`SerializedForest`] in regression mode
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SerializedRegressionNode {
    /// Tree index. 1-indexed.
    pub tree_idx: usize,
    /// Node index. 1-indexed.
    pub node_idx: usize,
    /// Pointer to left branch node
    #[serde(rename = "left daughter")]
    pub left: u32,
    /// Pointer to right branch node
    #[serde(rename = "right daughter")]
    pub right: u32,
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
    pub prediction: Option<f32>,
}

impl SerializedRegressionNode {
    /// Find the feature ID of this node's split variable
    pub fn feature_id(&self, features_map: &Map) -> Option<u32> {
        features_map.get(self.split_on.as_ref()?).copied()
    }

    /// Find this node's prediction
    pub fn target(&self) -> Option<f32> {
        self.prediction
    }
}

impl Sealed for SerializedRegressionNode {}

impl SerializedNode for SerializedRegressionNode {
    type ProblemType = Regression;

    fn deserialize<R: io::Read>(
        problem: &mut Self::ProblemType,
        rdr: &mut csv::Reader<R>,
    ) -> Result<Vec<Self>> {
        let mut feat_count = 0;
        let mut nodes = Vec::new();

        for result in rdr.deserialize() {
            let record: SerializedRegressionNode = result?;

            if let Some(feat) = &record.split_on {
                assert_ne!(record.left, 0, "Node doesn't have a left daughter");
                assert_ne!(record.right, 0, "Node doesn't have a right daughter");

                // Map all available features and assign an index to each
                if let Entry::Vacant(e) = problem.features_mut().entry(feat.clone()) {
                    e.insert(feat_count);
                    feat_count += 1;
                }
            }

            nodes.push(record);
        }

        Ok(nodes)
    }

    fn normalize(self, problem: &Self::ProblemType) -> Result<Node<Self::ProblemType>> {
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
                prediction: self.prediction.ok_or_eyre("Prediction missing")?,
            };

            return Ok(Node::Leaf(leaf));
        }
        Err(eyre!("Node is not a branch nor a leaf"))
    }

    fn node_idx(&self) -> usize {
        self.node_idx
    }

    fn tree_idx(&self) -> usize {
        self.tree_idx
    }
}

#[derive(Debug)]
pub struct SerializedForest<N: SerializedNode> {
    nodes: Vec<N>,
    problem: N::ProblemType,
}

impl<N: SerializedNode> SerializedForest<N> {
    pub fn problem(&self) -> &N::ProblemType {
        &self.problem
    }

    /// Get the features of this forest
    pub fn features(&self) -> &Map {
        self.problem.features()
    }

    pub fn nodes(&self) -> &[N] {
        &self.nodes
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        Self::validate_header(&path)?;

        let rdr = fs::File::open(path.as_ref())?;
        let mut rdr = csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .from_reader(rdr);

        let mut problem = N::ProblemType::default();

        let nodes = N::deserialize(&mut problem, &mut rdr)?;

        Ok(SerializedForest { nodes, problem })
    }

    fn validate_header(path: impl AsRef<Path>) -> Result<()> {
        let rdr = BufReader::new(fs::File::open(path.as_ref())?);

        let header = rdr
            .lines()
            .take(1)
            .collect::<Result<Vec<_>, _>>()?
            .join(" ");

        let header = header
            .strip_prefix("#")
            .context("Malformed forest definition file. First line doesn't start with '#'.")?;

        let prediction_type = &serde_json::from_str::<serde_json::Value>(header)
            .context("Malformed forest definition file. First line doesn't contain valid json")?["problem_type"];

        let prediction_type: PredictionType = serde_json::from_value(prediction_type.clone())?;
        if prediction_type != N::ProblemType::TYPE {
            return Err(color_eyre::eyre::eyre!(
                "You are trying to solve a regression problem with classification methods, or a classification problem with regression methods!"
            ));
        }

        Ok(())
    }
}

impl SerializedForest<SerializedClassificationNode> {
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
