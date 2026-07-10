use crate::compiled_model;
use crate::compiler::{BranchNode, LeafNode, Node, Tree};
use crate::problem::{Map, ProblemDefinition};
use std::fmt::Debug;
use std::{
    fs::File,
    io::{self, Write},
    path::Path,
};

use color_eyre::eyre::{Context, OptionExt, Result, eyre};
use half::bf16;
use indexmap::map::Entry;
use serde::{Deserialize, Deserializer};

use lumberjack_model::model::Model;

use crate::compiler::ForestModel;

/// A single node of a [`SerializedForest`] in classification mode
#[derive(Debug, Clone, serde::Deserialize)]
struct CsvNode {
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
    fn feature_id(&self, features_map: &Map) -> Option<u16> {
        features_map.get(self.split_on.as_ref()?).copied()
    }

    /// Find the target ID of this node's prediction
    fn target_id(&self, targets_map: &Map) -> Option<u16> {
        targets_map.get(self.prediction.as_ref()?).copied()
    }

    fn deserialize<R: io::Read>(
        problem: &mut ProblemDefinition,
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

    fn normalize(self, problem: &ProblemDefinition) -> Result<Node> {
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
    problem: ProblemDefinition,
}

impl CsvForest {
    pub fn problem(&self) -> &ProblemDefinition {
        &self.problem
    }

    /// Get the features of this forest
    pub fn features(&self) -> &Map {
        self.problem.features()
    }

    fn nodes(&self) -> &[CsvNode] {
        &self.nodes
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let rdr =
            File::open(path.as_ref()).context(format!("path: {}", path.as_ref().display()))?;
        let mut rdr = csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .from_reader(rdr);

        let mut problem = ProblemDefinition::default();

        let nodes = CsvNode::deserialize(&mut problem, &mut rdr)?;
        Ok(CsvForest { nodes, problem })
    }

    /// Get the targets of this forest
    pub fn targets(&self) -> &Map {
        self.problem.targets()
    }

    /// Convert a [`SerializedForest`] into a [`Forest`].
    ///
    /// In practice, this method flattens the nodes, putting all tree roots in
    /// front of the array.
    pub fn into_forest_model(self) -> Result<ForestModel> {
        let problem = self.problem();

        // Find all nodes which have an index of 1. These are our tree roots.
        let mut tree_roots: Vec<_> = self
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
                let mut nodes = self
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

        Ok(ForestModel::new(trees, self.problem().clone()))
    }
}

/// Takes a path to a file containing a model using the CSV specification, and
/// writes a compiled model to the output path.
pub fn compile_from_csv(
    input: impl AsRef<Path>,
    output: Option<impl AsRef<Path>>,
    num_cells: u8,
    analyze: bool,
) -> Result<()> {
    // Read the input file
    let serialized =
        CsvForest::read(input).context("Could not read forest definition file (CSV).")?;
    let forest = serialized.into_forest_model()?;

    // Compile the forest model
    let nodes = forest.compile(num_cells)?;
    let compiled = Model::new(
        forest.num_trees().try_into().unwrap(),
        num_cells,
        &nodes,
        u16::try_from(forest.num_features())?.try_into()?,
        u16::try_from(forest.num_targets())?.try_into()?,
    )
    .map_err(|e| eyre!("Malformed forest: {e:?}"))?;

    if analyze {
        compiled_model::analyze(&compiled);
    }

    let serialized = compiled.serialize();
    let ptr = serialized.as_ptr();
    assert!((ptr as usize).is_multiple_of(align_of_val(&compiled)));

    // Write the transformed data to the output file
    if let Some(out_path) = output {
        let mut output_file = File::create(out_path).context("Could not create output file")?;
        output_file.write_all(&serialized)?;
    }

    Ok(())
}

/// Compile the model, splitting it up into one file per cell cache
pub fn compile_split_caches_from_csv(
    input: impl AsRef<Path>,
    dir: impl AsRef<Path>,
    prefix: Option<String>,
    num_cells: u8,
    analyze: bool,
) -> Result<()> {
    let dir = dir.as_ref();
    if !dir.is_dir() || !std::fs::exists(dir)? {
        return Err(eyre!(
            "{} is not a directory or does not exist",
            dir.display()
        ));
    }

    // Read the input file
    let serialized =
        CsvForest::read(input).context("Could not read forest definition file (CSV).")?;
    let forest = serialized.into_forest_model()?;

    // Compile the forest
    let nodes = forest.compile(num_cells)?;
    let compiled = Model::new(
        forest.num_trees().try_into().unwrap(),
        num_cells,
        &nodes,
        u16::try_from(forest.num_features())?.try_into()?,
        u16::try_from(forest.num_targets())?.try_into()?,
    )
    .map_err(|e| eyre!("Malformed forest: {e:?}"))?;

    if analyze {
        compiled_model::analyze(&compiled);
    }

    for (i, cache_data) in compiled
        .split_cache_data::<{ u8::MAX as usize }>()
        .iter()
        .enumerate()
    {
        let prefix = prefix.as_deref().unwrap_or("cache_data");
        let out_path = dir.join(format!("{prefix}_{i}.lj_data"));
        let mut output_file = File::create(out_path).context("Could not create output file")?;

        output_file.write_all(lumberjack_model::model::Node::slice_as_bytes(cache_data))?;
    }

    Ok(())
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
