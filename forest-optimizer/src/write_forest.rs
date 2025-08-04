use color_eyre::{
    Result,
    eyre::{Context, eyre},
};

use std::{fs::File, io::Write, path::Path};

use embedded_rforest::forest::{Classification, OptimizedForest, Regression};

use crate::{
    forest::Forest,
    serialized_forest::{SerializedClassificationNode, SerializedForest, SerializedRegressionNode},
};

pub fn write_classification(input: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
    // Read the input file
    let serialized = SerializedForest::<SerializedClassificationNode>::read(input)
        .context("Could not read forest definition file (CSV).")?;
    let forest = Forest::from_serialized(serialized)?;

    // Optimize the forest
    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::<Classification>::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        forest.num_features().try_into().unwrap(),
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    let serialized = optimized.to_bytes();
    let ptr = serialized.as_ptr();
    assert!(ptr as usize % align_of_val(&optimized) == 0);

    // Write the transformed data to the output file
    let mut output_file = File::create(output).context("Could not create output file")?;
    output_file.write_all(&serialized)?;

    Ok(())
}

pub fn write_regression(input: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
    // Read the input file
    let serialized = SerializedForest::<SerializedRegressionNode>::read(input)
        .context("Could not read forest definition file (CSV).")?;
    let forest = Forest::from_serialized(serialized)?;

    // Optimize the forest
    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::<Regression>::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        forest.num_features().try_into().unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    let serialized = optimized.to_bytes();
    let ptr = serialized.as_ptr();
    assert!(ptr as usize % align_of_val(&optimized) == 0);

    // Write the transformed data to the output file
    let mut output_file = File::create(output).context("Could not create output file")?;
    output_file.write_all(&serialized)?;

    Ok(())
}
