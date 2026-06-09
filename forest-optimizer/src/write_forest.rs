use color_eyre::{
    Result,
    eyre::{Context, eyre},
};

use std::{fs::File, io::Write, num::NonZeroU8, path::Path};

use embedded_rforest::forest::{Classification, OptimizedForest};

use crate::{csv_forest::CsvForest, forest::Forest};

pub fn write_forest(input: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
    // Read the input file
    let serialized =
        CsvForest::read(input).context("Could not read forest definition file (CSV).")?;
    let forest = Forest::from_serialized(serialized)?;

    // Optimize the forest
    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        NonZeroU8::new(
            forest
                .num_features()
                .try_into()
                .expect("Features must fit into an u8."),
        )
        .expect("Number of features must be non-zero."),
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    let serialized = optimized.to_bytes();
    let ptr = serialized.as_ptr();
    assert!((ptr as usize).is_multiple_of(align_of_val(&optimized)));

    // Write the transformed data to the output file
    let mut output_file = File::create(output).context("Could not create output file")?;
    output_file.write_all(&serialized)?;

    Ok(())
}
