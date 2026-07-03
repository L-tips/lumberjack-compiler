use std::num::NonZeroU16;

use color_eyre::Result;
use color_eyre::eyre::eyre;
use lumberjack_compiler::serialize::to_bytes;
use lumberjack_model::forest::{Classification, OptimizedForest};

use crate::datasets::iris;
use crate::helpers::{get_forest, get_test_data};

#[test]
fn serialized_then_deserialized_classification_tree_is_accurate() -> Result<()> {
    let forest = get_forest("./tests/test-forests/forest_iris_5.csv")?;

    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        NonZeroU16::new(forest.num_features().try_into().unwrap()).unwrap(),
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    optimized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let serialized = to_bytes(&optimized);
    let optimized =
        OptimizedForest::deserialize(&serialized).map_err(|e| eyre!("Malfomed forest: {e:?}"))?;

    let test_data: Vec<iris::DataPoint> = get_test_data("./tests/test-data/iris.csv")?;

    for data_point in test_data {
        let features = data_point.transform_features(forest.features());
        let prediction = optimized.predict(&features);
        let target = forest.targets().get(&data_point.forest_prediction).unwrap();
        assert_eq!(prediction, *target);
    }

    Ok(())
}

#[test]
fn classification_static_storage_deserializes_correctly() -> Result<()> {
    let buf = lumberjack_model::static_storage!("../test-forests/forest_iris_5.rforest");

    let forest = get_forest("./tests/test-forests/forest_iris_5.csv")?;

    let deserialized = OptimizedForest::deserialize(buf)
        .map_err(|e| eyre!("Malformed forest detected upon deserialization: {e:?}"))?;

    deserialized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let test_data: Vec<iris::DataPoint> = get_test_data("./tests/test-data/iris.csv")?;

    for data_point in test_data {
        let features = data_point.transform_features(forest.features());
        let prediction = deserialized.predict(&features);
        let target = forest.targets().get(&data_point.forest_prediction).unwrap();
        assert_eq!(prediction, *target);
    }

    Ok(())
}
