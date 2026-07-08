use std::num::NonZeroU16;

use color_eyre::Result;
use color_eyre::eyre::eyre;
use lumberjack_model::model::{Classification, Model};

use crate::helpers::get_forest;

#[test]
fn serialized_then_deserialized_classification_tree_is_accurate() -> Result<()> {
    let forest = get_forest("./tests/test-forests/forest_iris_5.csv")?;

    let nodes = forest.compile();
    let compiled = Model::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        NonZeroU16::new(forest.num_features().try_into().unwrap()).unwrap(),
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    compiled
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let serialized = compiled.serialize();
    let optimized = Model::deserialize(&serialized).map_err(|e| eyre!("Malfomed forest: {e:?}"))?;

    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;

    for (features, target_prediction) in test_data {
        let prediction = optimized.predict(&features);
        assert_eq!(prediction, target_prediction);
    }

    Ok(())
}

#[test]
fn classification_static_storage_deserializes_correctly() -> Result<()> {
    let buf = lumberjack_model::static_storage!("../test-forests/forest_iris_5.rforest");

    let forest = get_forest("./tests/test-forests/forest_iris_5.csv")?;

    let deserialized = Model::deserialize(buf)
        .map_err(|e| eyre!("Malformed forest detected upon deserialization: {e:?}"))?;

    deserialized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;

    for (features, target_prediction) in test_data {
        let prediction = deserialized.predict(&features);
        assert_eq!(prediction, target_prediction);
    }

    Ok(())
}
