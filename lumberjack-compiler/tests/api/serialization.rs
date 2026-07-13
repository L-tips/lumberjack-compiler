use color_eyre::Result;
use color_eyre::eyre::eyre;
use lumberjack_model::model::Model;

use crate::helpers::parse_source;

#[test]
fn serialized_then_deserialized_classification_tree_is_accurate() -> Result<()> {
    let forest = parse_source("./tests/test-forests/forest_iris_5.csv")?;

    let nodes = forest.compile(0)?;
    let compiled = Model::new(
        forest.num_trees().try_into().unwrap(),
        0,
        &nodes,
        u16::try_from(forest.num_features())?.try_into()?,
        u16::try_from(forest.num_targets())?.try_into()?,
    )
    .map_err(|e| eyre!("Malformed forest: {e:?}"))?;

    compiled
        .verify()
        .map_err(|e| eyre!("Malformed forest detected upon verification: {e:?}"))?;

    let serialized = compiled.serialize();
    let optimized = Model::deserialize(&serialized).map_err(|e| eyre!("Malfomed forest: {e:?}"))?;

    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;

    for datapoint in test_data {
        let prediction = optimized.predict(&datapoint.features);
        assert_eq!(prediction, datapoint.reference_prediction);
    }

    Ok(())
}

#[test]
fn classification_static_storage_deserializes_correctly() -> Result<()> {
    let buf = lumberjack_model::static_storage!("../test-forests/forest_iris_5.ljmodel");

    let forest = parse_source("./tests/test-forests/forest_iris_5.csv")?;

    let deserialized = Model::deserialize(buf)
        .map_err(|e| eyre!("Malformed forest detected upon deserialization: {e:?}"))?;

    deserialized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;

    for datapoint in test_data {
        let prediction = deserialized.predict(&datapoint.features);
        assert_eq!(prediction, datapoint.reference_prediction);
    }

    Ok(())
}
