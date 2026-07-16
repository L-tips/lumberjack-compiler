use color_eyre::Result;
use color_eyre::eyre::eyre;
use lumberjack_compiler::PlacementStrategy;
use lumberjack_compiler::compiler::PartitionStrategy;
use lumberjack_compiler::feature_vectors::features_vector_from_csv;
use lumberjack_model::model::Model;

use crate::helpers::parse_source;

#[test]
fn serialized_then_deserialized_classification_tree_is_accurate() -> Result<()> {
    let model = parse_source("./tests/test-forests/forest_iris_5.csv")?;

    let nodes = model.compile(
        0,
        PlacementStrategy::ExecutionAware,
        PartitionStrategy::EqualRandom,
    )?;
    let compiled = Model::new(
        model.num_trees().try_into().unwrap(),
        0,
        &nodes,
        u16::try_from(model.num_features())?.try_into()?,
        u16::try_from(model.num_targets())?.try_into()?,
    )
    .map_err(|e| eyre!("Malformed forest: {e:?}"))?;

    compiled
        .verify_acyclic()
        .map_err(|e| eyre!("Malformed forest detected upon verification: {e:?}"))?;

    let serialized = compiled.serialize();
    let optimized = Model::deserialize(&serialized).map_err(|e| eyre!("Malfomed forest: {e:?}"))?;

    let test_data = features_vector_from_csv(model.problem(), "./tests/test-data/iris.csv")?;

    for datapoint in test_data {
        let prediction = optimized.predict(&datapoint.features);
        assert_eq!(prediction, datapoint.reference_prediction);
    }

    Ok(())
}

#[test]
fn classification_static_storage_deserializes_correctly() -> Result<()> {
    let buf = lumberjack_model::static_storage!("../test-forests/forest_iris_5.ljmodel");

    let ir = parse_source("./tests/test-forests/forest_iris_5.csv")?;

    let deserialized = Model::deserialize(buf)
        .map_err(|e| eyre!("Malformed forest detected upon deserialization: {e:?}"))?;

    deserialized
        .verify_acyclic()
        .map_err(|e| eyre!("Malformed forest detected upon verification: {e:?}"))?;

    let test_data = features_vector_from_csv(ir.problem(), "./tests/test-data/iris.csv")?;

    for datapoint in test_data {
        let prediction = deserialized.predict(&datapoint.features);
        assert_eq!(prediction, datapoint.reference_prediction);
    }

    Ok(())
}
