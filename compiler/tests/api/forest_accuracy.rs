use color_eyre::Result;
use color_eyre::eyre::eyre;
use lumberjack_compiler::PlacementStrategy;
use lumberjack_compiler::compiler::PartitionStrategy;
use lumberjack_compiler::feature_vectors::features_vector_from_csv;
use lumberjack_model::model::Model;

use crate::helpers::parse_source;

#[test]
fn ir_accuracy_iris_800_trees_f32() -> Result<()> {
    let ir = parse_source("./tests/test-forests/forest_iris_800.csv")?;
    let test_data = features_vector_from_csv(ir.problem(), "./tests/test-data/iris.csv")?;
    let targets_map = ir.targets();

    for datapoint in test_data {
        let prediction = ir.predict(&datapoint.features);
        let pred_idx = targets_map.get(prediction.0).expect("target not found");
        assert_eq!(*pred_idx, datapoint.reference_prediction);
    }

    Ok(())
}

#[test]
fn ir_accuracy_iris_800_trees_quantized() -> Result<()> {
    let ir = parse_source("./tests/test-forests/forest_iris_800.csv")?;
    let ir = ir.quantize_splits();

    let test_data = features_vector_from_csv(ir.problem(), "./tests/test-data/iris.csv")?;
    let targets_map = ir.targets();

    for datapoint in test_data {
        let prediction = ir.predict(&datapoint.features);
        let pred_idx = targets_map.get(prediction.0).expect("target not found");
        assert_eq!(*pred_idx, datapoint.reference_prediction);
    }

    Ok(())
}

#[test]
fn compiled_model_accuracy_iris_800_trees() -> Result<()> {
    let model = parse_source("./tests/test-forests/forest_iris_800.csv")?;

    let nodes = model.compile(
        0,
        PlacementStrategy::ExecutionAware,
        PartitionStrategy::EqualSorted,
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

    let test_data = features_vector_from_csv(model.problem(), "./tests/test-data/iris.csv")?;
    let targets_map = model.targets();

    for datapoint in test_data {
        let prediction = model.predict(&datapoint.features);
        let pred_idx = targets_map.get(prediction.0).expect("target not found");
        assert_eq!(*pred_idx, datapoint.reference_prediction);
    }

    Ok(())
}
