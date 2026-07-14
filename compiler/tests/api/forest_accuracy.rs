use color_eyre::Result;
use color_eyre::eyre::eyre;
use lumberjack_compiler::PlacementStrategy;
use lumberjack_compiler::compiler::PartitionStrategy;
use lumberjack_model::model::Model;

use crate::helpers::parse_source;

#[test]
fn ir_accuracy_iris_800_trees_f32() -> Result<()> {
    let ir = parse_source("./tests/test-forests/forest_iris_800.csv")?;
    let test_data = ir
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;
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

    let test_data = ir
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;
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
    let forest = parse_source("./tests/test-forests/forest_iris_800.csv")?;

    let nodes = forest.compile(
        0,
        PlacementStrategy::ExecutionAware,
        PartitionStrategy::EqualSorted,
    )?;
    let optimized = Model::new(
        forest.num_trees().try_into().unwrap(),
        0,
        &nodes,
        u16::try_from(forest.num_features())?.try_into()?,
        u16::try_from(forest.num_targets())?.try_into()?,
    )
    .map_err(|e| eyre!("Malformed forest: {e:?}"))?;

    optimized
        .verify_linear()
        .map_err(|e| eyre!("Malformed forest detected upon verification: {e:?}"))?;

    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;
    let targets_map = forest.targets();

    for datapoint in test_data {
        let prediction = forest.predict(&datapoint.features);
        let pred_idx = targets_map.get(prediction.0).expect("target not found");
        assert_eq!(*pred_idx, datapoint.reference_prediction);
    }

    Ok(())
}
