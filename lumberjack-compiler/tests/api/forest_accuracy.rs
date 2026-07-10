use color_eyre::Result;
use color_eyre::eyre::eyre;
use lumberjack_model::model::Model;

use crate::helpers::get_forest;

#[test]
fn raw_model_accuracy_iris_800_trees() -> Result<()> {
    let forest = get_forest("./tests/test-forests/forest_iris_800.csv")?;
    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;
    let targets_map = forest.targets();

    for datapoint in test_data {
        let prediction = forest.predict(&datapoint.features);
        let pred_idx = targets_map.get(&prediction).expect("target not found");
        assert_eq!(*pred_idx, datapoint.reference_prediction);
    }

    Ok(())
}

#[test]
fn compiled_model_accuracy_iris_800_trees() -> Result<()> {
    let forest = get_forest("./tests/test-forests/forest_iris_800.csv")?;

    let nodes = forest.compile(0)?;
    let optimized = Model::new(
        forest.num_trees().try_into().unwrap(),
        0,
        &nodes,
        u16::try_from(forest.num_features())?.try_into()?,
        u16::try_from(forest.num_targets())?.try_into()?,
    )
    .map_err(|e| eyre!("Malformed forest: {e:?}"))?;

    optimized
        .verify()
        .map_err(|e| eyre!("Malformed forest detected upon verification: {e:?}"))?;

    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;
    let targets_map = forest.targets();

    for datapoint in test_data {
        let prediction = forest.predict(&datapoint.features);
        let pred_idx = targets_map.get(&prediction).expect("target not found");
        assert_eq!(*pred_idx, datapoint.reference_prediction);
    }

    Ok(())
}
