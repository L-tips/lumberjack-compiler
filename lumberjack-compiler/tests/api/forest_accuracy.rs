use std::num::NonZeroU16;

use color_eyre::Result;
use color_eyre::eyre::eyre;
use lumberjack_model::model::{Classification, Model};

use crate::helpers::get_forest;

#[test]
fn raw_model_accuracy_iris_800_trees() -> Result<()> {
    let forest = get_forest("./tests/test-forests/forest_iris_800.csv")?;
    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;
    let targets_map = forest.targets();

    for (features, target_prediction) in test_data {
        let prediction = forest.predict(features.as_slice());
        let pred_idx = targets_map.get(&prediction).expect("target not found");
        assert_eq!(*pred_idx, target_prediction);
    }

    Ok(())
}

#[test]
fn compiled_model_accuracy_iris_800_trees() -> Result<()> {
    let forest = get_forest("./tests/test-forests/forest_iris_800.csv")?;

    let nodes = forest.compile();
    let optimized = Model::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        NonZeroU16::new(forest.num_features().try_into().unwrap()).unwrap(),
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    optimized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let test_data = forest
        .problem()
        .features_vector_from_csv("./tests/test-data/iris.csv")?;
    let targets_map = forest.targets();

    for (features, target_prediction) in test_data {
        let prediction = forest.predict(features.as_slice());
        let pred_idx = targets_map.get(&prediction).expect("target not found");
        assert_eq!(*pred_idx, target_prediction);
    }

    Ok(())
}
