use color_eyre::Result;
use color_eyre::eyre::eyre;
use embedded_rforest::forest::{Classification, OptimizedForest, Predict, Regression};
use forest_optimizer::serialized_forest::{SerializedClassificationNode, SerializedRegressionNode};

use crate::datasets::{airfoil, iris};
use crate::helpers::{assert_epsilon, get_forest, get_test_data};

#[test]
fn verify_regular_forest_accuracy_iris_800_trees() -> Result<()> {
    let forest =
        get_forest::<SerializedClassificationNode>("./tests/test-forests/forest_iris_800.csv")?;
    let test_data: Vec<iris::DataPoint> = get_test_data("./tests/test-data/iris.csv")?;

    for data_point in test_data {
        let features = data_point.transform_features(forest.features());
        let prediction = forest.predict(&features);
        assert_eq!(prediction, data_point.forest_prediction);
    }

    Ok(())
}

#[test]
fn verify_regular_forest_accuracy_airfoil_100_trees() -> Result<()> {
    let forest =
        get_forest::<SerializedRegressionNode>("./tests/test-forests/airfoil_100_200.csv")?;
    let test_data: Vec<airfoil::DataPoint> = get_test_data("./tests/test-data/airfoil.csv")?;

    for data_point in test_data {
        let features = data_point.transform_features(forest.features());
        let prediction = forest.predict(&features);
        assert_epsilon(prediction, data_point.forest_prediction, 2.5);
    }

    Ok(())
}

#[test]
fn verify_optimized_forest_accuracy_iris_880_trees() -> Result<()> {
    let forest =
        get_forest::<SerializedClassificationNode>("./tests/test-forests/forest_iris_800.csv")?;

    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::<Classification>::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        forest.num_features().try_into().unwrap(),
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    optimized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

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
fn verify_optimized_forest_accuracy_airfoil_100_trees() -> Result<()> {
    let forest =
        get_forest::<SerializedRegressionNode>("./tests/test-forests/airfoil_100_200.csv")?;

    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::<Regression>::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        forest.num_features().try_into().unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    optimized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let test_data: Vec<airfoil::DataPoint> = get_test_data("./tests/test-data/airfoil.csv")?;

    for data_point in test_data {
        let features = data_point.transform_features(forest.features());
        let prediction = optimized.predict(&features);
        assert_epsilon(prediction, data_point.forest_prediction, 2.5);
    }

    Ok(())
}
