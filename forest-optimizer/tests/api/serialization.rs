use color_eyre::Result;
use color_eyre::eyre::eyre;
use embedded_rforest::forest::{Classification, OptimizedForest, Predict, Regression};
use forest_optimizer::serialized_forest::{SerializedClassificationNode, SerializedRegressionNode};

use crate::datasets::{airfoil, iris};
use crate::helpers::{assert_epsilon, get_forest, get_test_data};

#[test]
fn serialized_then_deserialized_classification_tree_is_accurate() -> Result<()> {
    let forest =
        get_forest::<SerializedClassificationNode>("./tests/test-forests/forest_iris_5.csv")?;

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

    let serialized = optimized.to_bytes();
    let optimized = OptimizedForest::<Classification>::deserialize(&serialized)
        .map_err(|_| eyre!("Malfomed forest"))?;

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
#[ignore]
fn serialized_then_deserialized_regression_tree_is_accurate() -> Result<()> {
    let forest = get_forest::<SerializedRegressionNode>("./tests/test-forests/airfoil_100_50.csv")?;

    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::<Regression>::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        forest.num_features().try_into().unwrap(),
    )
    .map_err(|_| eyre!("Malformed forest"))?;

    let serialized = optimized.to_bytes();
    let optimized = OptimizedForest::<Regression>::deserialize(&serialized)
        .map_err(|_| eyre!("Malfomed forest"))?;

    optimized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let test_data: Vec<airfoil::DataPoint> = get_test_data("./tests/test-data/airfoil.csv")?;

    for data_point in test_data {
        let features = data_point.transform_features(forest.features());
        let prediction = optimized.predict(&features);
        assert_epsilon(prediction.to_f32(), data_point.forest_prediction, 2.5);
    }

    Ok(())
}

#[test]
fn classification_static_storage_deserializes_correctly() -> Result<()> {
    let buf = embedded_rforest::static_storage!("../test-forests/forest_iris_5.rforest");

    let forest =
        get_forest::<SerializedClassificationNode>("./tests/test-forests/forest_iris_5.csv")?;

    let deserialized = OptimizedForest::<Classification>::deserialize(buf)
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

#[test]
#[ignore]
fn regression_static_storage_deserializes_correctly() -> Result<()> {
    let buf = embedded_rforest::static_storage!("../test-forests/airfoil_100_50.rforest");

    let forest = get_forest::<SerializedRegressionNode>("./tests/test-forests/airfoil_100_50.csv")?;

    let deserialized =
        OptimizedForest::<Regression>::deserialize(buf).map_err(|_| eyre!("Malformed forest"))?;

    deserialized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let test_data: Vec<airfoil::DataPoint> = get_test_data("./tests/test-data/airfoil.csv")?;

    for data_point in test_data {
        let features = data_point.transform_features(forest.features());
        let prediction = deserialized.predict(&features);
        assert_epsilon(prediction.to_f32(), data_point.forest_prediction, 2.5);
    }

    Ok(())
}
