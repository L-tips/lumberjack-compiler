use color_eyre::{Result, eyre::eyre};
use embedded_rforest::forest::{Classification, OptimizedForest, Regression};
use forest_optimizer::serialized_forest::{SerializedClassificationNode, SerializedRegressionNode};

use crate::helpers::get_forest;

#[test]
fn serialized_classification_rejects_regression_deserialization() -> Result<()> {
    let forest =
        get_forest::<SerializedClassificationNode>("./tests/test-forests/forest_iris_5.csv")
            .unwrap();

    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::<Classification>::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        forest.num_features().try_into().unwrap(),
        Classification::new(forest.num_targets().try_into().unwrap()).unwrap(),
    )
    .unwrap();

    optimized
        .verify()
        .map_err(|_| eyre!("Malformed forest detected upon verification"))?;

    let serialized = optimized.to_bytes();
    assert!(OptimizedForest::<Regression>::deserialize(&serialized).is_err());

    Ok(())
}

#[test]
fn serialized_classification_rejects_regression_optimization() {
    let forest =
        get_forest::<SerializedClassificationNode>("./tests/test-forests/forest_iris_5.csv")
            .unwrap();

    let nodes = forest.optimize_nodes();
    assert!(
        OptimizedForest::<Regression>::new(
            forest.num_trees().try_into().unwrap(),
            &nodes,
            forest.num_features().try_into().unwrap(),
        )
        .is_err()
    );
}

#[test]
fn serialized_regression_rejects_classification_deserialization() -> Result<()> {
    let forest =
        get_forest::<SerializedRegressionNode>("./tests/test-forests/airfoil_100_200.csv").unwrap();

    let nodes = forest.optimize_nodes();
    let optimized = OptimizedForest::<Regression>::new(
        forest.num_trees().try_into().unwrap(),
        &nodes,
        forest.num_features().try_into().unwrap(),
    )
    .unwrap();

    let serialized = optimized.to_bytes();
    assert!(OptimizedForest::<Classification>::deserialize(&serialized).is_err());

    Ok(())
}

#[test]
fn serialized_regression_rejects_classification_optimization() {
    let forest =
        get_forest::<SerializedRegressionNode>("./tests/test-forests/airfoil_100_200.csv").unwrap();

    let nodes = forest.optimize_nodes();
    assert!(
        OptimizedForest::<Classification>::new(
            forest.num_trees().try_into().unwrap(),
            &nodes,
            forest.num_features().try_into().unwrap(),
            Classification::new(2).unwrap(),
        )
        .is_err()
    );
}

#[test]
fn serialization_rejects_wrong_type() -> Result<()> {
    assert!(
        get_forest::<SerializedRegressionNode>("./tests/test-forests/forest_iris_5.csv").is_err()
    );

    assert!(
        get_forest::<SerializedClassificationNode>("./tests/test-forests/airfoil_100_200.csv")
            .is_err()
    );

    Ok(())
}
