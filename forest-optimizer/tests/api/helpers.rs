use std::path::Path;

use color_eyre::Result;

use forest_optimizer::forest::Forest;
use forest_optimizer::serialized_forest::{SerializedForest, SerializedNode};
use serde::de::DeserializeOwned;

pub fn get_forest<N: SerializedNode>(path: impl AsRef<Path>) -> Result<Forest<N::ProblemType>> {
    let serialized = SerializedForest::<N>::read(path.as_ref())?;
    Forest::from_serialized(serialized)
}

pub fn get_test_data<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let mut reader = csv::Reader::from_path(path.as_ref())?;
    let mut data = Vec::new();
    for result in reader.deserialize() {
        data.push(result?);
    }

    Ok(data)
}
