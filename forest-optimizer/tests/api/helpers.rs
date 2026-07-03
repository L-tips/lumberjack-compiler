use std::path::Path;

use color_eyre::Result;

use forest_optimizer::csv_forest::CsvForest;
use forest_optimizer::forest::Forest;
use serde::de::DeserializeOwned;

pub fn get_forest(path: impl AsRef<Path>) -> Result<Forest> {
    let serialized = CsvForest::read(path.as_ref())?;
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
