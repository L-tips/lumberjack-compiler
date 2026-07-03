use std::path::Path;

use color_eyre::Result;

use lumberjack_compiler::csv_forest::CsvForest;
use lumberjack_compiler::forest_model::ForestModel;
use serde::de::DeserializeOwned;

pub fn get_forest(path: impl AsRef<Path>) -> Result<ForestModel> {
    let serialized = CsvForest::read(path.as_ref())?;
    serialized.into_forest_model()
}

pub fn get_test_data<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let mut reader = csv::Reader::from_path(path.as_ref())?;
    let mut data = Vec::new();
    for result in reader.deserialize() {
        data.push(result?);
    }

    Ok(data)
}
