use std::path::Path;

use color_eyre::Result;

use lumberjack_compiler::compiler::ForestModel;
use lumberjack_compiler::csv_forest::CsvForest;

pub fn get_forest(path: impl AsRef<Path>) -> Result<ForestModel> {
    let serialized = CsvForest::read(path.as_ref())?;
    serialized.into_forest_model()
}
