use std::path::Path;

use color_eyre::Result;

use lumberjack_compiler::compiler::IntermediateRepresentation;
use lumberjack_compiler::csv_source::CsvSource;

/// Parse a CSV source into an [`IntermediateRepresentation`], given a path to
/// the source.
pub fn parse_source(path: impl AsRef<Path>) -> Result<IntermediateRepresentation<f32>> {
    let serialized = CsvSource::read(path.as_ref())?;
    serialized.lower_to_ir()
}
