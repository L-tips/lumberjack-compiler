use std::{iter::once, path::Path, str::FromStr};

use color_eyre::{
    Result,
    eyre::{Context, eyre},
};
use half::bf16;

use crate::{Feature, compiler::InterRep, problem::ProblemDefinition};

pub struct DataPoint<F: Feature> {
    pub features: Vec<F>,
    /// Prediction as given by the reference software model
    pub reference_prediction: u16,
}

/// Read a list of feature vectors from a CSV file. Must contain a column
/// named `prediction`, which is the predicted class from the trained model.
/// This can be used to verify the fidelity of the compiled model.
pub fn features_vector_from_csv<F>(
    problem: &ProblemDefinition,
    csv_path: impl AsRef<Path>,
) -> Result<Vec<DataPoint<F>>>
where
    F: Feature + FromStr,
    <F as std::str::FromStr>::Err: Send + Sync + std::error::Error + 'static,
{
    let features_map = problem.features();
    let targets_map = problem.targets();

    let mut rdr = csv::ReaderBuilder::new()
        .from_path(csv_path.as_ref())
        .context(format!(
            "failed to open CSV: {}",
            csv_path.as_ref().display()
        ))?;

    let headers = rdr.headers().context("failed to read CSV headers")?;

    // Find prediction column if needed
    let prediction_col_idx = headers
        .iter()
        .position(|h| h == "prediction")
        .ok_or_else(|| eyre!("CSV must contain a 'prediction' column"))?;

    // Build a mapping from CSV column index to feature index
    let mut col_to_feature_idx: Vec<Option<u16>> = vec![None; headers.len()];
    for (csv_col_idx, header) in headers.iter().enumerate() {
        if header != "prediction"
            && let Some(&feature_idx) = features_map.get(header)
        {
            col_to_feature_idx[csv_col_idx] = Some(feature_idx);
        }
    }

    let num_features = features_map.len();
    let mut row_data = Vec::new();

    for record in rdr.records() {
        let record = record.context("bad CSV row")?;
        let mut row = vec![F::ZERO; num_features];

        let pred_str = record
            .get(prediction_col_idx)
            .ok_or_else(|| eyre!("missing prediction value"))?;
        let pred_idx = targets_map
            .get(pred_str)
            .ok_or_else(|| eyre!("unknown target: {}", pred_str))?;

        for (csv_col_idx, value) in record.iter().enumerate() {
            if let Some(Some(feature_idx)) = col_to_feature_idx.get(csv_col_idx) {
                let val: F = value.parse().context(format!("invalid float: {value}"))?;
                row[*feature_idx as usize] = val;
            }
        }
        let data_point = DataPoint {
            features: row,
            reference_prediction: *pred_idx,
        };
        row_data.push(data_point);
    }

    if row_data.is_empty() {
        return Err(eyre!("sample CSV must contain at least one row",));
    }

    Ok(row_data)
}

pub fn write_test_vectors(
    model: &InterRep<bf16>,
    vectors: &[DataPoint<bf16>],
    out_path: impl AsRef<Path>,
) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .from_path(out_path.as_ref())
        .context("Could not open output file")?;

    let headers = (0..vectors[0].features.len())
        .into_iter()
        .map(|c| c.to_string())
        .chain(once("prediction".to_string()))
        .chain(once("num_votes".to_string()));

    writer.write_record(headers)?;

    for datapoint in vectors {
        let prediction = model.predict(&datapoint.features);
        let prediction_idx = model
            .problem()
            .targets()
            .get(prediction.0)
            .expect("Could not find target");

        let record = datapoint
            .features
            .iter()
            .map(|f| format!("{:#06x}", f.to_bits()))
            .chain(once(prediction_idx.to_string()))
            .chain(once(prediction.1.to_string()));
        writer.write_record(record)?;
    }

    Ok(())
}
