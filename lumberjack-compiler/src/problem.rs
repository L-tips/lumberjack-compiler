use std::{fmt::Debug, path::Path};

use color_eyre::eyre::{Context, eyre};
use half::bf16;

pub type Map = indexmap::IndexMap<String, u16>;

pub struct DataPoint {
    pub features: Vec<bf16>,
    /// Prediction as given by the trained model
    pub trained_prediction: u16,
    /// Prediction as given by the imported model (after quantization)
    pub forest_prediction: u16,
}

#[derive(Default, Clone, Debug)]
pub struct Problem {
    targets: Map,
    features: Map,
}

impl Problem {
    pub fn targets(&self) -> &Map {
        &self.targets
    }

    pub fn features(&self) -> &Map {
        &self.features
    }

    pub(crate) fn features_mut(&mut self) -> &mut Map {
        &mut self.features
    }

    pub(crate) fn targets_mut(&mut self) -> &mut Map {
        &mut self.targets
    }

    /// Read a list of feature vectors from a CSV file. Must contain a column
    /// named `prediction`, which is the predicted class from the trained model.
    /// This can be used to verify the fidelity of the compiled model.
    ///
    /// # Returns
    ///
    /// - A `Vec<(Vec<bf16>, u16)>`, the `Vec<bf16>` being the feature vector
    ///   ordered by feature index, and the `u16` being the trained model's
    ///   prediction.
    pub fn features_vector_from_csv(
        &self,
        csv_path: impl AsRef<Path>,
    ) -> color_eyre::Result<Vec<(Vec<bf16>, u16)>> {
        let features_map = self.features();
        let targets_map = self.targets();

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
        let mut row_data: Vec<(Vec<bf16>, u16)> = Vec::new();

        for record in rdr.records() {
            let record = record.context("bad CSV row")?;
            let mut row = vec![bf16::ZERO; num_features];

            let pred_str = record
                .get(prediction_col_idx)
                .ok_or_else(|| eyre!("missing prediction value"))?;
            let pred_idx = targets_map
                .get(pred_str)
                .ok_or_else(|| eyre!("unknown target: {}", pred_str))?;

            for (csv_col_idx, value) in record.iter().enumerate() {
                if let Some(Some(feature_idx)) = col_to_feature_idx.get(csv_col_idx) {
                    let val: bf16 = value.parse().context(format!("invalid float: {value}"))?;
                    row[*feature_idx as usize] = val;
                }
            }
            row_data.push((row, *pred_idx));
        }

        if row_data.is_empty() {
            return Err(eyre!("sample CSV must contain at least one row",));
        }

        Ok(row_data)
    }
}
