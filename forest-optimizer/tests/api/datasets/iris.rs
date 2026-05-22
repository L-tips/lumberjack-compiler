use std::collections::HashMap;

use half::bf16;

/// Datapoints and forest generated using the `iris` R sample dataset
#[derive(serde::Deserialize, Debug)]
pub(crate) struct DataPoint {
    #[serde(rename = "Sepal.Length")]
    pub sepal_length: f32,
    #[serde(rename = "Petal.Length")]
    pub petal_length: f32,
    #[serde(rename = "Sepal.Width")]
    pub sepal_width: f32,
    #[serde(rename = "Petal.Width")]
    pub petal_width: f32,
    #[serde(rename = "Species")]
    #[expect(dead_code)]
    pub true_species: String,
    #[serde(rename = "Predicted")]
    pub forest_prediction: String,
}

impl DataPoint {
    pub fn transform_features(&self, feature_map: &HashMap<String, u16>) -> [bf16; 4] {
        let mut features = [bf16::ZERO; 4];

        let feats = [
            (self.sepal_length, "Sepal.Length"),
            (self.petal_length, "Petal.Length"),
            (self.sepal_width, "Sepal.Width"),
            (self.petal_width, "Petal.Width"),
        ];

        for feat in feats {
            let position = feature_map.get(feat.1).unwrap();
            features[*position as usize] = bf16::from_f32(feat.0);
        }

        features
    }
}
