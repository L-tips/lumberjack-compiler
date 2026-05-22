use std::collections::HashMap;

use half::bf16;

/// Datapoints and forest generated using the `iris` R sample dataset
#[derive(serde::Deserialize, Debug)]
pub(crate) struct DataPoint {
    pub alpha: f32,
    pub c: f32,
    #[serde(rename = "U_infinity")]
    pub u_inf: f32,
    pub delta: f32,
    #[serde(rename = "SSPL")]
    pub sspl: f32,

    #[serde(rename = "f")]
    #[expect(dead_code)]
    pub true_f: f32,
    #[serde(rename = "Predicted")]
    pub forest_prediction: f32,
}

impl DataPoint {
    pub fn transform_features(&self, feature_map: &HashMap<String, u16>) -> [bf16; 5] {
        let mut features = [bf16::ZERO; 5];

        let feats = [
            (self.alpha, "alpha"),
            (self.c, "c"),
            (self.u_inf, "U_infinity"),
            (self.delta, "delta"),
            (self.sspl, "SSPL"),
        ];

        for feat in feats {
            let position = feature_map.get(feat.1).unwrap();
            features[*position as usize] = bf16::from_f32(feat.0);
        }

        features
    }
}
