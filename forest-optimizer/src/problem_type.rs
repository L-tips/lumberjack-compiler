use std::{
    collections::HashMap,
    fmt::{Debug, Display},
};

pub type Map = HashMap<String, u16>;

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum PredictionType {
    #[serde(alias = "classification")]
    Classification,
    #[serde(alias = "regression")]
    Regression,
}

pub trait ProblemType: Default + Clone {
    type Output: Debug + Display + Copy;
    type OptimizedType: embedded_rforest::forest::ProblemType;

    const TYPE: PredictionType;

    fn features(&self) -> &Map;

    fn features_mut(&mut self) -> &mut Map;
}

#[derive(Default, Clone, Debug)]
pub struct Classification {
    targets: Map,
    features: Map,
}

impl Classification {
    pub fn targets(&self) -> &Map {
        &self.targets
    }

    pub(crate) fn targets_mut(&mut self) -> &mut Map {
        &mut self.targets
    }
}

impl ProblemType for Classification {
    type Output = u16;
    type OptimizedType = embedded_rforest::forest::Classification;

    const TYPE: PredictionType = PredictionType::Classification;

    fn features(&self) -> &Map {
        &self.features
    }

    fn features_mut(&mut self) -> &mut Map {
        &mut self.features
    }
}
