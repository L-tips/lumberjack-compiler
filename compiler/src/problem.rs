use std::fmt::Debug;

pub type Map = indexmap::IndexMap<String, u16>;

#[derive(Default, Clone, Debug)]
pub struct ProblemDefinition {
    targets: Map,
    features: Map,
}

impl ProblemDefinition {
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
}
