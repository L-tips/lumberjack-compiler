use std::{collections::HashMap, fmt::Debug};

pub type Map = HashMap<String, u16>;

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
}
