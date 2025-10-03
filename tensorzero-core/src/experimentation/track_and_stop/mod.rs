use std::{collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{error::Error, variant::VariantInfo};

use super::VariantSampler;

mod check_stopping;
mod estimate_optimal_probabilities;

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct TrackAndStopConfig {
    // TODO: validate all of these fields
    metric: String,
    candidate_variants: Vec<String>,
    fallback_variants: Vec<String>,
    min_samples_per_variant: usize,
    delta: f64,
    epsilon: f64,
}

impl VariantSampler for TrackAndStopConfig {
    async fn setup(&self) -> Result<(), Error> {
        todo!()
    }
    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        todo!()
    }
}
