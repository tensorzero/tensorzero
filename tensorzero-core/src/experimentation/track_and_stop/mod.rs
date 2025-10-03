use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
    time::Duration,
};

use estimate_optimal_probabilities::estimate_optimal_probabilities;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    db::{clickhouse::ClickHouseConnectionInfo, SelectQueries},
    error::Error,
    variant::VariantInfo,
};

use super::VariantSampler;

mod check_stopping;
mod estimate_optimal_probabilities;

const SLEEP_DURATION: Duration = Duration::from_secs(15 * 60);

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
    #[serde(default)]
    // this is undocumented please don't specify it, it will be overridden
    sampling_probabilities: Arc<RwLock<HashMap<String, f64>>>,
}

impl VariantSampler for TrackAndStopConfig {
    async fn setup(
        &self,
        clickhouse: &ClickHouseConnectionInfo,
        function_name: &str,
    ) -> Result<(), Error> {
        // First, let's write uniform probabilities to the `sampling_probabilites`
        let mut sampling_probabilities = self.sampling_probabilities.write().unwrap();
        for variant in self.candidate_variants.iter() {
            sampling_probabilities
                .insert(variant.clone(), 1.0 / self.candidate_variants.len() as f64);
        }
        // Next, let's spawn a task to estimate the optimal probabilities
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

async fn probability_update_task(
    clickhouse: ClickHouseConnectionInfo,
    metric_name: String,
    function_name: String,
    sampling_probabilities: Arc<RwLock<HashMap<String, f64>>>,
) {
    loop {
        let result = clickhouse
            .get_feedback_by_variant(&metric_name, &function_name, None)
            .await;
        let Ok(variant_performances) = result else {
            // sleep some
            todo!()
        };
        // TODO: block in place
        let updated_sampling_probabilities = tokio::task::spawn_blocking(move || {
            estimate_optimal_probabilities(variant_performances, None, None, None, None).unwrap()
        })
        .await
        .unwrap();
        let mut sampling_probabilities_write = sampling_probabilities.write().unwrap();
        *sampling_probabilities_write = updated_sampling_probabilities;
        // TODO: sleep
    }
}
