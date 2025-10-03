use arc_swap::ArcSwap;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
    time::Duration,
};

use estimate_optimal_probabilities::estimate_optimal_probabilities;
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    db::{
        clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo, FeedbackByVariant,
        SelectQueries,
    },
    error::Error,
    variant::VariantInfo,
};

use super::VariantSampler;

mod check_stopping;
mod estimate_optimal_probabilities;

const SLEEP_DURATION: Duration = Duration::from_secs(15 * 60);
const NURSERY_PROBABILITY: f64 = 0.1; // placeholder, can decide later

#[derive(Debug, Serialize)]
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
    // #[serde(default)]
    // this is undocumented please don't specify it, it will be overridden
    // TODO: use an UnitializedTrackAndStopConfig for config parsing
    // TODO: use an enum instead of HashMap<String, f64>
    // This enum should contain two states: Stopped and Running
    // in the Stopped state we should simply sample the winner
    // in the Running state we should have a Nursery that contains the variants that are below min pull count and an AtomicU64
    //  or, if there is a single variant with above that pull count, it as well (since we need >= 2 to do the track-and-stop)
    // we should also have a set of sampling probabilites HashMap<String, f64>
    // if the both the nursery and the sampling probabilities are nonempty we should sample using
    // round-robin from the nursery with probability NURSERY_PROBABILITY
    // and from the sampling probabilities with probability 1 - NURSERY_PROBABILITY
    // if it is empty we should sample using the sampling probabilities always
    // if the sampling probabilities are empty we should sample using the nursery always
    #[serde(skip)]
    state: Arc<ArcSwap<HashMap<String, f64>>>,
}

#[derive(Debug, Deserialize)]
pub struct UninitializedTrackAndStopConfig {
    metric: String,
    candidate_variants: Vec<String>,
    fallback_variants: Vec<String>,
    min_samples_per_variant: usize,
    delta: f64,
    epsilon: f64,
}

impl UninitializedTrackAndStopConfig {
    pub fn load(self) -> TrackAndStopConfig {
        TrackAndStopConfig {
            metric: self.metric,
            candidate_variants: self.candidate_variants,
            fallback_variants: self.fallback_variants,
            min_samples_per_variant: self.min_samples_per_variant,
            delta: self.delta,
            epsilon: self.epsilon,
            state: Arc::new(ArcSwap::new(Arc::new(HashMap::new()))),
        }
    }
}

impl VariantSampler for TrackAndStopConfig {
    async fn setup(
        &self,
        clickhouse: &ClickHouseConnectionInfo,
        function_name: &str,
    ) -> Result<(), Error> {
        // First, let's write uniform probabilities to the `sampling_probabilites`
        // TODO: consider eagerly computing the sampling probabilities on startup
        // or simply verifying that clickhouse is healthy
        let mut sampling_probabilities = HashMap::new();
        for variant in self.candidate_variants.iter() {
            sampling_probabilities
                .insert(variant.clone(), 1.0 / self.candidate_variants.len() as f64);
        }
        self.state.store(Arc::new(sampling_probabilities));
        let variant_performances = clickhouse
            .get_feedback_by_variant(&self.metric, &function_name, None)
            .await?;
        // Next, let's spawn a task to estimate the optimal probabilities
        tokio::spawn(probability_update_task(
            clickhouse.clone(),
            self.metric.clone(),
            function_name.to_string(),
            self.state.clone(),
            variant_performances,
            SLEEP_DURATION,
        ));
        Ok(())
    }
    async fn sample(
        &self,
        function_name: &str,
        episode_id: Uuid,
        active_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
        postgres: &PostgresConnectionInfo,
    ) -> Result<(String, Arc<VariantInfo>), Error> {
        let sampling_probabilities = self.state.load();
        let mut rng = rand::thread_rng();
        let total_active_probability = active_variants
            .keys()
            .map(|variant_name| sampling_probabilities.get(variant_name).unwrap_or(&0.0))
            .sum::<f64>();
        // TODO: handle the case where the total_active_probability is 0
        // Sample a uniform random value
        let random_threshold = rng.gen() * total_active_probability;
        let variant_to_remove = {
            let mut cumulative_weight = 0.0;
            active_variants
                .iter()
                .find(|(variant_name, _)| {
                    cumulative_weight += sampling_probabilities
                        .get(variant_name.as_str())
                        .unwrap_or(&0.0);
                    cumulative_weight > random_threshold
                })
                .map(|(name, _)| name.clone()) // Clone the key
        };

        //
        todo!()
    }
}

async fn probability_update_task(
    clickhouse: ClickHouseConnectionInfo,
    metric_name: String,
    function_name: String,
    sampling_probabilities: Arc<ArcSwap<HashMap<String, f64>>>,
    mut variant_performances: Vec<FeedbackByVariant>,
    sleep_duration: Duration,
) {
    loop {
        // TODO: check the stopping condition here
        // TODO: handle variants with too few pulls
        let updated_sampling_probabilities = tokio::task::spawn_blocking(move || {
            estimate_optimal_probabilities(variant_performances, None, None, None, None).unwrap()
        })
        .await
        .unwrap();
        sampling_probabilities.store(Arc::new(updated_sampling_probabilities));
        tokio::time::sleep(sleep_duration).await;
        // TODO: make this a while loop that maybe polls more often
        let result = clickhouse
            .get_feedback_by_variant(&metric_name, &function_name, None)
            .await;
        variant_performances = match result {
            Ok(new_data) => new_data,
            Err(_) => {
                todo!()
            }
        };
    }
}
