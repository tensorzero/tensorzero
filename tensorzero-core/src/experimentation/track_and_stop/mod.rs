use arc_swap::ArcSwap;
use check_stopping::{check_stopping, CheckStoppingArgs, StoppingResult};
use error::TrackAndStopError;
use sha2::digest::Update;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use estimate_optimal_probabilities::{
    estimate_optimal_probabilities, EstimateOptimalProbabilitiesArgs,
};
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
mod error;
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
    // Includes candidate and fallback variants
    all_variants: Vec<String>,
    min_samples_per_variant: u64,
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
    state: Arc<ArcSwap<TrackAndStopState>>,
}

#[derive(Debug)]
enum TrackAndStopState {
    Stopped {
        winner_variant_name: String,
    },
    NurseryOnly(Nursery),
    BanditsOnly {
        sampling_probabilities: HashMap<String, f64>,
    },
    NurseryAndBandits {
        nursery: Nursery,
        sampling_probabilities: HashMap<String, f64>,
    },
    NurseryAndStopped {
        nursery: Nursery,
        stopped_variant_name: String,
    },
}

#[derive(Debug)]
struct Nursery {
    variants: Vec<String>,
    index: AtomicU64,
}

impl Nursery {
    pub fn new(variants: Vec<String>) -> Self {
        Nursery {
            variants,
            index: AtomicU64::new(0),
        }
    }

    // increments the index and returns the variant name corresponding to the index
    pub fn get_variant_round_robin<'a>(&'a self) -> &'a str {
        let index = self.index.fetch_add(1, Ordering::Relaxed) % self.variants.len() as u64;
        &self.variants[index as usize]
    }
}

#[derive(Debug, Deserialize)]
pub struct UninitializedTrackAndStopConfig {
    metric: String,
    candidate_variants: Vec<String>,
    fallback_variants: Vec<String>,
    min_samples_per_variant: u64,
    delta: f64,
    epsilon: f64,
}

impl UninitializedTrackAndStopConfig {
    pub fn load(self, variants: &HashMap<String, Arc<VariantInfo>>) -> TrackAndStopConfig {
        TrackAndStopConfig {
            metric: self.metric,
            candidate_variants: self.candidate_variants,
            fallback_variants: self.fallback_variants,
            min_samples_per_variant: self.min_samples_per_variant,
            delta: self.delta,
            epsilon: self.epsilon,
            state: Arc::new(ArcSwap::new(Arc::new(
                TrackAndStopState::nursery_from_variants(variants.keys().cloned().collect()),
            ))),
        }
    }
}

impl VariantSampler for TrackAndStopConfig {
    async fn setup(
        &self,
        clickhouse: &ClickHouseConnectionInfo,
        function_name: &str,
    ) -> Result<(), Error> {
        // TODO: validate all fields
        // Spawn a task to estimate the optimal probabilities
        tokio::spawn(probability_update_task(ProbabilityUpdateTaskArgs {
            clickhouse: clickhouse.clone(),
            candidate_variants: self.candidate_variants.clone().into(),
            metric_name: self.metric.clone(),
            function_name: function_name.to_string(),
            sampling_probabilities: self.state.clone(),
            sleep_duration: SLEEP_DURATION,
            min_samples_per_variant: self.min_samples_per_variant,
            epsilon: self.epsilon,
            delta: self.delta,
        }));
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
        let mut rng = rand::rng();
        let total_active_probability = active_variants
            .keys()
            .map(|variant_name| sampling_probabilities.get(variant_name).unwrap_or(&0.0))
            .sum::<f64>();
        // TODO: handle the case where the total_active_probability is 0
        // Sample a uniform random value
        let random_threshold = rng.random::<f64>() * total_active_probability;
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

struct ProbabilityUpdateTaskArgs {
    clickhouse: ClickHouseConnectionInfo,
    candidate_variants: Arc<Vec<String>>,
    metric_name: String,
    function_name: String,
    sampling_probabilities: Arc<ArcSwap<TrackAndStopState>>,
    sleep_duration: Duration,
    min_samples_per_variant: u64,
    epsilon: f64,
    delta: f64,
}

async fn probability_update_task(args: ProbabilityUpdateTaskArgs) {
    let ProbabilityUpdateTaskArgs {
        clickhouse,
        candidate_variants,
        metric_name,
        function_name,
        sampling_probabilities,
        sleep_duration,
        min_samples_per_variant,
        epsilon,
        delta,
    } = args;

    loop {
        let result = update_probabilities(
            &clickhouse,
            &candidate_variants,
            &metric_name,
            &function_name,
            &sampling_probabilities,
            min_samples_per_variant,
            epsilon,
            delta,
        )
        .await;

        match result {
            Ok(()) => {}
            Err(e) => {
                tracing::warn!("Failed to update probabilities for {function_name}: {e}");
            }
        }

        tokio::time::sleep(sleep_duration).await;
    }
}

async fn update_probabilities(
    clickhouse: &ClickHouseConnectionInfo,
    candidate_variants: &Arc<Vec<String>>,
    metric_name: &str,
    function_name: &str,
    sampling_probabilities: &Arc<ArcSwap<TrackAndStopState>>,
    min_samples_per_variant: u64,
    epsilon: f64,
    delta: f64,
) -> Result<(), TrackAndStopError> {
    // Fetch feedback from database
    let variant_performances = clickhouse
        .get_feedback_by_variant(metric_name, function_name, None)
        .await?;

    // Compute new state in blocking task (CPU-bound work)
    let candidate_variants_clone = candidate_variants.clone();
    let new_state = tokio::task::spawn_blocking(move || {
        TrackAndStopState::new(
            &candidate_variants_clone,
            variant_performances,
            min_samples_per_variant,
            delta,
            epsilon,
        )
    })
    .await??; // First ? for JoinError, second ? for TrackAndStopError

    // Store the new state
    sampling_probabilities.store(Arc::new(new_state));

    Ok(())
}

/// For each variant in `candidate_variants`, get the count from variant_performances if it exists.
/// If it doesn't exist, return 0.
fn get_count_by_variant<'a>(
    candidate_variants: &'a Vec<String>,
    variant_performances: &Vec<FeedbackByVariant>,
) -> HashMap<&'a str, usize> {
    candidate_variants
        .iter()
        .map(|variant| {
            let count = variant_performances
                .iter()
                .filter(|p| &p.variant_name == variant)
                .count();
            (variant.as_str(), count)
        })
        .collect()
}

impl TrackAndStopState {
    /// For a quick initialization of a TrackAndStopState instance with only a nursery.
    /// Should be called on startup.
    fn nursery_from_variants(variants: Vec<String>) -> Self {
        TrackAndStopState::NurseryOnly(Nursery::new(variants))
    }

    /// Initializes a new TrackAndStopState instance based on the current statistics
    /// and configured parameters.
    /// NOTE: This function may do some CPU-bound work to compute probabilities
    fn new(
        candidate_variants: &Vec<String>,
        variant_performances: Vec<FeedbackByVariant>,
        min_samples_per_variant: u64,
        delta: f64,
        epsilon: f64,
    ) -> Result<Self, TrackAndStopError> {
        let variant_counts = get_count_by_variant(candidate_variants, &variant_performances);
        let num_variants_above_cutoff = variant_counts
            .values()
            .filter(|&count| *count as u64 >= min_samples_per_variant)
            .count();
        let num_variants_below_cutoff = variant_counts.len() - num_variants_above_cutoff;
        let need_nursery = num_variants_below_cutoff > 0;
        let need_bandits = num_variants_above_cutoff >= 2;
        match (need_nursery, need_bandits) {
            (true, false) => Ok(TrackAndStopState::NurseryOnly(Nursery::new(
                candidate_variants.clone(),
            ))),
            (false, true) => {
                // Check for stopping using all variants
                match check_stopping(CheckStoppingArgs {
                    feedback: &variant_performances,
                    min_pulls: min_samples_per_variant,
                    ridge_variance: None,
                    delta: Some(delta),
                    epsilon: Some(epsilon),
                })? {
                    StoppingResult::Winner(winner_variant_name) => Ok(TrackAndStopState::Stopped {
                        winner_variant_name,
                    }),
                    StoppingResult::NotStopped => Ok(TrackAndStopState::BanditsOnly {
                        sampling_probabilities: estimate_optimal_probabilities(
                            EstimateOptimalProbabilitiesArgs {
                                feedback: variant_performances,
                                epsilon: Some(epsilon),
                                ridge_variance: None,
                                min_prob: None,
                                reg0: None,
                            },
                        )?,
                    }),
                }
            }
            (false, false) => {
                // this case implies there are zero variants
                // or one with > min_samples_per_variant samples
                // Since we validate there are > 0 variants we must have a single variant
                // In this case, we'll simply use it
                Ok(TrackAndStopState::Stopped {
                    winner_variant_name: candidate_variants[0].clone(),
                })
            }
            (true, true) => {
                // If we need both a nursery and a bandit
                // we can separate them by filtering the variants based on their counts
                let nursery_variants: Vec<String> = variant_counts
                    .iter()
                    .filter(|(_, &count)| (count as u64) < min_samples_per_variant)
                    .map(|(key, _)| key.to_string())
                    .collect();
                let bandit_feedback: Vec<FeedbackByVariant> = variant_performances
                    .into_iter()
                    .filter(|feedback| feedback.count >= min_samples_per_variant)
                    .collect();

                match check_stopping(CheckStoppingArgs {
                    feedback: &bandit_feedback,
                    min_pulls: min_samples_per_variant,
                    ridge_variance: None,
                    delta: Some(delta),
                    epsilon: Some(epsilon),
                })? {
                    StoppingResult::Winner(winner) => Ok(TrackAndStopState::NurseryAndStopped {
                        nursery: Nursery::new(nursery_variants),
                        stopped_variant_name: winner,
                    }),
                    StoppingResult::NotStopped => Ok(TrackAndStopState::NurseryAndBandits {
                        nursery: Nursery::new(nursery_variants),
                        sampling_probabilities: estimate_optimal_probabilities(
                            EstimateOptimalProbabilitiesArgs {
                                feedback: bandit_feedback,
                                epsilon: Some(epsilon),
                                ridge_variance: None,
                                min_prob: None,
                                reg0: None,
                            },
                        )?,
                    }),
                }
            }
        }
    }

    // Note: this function does __not__ pop
    fn sample<'a>(
        &self,
        &'a active_variants: BTreeMap<String, Arc<VariantInfo>>,
    ) -> Option<&'a str> {
        todo!()
    }
}
