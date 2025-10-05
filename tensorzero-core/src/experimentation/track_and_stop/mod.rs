use arc_swap::ArcSwap;
use check_stopping::{check_stopping, CheckStoppingArgs, StoppingResult};
use error::TrackAndStopError;
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
        clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo,
        ExperimentationQueries, FeedbackByVariant, SelectQueries,
    },
    error::{Error, ErrorDetails},
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
    fallback_variants: Vec<String>,
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
    fn get_variant_round_robin(&self) -> &str {
        let index = self.index.fetch_add(1, Ordering::Relaxed) % self.variants.len() as u64;
        &self.variants[index as usize]
    }

    /// Try to sample an active variant from the nursery using round-robin.
    /// Tries up to N times (where N = number of variants) to find one in active_variants.
    /// Returns None if no intersection exists.
    pub fn sample_active<'a>(
        &'a self,
        active_variants: &'a BTreeMap<String, Arc<VariantInfo>>,
    ) -> Option<&'a str> {
        // Try up to N times to find an active variant
        for _ in 0..self.variants.len() {
            let variant = self.get_variant_round_robin();
            if active_variants.contains_key(variant) {
                return Some(variant);
            }
        }
        None
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
        let mut all_variants = self.candidate_variants.clone();
        all_variants.extend(self.fallback_variants.clone());
        TrackAndStopConfig {
            metric: self.metric,
            candidate_variants: self.candidate_variants,
            all_variants,
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
        let state = self.state.load();

        // Generate random value and drop the RNG before any await points
        let uniform_sample = {
            let mut rng = rand::rng();
            rng.random::<f64>()
        };

        // Try to sample from the current state
        let variant_name =
            if let Some(candidate_name) = state.sample(active_variants, uniform_sample) {
                // Check and set the variant in Postgres (ensures consistency for the episode)
                let set_variant = postgres
                    .check_and_set_variant_by_episode(episode_id, function_name, candidate_name)
                    .await?;

                // Check if the returned variant is active
                if active_variants.contains_key(&set_variant) {
                    set_variant
                } else {
                    // The variant that was already set for this episode is not active, fall back
                    fallback_sample(active_variants, &self.fallback_variants, uniform_sample)?
                }
            } else {
                // State couldn't provide a variant, fall back to uniform sampling from fallback_variants
                fallback_sample(active_variants, &self.fallback_variants, uniform_sample)?
            };

        // Remove and return the sampled variant
        active_variants.remove_entry(&variant_name).ok_or_else(|| {
            Error::new(ErrorDetails::Inference {
                message: format!(
                    "Sampled variant {variant_name} not found in active_variants. This should never happen."
                ),
            })
        })
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
        let result = update_probabilities(UpdateProbabilitiesArgs {
            clickhouse: &clickhouse,
            candidate_variants: &candidate_variants,
            metric_name: &metric_name,
            function_name: &function_name,
            sampling_probabilities: &sampling_probabilities,
            min_samples_per_variant,
            epsilon,
            delta,
        })
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

struct UpdateProbabilitiesArgs<'a> {
    clickhouse: &'a ClickHouseConnectionInfo,
    candidate_variants: &'a Arc<Vec<String>>,
    metric_name: &'a str,
    function_name: &'a str,
    sampling_probabilities: &'a Arc<ArcSwap<TrackAndStopState>>,
    min_samples_per_variant: u64,
    epsilon: f64,
    delta: f64,
}

async fn update_probabilities(args: UpdateProbabilitiesArgs<'_>) -> Result<(), TrackAndStopError> {
    let UpdateProbabilitiesArgs {
        clickhouse,
        candidate_variants,
        metric_name,
        function_name,
        sampling_probabilities,
        min_samples_per_variant,
        epsilon,
        delta,
    } = args;

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
    candidate_variants: &'a [String],
    variant_performances: &[FeedbackByVariant],
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

/// Perform uniform sampling from fallback_variants that are in active_variants.
/// Returns an error if no fallback variants are active.
fn fallback_sample(
    active_variants: &BTreeMap<String, Arc<VariantInfo>>,
    fallback_variants: &[String],
    uniform_sample: f64,
) -> Result<String, Error> {
    let intersection: Vec<&String> = active_variants
        .keys()
        .filter(|variant_name| fallback_variants.contains(variant_name))
        .collect();

    if intersection.is_empty() {
        return Err(ErrorDetails::NoFallbackVariantsRemaining.into());
    }

    let random_index = (uniform_sample * intersection.len() as f64).floor() as usize;
    intersection
        .get(random_index)
        .ok_or_else(|| {
            Error::new(ErrorDetails::Inference {
                message:
                    "Failed to sample variant from nonempty intersection. This should never happen."
                        .to_string(),
            })
        })
        .map(std::string::ToString::to_string)
}

/// Sample a variant from active_variants using weighted probabilities.
/// Returns None if no active variant has positive probability.
fn sample_with_probabilities<'a>(
    active_variants: &'a BTreeMap<String, Arc<VariantInfo>>,
    sampling_probabilities: &HashMap<String, f64>,
    uniform_sample: f64,
) -> Option<&'a str> {
    // Compute the total probability of active variants
    let total_probability: f64 = active_variants
        .keys()
        .map(|variant_name| sampling_probabilities.get(variant_name).unwrap_or(&0.0))
        .sum();

    if total_probability <= 0.0 {
        return None;
    }

    // Use weighted sampling
    let random_threshold = uniform_sample * total_probability;
    let mut cumulative_probability = 0.0;

    active_variants
        .keys()
        .find(|variant_name| {
            cumulative_probability += sampling_probabilities
                .get(variant_name.as_str())
                .unwrap_or(&0.0);
            cumulative_probability > random_threshold
        })
        .map(std::string::String::as_str)
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
        candidate_variants: &[String],
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
                candidate_variants.to_vec(),
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
        &'a self,
        active_variants: &'a BTreeMap<String, Arc<VariantInfo>>,
        uniform_sample: f64,
    ) -> Option<&'a str> {
        match self {
            TrackAndStopState::Stopped {
                winner_variant_name,
            } => {
                if active_variants.contains_key(winner_variant_name) {
                    Some(winner_variant_name)
                } else {
                    None
                }
            }
            TrackAndStopState::NurseryOnly(nursery) => {
                // Do round-robin sampling from the variants until we find one that is active
                // If there is no intersection, return none
                nursery.sample_active(active_variants)
            }
            TrackAndStopState::NurseryAndBandits {
                sampling_probabilities,
                nursery,
            } => {
                // With probability `NURSERY_PROBABILITY`, sample from the nursery using
                // round-robin sampling
                // Otherwise sample from the bandits using probability sampling
                if uniform_sample < NURSERY_PROBABILITY {
                    nursery.sample_active(active_variants)
                } else {
                    sample_with_probabilities(
                        active_variants,
                        sampling_probabilities,
                        uniform_sample,
                    )
                }
            }
            TrackAndStopState::BanditsOnly {
                sampling_probabilities,
            } => {
                // Sample from the active bandits using probability sampling
                sample_with_probabilities(active_variants, sampling_probabilities, uniform_sample)
            }
            TrackAndStopState::NurseryAndStopped {
                nursery,
                stopped_variant_name,
            } => {
                // with probability `NURSERY_PROBABILITY`, sample from the nursery using
                // round-robin sampling until we find one that is active
                // if there is no intersection, return none
                //
                // with probability 1 - NURSERY_PROBABILITY,
                // return the stopped variant name if it's active and None otherwise
                if uniform_sample < NURSERY_PROBABILITY {
                    nursery.sample_active(active_variants)
                } else if active_variants.contains_key(stopped_variant_name) {
                    Some(stopped_variant_name)
                } else {
                    None
                }
            }
        }
    }
}
