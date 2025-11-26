//! Track-and-Stop experimentation strategy for adaptive A/B testing.
//!
//! This module implements a bandit-based approach to variant selection that dynamically
//! adjusts sampling probabilities based on observed performance metrics. It can automatically
//! detect when a clear winner emerges and stop the experiment.
//!
//! # How it works
//!
//! 1. **Nursery Phase**: Variants start in a "nursery" where they're sampled round-robin
//!    until they reach the minimum sample threshold (`min_samples_per_variant`).
//!
//! 2. **Bandit Phase**: Once variants have sufficient data, the system estimates the asymptotically
//!    optimal sampling probabilities for the arms (see `estimate_optimal_probabilities`).
//!    A background task periodically updates these probabilities based on accumulated feedback.
//!    These estimates converge to the true optimal sampling probabilities as the sample statistics
//!    converge.
//!
//! 3. **Stopping Phase**: When statistical analysis determines a clear winner (within
//!    confidence bounds specified by `delta` and `epsilon`), the experiment stops and
//!    only the winning variant is selected (see `check_stopping`) going forward, unless
//!    new variants are introduced.
//!
//! 4. **Re-exploration Phase**: If new variants are introduced after a winner is selected,
//!    they will be put in the "nursery", and eventually the whole set of active variants will
//!    re-enter the Bandit Phase, meaning the system will have lost track of the winner. They
//!    way to avoid this is to remove non-winning variants from the set of active variants in
//!    config before introducing new variants.
//!
//! # Module structure
//!
//! - **Main type**: `TrackAndStopConfig` - the public API for track-and-stop experiments
//! - **State machine**: `TrackAndStopState` - tracks whether we're in nursery, bandit, stopped phase,
//!   or somewhere in between (e.g. some arms still in nursery with some arms in the bandit phases)
//! - **Nursery**: Round-robin sampling for cold-start variants
//! - **Background task**: `probability_update_task` - periodically recomputes sampling probabilities
//!
//! ## Submodules
//!
//! - `check_stopping`: Algorithm for detecting when to stop the experiment
//! - `estimate_optimal_probabilities`: Optimization procedure for estimating the optimal sampling probabilities
//! - `error`: Error types specific to track-and-stop

use arc_swap::ArcSwap;
use check_stopping::{check_stopping, CheckStoppingArgs, StoppingResult};
use error::TrackAndStopError;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio_util::sync::CancellationToken;

use estimate_optimal_probabilities::{
    estimate_optimal_probabilities, EstimateOptimalProbabilitiesArgs,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    config::{MetricConfig, MetricConfigOptimize},
    db::{
        feedback::{FeedbackByVariant, FeedbackQueries},
        postgres::PostgresConnectionInfo,
        ExperimentationQueries, HealthCheckable,
    },
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    utils::spawn_ignoring_shutdown,
    variant::VariantInfo,
};

use super::{check_duplicates_across, check_duplicates_within, VariantSampler};

mod check_stopping;
mod error;
pub mod estimate_optimal_probabilities;

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TrackAndStopConfig {
    metric: String,
    candidate_variants: Vec<String>,
    fallback_variants: Vec<String>,
    min_samples_per_variant: u64,
    delta: f64,
    epsilon: f64,
    #[ts(skip)]
    update_period: Duration,
    min_prob: Option<f64>,
    #[serde(skip)]
    metric_optimize: MetricConfigOptimize,
    #[serde(skip)]
    state: Arc<ArcSwap<TrackAndStopState>>,
    #[serde(skip)]
    task_spawned: AtomicBool,
}

/// Computes the Track-and-Stop state from configuration and feedback data.
///
/// This is a helper function primarily for UI/observability use cases where you have
/// access to the config and feedback data but not the running gateway instance.
/// It reconstructs what the state would be given the current feedback.
///
/// This function is the single source of truth for computing the experiment state in the UI.
/// It handles all the logic including:
/// - Determining which variants are in nursery vs bandit phase
/// - Computing optimal probabilities for bandit variants
/// - Detecting if a winner has been found (stopped state)
/// - Correctly allocating probability mass to nursery and bandit variants
///
/// # Arguments
/// * `candidate_variants` - List of candidate variant names
/// * `variant_performances` - Feedback data for each variant
/// * `min_samples_per_variant` - Minimum samples before a variant graduates from nursery
/// * `delta` - Confidence parameter for stopping detection
/// * `epsilon` - Tolerance parameter for stopping detection
/// * `min_prob` - Minimum probability floor for bandit variants
/// * `metric_optimize` - Whether to maximize or minimize the metric
///
/// # Returns
/// * `Ok(TrackAndStopState)` with computed sampling probabilities
/// * `Err` if the computation fails
pub fn compute_track_and_stop_state(
    candidate_variants: &[String],
    variant_performances: Vec<FeedbackByVariant>,
    min_samples_per_variant: u64,
    delta: f64,
    epsilon: f64,
    min_prob: Option<f64>,
    metric_optimize: MetricConfigOptimize,
) -> Result<TrackAndStopState, TrackAndStopError> {
    TrackAndStopState::new(
        candidate_variants,
        variant_performances,
        min_samples_per_variant,
        delta,
        epsilon,
        min_prob,
        metric_optimize,
    )
}

/// Public representation of Track-and-Stop state for external callers (tests, UI, monitoring).
/// This type exposes sampling probabilities but hides internal implementation details like
/// the `Nursery` struct and atomic counters.
#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrackAndStopState {
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

/// Round-robin sampler for variants in the cold-start phase.
///
/// The nursery holds variants that don't yet have enough data for statistical
/// analysis (i.e., fewer than `min_samples_per_variant` samples). It uses an
/// atomic counter to implement thread-safe round-robin sampling, ensuring each
/// variant gets sampled until it graduates to the bandit phase.
///
/// When serialized for external use, only the variant list is exposed (not the atomic counter).
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct Nursery {
    pub variants: Vec<String>,
    #[serde(skip)]
    #[serde(default)]
    #[ts(skip)]
    index: AtomicU64,
}

impl Clone for Nursery {
    fn clone(&self) -> Self {
        Nursery {
            variants: self.variants.clone(),
            index: AtomicU64::new(self.index.load(Ordering::Relaxed)),
        }
    }
}

impl Nursery {
    fn new(variants: Vec<String>) -> Self {
        Nursery {
            variants,
            index: AtomicU64::new(0),
        }
    }

    // Increments the index at which to start searching for an active variant
    fn get_variant_idx_round_robin(&self) -> u64 {
        self.index.fetch_add(1, Ordering::Relaxed) % self.variants.len() as u64
    }

    /// Returns the total probability mass that should be allocated to the nursery
    /// when there are `num_other_variants` non-nursery variants.
    ///
    /// The allocation formula is: nursery_variants / (nursery_variants + other_variants)
    /// This gives each variant (nursery or not) equal 1 / K probability, where K is the
    /// total number of variants.
    pub fn nursery_total_mass(&self, num_other_variants: usize) -> f64 {
        let num_nursery = self.variants.len();
        let total = num_nursery + num_other_variants;
        if total == 0 {
            0.0
        } else {
            num_nursery as f64 / total as f64
        }
    }

    /// Try to sample an active variant from the nursery using round-robin.
    /// Tries up to N times (where N = number of variants) to find one in active_variants.
    /// Returns None if no intersection exists.
    fn sample_active<'a>(
        &'a self,
        active_variants: &'a BTreeMap<String, Arc<VariantInfo>>,
    ) -> Option<&'a str> {
        // Handle empty nursery case
        if self.variants.is_empty() {
            return None;
        }

        // Try up to N times to find an active variant. We choose the starting
        // index and then search over the variants, wrapping around the array,
        // as opposed to incrementing self.index within the loop below,
        // so that the sampling is truly round robin even when there are concurrent
        // sampling calls.
        let start_idx = self.get_variant_idx_round_robin();
        for idx in start_idx..(start_idx + self.variants.len() as u64) {
            let index = idx % (self.variants.len() as u64);
            let variant = &self.variants[index as usize];
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
    #[serde(default)]
    fallback_variants: Vec<String>,
    #[serde(default = "default_min_samples_per_variant")]
    min_samples_per_variant: u64,
    #[serde(default = "default_delta")]
    delta: f64,
    #[serde(default)]
    epsilon: f64,
    #[serde(default = "default_update_period_s")]
    update_period_s: u64,
    #[serde(default = "default_min_prob")]
    min_prob: Option<f64>,
}

fn default_min_samples_per_variant() -> u64 {
    10
}

fn default_delta() -> f64 {
    0.05
}

fn default_update_period_s() -> u64 {
    300
}

/// Default minimum probability is 0.0, but it will be treated as 1e-6 internally
/// in the optimization algorithm for numerical stability.
#[expect(clippy::unnecessary_wraps)]
fn default_min_prob() -> Option<f64> {
    Some(0.0)
}

impl UninitializedTrackAndStopConfig {
    pub fn load(
        self,
        variants: &HashMap<String, Arc<VariantInfo>>,
        metrics: &HashMap<String, MetricConfig>,
    ) -> Result<TrackAndStopConfig, Error> {
        // Validate metric exists
        if !metrics.contains_key(&self.metric) {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Track-and-Stop experiment references unknown metric '{}'. Available metrics: {:?}",
                    self.metric,
                    metrics.keys().collect::<Vec<_>>()
                ),
            }));
        }

        // Validate candidate_variants is non-empty
        if self.candidate_variants.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: "Track-and-Stop candidate_variants cannot be empty".to_string(),
            }));
        }

        // Check for duplicates within candidate_variants
        check_duplicates_within(&self.candidate_variants, "candidate_variants")?;

        // Validate candidate_variants are a subset of available variants
        for variant in &self.candidate_variants {
            if !variants.contains_key(variant) {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Track-and-Stop candidate_variants includes unknown variant '{}'. Available variants: {:?}",
                        variant,
                        variants.keys().collect::<Vec<_>>()
                    ),
                }));
            }
        }

        // Check for duplicates within fallback_variants
        check_duplicates_within(&self.fallback_variants, "fallback_variants")?;

        // Validate fallback_variants are a subset of available variants
        for variant in &self.fallback_variants {
            if !variants.contains_key(variant) {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Track-and-Stop fallback_variants includes unknown variant '{}'. Available variants: {:?}",
                        variant,
                        variants.keys().collect::<Vec<_>>()
                    ),
                }));
            }
        }

        // Check for duplicates across both lists
        check_duplicates_across(&self.candidate_variants, &self.fallback_variants)?;

        // Validate min_samples_per_variant >= 1
        if self.min_samples_per_variant < 1 {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Track-and-Stop min_samples_per_variant must be >= 1, got {}",
                    self.min_samples_per_variant
                ),
            }));
        }

        // Validate delta is in (0, 1)
        if self.delta <= 0.0 || self.delta >= 1.0 {
            return Err(Error::new(ErrorDetails::Config {
                message: format!("Track-and-Stop delta must be in (0, 1), got {}", self.delta),
            }));
        }

        // Validate epsilon >= 0
        if self.epsilon < 0.0 {
            return Err(Error::new(ErrorDetails::Config {
                message: format!("Track-and-Stop epsilon must be >= 0, got {}", self.epsilon),
            }));
        }

        // Validate min_prob if provided
        // Note: min_prob can be 0.0, but it will be treated as 1e-6 internally for numerical stability
        if let Some(min_prob) = self.min_prob {
            // Check non-negative
            if min_prob < 0.0 {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!("Track-and-Stop `min_prob` must be >= 0.0, got {min_prob}"),
                }));
            }

            // Check finite
            if !min_prob.is_finite() {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!("Track-and-Stop `min_prob` must be finite, got {min_prob}"),
                }));
            }

            // Check that min_prob * num_candidate_variants <= 1.0
            // Only candidate variants get probability mass, not fallback variants
            // Note: This check uses the configured min_prob value (which can be 0.0).
            // The actual optimization uses max(min_prob, 1e-6) for numerical stability.
            let num_candidate_variants = self.candidate_variants.len();
            let min_total_prob = min_prob * (num_candidate_variants as f64);
            if min_total_prob > 1.0 + 1e-9 {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Track-and-Stop min_prob is too large: min_prob ({min_prob}) * num_candidate_variants ({num_candidate_variants}) = {min_total_prob} > 1.0. \
                        The sum of minimum probabilities for candidate variants cannot exceed 1.0."
                    ),
                }));
            }
        }

        let keep_variants: Vec<String> = self
            .candidate_variants
            .iter()
            .cloned()
            .chain(self.fallback_variants.iter().cloned())
            .collect();

        // Get the metric's optimization direction
        let metric_config = metrics.get(&self.metric).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("Metric '{}' not found in metrics config", self.metric),
            })
        })?;

        Ok(TrackAndStopConfig {
            metric: self.metric,
            candidate_variants: self.candidate_variants,
            fallback_variants: self.fallback_variants,
            min_samples_per_variant: self.min_samples_per_variant,
            delta: self.delta,
            epsilon: self.epsilon,
            update_period: Duration::from_secs(self.update_period_s),
            min_prob: self.min_prob,
            metric_optimize: metric_config.optimize,
            state: Arc::new(ArcSwap::new(Arc::new(
                TrackAndStopState::nursery_from_variants(keep_variants),
            ))),
            task_spawned: AtomicBool::new(false),
        })
    }
}

impl VariantSampler for TrackAndStopConfig {
    async fn setup(
        &self,
        db: Arc<dyn FeedbackQueries + Send + Sync>,
        function_name: &str,
        postgres: &PostgresConnectionInfo,
        cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        // Track-and-Stop requires PostgreSQL for episode-to-variant mapping
        match postgres {
            PostgresConnectionInfo::Disabled => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Track-and-Stop experimentation is configured for function '{function_name}' but PostgreSQL is not available. \
                        Track-and-Stop requires PostgreSQL for episode-to-variant consistency. \
                        Please set the `TENSORZERO_POSTGRES_URL` environment variable.",
                    ),
                }));
            }
            PostgresConnectionInfo::Enabled { .. } => {}
            // Accept Mock postgres for testing purposes
            #[cfg(test)]
            PostgresConnectionInfo::Mock { .. } => {}
        }

        // Check if postgres is healthy
        postgres.health().await.map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Track-and-Stop experimentation is configured for function '{function_name}' but PostgreSQL is unhealthy: {e}. \
                    Track-and-Stop requires a healthy PostgreSQL connection for episode-to-variant consistency.",
                ),
            })
        })?;

        // Check if a task has already been spawned for this function
        // Use compare_exchange to atomically check and set the flag
        if self
            .task_spawned
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Track-and-Stop probability update task has already been spawned for function '{function_name}'"
                ),
            }));
        }

        // Spawn a background task that continuously updates sampling probabilities.
        // This task:
        // 1. Runs independently for the lifetime of the application
        // 2. Periodically queries the database for feedback data
        // 3. Computes new optimal sampling probabilities based on observed performance
        // 4. Updates the shared `self.state` via ArcSwap (lock-free concurrent updates)
        // 5. Concurrent `sample()` calls read the latest state without blocking
        spawn_ignoring_shutdown(probability_update_task(ProbabilityUpdateTaskArgs {
            db,
            candidate_variants: self.candidate_variants.clone().into(),
            metric_name: self.metric.clone(),
            function_name: function_name.to_string(),
            sampling_probabilities: self.state.clone(),
            update_period: self.update_period,
            min_samples_per_variant: self.min_samples_per_variant,
            epsilon: self.epsilon,
            delta: self.delta,
            min_prob: self.min_prob,
            metric_optimize: self.metric_optimize,
            cancel_token,
        }));
        Ok(())
    }
    fn allowed_variants(&self) -> impl Iterator<Item = &str> + '_ {
        self.candidate_variants
            .iter()
            .map(String::as_str)
            .chain(self.fallback_variants.iter().map(String::as_str))
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
        let candidate_name_opt = state.sample(active_variants, uniform_sample).map_err(|e| {
            Error::new(ErrorDetails::Inference {
                message: format!("Error sampling variant: {e}"),
            })
        })?;

        let variant_name = if let Some(candidate_name) = candidate_name_opt {
            // Check and set the variant in Postgres (ensures consistency for the episode)
            let set_variant = postgres
                .check_and_set_variant_by_episode(episode_id, function_name, candidate_name)
                .await?;

            // Check if the returned variant is active
            if active_variants.contains_key(&set_variant) {
                set_variant
            } else {
                // The variant that was already set for this episode is not active, fall back
                fallback_sample(active_variants, &self.fallback_variants)?
            }
        } else {
            // State couldn't provide a variant, fall back to fallback_variants
            fallback_sample(active_variants, &self.fallback_variants)?
        };

        // Remove and return the sampled variant
        active_variants.remove_entry(&variant_name).ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Sampled variant {variant_name} not found in active_variants. {IMPOSSIBLE_ERROR_MESSAGE}."
                ),
            })
        })
    }

    fn get_current_display_probabilities<'a>(
        &self,
        _function_name: &str,
        active_variants: &'a HashMap<String, Arc<VariantInfo>>,
        _postgres: &PostgresConnectionInfo,
    ) -> Result<HashMap<&'a str, f64>, Error> {
        let state = self.state.load();
        match state.as_ref() {
            TrackAndStopState::Stopped {
                winner_variant_name,
            } => {
                let mut probs = HashMap::new();
                if let Some(key) = active_variants.keys().find(|k| *k == winner_variant_name) {
                    probs.insert(key.as_str(), 1.0);
                }
                Ok(probs)
            }
            TrackAndStopState::NurseryOnly(nursery) => {
                // Count only active nursery variants
                let active_nursery_variants: Vec<_> = active_variants
                    .keys()
                    .filter(|k| nursery.variants.contains(k))
                    .collect();
                let num_active_nursery = active_nursery_variants.len();
                let uniform_prob = if num_active_nursery > 0 {
                    1.0 / (num_active_nursery as f64)
                } else {
                    0.0
                };
                let probs = active_nursery_variants
                    .into_iter()
                    .map(|k| (k.as_str(), uniform_prob))
                    .collect();
                Ok(probs)
            }
            TrackAndStopState::BanditsOnly {
                sampling_probabilities,
            } => {
                // Collect active bandit probabilities
                let mut probs: HashMap<&str, f64> = active_variants
                    .keys()
                    .filter_map(|k| {
                        sampling_probabilities
                            .get(k)
                            .map(|&prob| (k.as_str(), prob))
                    })
                    .collect();

                // Renormalize if some variants are inactive
                let total: f64 = probs.values().sum();
                if total > 0.0 && (total - 1.0).abs() > 1e-9 {
                    for prob in probs.values_mut() {
                        *prob /= total;
                    }
                }

                Ok(probs)
            }
            TrackAndStopState::NurseryAndBandits {
                nursery,
                sampling_probabilities,
            } => {
                // Count only active nursery and bandit variants
                let active_nursery_variants: Vec<_> = active_variants
                    .keys()
                    .filter(|k| nursery.variants.contains(k))
                    .collect();
                let active_bandit_variants: Vec<_> = active_variants
                    .keys()
                    .filter(|k| sampling_probabilities.contains_key(k.as_str()))
                    .collect();

                let num_active_nursery = active_nursery_variants.len();
                let num_active_bandits = active_bandit_variants.len();

                // Compute mass allocation based on active variants
                let nursery_mass = nursery.nursery_total_mass(num_active_bandits);
                let bandit_mass = 1.0 - nursery_mass;

                let mut probs = HashMap::new();

                // Assign uniform probability to each active nursery variant
                let uniform_nursery_prob = if num_active_nursery > 0 {
                    nursery_mass / (num_active_nursery as f64)
                } else {
                    0.0
                };
                for key in active_nursery_variants {
                    probs.insert(key.as_str(), uniform_nursery_prob);
                }

                // For active bandit variants, renormalize and scale by bandit_mass
                let active_bandit_probs: Vec<_> = active_bandit_variants
                    .iter()
                    .filter_map(|k| sampling_probabilities.get(k.as_str()).map(|&p| (*k, p)))
                    .collect();

                let bandit_total: f64 = active_bandit_probs.iter().map(|(_, p)| p).sum();
                if bandit_total > 0.0 {
                    for (key, prob) in active_bandit_probs {
                        probs.insert(key.as_str(), (prob / bandit_total) * bandit_mass);
                    }
                }

                Ok(probs)
            }
            TrackAndStopState::NurseryAndStopped {
                nursery,
                stopped_variant_name,
            } => {
                // Count only active nursery variants and check if stopped variant is active
                let active_nursery_variants: Vec<_> = active_variants
                    .keys()
                    .filter(|k| nursery.variants.contains(k))
                    .collect();
                let stopped_is_active = active_variants.contains_key(stopped_variant_name);

                let num_active_nursery = active_nursery_variants.len();
                let num_active_other = if stopped_is_active { 1 } else { 0 };

                // Compute mass allocation based on active variants
                let nursery_mass = nursery.nursery_total_mass(num_active_other);
                let stopped_mass = 1.0 - nursery_mass;

                let mut probs = HashMap::new();

                // Assign uniform probability to each active nursery variant
                let uniform_nursery_prob = if num_active_nursery > 0 {
                    nursery_mass / (num_active_nursery as f64)
                } else {
                    0.0
                };
                for key in active_nursery_variants {
                    probs.insert(key.as_str(), uniform_nursery_prob);
                }

                // Assign probability to stopped variant if active
                if let Some(key) = active_variants.keys().find(|k| *k == stopped_variant_name) {
                    probs.insert(key.as_str(), stopped_mass);
                }

                Ok(probs)
            }
        }
    }
}

struct ProbabilityUpdateTaskArgs {
    db: Arc<dyn FeedbackQueries + Send + Sync>,
    candidate_variants: Arc<Vec<String>>,
    metric_name: String,
    function_name: String,
    sampling_probabilities: Arc<ArcSwap<TrackAndStopState>>,
    update_period: Duration,
    min_samples_per_variant: u64,
    epsilon: f64,
    delta: f64,
    min_prob: Option<f64>,
    metric_optimize: MetricConfigOptimize,
    cancel_token: CancellationToken,
}

/// Background task that continuously updates sampling probabilities for track-and-stop experiments.
///
/// This task runs in an infinite loop with the following behavior:
/// - Sleeps for `update_period` between iterations
/// - Queries ClickHouse for feedback data on all candidate variants
/// - Computes optimal sampling probabilities (or detects a winner to stop)
/// - Updates the shared state via `ArcSwap::store()` for lock-free reads
/// - Logs warnings on errors but never crashes (continues retrying)
///
/// The task is spawned once per function during `setup()` and runs for the application's lifetime.
async fn probability_update_task(args: ProbabilityUpdateTaskArgs) {
    let ProbabilityUpdateTaskArgs {
        db,
        candidate_variants,
        metric_name,
        function_name,
        sampling_probabilities,
        update_period,
        min_samples_per_variant,
        epsilon,
        delta,
        min_prob,
        metric_optimize,
        cancel_token,
    } = args;

    let mut interval = tokio::time::interval(update_period);
    loop {
        tokio::select! {
            () = cancel_token.cancelled() => {
                break;
            }
            _ = interval.tick() => {}
        }

        let result = update_probabilities(UpdateProbabilitiesArgs {
            db: db.as_ref(),
            candidate_variants: &candidate_variants,
            metric_name: &metric_name,
            function_name: &function_name,
            sampling_probabilities: &sampling_probabilities,
            min_samples_per_variant,
            epsilon,
            delta,
            min_prob,
            metric_optimize,
        })
        .await;

        match result {
            Ok(()) => {}
            Err(e) => {
                tracing::warn!("Failed to update probabilities for {function_name}: {e}");
            }
        }
    }
}

struct UpdateProbabilitiesArgs<'a> {
    db: &'a (dyn FeedbackQueries + Send + Sync),
    candidate_variants: &'a Arc<Vec<String>>,
    metric_name: &'a str,
    function_name: &'a str,
    sampling_probabilities: &'a Arc<ArcSwap<TrackAndStopState>>,
    min_samples_per_variant: u64,
    epsilon: f64,
    delta: f64,
    min_prob: Option<f64>,
    metric_optimize: MetricConfigOptimize,
}

async fn update_probabilities(args: UpdateProbabilitiesArgs<'_>) -> Result<(), TrackAndStopError> {
    let UpdateProbabilitiesArgs {
        db,
        candidate_variants,
        metric_name,
        function_name,
        sampling_probabilities,
        min_samples_per_variant,
        epsilon,
        delta,
        min_prob,
        metric_optimize,
    } = args;

    // Fetch feedback from database
    let variant_performances = db
        .get_feedback_by_variant(metric_name, function_name, Some(candidate_variants))
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
            min_prob,
            metric_optimize,
        )
    })
    .await??; // First ? for JoinError, second ? for TrackAndStopError

    // Store the new state
    sampling_probabilities.store(Arc::new(new_state));

    Ok(())
}

/// For each variant in `candidate_variants`, get the count from variant_performances if it exists.
/// If it doesn't exist, return 0.
/// Returns an error if multiple entries exist for the same variant.
fn get_count_by_variant<'a>(
    candidate_variants: &'a [String],
    variant_performances: &[FeedbackByVariant],
) -> Result<HashMap<&'a str, u64>, TrackAndStopError> {
    candidate_variants
        .iter()
        .map(|variant| {
            let matching: Vec<_> = variant_performances
                .iter()
                .filter(|p| &p.variant_name == variant)
                .collect();

            match matching.len() {
                0 => Ok((variant.as_str(), 0)),
                1 => Ok((variant.as_str(), matching[0].count)),
                n => Err(TrackAndStopError::MultipleEntriesForVariant {
                    variant_name: variant.to_string(),
                    num_entries: n,
                }),
            }
        })
        .collect()
}

/// Select the first variant from the ranked fallback_variants list that is active.
/// Returns an error if no fallback variants are active.
fn fallback_sample(
    active_variants: &BTreeMap<String, Arc<VariantInfo>>,
    fallback_variants: &[String],
) -> Result<String, Error> {
    // Select the first variant from the ranked fallback_variants list that is active
    for variant_name in fallback_variants {
        if active_variants.contains_key(variant_name) {
            return Ok(variant_name.clone());
        }
    }

    // No active fallback variants found
    Err(ErrorDetails::NoFallbackVariantsRemaining.into())
}

/// Sample a variant from active_variants using weighted probabilities.
/// Returns None if no active variant has positive probability.
/// Returns an error if any probability is negative.
/// Note: `unform_sample` must be in [0, 1], but `sampling_probabilities`
/// just need to be non-negative, since they're summed and normalized.
fn sample_with_probabilities<'a>(
    active_variants: &'a BTreeMap<String, Arc<VariantInfo>>,
    sampling_probabilities: &HashMap<String, f64>,
    uniform_sample: f64,
) -> Result<Option<&'a str>, TrackAndStopError> {
    // Check for negative probabilities
    for (variant_name, &prob) in sampling_probabilities {
        if prob < 0.0 {
            return Err(TrackAndStopError::NegativeProbability {
                variant_name: variant_name.clone(),
                probability: prob,
            });
        }
    }

    // Compute the total probability of active variants
    let total_probability: f64 = active_variants
        .keys()
        .map(|variant_name| sampling_probabilities.get(variant_name).unwrap_or(&0.0))
        .sum();

    if total_probability <= 0.0 {
        return Ok(None);
    }

    // Use weighted sampling
    let random_threshold = uniform_sample * total_probability;
    let mut cumulative_probability = 0.0;

    Ok(active_variants
        .keys()
        .find(|variant_name| {
            cumulative_probability += sampling_probabilities
                .get(variant_name.as_str())
                .unwrap_or(&0.0);
            cumulative_probability > random_threshold
        })
        .map(std::string::String::as_str))
}

impl TrackAndStopState {
    /// For a quick initialization of a TrackAndStopState instance with only a nursery.
    /// Should be called on startup.
    fn nursery_from_variants(variants: Vec<String>) -> Self {
        TrackAndStopState::NurseryOnly(Nursery::new(variants))
    }

    /// Initializes a new TrackAndStopState instance based on the current statistics
    /// and configured parameters.
    ///
    /// NOTE: This function may do some CPU-bound work to compute probabilities.
    ///
    /// # Assumptions
    /// - `candidate_variants` is non-empty (validated in `UninitializedTrackAndStopConfig::load()`)
    /// - `variant_performances` may contain feedback for non-candidate variants (these are filtered out)
    fn new(
        candidate_variants: &[String],
        variant_performances: Vec<FeedbackByVariant>,
        min_samples_per_variant: u64,
        delta: f64,
        epsilon: f64,
        min_prob: Option<f64>,
        metric_optimize: MetricConfigOptimize,
    ) -> Result<Self, TrackAndStopError> {
        // Filter out feedback for non-candidate variants (can happen if config changed)
        let variant_performances = if variant_performances.len() > candidate_variants.len() {
            tracing::warn!("Feedback is being filtered out for non-candidate variants. Current candidate variants: {candidate_variants:?}");
            variant_performances
                .into_iter()
                .filter(|feedback| candidate_variants.contains(&feedback.variant_name))
                .collect()
        } else {
            variant_performances
        };

        // If we only have one variant, we'll simply use it
        if candidate_variants.len() == 1 {
            return Ok(TrackAndStopState::Stopped {
                winner_variant_name: candidate_variants[0].clone(),
            });
        }
        let variant_counts = get_count_by_variant(candidate_variants, &variant_performances)?;
        let num_variants_above_cutoff = variant_counts
            .values()
            .filter(|&count| *count >= min_samples_per_variant)
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
                    variance_floor: None,
                    delta: Some(delta),
                    epsilon: Some(epsilon),
                    metric_optimize,
                })? {
                    StoppingResult::Winner(winner_variant_name) => {
                        let competitors: Vec<&str> = candidate_variants
                            .iter()
                            .filter(|v| v.as_str() != winner_variant_name)
                            .map(std::string::String::as_str)
                            .collect();
                        tracing::info!(
                            winner = winner_variant_name,
                            competitors = ?competitors,
                            "Track-and-Stop experiment stopped: winner identified. This variant will be used exclusively
                            going forward, unless new variants are introduced, or unless additional feedback data suggests
                            that this variant is not actually a winning variant."
                        );
                        Ok(TrackAndStopState::Stopped {
                            winner_variant_name,
                        })
                    }
                    StoppingResult::NotStopped => Ok(TrackAndStopState::BanditsOnly {
                        sampling_probabilities: estimate_optimal_probabilities(
                            EstimateOptimalProbabilitiesArgs {
                                feedback: variant_performances,
                                epsilon: Some(epsilon),
                                variance_floor: None,
                                min_prob,
                                reg0: None,
                                metric_optimize,
                            },
                        )?,
                    }),
                }
            }
            (false, false) => {
                // This case should be unreachable since we handle single variant early
                // If we get here, something is wrong with the logic
                Err(TrackAndStopError::NoArmsDetected)
            }
            (true, true) => {
                // If we need both a nursery and a bandit
                // we can separate them by filtering the variants based on their counts
                let nursery_variants: Vec<String> = variant_counts
                    .iter()
                    .filter(|(_, &count)| count < min_samples_per_variant)
                    .map(|(key, _)| key.to_string())
                    .collect();
                let bandit_feedback: Vec<FeedbackByVariant> = variant_performances
                    .into_iter()
                    .filter(|feedback| feedback.count >= min_samples_per_variant)
                    .collect();

                match check_stopping(CheckStoppingArgs {
                    feedback: &bandit_feedback,
                    min_pulls: min_samples_per_variant,
                    variance_floor: None,
                    delta: Some(delta),
                    epsilon: Some(epsilon),
                    metric_optimize,
                })? {
                    StoppingResult::Winner(winner) => {
                        let bandit_competitors: Vec<&str> = bandit_feedback
                            .iter()
                            .filter(|f| f.variant_name != winner)
                            .map(|f| f.variant_name.as_str())
                            .collect();
                        tracing::info!(
                            winner = winner.as_str(),
                            competitors = ?bandit_competitors,
                            nursery_variants = ?nursery_variants,
                            "Track-and-Stop experiment stopped among bandit variants, with nursery variants remaining"
                        );
                        tracing::warn!(
                            winner = winner.as_str(),
                            competitors = ?bandit_competitors,
                            nursery_variants = ?nursery_variants,
                            "Winner identified among bandit arms, but nursery variants will continue being explored.
                            The experiment will eventually lose track of the winner when nursery variants graduate to bandit status.
                            Recommendation: remove the previous competitors and start a new experiment to test the previous winner
                            against the current set of nursery variants."
                        );
                        Ok(TrackAndStopState::NurseryAndStopped {
                            nursery: Nursery::new(nursery_variants),
                            stopped_variant_name: winner,
                        })
                    }
                    StoppingResult::NotStopped => Ok(TrackAndStopState::NurseryAndBandits {
                        nursery: Nursery::new(nursery_variants),
                        sampling_probabilities: estimate_optimal_probabilities(
                            EstimateOptimalProbabilitiesArgs {
                                feedback: bandit_feedback,
                                epsilon: Some(epsilon),
                                variance_floor: None,
                                min_prob,
                                reg0: None,
                                metric_optimize,
                            },
                        )?,
                    }),
                }
            }
        }
    }

    /// Samples an active variant from the current track-and-stop state.
    /// Returns `Ok(Some(..))` if it is possible to sample an active variant from the "happy path"
    /// of experiment execution, `Ok(None)` otherwise.
    /// Returns an error if invalid probabilities are detected.
    /// Note: this function does __not__ pop
    fn sample<'a>(
        &'a self,
        active_variants: &'a BTreeMap<String, Arc<VariantInfo>>,
        uniform_sample: f64,
    ) -> Result<Option<&'a str>, TrackAndStopError> {
        match self {
            TrackAndStopState::Stopped {
                winner_variant_name,
            } => {
                if active_variants.contains_key(winner_variant_name) {
                    Ok(Some(winner_variant_name))
                } else {
                    Ok(None)
                }
            }
            TrackAndStopState::NurseryOnly(nursery) => {
                // Do round-robin sampling from the variants until we find one that is active
                Ok(nursery.sample_active(active_variants))
            }
            TrackAndStopState::NurseryAndBandits {
                sampling_probabilities,
                nursery,
            } => {
                // Total probability mass assigned to nursery variants
                let nursery_mass = nursery.nursery_total_mass(sampling_probabilities.len());
                // Total probability mass assigned to bandit variants
                let bandit_mass = 1.0 - nursery_mass;
                // If `uniform_sample` has chosen the nursery, or if `bandit_mass`` is numerically essentially 0, sample a nursery variant
                if (bandit_mass <= f64::EPSILON) || (uniform_sample < nursery_mass) {
                    Ok(nursery.sample_active(active_variants))
                // Otherwise sample a bandit variant
                } else {
                    let bandit_sample = (uniform_sample - nursery_mass) / bandit_mass;
                    sample_with_probabilities(
                        active_variants,
                        sampling_probabilities,
                        bandit_sample,
                    )
                }
            }
            TrackAndStopState::BanditsOnly {
                sampling_probabilities,
            } => sample_with_probabilities(active_variants, sampling_probabilities, uniform_sample),
            TrackAndStopState::NurseryAndStopped {
                nursery,
                stopped_variant_name,
            } => {
                let nursery_mass = nursery.nursery_total_mass(1); // 1 other variant (winner variant)
                if uniform_sample < nursery_mass {
                    Ok(nursery.sample_active(active_variants))
                } else if active_variants.contains_key(stopped_variant_name) {
                    Ok(Some(stopped_variant_name))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ErrorContext, MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType,
        SchemaData,
    };
    use crate::db::clickhouse::ClickHouseConnectionInfo;
    use crate::db::feedback::FeedbackByVariant;
    use crate::variant::{chat_completion::UninitializedChatCompletionConfig, VariantConfig};

    // Helper function to create test variants
    fn create_test_variants(names: &[&str]) -> BTreeMap<String, Arc<VariantInfo>> {
        names
            .iter()
            .map(|&name| {
                (
                    name.to_string(),
                    Arc::new(VariantInfo {
                        inner: VariantConfig::ChatCompletion(
                            UninitializedChatCompletionConfig {
                                weight: None,
                                model: "model-name".into(),
                                ..Default::default()
                            }
                            .load(&SchemaData::default(), &ErrorContext::new_test())
                            .unwrap(),
                        ),
                        timeouts: Default::default(),
                    }),
                )
            })
            .collect()
    }

    // Helper function to create test feedback
    fn create_feedback(
        variant_name: &str,
        count: u64,
        mean: f32,
        variance: f32,
    ) -> FeedbackByVariant {
        FeedbackByVariant {
            variant_name: variant_name.to_string(),
            count,
            mean,
            variance: Some(variance),
        }
    }

    // Tests for get_count_by_variant
    #[test]
    fn test_get_count_by_variant_all_present() {
        let candidates = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let performances = vec![
            create_feedback("A", 10, 0.5, 0.1),
            create_feedback("B", 20, 0.6, 0.2),
            create_feedback("C", 15, 0.7, 0.15),
        ];

        let counts = get_count_by_variant(&candidates, &performances).unwrap();

        assert_eq!(counts.get("A"), Some(&10));
        assert_eq!(counts.get("B"), Some(&20));
        assert_eq!(counts.get("C"), Some(&15));
    }

    #[test]
    fn test_get_count_by_variant_some_missing() {
        let candidates = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let performances = vec![
            create_feedback("A", 10, 0.5, 0.1),
            create_feedback("C", 15, 0.7, 0.15),
        ];

        let counts = get_count_by_variant(&candidates, &performances).unwrap();

        assert_eq!(counts.get("A"), Some(&10));
        assert_eq!(counts.get("B"), Some(&0)); // B is missing
        assert_eq!(counts.get("C"), Some(&15));
    }

    #[test]
    fn test_get_count_by_variant_empty_performances() {
        let candidates = vec!["A".to_string(), "B".to_string()];
        let performances = vec![];

        let counts = get_count_by_variant(&candidates, &performances).unwrap();

        assert_eq!(counts.get("A"), Some(&0));
        assert_eq!(counts.get("B"), Some(&0));
    }

    #[test]
    fn test_get_count_by_variant_empty_candidates() {
        let candidates = vec![];
        let performances = vec![create_feedback("A", 10, 0.5, 0.1)];

        let counts = get_count_by_variant(&candidates, &performances).unwrap();

        assert_eq!(counts.len(), 0);
    }

    #[test]
    fn test_get_count_by_variant_multiple_entries_error() {
        let candidates = vec!["A".to_string(), "B".to_string()];
        let performances = vec![
            create_feedback("A", 10, 0.5, 0.1),
            create_feedback("A", 5, 0.6, 0.2), // Duplicate entry for A
            create_feedback("B", 20, 0.7, 0.15),
        ];

        let result = get_count_by_variant(&candidates, &performances);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(
            err,
            TrackAndStopError::MultipleEntriesForVariant { .. }
        ));
    }

    // Tests for fallback_sample
    #[test]
    fn test_fallback_sample_all_active() {
        let active = create_test_variants(&["A", "B", "C"]);
        let fallbacks = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        // With ranked list behavior, always returns the first active variant
        let result = fallback_sample(&active, &fallbacks);
        assert_eq!(result.unwrap(), "A");
    }

    #[test]
    fn test_fallback_sample_some_active() {
        let active = create_test_variants(&["A", "C"]);
        let fallbacks = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        // With ranked list, returns the first active variant (A)
        let result = fallback_sample(&active, &fallbacks);
        assert_eq!(result.unwrap(), "A");
    }

    #[test]
    fn test_fallback_sample_first_inactive() {
        let active = create_test_variants(&["C"]);
        let fallbacks = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        // With ranked list, returns the first active variant (C, skipping A and B)
        let result = fallback_sample(&active, &fallbacks);
        assert_eq!(result.unwrap(), "C");
    }

    #[test]
    fn test_fallback_sample_single_active() {
        let active = create_test_variants(&["B"]);
        let fallbacks = vec!["B".to_string(), "C".to_string()];

        // Only B is active, should always return B
        let result = fallback_sample(&active, &fallbacks);
        assert_eq!(result.unwrap(), "B");
    }

    #[test]
    fn test_fallback_sample_no_intersection() {
        let active = create_test_variants(&["A", "B"]);
        let fallbacks = vec!["C".to_string(), "D".to_string()];

        let result = fallback_sample(&active, &fallbacks);
        assert!(result.is_err());
    }

    #[test]
    fn test_fallback_sample_empty_fallbacks() {
        let active = create_test_variants(&["A", "B"]);
        let fallbacks = vec![];

        let result = fallback_sample(&active, &fallbacks);
        assert!(result.is_err());
    }

    #[test]
    fn test_fallback_sample_empty_active() {
        let active: BTreeMap<String, Arc<VariantInfo>> = create_test_variants(&[]);
        let fallbacks = vec!["A".to_string(), "B".to_string()];

        let result = fallback_sample(&active, &fallbacks);
        assert!(result.is_err());
    }

    // Tests for sample_with_probabilities
    #[test]
    fn test_sample_with_probabilities_weighted() {
        let active = create_test_variants(&["A", "B", "C"]);
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.1);
        probs.insert("B".to_string(), 0.3);
        probs.insert("C".to_string(), 0.6);

        // Total probability = 1.0
        // A: [0.0, 0.1)
        // B: [0.1, 0.4)
        // C: [0.4, 1.0)

        let result = sample_with_probabilities(&active, &probs, 0.05).unwrap();
        assert_eq!(result, Some("A"));

        let result = sample_with_probabilities(&active, &probs, 0.2).unwrap();
        assert_eq!(result, Some("B"));

        let result = sample_with_probabilities(&active, &probs, 0.7).unwrap();
        assert_eq!(result, Some("C"));
    }

    #[test]
    fn test_sample_with_probabilities_some_inactive() {
        let active = create_test_variants(&["A", "C"]);
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.2);
        probs.insert("B".to_string(), 0.3); // Not active
        probs.insert("C".to_string(), 0.5);

        // Total active probability = 0.2 + 0.5 = 0.7
        // A: [0.0, 0.2/0.7) -> [0.0, 0.286...)
        // C: [0.2/0.7, 0.7/0.7) -> [0.286..., 1.0)

        let result = sample_with_probabilities(&active, &probs, 0.1).unwrap();
        assert_eq!(result, Some("A"));

        let result = sample_with_probabilities(&active, &probs, 0.5).unwrap();
        assert_eq!(result, Some("C"));
    }

    #[test]
    fn test_sample_with_probabilities_all_zero() {
        let active = create_test_variants(&["A", "B"]);
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.0);
        probs.insert("B".to_string(), 0.0);

        let result = sample_with_probabilities(&active, &probs, 0.5).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_sample_with_probabilities_missing_variants() {
        let active = create_test_variants(&["A", "B", "C"]);
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.5);
        // B and C have no probabilities, treated as 0.0

        let result = sample_with_probabilities(&active, &probs, 0.2).unwrap();
        assert_eq!(result, Some("A"));

        let result = sample_with_probabilities(&active, &probs, 0.6).unwrap();
        assert_eq!(result, Some("A")); // Still A since it's the only one with probability
    }

    #[test]
    fn test_sample_with_probabilities_single_variant() {
        let active = create_test_variants(&["A"]);
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 1.0);

        let result = sample_with_probabilities(&active, &probs, 0.0).unwrap();
        assert_eq!(result, Some("A"));

        let result = sample_with_probabilities(&active, &probs, 0.999).unwrap();
        assert_eq!(result, Some("A"));
    }

    #[test]
    fn test_sample_with_probabilities_edge_cases() {
        let active = create_test_variants(&["A", "B"]);
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.5);
        probs.insert("B".to_string(), 0.5);

        // Test at exact boundaries
        let result = sample_with_probabilities(&active, &probs, 0.0).unwrap();
        assert_eq!(result, Some("A"));

        let result = sample_with_probabilities(&active, &probs, 0.5).unwrap();
        assert_eq!(result, Some("B"));

        let result = sample_with_probabilities(&active, &probs, 0.9999).unwrap();
        assert_eq!(result, Some("B"));
    }

    #[test]
    fn test_sample_with_probabilities_negative_error() {
        let active = create_test_variants(&["A", "B"]);
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.5);
        probs.insert("B".to_string(), -0.1); // Negative probability

        let result = sample_with_probabilities(&active, &probs, 0.5);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, TrackAndStopError::NegativeProbability { .. }));
    }

    // Tests for Nursery
    #[test]
    fn test_nursery_new() {
        let variants = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let nursery = Nursery::new(variants.clone());

        assert_eq!(nursery.variants, variants);
        assert_eq!(nursery.index.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_nursery_get_variant_round_robin_single() {
        let nursery = Nursery::new(vec!["A".to_string()]);

        // Should always return index 0 (which maps to "A") regardless of how many times called
        for _ in 0..10 {
            assert_eq!(nursery.get_variant_idx_round_robin(), 0);
        }
    }

    #[test]
    fn test_nursery_get_variant_round_robin_multiple() {
        let nursery = Nursery::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]);

        // First cycle - returns indices 0, 1, 2
        assert_eq!(nursery.get_variant_idx_round_robin(), 0); // "A"
        assert_eq!(nursery.get_variant_idx_round_robin(), 1); // "B"
        assert_eq!(nursery.get_variant_idx_round_robin(), 2); // "C"

        // Second cycle - should wrap around
        assert_eq!(nursery.get_variant_idx_round_robin(), 0); // "A"
        assert_eq!(nursery.get_variant_idx_round_robin(), 1); // "B"
        assert_eq!(nursery.get_variant_idx_round_robin(), 2); // "C"
    }

    #[test]
    fn test_nursery_get_variant_round_robin_two_variants() {
        let nursery = Nursery::new(vec!["X".to_string(), "Y".to_string()]);

        assert_eq!(nursery.get_variant_idx_round_robin(), 0); // "X"
        assert_eq!(nursery.get_variant_idx_round_robin(), 1); // "Y"
        assert_eq!(nursery.get_variant_idx_round_robin(), 0); // "X"
        assert_eq!(nursery.get_variant_idx_round_robin(), 1); // "Y"
    }

    #[test]
    fn test_nursery_sample_active_all_active() {
        let nursery = Nursery::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        let active = create_test_variants(&["A", "B", "C"]);

        // Index starts at 0, first variant is "A"
        let result = nursery.sample_active(&active);
        assert_eq!(result, Some("A"));
    }

    #[test]
    fn test_nursery_sample_active_some_active() {
        let nursery = Nursery::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        let active = create_test_variants(&["B", "C"]);

        // Index starts at 0
        // Try 1: "A" (index 0) - not active, increment to 1
        // Try 2: "B" (index 1) - active! Return "B"
        let result = nursery.sample_active(&active);
        assert_eq!(result, Some("B"));
    }

    #[test]
    fn test_nursery_sample_active_single_active() {
        let nursery = Nursery::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        let active = create_test_variants(&["B"]);

        // Index starts at 0
        // Try 1: "A" (index 0) - not active, increment to 1
        // Try 2: "B" (index 1) - active! Return "B"
        let result = nursery.sample_active(&active);
        assert_eq!(result, Some("B"));
    }

    #[test]
    fn test_nursery_sample_active_none_active() {
        let nursery = Nursery::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        let active = create_test_variants(&["D", "E", "F"]); // No intersection

        let result = nursery.sample_active(&active);
        assert_eq!(result, None);
    }

    #[test]
    fn test_nursery_sample_active_empty_active() {
        let nursery = Nursery::new(vec!["A".to_string(), "B".to_string()]);
        let active = create_test_variants(&[]); // Empty active variants

        let result = nursery.sample_active(&active);
        assert_eq!(result, None);
    }

    #[test]
    fn test_nursery_sample_active_empty_nursery() {
        let nursery = Nursery::new(vec![]); // Empty nursery
        let active = create_test_variants(&["A", "B"]);

        // This is a degenerate case - with empty nursery, the function will try 0 times
        // and return None (since the loop runs 0..0)
        let result = nursery.sample_active(&active);
        assert_eq!(result, None);
    }

    #[test]
    fn test_nursery_sample_active_deterministic_position() {
        // Test that sample_active respects the round-robin position
        let nursery = Nursery::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        let active = create_test_variants(&["A", "B", "C"]);

        // First call should start at position 0 (A)
        let result1 = nursery.sample_active(&active);
        assert_eq!(result1, Some("A"));

        // Next call should try from position 3 % 3 = 0, but index has advanced
        // Actually after the first call, index is at 1
        let result2 = nursery.sample_active(&active);
        assert_eq!(result2, Some("B"));

        let result3 = nursery.sample_active(&active);
        assert_eq!(result3, Some("C"));

        // Should wrap around
        let result4 = nursery.sample_active(&active);
        assert_eq!(result4, Some("A"));
    }

    // Tests for TrackAndStopState
    #[test]
    fn test_state_nursery_from_variants() {
        let variants = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let state = TrackAndStopState::nursery_from_variants(variants.clone());

        match state {
            TrackAndStopState::NurseryOnly(nursery) => {
                assert_eq!(nursery.variants, variants);
            }
            _ => panic!("Expected NurseryOnly state"),
        }
    }

    #[test]
    fn test_state_new_all_below_cutoff() {
        // All variants below min_samples_per_variant → NurseryOnly
        let candidates = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let performances = vec![
            create_feedback("A", 5, 0.5, 0.1),
            create_feedback("B", 3, 0.6, 0.2),
            create_feedback("C", 4, 0.7, 0.15),
        ];

        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            None,
            MetricConfigOptimize::Max,
        )
        .unwrap();

        match state {
            TrackAndStopState::NurseryOnly(nursery) => {
                assert_eq!(nursery.variants, candidates);
            }
            _ => panic!("Expected NurseryOnly state, got {state:?}"),
        }
    }

    #[test]
    fn test_state_new_all_above_cutoff_no_stopping() {
        // All variants above cutoff but not stopped → BanditsOnly
        let candidates = vec!["A".to_string(), "B".to_string()];
        let performances = vec![
            create_feedback("A", 20, 0.5, 0.1),
            create_feedback("B", 20, 0.6, 0.2),
        ];

        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            None,
            MetricConfigOptimize::Max,
        )
        .unwrap();

        match state {
            TrackAndStopState::BanditsOnly {
                sampling_probabilities,
            } => {
                // Should have probabilities for both variants
                assert_eq!(sampling_probabilities.len(), 2);
                assert!(sampling_probabilities.contains_key("A"));
                assert!(sampling_probabilities.contains_key("B"));
                // Probabilities should sum to 1.0
                let sum: f64 = sampling_probabilities.values().sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected BanditsOnly state, got {state:?}"),
        }
    }

    #[test]
    fn test_state_new_all_above_cutoff_stopping() {
        // All variants above cutoff but not stopped → BanditsOnly
        let candidates = vec!["A".to_string(), "B".to_string()];
        let performances = vec![
            create_feedback("A", 50, 0.5, 0.05),
            create_feedback("B", 50, 0.7, 0.1),
        ];

        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            None,
            MetricConfigOptimize::Min,
        )
        .unwrap();

        match state {
            TrackAndStopState::Stopped {
                winner_variant_name,
            } => {
                assert_eq!(winner_variant_name, "A");
            }
            _ => panic!("Expected Stopped state, got {state:?}"),
        }
    }

    #[test]
    fn test_state_new_single_variant() {
        // Single variant case → Stopped
        let candidates = vec!["A".to_string()];
        let performances = vec![create_feedback("A", 5, 0.5, 0.1)];

        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            None,
            MetricConfigOptimize::Max,
        )
        .unwrap();

        match state {
            TrackAndStopState::Stopped {
                winner_variant_name,
            } => {
                assert_eq!(winner_variant_name, "A");
            }
            _ => panic!("Expected Stopped state, got {state:?}"),
        }
    }

    #[test]
    fn test_state_new_nursery_and_bandits() {
        // Some variants above cutoff, some below -> here should produce NurseryAndBandits state
        let candidates = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let performances = vec![
            create_feedback("A", 20, 0.5, 0.1), // Above cutoff
            create_feedback("B", 20, 0.6, 0.2), // Above cutoff
            create_feedback("C", 5, 0.7, 0.15), // Below cutoff
        ];

        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            None,
            MetricConfigOptimize::Min,
        )
        .unwrap();

        match state {
            TrackAndStopState::NurseryAndBandits {
                nursery,
                sampling_probabilities,
            } => {
                // Nursery should have variant C
                assert_eq!(nursery.variants, vec!["C".to_string()]);
                // Bandits should have A and B
                assert_eq!(sampling_probabilities.len(), 2);
                assert!(sampling_probabilities.contains_key("A"));
                assert!(sampling_probabilities.contains_key("B"));
            }
            _ => panic!("Expected NurseryAndBandits state, got {state:?}"),
        }
    }

    #[test]
    fn test_state_new_nursery_and_stopped() {
        // Some variants above cutoff, some below -> here should produce NurseryAndStopped state
        let candidates = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let performances = vec![
            create_feedback("A", 20, 0.5, 0.1), // Above cutoff
            create_feedback("B", 20, 1.0, 0.2), // Above cutoff; clear winner
            create_feedback("C", 5, 0.7, 0.15), // Below cutoff
        ];

        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            None,
            MetricConfigOptimize::Max,
        )
        .unwrap();

        match state {
            TrackAndStopState::NurseryAndStopped {
                nursery,
                stopped_variant_name,
            } => {
                assert_eq!(nursery.variants, vec!["C".to_string()]);
                assert_eq!(stopped_variant_name, "B");
            }
            _ => panic!("Expected NurseryAndBandits state, got {state:?}"),
        }
    }

    #[test]
    fn test_state_new_zero_variants() {
        // Edge case: zero candidates should return NoArmsDetected error
        let candidates = vec![];
        let performances = vec![];

        let result = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            None,
            MetricConfigOptimize::Min,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            TrackAndStopError::NoArmsDetected => {
                // Expected
            }
            other => panic!("Expected NoArmsDetected error, got {other:?}"),
        }
    }

    // Tests for TrackAndStopState::sample()
    #[test]
    fn test_sample_stopped_winner_active() {
        let state = TrackAndStopState::Stopped {
            winner_variant_name: "A".to_string(),
        };
        let active = create_test_variants(&["A", "B", "C"]);

        let result = state.sample(&active, 0.5).unwrap();
        assert_eq!(result, Some("A"));
    }

    #[test]
    fn test_sample_stopped_winner_inactive() {
        let state = TrackAndStopState::Stopped {
            winner_variant_name: "D".to_string(),
        };
        let active = create_test_variants(&["A", "B", "C"]);

        let result = state.sample(&active, 0.5).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_sample_nursery_only() {
        let state = TrackAndStopState::NurseryOnly(Nursery::new(vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
        ]));
        let active = create_test_variants(&["A", "B", "C"]);

        // First call should return "A" (index starts at 0)
        let result = state.sample(&active, 0.5).unwrap();
        assert_eq!(result, Some("A"));
    }

    #[test]
    fn test_sample_nursery_only_partial_active() {
        let state = TrackAndStopState::NurseryOnly(Nursery::new(vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
        ]));
        let active = create_test_variants(&["B", "C"]);

        // Index starts at 0
        // Try 1: "A" (index 0) - not active
        // Try 2: "B" (index 1) - active! Return "B"
        let result = state.sample(&active, 0.5).unwrap();
        assert_eq!(result, Some("B"));
    }

    #[test]
    fn test_sample_bandits_only() {
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.3);
        probs.insert("B".to_string(), 0.7);

        let state = TrackAndStopState::BanditsOnly {
            sampling_probabilities: probs,
        };
        let active = create_test_variants(&["A", "B"]);

        // With uniform_sample = 0.2, should select A (0.0 <= 0.2 < 0.3)
        let result = state.sample(&active, 0.2).unwrap();
        assert_eq!(result, Some("A"));

        // With uniform_sample = 0.5, should select B (0.3 <= 0.5 < 1.0)
        let result = state.sample(&active, 0.5).unwrap();
        assert_eq!(result, Some("B"));
    }

    #[test]
    fn test_sample_nursery_and_bandits_nursery_selected() {
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.5);
        probs.insert("B".to_string(), 0.5);

        let state = TrackAndStopState::NurseryAndBandits {
            nursery: Nursery::new(vec!["C".to_string()]),
            sampling_probabilities: probs,
        };
        let active = create_test_variants(&["A", "B", "C"]);

        // Total variants = 2 bandits + 1 nursery = 3
        // Nursery probability = 1/3 ≈ 0.333
        // With uniform_sample = 0.2 < 0.333, should select from nursery (C)
        let result = state.sample(&active, 0.2).unwrap();
        assert_eq!(result, Some("C"));
    }

    #[test]
    fn test_sample_nursery_and_bandits_bandits_selected() {
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.3);
        probs.insert("B".to_string(), 0.7);

        let state = TrackAndStopState::NurseryAndBandits {
            nursery: Nursery::new(vec!["C".to_string()]),
            sampling_probabilities: probs,
        };
        let active = create_test_variants(&["A", "B", "C"]);

        // Total variants = 2 bandits + 1 nursery = 3
        // Nursery probability = 1/3 ≈ 0.333
        // With uniform_sample = 0.6 > 0.333, should select from bandits
        // Rescaling to bandit probabilities: 0.5 should map to A
        let result = state.sample(&active, 0.5).unwrap();
        assert_eq!(result, Some("A"));
    }

    #[test]
    fn test_sample_nursery_and_stopped_nursery_selected() {
        let state = TrackAndStopState::NurseryAndStopped {
            nursery: Nursery::new(vec!["C".to_string()]),
            stopped_variant_name: "A".to_string(),
        };
        let active = create_test_variants(&["A", "C"]);

        // Total variants = 1 stopped + 1 nursery = 2
        // Nursery probability = 1/2 = 0.5
        // With uniform_sample = 0.3 < 0.5, should select from nursery (C)
        let result = state.sample(&active, 0.3).unwrap();
        assert_eq!(result, Some("C"));
    }

    #[test]
    fn test_sample_nursery_and_stopped_stopped_selected() {
        let state = TrackAndStopState::NurseryAndStopped {
            nursery: Nursery::new(vec!["C".to_string()]),
            stopped_variant_name: "A".to_string(),
        };
        let active = create_test_variants(&["A", "C"]);

        // Total variants = 1 stopped + 1 nursery = 2
        // Nursery probability = 1/2 = 0.5
        // With uniform_sample = 0.7 > 0.5, should select stopped variant (A)
        let result = state.sample(&active, 0.7).unwrap();
        assert_eq!(result, Some("A"));
    }

    #[test]
    fn test_sample_nursery_and_stopped_stopped_inactive() {
        let state = TrackAndStopState::NurseryAndStopped {
            nursery: Nursery::new(vec!["C".to_string()]),
            stopped_variant_name: "A".to_string(),
        };
        let active = create_test_variants(&["C"]); // A is not active

        // With uniform_sample = 0.7 > 0.5, would try to select stopped variant
        // But A is not active, so should return None
        let result = state.sample(&active, 0.7).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_sample_bandits_negative_probability_error() {
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.5);
        probs.insert("B".to_string(), -0.1); // Negative!

        let state = TrackAndStopState::BanditsOnly {
            sampling_probabilities: probs,
        };
        let active = create_test_variants(&["A", "B"]);

        let result = state.sample(&active, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TrackAndStopError::NegativeProbability { .. }
        ));
    }

    #[test]
    fn test_track_and_stop_state_bandits_only_excludes_fallback_feedback() {
        let candidates = vec!["A".to_string(), "B".to_string()];
        let feedback = vec![
            create_feedback("A", 20, 0.5, 0.1),
            create_feedback("B", 20, 0.5, 0.1),
            create_feedback("fallback", 20, 0.9, 0.1),
        ];

        let state = TrackAndStopState::new(
            &candidates,
            feedback,
            5,
            0.1,
            0.0,
            None,
            MetricConfigOptimize::Max,
        )
        .expect("state builds");

        let TrackAndStopState::BanditsOnly {
            sampling_probabilities,
        } = state
        else {
            panic!("expected BanditsOnly state");
        };

        assert_eq!(sampling_probabilities.len(), 2);
        assert!(sampling_probabilities.contains_key("A"));
        assert!(sampling_probabilities.contains_key("B"));
        assert!(
            !sampling_probabilities.contains_key("fallback"),
            "fallback variants should be filtered out before probability estimation"
        );
    }

    #[test]
    fn test_track_and_stop_state_nursery_and_bandits_filters_fallbacks() {
        let candidates = vec!["A".to_string(), "B".to_string(), "D".to_string()];
        let feedback = vec![
            create_feedback("A", 20, 0.5, 0.1),
            create_feedback("B", 20, 0.4, 0.1),
            create_feedback("D", 2, 0.3, 0.1), // still in nursery
            create_feedback("fallback", 30, 0.9, 0.1),
        ];

        let state = TrackAndStopState::new(
            &candidates,
            feedback,
            5,
            0.1,
            0.0,
            None,
            MetricConfigOptimize::Min,
        )
        .expect("state builds");

        let TrackAndStopState::NurseryAndBandits {
            nursery,
            sampling_probabilities,
        } = state
        else {
            panic!("expected NurseryAndBandits state");
        };

        assert_eq!(nursery.variants, vec!["D".to_string()]);
        assert!(sampling_probabilities.contains_key("A"));
        assert!(sampling_probabilities.contains_key("B"));
        assert!(
            !sampling_probabilities.contains_key("fallback"),
            "fallback variants should not appear in bandit probabilities"
        );
    }

    #[tokio::test]
    async fn test_setup_prevents_duplicate_task_spawning() {
        // Create a TrackAndStopConfig instance
        let config = TrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string()],
            fallback_variants: vec!["C".to_string()],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.1,
            update_period: Duration::from_secs(60),
            min_prob: Some(0.001),
            metric_optimize: MetricConfigOptimize::Min,
            state: Arc::new(ArcSwap::new(Arc::new(
                TrackAndStopState::nursery_from_variants(vec!["A".to_string(), "B".to_string()]),
            ))),
            task_spawned: AtomicBool::new(false),
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let postgres = PostgresConnectionInfo::new_mock(true);
        let cancel_token = CancellationToken::new();

        // First call to setup should succeed
        let result1 = config
            .setup(db.clone(), "test_function", &postgres, cancel_token.clone())
            .await;
        assert!(result1.is_ok(), "First setup call should succeed");

        // Second call to setup should fail with an error
        let result2 = config
            .setup(db, "test_function", &postgres, cancel_token.clone())
            .await;
        assert!(result2.is_err(), "Second setup call should fail");

        // Verify the error message mentions the task has already been spawned
        let err = result2.unwrap_err();
        assert!(
            err.to_string().contains("already been spawned"),
            "Error should mention task already spawned, got: {err}"
        );

        // Clean up: cancel the spawned task
        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_setup_requires_postgres() {
        // Create a TrackAndStopConfig instance
        let config = TrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string()],
            fallback_variants: vec!["C".to_string()],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.1,
            update_period: Duration::from_secs(60),
            min_prob: Some(0.001),
            metric_optimize: MetricConfigOptimize::Max,
            state: Arc::new(ArcSwap::new(Arc::new(
                TrackAndStopState::nursery_from_variants(vec!["A".to_string(), "B".to_string()]),
            ))),
            task_spawned: AtomicBool::new(false),
        };

        let db = Arc::new(ClickHouseConnectionInfo::new_disabled())
            as Arc<dyn FeedbackQueries + Send + Sync>;
        let cancel_token = CancellationToken::new();

        // Test with disabled Postgres
        let postgres_disabled = PostgresConnectionInfo::new_disabled();
        let result = config
            .setup(
                db.clone(),
                "test_function",
                &postgres_disabled,
                cancel_token.clone(),
            )
            .await;

        assert!(
            result.is_err(),
            "Setup should fail when Postgres is disabled"
        );
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Track-and-Stop"),
            "Error message should mention Track-and-Stop, got: {err_msg}"
        );
        assert!(
            err_msg.contains("PostgreSQL"),
            "Error message should mention PostgreSQL, got: {err_msg}"
        );
        assert!(
            err_msg.contains("test_function"),
            "Error message should mention the function name, got: {err_msg}"
        );

        // Test with mock Postgres (unhealthy)
        let postgres_unhealthy = PostgresConnectionInfo::new_mock(false);
        let result = config
            .setup(
                db.clone(),
                "test_function",
                &postgres_unhealthy,
                cancel_token.clone(),
            )
            .await;

        assert!(
            result.is_err(),
            "Setup should fail when Postgres is unhealthy"
        );
    }

    // Tests for min_prob parameter being passed through correctly
    #[test]
    fn test_min_prob_passed_to_estimate_optimal_probabilities() {
        // Test that min_prob is passed through to estimate_optimal_probabilities
        // by checking that all probabilities respect the min_prob constraint
        let candidates = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let performances = vec![
            create_feedback("A", 20, 0.5, 0.1),
            create_feedback("B", 20, 0.6, 0.2),
            create_feedback("C", 20, 0.55, 0.15),
        ];

        let min_prob = 0.15; // Set a high min_prob to ensure it's enforced
        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            Some(min_prob),
            MetricConfigOptimize::Max,
        )
        .unwrap();

        match state {
            TrackAndStopState::BanditsOnly {
                sampling_probabilities,
            } => {
                // All probabilities should be >= min_prob
                for (variant_name, &prob) in &sampling_probabilities {
                    assert!(
                        prob >= min_prob - 1e-6,
                        "Variant {variant_name} has probability {prob} which violates min_prob {min_prob}"
                    );
                }
            }
            _ => panic!("Expected BanditsOnly state, got {state:?}"),
        }
    }

    #[test]
    fn test_min_prob_none_uses_default() {
        // Test that when min_prob is None, the default config value (0.0) is used,
        // but the optimization algorithm applies a floor of 1e-6 for numerical stability
        let candidates = vec!["A".to_string(), "B".to_string()];
        let performances = vec![
            create_feedback("A", 20, 0.5, 0.1),
            create_feedback("B", 20, 0.6, 0.2),
        ];

        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            None, // min_prob is None, defaults to 0.0 in config, but 1e-6 is applied in optimization
            MetricConfigOptimize::Max,
        )
        .unwrap();

        match state {
            TrackAndStopState::BanditsOnly {
                sampling_probabilities,
            } => {
                // All probabilities should be >= 1e-6 (the floor applied in optimization)
                for (variant_name, &prob) in &sampling_probabilities {
                    assert!(
                        prob >= 1e-6 - 1e-9,
                        "Variant {variant_name} has probability {prob} which is less than the optimization floor (1e-6)"
                    );
                }
            }
            _ => panic!("Expected BanditsOnly state, got {state:?}"),
        }
    }

    #[test]
    fn test_min_prob_zero_accepted_and_uses_floor() {
        // Test that min_prob=0.0 is accepted in config but the optimization
        // algorithm applies a floor of 1e-6 for numerical stability
        let candidates = vec!["A".to_string(), "B".to_string()];
        let performances = vec![
            create_feedback("A", 20, 0.5, 0.1),
            create_feedback("B", 20, 0.6, 0.2),
        ];

        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            Some(0.0), // min_prob is explicitly set to 0.0
            MetricConfigOptimize::Max,
        )
        .unwrap();

        match state {
            TrackAndStopState::BanditsOnly {
                sampling_probabilities,
            } => {
                // All probabilities should be >= 1e-6 (the floor applied in optimization)
                // even though min_prob was set to 0.0
                for (variant_name, &prob) in &sampling_probabilities {
                    assert!(
                        prob >= 1e-6 - 1e-9,
                        "Variant {variant_name} has probability {prob} which is less than the optimization floor (1e-6)"
                    );
                }
            }
            _ => panic!("Expected BanditsOnly state, got {state:?}"),
        }
    }

    #[test]
    fn test_min_prob_in_nursery_and_bandits_state() {
        // Test that min_prob is respected in NurseryAndBandits state
        let candidates = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let performances = vec![
            create_feedback("A", 20, 0.5, 0.1),  // Above cutoff
            create_feedback("B", 20, 0.6, 0.2),  // Above cutoff
            create_feedback("C", 5, 0.55, 0.15), // Below cutoff, in nursery
        ];

        let min_prob = 0.2;
        let state = TrackAndStopState::new(
            &candidates,
            performances,
            10,
            0.05,
            0.0,
            Some(min_prob),
            MetricConfigOptimize::Max,
        )
        .unwrap();

        match state {
            TrackAndStopState::NurseryAndBandits {
                nursery,
                sampling_probabilities,
            } => {
                // Nursery should have variant C
                assert_eq!(nursery.variants, vec!["C".to_string()]);

                // All bandit probabilities should be >= min_prob
                for (variant_name, &prob) in &sampling_probabilities {
                    assert!(
                        prob >= min_prob - 1e-6,
                        "Variant {variant_name} has probability {prob} which violates min_prob {min_prob}"
                    );
                }
            }
            _ => panic!("Expected NurseryAndBandits state, got {state:?}"),
        }
    }

    // Tests for get_current_display_probabilities
    #[test]
    fn test_get_current_display_probabilities_stopped() {
        let config = TrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string()],
            fallback_variants: vec![],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period: Duration::from_secs(60),
            min_prob: None,
            metric_optimize: MetricConfigOptimize::Max,
            state: Arc::new(ArcSwap::new(Arc::new(TrackAndStopState::Stopped {
                winner_variant_name: "A".to_string(),
            }))),
            task_spawned: AtomicBool::new(false),
        };

        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Only the winner should have probability 1.0
        assert_eq!(probs.len(), 1);
        assert_eq!(probs.get("A"), Some(&1.0));
    }

    #[test]
    fn test_get_current_display_probabilities_nursery_only() {
        let config = TrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            fallback_variants: vec![],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period: Duration::from_secs(60),
            min_prob: None,
            metric_optimize: MetricConfigOptimize::Max,
            state: Arc::new(ArcSwap::new(Arc::new(TrackAndStopState::NurseryOnly(
                Nursery::new(vec!["A".to_string(), "B".to_string(), "C".to_string()]),
            )))),
            task_spawned: AtomicBool::new(false),
        };

        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Should have uniform probabilities
        assert_eq!(probs.len(), 3);
        assert!((probs["A"] - 1.0 / 3.0).abs() < 1e-9);
        assert!((probs["B"] - 1.0 / 3.0).abs() < 1e-9);
        assert!((probs["C"] - 1.0 / 3.0).abs() < 1e-9);

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_bandits_only() {
        let mut sampling_probs = HashMap::new();
        sampling_probs.insert("A".to_string(), 0.3);
        sampling_probs.insert("B".to_string(), 0.7);

        let config = TrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string()],
            fallback_variants: vec![],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period: Duration::from_secs(60),
            min_prob: None,
            metric_optimize: MetricConfigOptimize::Max,
            state: Arc::new(ArcSwap::new(Arc::new(TrackAndStopState::BanditsOnly {
                sampling_probabilities: sampling_probs,
            }))),
            task_spawned: AtomicBool::new(false),
        };

        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        assert_eq!(probs.len(), 2);
        assert!((probs["A"] - 0.3).abs() < 1e-9);
        assert!((probs["B"] - 0.7).abs() < 1e-9);

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_nursery_and_bandits() {
        let mut sampling_probs = HashMap::new();
        sampling_probs.insert("A".to_string(), 0.3);
        sampling_probs.insert("B".to_string(), 0.7);

        let config = TrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            fallback_variants: vec![],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period: Duration::from_secs(60),
            min_prob: None,
            metric_optimize: MetricConfigOptimize::Max,
            state: Arc::new(ArcSwap::new(Arc::new(
                TrackAndStopState::NurseryAndBandits {
                    nursery: Nursery::new(vec!["C".to_string()]),
                    sampling_probabilities: sampling_probs,
                },
            ))),
            task_spawned: AtomicBool::new(false),
        };

        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "B", "C"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        assert_eq!(probs.len(), 3);

        // Total: 2 bandits + 1 nursery = 3 variants
        // Nursery gets 1/3 of probability mass
        // Bandits get 2/3 of probability mass
        let nursery_mass = 1.0 / 3.0;
        let bandit_mass = 2.0 / 3.0;

        // C gets all nursery mass (1 nursery variant)
        assert!(
            (probs["C"] - nursery_mass).abs() < 1e-9,
            "C should get {}, but got {}",
            nursery_mass,
            probs["C"]
        );

        // A and B get their proportional shares of bandit mass
        let expected_a = 0.3 * bandit_mass;
        let expected_b = 0.7 * bandit_mass;

        assert!(
            (probs["A"] - expected_a).abs() < 1e-9,
            "A should get {}, but got {}",
            expected_a,
            probs["A"]
        );
        assert!(
            (probs["B"] - expected_b).abs() < 1e-9,
            "B should get {}, but got {}",
            expected_b,
            probs["B"]
        );

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_nursery_and_stopped() {
        let config = TrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "C".to_string()],
            fallback_variants: vec![],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period: Duration::from_secs(60),
            min_prob: None,
            metric_optimize: MetricConfigOptimize::Max,
            state: Arc::new(ArcSwap::new(Arc::new(
                TrackAndStopState::NurseryAndStopped {
                    nursery: Nursery::new(vec!["C".to_string()]),
                    stopped_variant_name: "A".to_string(),
                },
            ))),
            task_spawned: AtomicBool::new(false),
        };

        let active_variants: HashMap<_, _> =
            create_test_variants(&["A", "C"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        assert_eq!(probs.len(), 2);

        // Total: 1 stopped + 1 nursery = 2 variants
        // Each gets 1/2
        assert!((probs["C"] - 0.5).abs() < 1e-9);
        assert!((probs["A"] - 0.5).abs() < 1e-9);

        // Check sum
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_partial_active() {
        // Test with only some variants active
        let mut sampling_probs = HashMap::new();
        sampling_probs.insert("A".to_string(), 0.4);
        sampling_probs.insert("B".to_string(), 0.6);

        let config = TrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string()],
            fallback_variants: vec![],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period: Duration::from_secs(60),
            min_prob: None,
            metric_optimize: MetricConfigOptimize::Max,
            state: Arc::new(ArcSwap::new(Arc::new(TrackAndStopState::BanditsOnly {
                sampling_probabilities: sampling_probs,
            }))),
            task_spawned: AtomicBool::new(false),
        };

        // Only A is active, B is not
        let active_variants: HashMap<_, _> = create_test_variants(&["A"]).into_iter().collect();
        let postgres = PostgresConnectionInfo::new_disabled();

        let probs = config
            .get_current_display_probabilities("test", &active_variants, &postgres)
            .unwrap();

        // Only A should appear, and since it's the only active variant,
        // it should be renormalized to have probability 1.0
        assert_eq!(probs.len(), 1);
        assert!((probs["A"] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_current_display_probabilities_inactive_variants_sum_to_one() {
        // This test verifies that probabilities always sum to 1.0 when some configured
        // variants are inactive, across all state types.
        let postgres = PostgresConnectionInfo::new_disabled();

        // Test 1: NurseryOnly with some inactive nursery variants
        {
            let nursery = Nursery {
                variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                index: AtomicU64::new(0),
            };
            let config = TrackAndStopConfig {
                metric: "test_metric".to_string(),
                candidate_variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                fallback_variants: vec![],
                min_samples_per_variant: 10,
                delta: 0.05,
                epsilon: 0.0,
                update_period: Duration::from_secs(60),
                min_prob: None,
                metric_optimize: MetricConfigOptimize::Max,
                state: Arc::new(ArcSwap::new(Arc::new(TrackAndStopState::NurseryOnly(
                    nursery,
                )))),
                task_spawned: AtomicBool::new(false),
            };
            // Only A is active
            let active_variants: HashMap<_, _> = create_test_variants(&["A"]).into_iter().collect();
            let probs = config
                .get_current_display_probabilities("test", &active_variants, &postgres)
                .unwrap();
            let sum: f64 = probs.values().sum();
            assert!((sum - 1.0).abs() < 1e-9, "NurseryOnly: sum={sum}");
        }

        // Test 2: BanditsOnly with some inactive bandit variants
        {
            let mut sampling_probs = HashMap::new();
            sampling_probs.insert("A".to_string(), 0.2);
            sampling_probs.insert("B".to_string(), 0.3);
            sampling_probs.insert("C".to_string(), 0.5);
            let config = TrackAndStopConfig {
                metric: "test_metric".to_string(),
                candidate_variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                fallback_variants: vec![],
                min_samples_per_variant: 10,
                delta: 0.05,
                epsilon: 0.0,
                update_period: Duration::from_secs(60),
                min_prob: None,
                metric_optimize: MetricConfigOptimize::Max,
                state: Arc::new(ArcSwap::new(Arc::new(TrackAndStopState::BanditsOnly {
                    sampling_probabilities: sampling_probs,
                }))),
                task_spawned: AtomicBool::new(false),
            };
            // Only A is active
            let active_variants: HashMap<_, _> = create_test_variants(&["A"]).into_iter().collect();
            let probs = config
                .get_current_display_probabilities("test", &active_variants, &postgres)
                .unwrap();
            let sum: f64 = probs.values().sum();
            assert!((sum - 1.0).abs() < 1e-9, "BanditsOnly: sum={sum}");
        }

        // Test 3: NurseryAndBandits with some inactive variants in both
        {
            let nursery = Nursery {
                variants: vec!["A".to_string(), "B".to_string()],
                index: AtomicU64::new(0),
            };
            let mut sampling_probs = HashMap::new();
            sampling_probs.insert("C".to_string(), 0.4);
            sampling_probs.insert("D".to_string(), 0.6);
            let config = TrackAndStopConfig {
                metric: "test_metric".to_string(),
                candidate_variants: vec![
                    "A".to_string(),
                    "B".to_string(),
                    "C".to_string(),
                    "D".to_string(),
                ],
                fallback_variants: vec![],
                min_samples_per_variant: 10,
                delta: 0.05,
                epsilon: 0.0,
                update_period: Duration::from_secs(60),
                min_prob: None,
                metric_optimize: MetricConfigOptimize::Max,
                state: Arc::new(ArcSwap::new(Arc::new(
                    TrackAndStopState::NurseryAndBandits {
                        nursery,
                        sampling_probabilities: sampling_probs,
                    },
                ))),
                task_spawned: AtomicBool::new(false),
            };
            // Only A and C are active
            let active_variants: HashMap<_, _> =
                create_test_variants(&["A", "C"]).into_iter().collect();
            let probs = config
                .get_current_display_probabilities("test", &active_variants, &postgres)
                .unwrap();
            let sum: f64 = probs.values().sum();
            assert!((sum - 1.0).abs() < 1e-9, "NurseryAndBandits: sum={sum}");
        }

        // Test 4: NurseryAndStopped with some inactive nursery variants
        {
            let nursery = Nursery {
                variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                index: AtomicU64::new(0),
            };
            let config = TrackAndStopConfig {
                metric: "test_metric".to_string(),
                candidate_variants: vec![
                    "A".to_string(),
                    "B".to_string(),
                    "C".to_string(),
                    "D".to_string(),
                ],
                fallback_variants: vec![],
                min_samples_per_variant: 10,
                delta: 0.05,
                epsilon: 0.0,
                update_period: Duration::from_secs(60),
                min_prob: None,
                metric_optimize: MetricConfigOptimize::Max,
                state: Arc::new(ArcSwap::new(Arc::new(
                    TrackAndStopState::NurseryAndStopped {
                        nursery,
                        stopped_variant_name: "D".to_string(),
                    },
                ))),
                task_spawned: AtomicBool::new(false),
            };
            // Only A and D are active
            let active_variants: HashMap<_, _> =
                create_test_variants(&["A", "D"]).into_iter().collect();
            let probs = config
                .get_current_display_probabilities("test", &active_variants, &postgres)
                .unwrap();
            let sum: f64 = probs.values().sum();
            assert!((sum - 1.0).abs() < 1e-9, "NurseryAndStopped: sum={sum}");
        }
    }

    #[test]
    fn test_load_error_empty_candidate_variants() {
        let config = UninitializedTrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec![], // Empty!
            fallback_variants: vec!["A".to_string()],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period_s: 60,
            min_prob: None,
        };

        let mut variants = HashMap::new();
        variants.insert(
            "A".to_string(),
            Arc::new(VariantInfo {
                inner: VariantConfig::ChatCompletion(
                    UninitializedChatCompletionConfig {
                        weight: None,
                        model: "model-name".into(),
                        ..Default::default()
                    }
                    .load(&SchemaData::default(), &ErrorContext::new_test())
                    .unwrap(),
                ),
                timeouts: Default::default(),
            }),
        );

        let mut metrics = HashMap::new();
        metrics.insert(
            "test_metric".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
            },
        );

        let result = config.load(&variants, &metrics);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("candidate_variants cannot be empty"));
    }

    #[test]
    fn test_load_error_duplicate_candidates() {
        let config = UninitializedTrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string(), "A".to_string()],
            fallback_variants: vec![],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period_s: 60,
            min_prob: None,
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let mut metrics = HashMap::new();
        metrics.insert(
            "test_metric".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
            },
        );

        let result = config.load(&variants, &metrics);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("candidate_variants"));
        assert!(err_msg.contains("duplicate entries"));
        assert!(err_msg.contains("A"));
    }

    #[test]
    fn test_load_error_duplicate_fallbacks() {
        let config = UninitializedTrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string()],
            fallback_variants: vec!["B".to_string(), "C".to_string(), "B".to_string()],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period_s: 60,
            min_prob: None,
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let mut metrics = HashMap::new();
        metrics.insert(
            "test_metric".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
            },
        );

        let result = config.load(&variants, &metrics);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("fallback_variants"));
        assert!(err_msg.contains("duplicate entries"));
        assert!(err_msg.contains("B"));
    }

    #[test]
    fn test_load_error_duplicate_across_lists() {
        let config = UninitializedTrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string()],
            fallback_variants: vec!["B".to_string(), "C".to_string()],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period_s: 60,
            min_prob: None,
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C"]).into_iter().collect();

        let mut metrics = HashMap::new();
        metrics.insert(
            "test_metric".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
            },
        );

        let result = config.load(&variants, &metrics);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot appear in both `candidate_variants` and `fallback_variants`")
        );
        assert!(err_msg.contains("B"));
    }

    #[test]
    fn test_load_error_multiple_duplicates_across_lists() {
        let config = UninitializedTrackAndStopConfig {
            metric: "test_metric".to_string(),
            candidate_variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            fallback_variants: vec!["B".to_string(), "C".to_string(), "D".to_string()],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.0,
            update_period_s: 60,
            min_prob: None,
        };

        let variants: HashMap<_, _> = create_test_variants(&["A", "B", "C", "D"])
            .into_iter()
            .collect();

        let mut metrics = HashMap::new();
        metrics.insert(
            "test_metric".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
            },
        );

        let result = config.load(&variants, &metrics);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot appear in both `candidate_variants` and `fallback_variants`")
        );
        assert!(err_msg.contains("B"));
        assert!(err_msg.contains("C"));
    }
}
