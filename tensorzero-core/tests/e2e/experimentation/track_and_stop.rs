use crate::clickhouse::{get_clean_clickhouse, DeleteDbOnDrop};
use futures::future::join_all;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tempfile::NamedTempFile;
use tensorzero::{Client, ClientBuilder, ClientBuilderMode};
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    FeedbackParams, InferenceOutput, InferenceResponse, Role,
};
use tensorzero_core::db::clickhouse::test_helpers::clickhouse_flush_async_insert;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::inference::types::TextKind;
use tokio::time::Duration;
use url::Url;
use uuid::Uuid;

// ============================================================================
// Test Helpers
// ============================================================================

/// Build an embedded gateway backed by a fresh ClickHouse database.
/// Returns the client, the ClickHouse connection, and a guard that drops the database.
async fn make_embedded_gateway_with_clean_clickhouse(
    config: &str,
) -> (Client, ClickHouseConnectionInfo, DeleteDbOnDrop) {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for tests that require Postgres");

    let (clickhouse, guard) = get_clean_clickhouse(false).await;

    clickhouse
        .create_database_and_migrations_table()
        .await
        .expect("failed to create ClickHouse database for embedded gateway tests");

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    // Point the embedded gateway at the freshly-created database.
    let database = clickhouse.database();
    let mut clickhouse_url = Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path(database);
    let clickhouse_url_string = clickhouse_url.to_string();

    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(clickhouse_url_string),
        postgres_url: Some(postgres_url),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();

    (client, clickhouse, guard)
}

/// Build an embedded gateway backed by an existing ClickHouse database.
/// This is useful for testing cold-start behavior where a new gateway needs to read
/// existing data from the database.
async fn make_embedded_gateway_with_existing_clickhouse(
    config: &str,
    existing_clickhouse: &ClickHouseConnectionInfo,
) -> (Client, ClickHouseConnectionInfo) {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for tests that require Postgres");

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    // Point the embedded gateway at the existing database.
    let database = existing_clickhouse.database();
    let mut clickhouse_url = Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path(database);
    let clickhouse_url_string = clickhouse_url.to_string();

    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(clickhouse_url_string),
        postgres_url: Some(postgres_url),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();

    (client, existing_clickhouse.clone())
}

// ============================================================================
// Test Constants
// ============================================================================

/// Threshold for detecting stopped state: fraction of pulls that must go to one variant.
/// Set to 1.0 to ensure deterministic stopping behavior.
const STOPPED_THRESHOLD: f64 = 1.0;

/// Minimum fraction of pulls a variant should receive to be considered "explored".
const MIN_EXPLORATION_FRACTION: f64 = 0.10;

/// Delay after ClickHouse flush for inferences (milliseconds).
const CLICKHOUSE_FLUSH_DELAY_MS: u64 = 1000;

/// Delay after ClickHouse flush for feedback to allow background update (milliseconds).
const BACKGROUND_UPDATE_DELAY_MS: u64 = 1000;

/// Delay for background task initialization when creating a new client (milliseconds).
const BACKGROUND_TASK_INIT_DELAY_MS: u64 = 1000;

// ============================================================================
// Test Configuration
// ============================================================================

/// Configuration parameters for generating a Track-and-Stop test config.
#[derive(Debug, Clone)]
pub struct TrackAndStopTestConfig<'a> {
    pub metric_name: &'a str,
    pub metric_type: &'a str,
    pub optimize: &'a str,
    pub candidate_variants: &'a [&'a str],
    pub fallback_variants: &'a [&'a str],
    pub min_samples_per_variant: u64,
    pub delta: f64,
    pub epsilon: f64,
    pub update_period_s: u64,
    pub min_prob: Option<f64>,
}

impl Default for TrackAndStopTestConfig<'static> {
    fn default() -> Self {
        Self {
            metric_name: "test_metric",
            metric_type: "boolean",
            optimize: "max",
            candidate_variants: &["variant_a", "variant_b"],
            fallback_variants: &["variant_fallback"],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.1,
            update_period_s: 300,
            min_prob: None,
        }
    }
}

/// Helper function to generate a minimal Track-and-Stop config for testing.
///
/// Returns a complete TOML config string with:
/// - A dummy model provider
/// - A function with the specified variants
/// - A metric
/// - Track-and-Stop experimentation config
pub fn make_track_and_stop_config(config: TrackAndStopTestConfig) -> String {
    let TrackAndStopTestConfig {
        metric_name,
        metric_type,
        optimize,
        candidate_variants,
        fallback_variants,
        min_samples_per_variant,
        delta,
        epsilon,
        update_period_s,
        min_prob,
    } = config;
    // Create variant definitions for the union of candidate and fallback variants
    let all_variants: std::collections::HashSet<&str> = candidate_variants
        .iter()
        .chain(fallback_variants.iter())
        .copied()
        .collect();

    let variant_configs: Vec<String> = all_variants
        .iter()
        .map(|variant| {
            format!(
                r#"
[functions.test_function.variants.{variant}]
type = "chat_completion"
model = "test_model"
"#
            )
        })
        .collect();

    let candidate_variants_list = candidate_variants
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(", ");

    let fallback_variants_list = fallback_variants
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(", ");

    let min_prob_line = min_prob
        .map(|value| format!("min_prob = {value}\n"))
        .unwrap_or_default();

    format!(
        r#"
gateway.unstable_disable_feedback_target_validation = true

[models.test_model]
routing = ["dummy"]

[models.test_model.providers.dummy]
type = "dummy"
model_name = "test"

[metrics.{metric_name}]
type = "{metric_type}"
optimize = "{optimize}"
level = "inference"

[functions.test_function]
type = "chat"
{variants}

[functions.test_function.experimentation]
type = "track_and_stop"
metric = "{metric_name}"
candidate_variants = [{candidate_variants_list}]
fallback_variants = [{fallback_variants_list}]
min_samples_per_variant = {min_samples_per_variant}
delta = {delta}
epsilon = {epsilon}
update_period_s = {update_period_s}
{min_prob_line}"#,
        variants = variant_configs.join("\n"),
    )
}

/// Bernoulli bandit environment for testing.
///
/// Each arm (variant) returns true with probability p or false with probability (1-p).
/// Use this to generate rewards for boolean metrics.
pub struct BernoulliBandit {
    probabilities: std::collections::HashMap<String, f64>,
    rng: Option<Mutex<rand::rngs::StdRng>>,
}

impl BernoulliBandit {
    /// Create a new Bernoulli bandit environment with an optional seeded RNG.
    ///
    /// # Arguments
    /// * `variant_probs` - List of (variant_name, success_probability) tuples
    /// * `seed` - Optional seed for the random number generator. If None, uses entropy.
    ///
    /// # Panics
    /// Panics if any probability is outside [0, 1]
    pub fn new(variant_probs: Vec<(&str, f64)>, seed: Option<u64>) -> Self {
        let probabilities = variant_probs
            .into_iter()
            .map(|(name, prob)| {
                assert!(
                    (0.0..=1.0).contains(&prob),
                    "Probability must be in [0, 1], got {prob}"
                );
                (name.to_string(), prob)
            })
            .collect();

        let rng = seed.map(|s| Mutex::new(rand::rngs::StdRng::seed_from_u64(s)));

        Self { probabilities, rng }
    }

    /// Sample a reward (true/false) for the given variant.
    ///
    /// # Panics
    /// Panics if variant_name is not in the environment
    pub fn sample(&self, variant_name: &str) -> bool {
        let prob = self
            .probabilities
            .get(variant_name)
            .unwrap_or_else(|| panic!("Unknown variant: {variant_name}"));
        match &self.rng {
            Some(rng) => {
                let mut rng = rng.lock().unwrap();
                rand::Rng::random_bool(&mut *rng, *prob)
            }
            None => rand::rng().random_bool(*prob),
        }
    }
}

/// Gaussian bandit environment for testing.
///
/// Each arm (variant) returns a sample from N(μ, σ²).
/// Use this to generate rewards for float metrics.
pub struct GaussianBandit {
    distributions: std::collections::HashMap<String, (f64, f64)>, // (mean, stddev)
    rng: Option<Mutex<rand::rngs::StdRng>>,
}

impl GaussianBandit {
    /// Create a new Gaussian bandit environment with an optional seeded RNG.
    ///
    /// # Arguments
    /// * `variant_distributions` - List of (variant_name, mean, stddev) tuples
    /// * `seed` - Optional seed for the random number generator. If None, uses entropy.
    ///
    /// # Panics
    /// Panics if any stddev is negative
    pub fn new(variant_distributions: Vec<(&str, f64, f64)>, seed: Option<u64>) -> Self {
        let distributions = variant_distributions
            .into_iter()
            .map(|(name, mean, stddev)| {
                assert!(
                    stddev >= 0.0,
                    "Standard deviation must be non-negative, got {stddev}"
                );
                (name.to_string(), (mean, stddev))
            })
            .collect();

        let rng = seed.map(|s| Mutex::new(rand::rngs::StdRng::seed_from_u64(s)));

        Self { distributions, rng }
    }

    /// Sample a reward (float) for the given variant.
    ///
    /// # Panics
    /// Panics if variant_name is not in the environment
    pub fn sample(&self, variant_name: &str) -> f64 {
        let (mean, stddev) = self
            .distributions
            .get(variant_name)
            .unwrap_or_else(|| panic!("Unknown variant: {variant_name}"));

        let normal = rand_distr::Normal::new(*mean, *stddev)
            .unwrap_or_else(|e| panic!("Failed to create normal distribution: {e}"));
        match &self.rng {
            Some(rng) => {
                let mut rng = rng.lock().unwrap();
                normal.sample(&mut *rng)
            }
            None => normal.sample(&mut rand::rng()),
        }
    }
}

// ============================================================================
// Test Helper Functions
// ============================================================================

/// Run a batch of inferences and return the inference IDs and variant names.
async fn run_inference_batch(client: &Arc<Client>, count: usize) -> Vec<(Uuid, String)> {
    let inference_tasks: Vec<_> = (0..count)
        .map(|_| {
            let client = client.clone();
            async move {
                let output = client
                    .inference(ClientInferenceParams {
                        function_name: Some("test_function".to_string()),
                        input: ClientInput {
                            system: None,
                            messages: vec![ClientInputMessage {
                                role: Role::User,
                                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                                    text: "test input".to_string(),
                                })],
                            }],
                        },
                        ..Default::default()
                    })
                    .await
                    .unwrap();

                let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = output
                else {
                    panic!("Expected non-streaming chat response");
                };

                (response.inference_id, response.variant_name.clone())
            }
        })
        .collect();

    join_all(inference_tasks).await
}

/// Enum wrapper for different bandit types.
#[derive(Clone)]
enum Bandit {
    Bernoulli(Arc<BernoulliBandit>),
    Gaussian(Arc<GaussianBandit>),
}

impl Bandit {
    fn sample(&self, variant_name: &str) -> serde_json::Value {
        match self {
            Bandit::Bernoulli(b) => json!(b.sample(variant_name)),
            Bandit::Gaussian(b) => json!(b.sample(variant_name)),
        }
    }
}

/// Send feedback for a batch of inferences.
async fn send_feedback(
    client: &Arc<Client>,
    inference_results: &[(Uuid, String)],
    bandit: &Bandit,
    metric_name: &str,
) -> Vec<String> {
    let feedback_tasks: Vec<_> = inference_results
        .iter()
        .map(|(inference_id, variant_name)| {
            let client = client.clone();
            let bandit = bandit.clone();
            let metric_name = metric_name.to_string();
            let inference_id = *inference_id;
            let variant_name = variant_name.clone();
            async move {
                let reward = bandit.sample(&variant_name);
                client
                    .feedback(FeedbackParams {
                        inference_id: Some(inference_id),
                        metric_name,
                        value: reward,
                        ..Default::default()
                    })
                    .await
                    .unwrap();

                variant_name
            }
        })
        .collect();

    join_all(feedback_tasks).await
}

// ============================================================================
// Config Validation Tests
// ============================================================================

/// Helper to build a client and expect an error
async fn expect_config_error(config: &str) -> String {
    let tmp_config = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL.clone()),
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap_err()
    .to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_invalid_metric() {
    let config = r#"
  [models.test_model]
  routing = ["dummy"]

  [models.test_model.providers.dummy]
  type = "dummy"
  model_name = "test"

  # Note: NO [metrics.nonexistent_metric] section!

  [functions.test_function]
  type = "chat"

  [functions.test_function.variants.variant_a]
  type = "chat_completion"
  model = "test_model"

  [functions.test_function.experimentation]
  type = "track_and_stop"
  metric = "nonexistent_metric"  # References a metric that doesn't exist
  candidate_variants = ["variant_a"]
  fallback_variants = []
  min_samples_per_variant = 10
  delta = 0.05
  epsilon = 0.1
  update_period_s = 2
  "#;

    let err = expect_config_error(config).await;
    assert!(
        err.contains("unknown metric"),
        "Expected error about unknown metric, got: {err}"
    );
    assert!(err.contains("nonexistent_metric"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_invalid_candidate_variant() {
    let config = r#"
[models.test_model]
routing = ["dummy"]

[models.test_model.providers.dummy]
type = "dummy"
model_name = "test"

[metrics.accuracy]
type = "float"
optimize = "min"
level = "inference"

[functions.test_function]
type = "chat"

[functions.test_function.variants.variant_a]
type = "chat_completion"
model = "test_model"

[functions.test_function.experimentation]
type = "track_and_stop"
metric = "accuracy"
candidate_variants = ["variant_a", "nonexistent_variant"]
fallback_variants = []
min_samples_per_variant = 10
delta = 0.05
epsilon = 0.1
update_period_s = 300
"#;

    let err = expect_config_error(config).await;
    assert!(
        err.contains("unknown variant"),
        "Expected error about unknown variant, got: {err}"
    );
    assert!(err.contains("nonexistent_variant"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_invalid_fallback_variant() {
    let config = r#"
[models.test_model]
routing = ["dummy"]

[models.test_model.providers.dummy]
type = "dummy"
model_name = "test"

[metrics.accuracy]
type = "float"
optimize = "max"
level = "inference"

[functions.test_function]
type = "chat"

[functions.test_function.variants.variant_a]
type = "chat_completion"
model = "test_model"

[functions.test_function.experimentation]
type = "track_and_stop"
metric = "accuracy"
candidate_variants = ["variant_a"]
fallback_variants = ["nonexistent_variant"]
min_samples_per_variant = 10
delta = 0.05
epsilon = 0.1
update_period_s = 300
"#;

    let err = expect_config_error(config).await;
    assert!(
        err.contains("unknown variant"),
        "Expected error about unknown variant, got: {err}"
    );
    assert!(err.contains("nonexistent_variant"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_invalid_min_samples() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_type: "float",
        optimize: "min",
        candidate_variants: &["variant_a"],
        fallback_variants: &[],
        min_samples_per_variant: 0, // Invalid: must be >= 1
        ..Default::default()
    });

    let err = expect_config_error(&config).await;
    assert!(
        err.contains("min_samples_per_variant"),
        "Expected error about min_samples_per_variant, got: {err}"
    );
    assert!(err.contains(">= 1"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_invalid_delta() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_a"],
        fallback_variants: &[],
        delta: 1.5, // Invalid: must be in (0, 1)
        ..Default::default()
    });

    let err = expect_config_error(&config).await;
    assert!(
        err.contains("delta"),
        "Expected error about delta, got: {err}"
    );
    assert!(err.contains("(0, 1)"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_invalid_epsilon() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_type: "float",
        optimize: "min",
        candidate_variants: &["variant_a"],
        fallback_variants: &[],
        epsilon: -0.1, // Invalid: must be >= 0
        ..Default::default()
    });

    let err = expect_config_error(&config).await;
    assert!(
        err.contains("epsilon"),
        "Expected error about epsilon, got: {err}"
    );
    assert!(err.contains(">= 0"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_config_valid() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "accuracy",
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &["variant_d"],
        min_samples_per_variant: 10,
        delta: 0.05,
        epsilon: 0.1,
        update_period_s: 300,
        min_prob: None,
    });

    // This should not error
    let _client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
}

// ============================================================================
// Bandit Behavior Tests
// ============================================================================
// Note that all the tests below have some chance of failure due to randomness
// in the arm pulls and bandit rewards. Reducing this would involve averaging
// over multiple runs, which is costly due to the need to wait for the background
// task to update the sampling probabilities. Currently those updates can't
// happen more often than once per second.

/// Test the round-robin sampling in the nursery phase: each variant should
/// get `min_samples_per_variant` pulls, plus or minus 1 to allow for the
/// possibility that the background task update happens before the pulls are
/// complete. If failures occur frequently, `update_period_s` could be increased.
#[tokio::test(flavor = "multi_thread")]
async fn test_min_pulls() {
    // Setup: Create config with specific min_samples_per_variant
    let min_samples = 20;
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "min",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &[],
        min_samples_per_variant: min_samples,
        delta: 0.05,
        epsilon: 0.1,
        update_period_s: 1,
        min_prob: None,
    });

    let (client, clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;
    let client = std::sync::Arc::new(client);

    // Run exactly enough inferences to complete the nursery phase
    let num_variants = 3;
    let total_inferences = num_variants * min_samples as usize;

    // Inferences
    let inference_results = run_inference_batch(&client, total_inferences).await;

    // Wait for ClickHouse to flush
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(2 * BACKGROUND_UPDATE_DELAY_MS)).await;

    // Count how many times each variant was selected
    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for (_, variant_name) in &inference_results {
        *variant_counts.entry(variant_name.clone()).or_insert(0) += 1;
    }

    // Verify each variant was sampled min_samples times plus or minus 1
    for variant in ["variant_a", "variant_b", "variant_c"] {
        let count = variant_counts.get(variant).copied().unwrap_or(0);
        assert!(count.abs_diff(min_samples as usize) <= 1,
            "Expected variant {variant} to be sampled {min_samples} times (+/-) in nursery phase, got {count}"
        );
    }
}

// ============================================================================
// Stopping Behavior Tests
// ============================================================================

/// Test that after stopping, only the winning arm is pulled, with optimize="max"
#[tokio::test(flavor = "multi_thread")]
async fn test_winner_arm_pulled_after_stopping_optimize_max() {
    // Experiment parameters
    let num_initial_batches = 2; // Number of batches in Phase 1
    let inferences_per_batch = 300; // Number of inferences per batch in Phase 1
    let verification_inferences: usize = 100; // Number of inferences in Phase 2, after stopping

    // Use a very clear winner to ensure stopping happens quickly
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "boolean",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &[],
        min_samples_per_variant: 100,
        delta: 0.05,   // Reasonable confidence level
        epsilon: 0.01, // Very small epsilon - we want the clear best arm
        update_period_s: 1,
        min_prob: None,
    });

    // Set up bandit with very clear winner (variant_c has much higher success rate)
    let bandit_distribution = vec![
        ("variant_a", 0.30),
        ("variant_b", 0.40),
        ("variant_c", 0.95), // Clear winner
    ];

    let (client, clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;
    let bandit = BernoulliBandit::new(bandit_distribution.clone(), Some(42));
    let client = std::sync::Arc::new(client);
    let bandit = std::sync::Arc::new(bandit);

    // Phase 1: Run enough batches to trigger stopping

    for _batch in 0..num_initial_batches {
        // Phase 1a: Run all inferences
        let inference_results = run_inference_batch(&client, inferences_per_batch).await;

        // Wait for ClickHouse to flush inferences
        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

        // Phase 1b: Send feedback for all inferences
        let variant_names = send_feedback(
            &client,
            &inference_results,
            &Bandit::Bernoulli(bandit.clone()),
            "performance_score",
        )
        .await;

        // Count variants in this batch
        let mut variant_counts: HashMap<String, usize> = HashMap::new();
        for name in &variant_names {
            *variant_counts.entry(name.clone()).or_insert(0) += 1;
        }

        // Wait for ClickHouse and background update
        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;
    }

    // Phase 2: Run additional inferences and verify they all go to the winner
    let inference_results = run_inference_batch(&client, verification_inferences).await;
    let verification_variants: Vec<String> = inference_results
        .into_iter()
        .map(|(_, variant_name)| variant_name)
        .collect();

    // Count variants in verification phase
    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for name in &verification_variants {
        *variant_counts.entry(name.clone()).or_insert(0) += 1;
    }

    // After stopping, we expect ALL inferences to go to variant_c (highest success rate)
    let variant_c_count = variant_counts.get("variant_c").copied().unwrap_or(0);

    assert_eq!(
        variant_c_count, verification_inferences,
        "Expected 100% of inferences to go to variant_c (winner with highest mean for optimize=max). Distribution: {variant_counts:?}"
    );
}

/// Test that after stopping, only the winning arm is pulled, with optimize="min"
#[tokio::test(flavor = "multi_thread")]
async fn test_winner_arm_pulled_after_stopping_optimize_min() {
    // Experiment parameters
    let num_initial_batches = 2; // Number of batches in Phase 1
    let inferences_per_batch = 300; // Number of inferences per batch in Phase 1
    let verification_inferences: usize = 100; // Number of inferences in Phase 2, after stopping

    // Use a very clear winner to ensure stopping happens quickly
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "min",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &[],
        min_samples_per_variant: 100,
        delta: 0.05,   // Reasonable confidence level
        epsilon: 0.01, // Very small epsilon - we want the clear best arm
        update_period_s: 1,
        min_prob: None,
    });

    // Set up bandit with very clear winner (variant_a has much lower mean)
    let bandit_distribution = vec![
        ("variant_a", 0.30, 0.10), // Clear winner
        ("variant_b", 0.80, 0.10),
        ("variant_c", 1.0, 0.10),
    ];

    let (client, clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;
    let bandit = GaussianBandit::new(bandit_distribution.clone(), Some(42));
    let client = std::sync::Arc::new(client);
    let bandit = std::sync::Arc::new(bandit);

    // Phase 1: Run enough batches to trigger stopping

    for _batch in 0..num_initial_batches {
        // Phase 1a: Run all inferences
        let inference_results = run_inference_batch(&client, inferences_per_batch).await;

        // Wait for ClickHouse to flush inferences
        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

        // Phase 1b: Send feedback for all inferences
        let variant_names = send_feedback(
            &client,
            &inference_results,
            &Bandit::Gaussian(bandit.clone()),
            "performance_score",
        )
        .await;

        // Count variants in this batch
        let mut variant_counts: HashMap<String, usize> = HashMap::new();
        for name in &variant_names {
            *variant_counts.entry(name.clone()).or_insert(0) += 1;
        }

        // Wait for ClickHouse and background update
        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;
    }

    // Phase 2: Run additional inferences and verify they all go to the winner
    let inference_results = run_inference_batch(&client, verification_inferences).await;
    let verification_variants: Vec<String> = inference_results
        .into_iter()
        .map(|(_, variant_name)| variant_name)
        .collect();

    // Count variants in verification phase
    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for name in &verification_variants {
        *variant_counts.entry(name.clone()).or_insert(0) += 1;
    }

    // After stopping, we expect ALL inferences to go to variant_a (lowest mean for optimize=min)
    let variant_a_count = variant_counts.get("variant_a").copied().unwrap_or(0);

    assert_eq!(
        variant_a_count, verification_inferences,
        "Expected 100% of inferences to go to variant_a (winner with lowest mean for optimize=min). Distribution: {variant_counts:?}"
    );
}

/// Test that large delta (low confidence requirement) allows stopping with less evidence
/// than small delta (high confidence requirement).
///
/// This test sets up borderline stopping conditions, then verifies that a large delta
/// stops immediately while a small delta continues exploring.
#[tokio::test(flavor = "multi_thread")]
async fn test_effect_of_delta_on_stopping() {
    // Phase 1: Set up borderline stopping data using a temporary client
    let setup_config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b"],
        fallback_variants: &[],
        min_samples_per_variant: 50,
        delta: 0.05,
        epsilon: 0.05,
        update_period_s: 1,
        min_prob: None,
    });

    let (setup_client, clickhouse, guard) =
        make_embedded_gateway_with_clean_clickhouse(&setup_config).await;
    let setup_client = std::sync::Arc::new(setup_client);

    // Create borderline data: variant_b better than variant_a
    let bandit = GaussianBandit::new(
        vec![("variant_a", 1.0, 0.30), ("variant_b", 1.5, 0.30)],
        Some(42),
    );
    let bandit = std::sync::Arc::new(bandit);

    // Run one batch to populate data
    let inference_results = run_inference_batch(&setup_client, 120).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    send_feedback(
        &setup_client,
        &inference_results,
        &Bandit::Gaussian(bandit.clone()),
        "performance_score",
    )
    .await;
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;

    // Drop setup client to allow new clients with different configs
    drop(setup_client);
    drop(bandit);

    // Phase 2: Test with large delta (should stop)
    let config_large = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b"],
        fallback_variants: &[],
        min_samples_per_variant: 50,
        delta: 0.50, // Large delta = stops easily
        epsilon: 0.05,
        update_period_s: 1,
        min_prob: None,
    });

    let (client_large, _) =
        make_embedded_gateway_with_existing_clickhouse(&config_large, &clickhouse).await;
    let client_large = std::sync::Arc::new(client_large);

    // Give time for background task to process existing data
    tokio::time::sleep(Duration::from_millis(BACKGROUND_TASK_INIT_DELAY_MS)).await;

    // Run one batch and check if stopped
    let results_large = run_inference_batch(&client_large, 50).await;
    let variants_large: Vec<String> = results_large.into_iter().map(|(_, v)| v).collect();
    let mut counts_large: HashMap<String, usize> = HashMap::new();
    for v in &variants_large {
        *counts_large.entry(v.clone()).or_insert(0) += 1;
    }

    let max_count_large = counts_large.values().max().copied().unwrap_or(0);
    let stopped_large = max_count_large as f64 / variants_large.len() as f64 >= STOPPED_THRESHOLD;

    // Phase 3: Test with small delta (should NOT stop)
    let config_small = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b"],
        fallback_variants: &[],
        min_samples_per_variant: 50,
        delta: 1e-6, // Small delta = needs strong evidence
        epsilon: 0.05,
        update_period_s: 1,
        min_prob: None,
    });

    let (client_small, _) =
        make_embedded_gateway_with_existing_clickhouse(&config_small, &clickhouse).await;
    let client_small = std::sync::Arc::new(client_small);

    tokio::time::sleep(Duration::from_millis(BACKGROUND_TASK_INIT_DELAY_MS)).await;

    let results_small = run_inference_batch(&client_small, 50).await;
    let variants_small: Vec<String> = results_small.into_iter().map(|(_, v)| v).collect();
    let min_count_small = variants_small
        .iter()
        .fold(HashMap::new(), |mut map, v| {
            *map.entry(v.clone()).or_insert(0) += 1;
            map
        })
        .values()
        .min()
        .copied()
        .unwrap_or(0);

    let stopped_small = min_count_small == 0; // At least one variant got zero samples

    assert!(
        stopped_large,
        "Expected large delta to stop, but got distribution: {counts_large:?}"
    );
    assert!(
        !stopped_small,
        "Expected small delta to continue exploring, but it stopped"
    );

    drop(guard);
}

/// Test that large epsilon (less strict optimality requirement) allows stopping with less evidence
/// than small epsilon (strict optimality requirement).
///
/// This test sets up borderline stopping conditions, then verifies that a large epsilon
/// stops immediately while a small epsilon continues exploring.
#[tokio::test(flavor = "multi_thread")]
async fn test_effect_of_epsilon_on_stopping() {
    // Phase 1: Set up borderline stopping data
    let setup_config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "boolean",
        optimize: "min",
        candidate_variants: &["variant_a", "variant_b"],
        fallback_variants: &[],
        min_samples_per_variant: 50,
        delta: 0.05,
        epsilon: 0.05,
        update_period_s: 1,
        min_prob: None,
    });

    let (setup_client, clickhouse, guard) =
        make_embedded_gateway_with_clean_clickhouse(&setup_config).await;
    let setup_client = std::sync::Arc::new(setup_client);

    // Create data where variant_b is slightly better (within epsilon range)
    let bandit = BernoulliBandit::new(vec![("variant_a", 0.6), ("variant_b", 0.7)], Some(42));
    let bandit = std::sync::Arc::new(bandit);

    let inference_results = run_inference_batch(&setup_client, 120).await;
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    send_feedback(
        &setup_client,
        &inference_results,
        &Bandit::Bernoulli(bandit.clone()),
        "performance_score",
    )
    .await;
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;

    drop(setup_client);
    drop(bandit);

    // Phase 2: Test with large epsilon: should stop
    let config_large = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "boolean",
        optimize: "min",
        candidate_variants: &["variant_a", "variant_b"],
        fallback_variants: &[],
        min_samples_per_variant: 50,
        delta: 0.05,
        epsilon: 0.20,
        update_period_s: 1,
        min_prob: None,
    });

    let (client_large, _) =
        make_embedded_gateway_with_existing_clickhouse(&config_large, &clickhouse).await;
    let client_large = std::sync::Arc::new(client_large);
    tokio::time::sleep(Duration::from_millis(BACKGROUND_TASK_INIT_DELAY_MS)).await;

    let results_large = run_inference_batch(&client_large, 50).await;
    let variants_large: Vec<String> = results_large.into_iter().map(|(_, v)| v).collect();
    let mut counts_large: HashMap<String, usize> = HashMap::new();
    for v in &variants_large {
        *counts_large.entry(v.clone()).or_insert(0) += 1;
    }

    let max_count_large = counts_large.values().max().copied().unwrap_or(0);
    let stopped_large = max_count_large as f64 / variants_large.len() as f64 >= STOPPED_THRESHOLD;

    // Phase 3: Test with small epsilon: should NOT stop
    let config_small = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "boolean",
        optimize: "min",
        candidate_variants: &["variant_a", "variant_b"],
        fallback_variants: &[],
        min_samples_per_variant: 50,
        delta: 0.05,
        epsilon: 1e-4,
        update_period_s: 1,
        min_prob: None,
    });

    let (client_small, _) =
        make_embedded_gateway_with_existing_clickhouse(&config_small, &clickhouse).await;
    let client_small = std::sync::Arc::new(client_small);
    tokio::time::sleep(Duration::from_millis(BACKGROUND_TASK_INIT_DELAY_MS)).await;

    let results_small = run_inference_batch(&client_small, 50).await;
    let variants_small: Vec<String> = results_small.into_iter().map(|(_, v)| v).collect();
    let min_count_small = variants_small
        .iter()
        .fold(HashMap::new(), |mut map, v| {
            *map.entry(v.clone()).or_insert(0) += 1;
            map
        })
        .values()
        .min()
        .copied()
        .unwrap_or(0);

    let stopped_small = min_count_small == 0;

    assert!(
        stopped_large,
        "Expected large epsilon to stop, but got distribution: {counts_large:?}"
    );
    assert!(
        !stopped_small,
        "Expected small epsilon to continue exploring, but it stopped"
    );

    drop(guard);
}

/// Test that a new client immediately enters stopped state when the database
/// already contains data sufficient for stopping
#[tokio::test(flavor = "multi_thread")]
async fn test_cold_start_with_stopped_experiment() {
    let bandit_distribution = vec![
        ("variant_a", 0.40),
        ("variant_b", 0.90), // Very clearly better
        ("variant_c", 0.45),
    ];
    let seed = 42;

    // Experiment parameters
    let max_batches = 10; // Target max number of updates from background task
    let inferences_per_batch = 60; // Number of inferences between updates from background task

    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "boolean",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &[],
        min_samples_per_variant: 20,
        delta: 0.05,
        epsilon: 0.05,
        update_period_s: 1,
        min_prob: None,
    });

    let (client, clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;
    let bandit = BernoulliBandit::new(bandit_distribution.clone(), Some(seed));
    let client = std::sync::Arc::new(client);
    let bandit = std::sync::Arc::new(bandit);
    let mut stopped_at_batch = None;
    let mut last_batch_variants: Vec<String> = Vec::new();

    // Run until stopping is detected
    for batch in 0..max_batches {
        // Run all inferences
        let inference_results = run_inference_batch(&client, inferences_per_batch).await;

        // Wait for ClickHouse to flush inferences
        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

        // Send feedback for all inferences
        let variant_names = send_feedback(
            &client,
            &inference_results,
            &Bandit::Bernoulli(bandit.clone()),
            "performance_score",
        )
        .await;

        let mut variant_counts: HashMap<String, usize> = HashMap::new();
        for name in &variant_names {
            *variant_counts.entry(name.clone()).or_insert(0) += 1;
        }

        // Save the variant names from this batch
        last_batch_variants = variant_names.clone();

        let max_count = variant_counts.values().max().copied().unwrap_or(0);
        if max_count as f64 / variant_names.len() as f64 >= STOPPED_THRESHOLD
            && stopped_at_batch.is_none()
        {
            stopped_at_batch = Some(batch);
            break;
        }

        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;
    }

    assert!(
        stopped_at_batch.is_some(),
        "Expected stopping to occur in Phase 1 with very well-separated arms"
    );

    // Identify the winner variant from the last batch
    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for name in &last_batch_variants {
        *variant_counts.entry(name.clone()).or_insert(0) += 1;
    }
    let winner_variant = variant_counts
        .iter()
        .max_by_key(|(_, &count)| count)
        .map(|(name, _)| name.clone())
        .expect("Expected at least one variant in last batch");

    // Drop the first client to simulate restart
    drop(client);
    drop(bandit);

    // Small delay to ensure everything is flushed
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // Create a new client with the same configuration but fresh gateway instance
    let (new_client, new_clickhouse) =
        make_embedded_gateway_with_existing_clickhouse(&config, &clickhouse).await;
    let new_client = std::sync::Arc::new(new_client);

    // Give the new client time to initialize and check the database
    tokio::time::sleep(Duration::from_millis(BACKGROUND_TASK_INIT_DELAY_MS)).await;

    // Run a batch of inferences with the new client
    let inference_results = run_inference_batch(&new_client, inferences_per_batch).await;
    let variant_names: Vec<String> = inference_results
        .into_iter()
        .map(|(_, variant_name)| variant_name)
        .collect();

    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for name in &variant_names {
        *variant_counts.entry(name.clone()).or_insert(0) += 1;
    }

    // Assert that the new client is already in stopped state pulling only the winner
    let winner_count = variant_counts.get(&winner_variant).copied().unwrap_or(0);
    let winner_fraction = winner_count as f64 / variant_names.len() as f64;
    let winner_percentage = winner_fraction * 100.0;

    assert_eq!(
        winner_fraction, 1.0,
        "Expected new client to immediately enter stopped state with 100% to winner variant '{winner_variant}', but got {winner_percentage:.2}% to that variant. Distribution: {variant_counts:?}"
    );

    // Also verify that specifically the winner variant from phase 1 is being pulled
    let total_pulls = variant_names.len();
    assert_eq!(
        winner_count,
        variant_names.len(),
        "Expected all {total_pulls} pulls to go to winner variant '{winner_variant}', but only {winner_count} went to that variant"
    );

    // Clean up
    clickhouse_flush_async_insert(&new_clickhouse).await;
}

/// Test that introducing a new variant after stopping causes the system to
/// re-enter exploration mode rather than staying stopped
#[tokio::test(flavor = "multi_thread")]
async fn test_new_variant_triggers_reexploration() {
    let initial_bandit_distribution = vec![
        ("variant_a", 0.40, 0.10),
        ("variant_b", 0.80, 0.10), // Very clearly better
        ("variant_c", 0.45, 0.10),
    ];
    let seed = 42;

    // Track-and-Stop config parameters
    let shared_min_samples_per_variant = 20;
    let shared_delta = 0.01;
    let shared_epsilon = 0.05;
    let shared_update_period_s = 1;

    // Experiment parameters
    let max_batches = 10; // Target max number of updates from background task
    let inferences_per_batch = 60; // Number of inferences between updates from background task

    let initial_config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &[],
        min_samples_per_variant: shared_min_samples_per_variant,
        delta: shared_delta,
        epsilon: shared_epsilon,
        update_period_s: shared_update_period_s,
        min_prob: None,
    });

    let (client, clickhouse, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&initial_config).await;
    let bandit = GaussianBandit::new(initial_bandit_distribution.clone(), Some(seed));
    let client = std::sync::Arc::new(client);
    let bandit = std::sync::Arc::new(bandit);
    let mut stopped_at_batch = None;

    // Run until stopping is detected
    for batch in 0..max_batches {
        // Run all inferences
        let inference_results = run_inference_batch(&client, inferences_per_batch).await;

        // Wait for ClickHouse to flush inferences
        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

        // Send feedback for all inferences
        let variant_names = send_feedback(
            &client,
            &inference_results,
            &Bandit::Gaussian(bandit.clone()),
            "performance_score",
        )
        .await;

        let mut variant_counts: HashMap<String, usize> = HashMap::new();
        for name in &variant_names {
            *variant_counts.entry(name.clone()).or_insert(0) += 1;
        }

        let max_count = variant_counts.values().max().copied().unwrap_or(0);
        if max_count as f64 / variant_names.len() as f64 >= STOPPED_THRESHOLD
            && stopped_at_batch.is_none()
        {
            stopped_at_batch = Some(batch);
            break;
        }

        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;
    }

    assert!(
        stopped_at_batch.is_some(),
        "Expected stopping to occur in Phase 1 with very well-separated arms"
    );

    // Phase 2: Launch new client with an additional variant

    // Drop the first client to simulate restart
    drop(client);
    drop(bandit);

    // Small delay to ensure everything is flushed
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // Create config with new variant added
    let new_config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b", "variant_c", "variant_d"], // Added variant_d
        fallback_variants: &[],
        min_samples_per_variant: shared_min_samples_per_variant,
        delta: shared_delta,
        epsilon: shared_epsilon,
        update_period_s: shared_update_period_s,
        min_prob: None,
    });

    // Create new bandit that includes the new variant
    let new_bandit_distribution = vec![
        ("variant_a", 0.40, 0.10),
        ("variant_b", 0.80, 0.10),
        ("variant_c", 0.45, 0.10),
        ("variant_d", 0.70, 0.10), // New variant with good performance
    ];
    let new_bandit = GaussianBandit::new(new_bandit_distribution.clone(), Some(seed));

    // Create a new client with the updated configuration
    let (new_client, new_clickhouse) =
        make_embedded_gateway_with_existing_clickhouse(&new_config, &clickhouse).await;
    let new_client = std::sync::Arc::new(new_client);
    let new_bandit = std::sync::Arc::new(new_bandit);

    // Give the new client time to initialize
    tokio::time::sleep(Duration::from_millis(BACKGROUND_TASK_INIT_DELAY_MS)).await;

    // Run all inferences
    let inference_results = run_inference_batch(&new_client, inferences_per_batch).await;

    // Wait for ClickHouse to flush inferences
    clickhouse_flush_async_insert(&new_clickhouse).await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // Send feedback for all inferences
    let variant_names = send_feedback(
        &new_client,
        &inference_results,
        &Bandit::Gaussian(new_bandit.clone()),
        "performance_score",
    )
    .await;

    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for name in &variant_names {
        *variant_counts.entry(name.clone()).or_insert(0) += 1;
    }

    let total_pulls = variant_names.len() as f64;
    let variant_b_count = *variant_counts.get("variant_b").unwrap_or(&0);
    let variant_d_count = *variant_counts.get("variant_d").unwrap_or(&0);
    let variant_a_count = *variant_counts.get("variant_a").unwrap_or(&0);
    let variant_c_count = *variant_counts.get("variant_c").unwrap_or(&0);

    let variant_b_fraction = variant_b_count as f64 / total_pulls;
    let variant_d_fraction = variant_d_count as f64 / total_pulls;

    // Verify that only variant_b (winner) and variant_d (new) receive pulls
    assert!(
        variant_a_count == 0 && variant_c_count == 0,
        "Expected only variant_b (winner) and variant_d (new) to receive pulls, but variant_a got {variant_a_count} and variant_c got {variant_c_count}"
    );

    // Verify that both variant_b and variant_d receive non-trivial percentages
    let variant_b_percentage = variant_b_fraction * 100.0;
    assert!(
        variant_b_fraction >= MIN_EXPLORATION_FRACTION,
        "Expected variant_b (winner) to receive non-trivial percentage of pulls, but got {variant_b_percentage:.1}%"
    );

    let variant_d_percentage = variant_d_fraction * 100.0;
    assert!(
        variant_d_fraction >= MIN_EXPLORATION_FRACTION,
        "Expected variant_d (new) to receive non-trivial percentage of pulls, but got {variant_d_percentage:.1}%"
    );

    // Clean up
    clickhouse_flush_async_insert(&new_clickhouse).await;
}

// ============================================================================
// Variant Removal Tests
// ============================================================================

/// Test that removing the winner variant after stopping causes re-exploration
/// or runner up to be declared the winner. (Either is possible depending on the
/// data that's generated before the winner is removed.)
#[tokio::test(flavor = "multi_thread")]
async fn test_remove_winner_variant_after_stopping() {
    let initial_bandit_distribution = vec![
        ("variant_a", 0.95, 0.05), // Strong winner
        ("variant_b", 0.70, 0.05), // Runner-up
        ("variant_c", 0.20, 0.05),
    ];
    let seed = 42;

    // Experiment parameters
    let max_batches = 10;
    let inferences_per_batch = 60;

    let initial_config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &[],
        min_samples_per_variant: 20,
        delta: 0.05,
        epsilon: 0.01,
        update_period_s: 1,
        min_prob: None,
    });

    let (client, clickhouse, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&initial_config).await;
    let bandit = GaussianBandit::new(initial_bandit_distribution.clone(), Some(seed));
    let client = std::sync::Arc::new(client);
    let bandit = std::sync::Arc::new(bandit);
    let mut stopped_at_batch = None;

    // Phase 1: Run until stopping
    for batch in 0..max_batches {
        let inference_results = run_inference_batch(&client, inferences_per_batch).await;

        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

        let variant_names = send_feedback(
            &client,
            &inference_results,
            &Bandit::Gaussian(bandit.clone()),
            "performance_score",
        )
        .await;

        let mut variant_counts: HashMap<String, usize> = HashMap::new();
        for name in &variant_names {
            *variant_counts.entry(name.clone()).or_insert(0) += 1;
        }

        let max_count = variant_counts.values().max().copied().unwrap_or(0);
        if max_count as f64 / variant_names.len() as f64 >= STOPPED_THRESHOLD
            && stopped_at_batch.is_none()
        {
            stopped_at_batch = Some(batch);
            break;
        }

        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;
    }

    assert!(
        stopped_at_batch.is_some(),
        "Expected stopping to occur in Phase 1"
    );

    // Phase 2: Remove variant_a (the original winner) and verify that variant_b becomes
    // the new winner or at least gets substantial probability mass (i.e. verify that
    // variant_c does not get labeled the winner)

    drop(client);
    drop(bandit);
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // Create new config without variant_a (the original winner)
    let new_config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        optimize: "max",
        candidate_variants: &["variant_b", "variant_c"], // Removed variant_a
        fallback_variants: &[],
        min_samples_per_variant: 20,
        delta: 0.01,
        epsilon: 0.01,
        update_period_s: 1,
        min_prob: None,
    });

    // Create new bandit for remaining variants
    let new_bandit_distribution = vec![("variant_b", 0.70, 0.05), ("variant_c", 0.20, 0.05)];
    let new_bandit = GaussianBandit::new(new_bandit_distribution, Some(seed));

    let (new_client, new_clickhouse) =
        make_embedded_gateway_with_existing_clickhouse(&new_config, &clickhouse).await;
    let new_client = std::sync::Arc::new(new_client);
    let new_bandit = std::sync::Arc::new(new_bandit);

    tokio::time::sleep(Duration::from_millis(BACKGROUND_TASK_INIT_DELAY_MS)).await;

    // Run a batch and verify the system immediately locks onto the new winner
    let inference_results = run_inference_batch(&new_client, inferences_per_batch).await;

    clickhouse_flush_async_insert(&new_clickhouse).await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    let variant_names = send_feedback(
        &new_client,
        &inference_results,
        &Bandit::Gaussian(new_bandit.clone()),
        "performance_score",
    )
    .await;

    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for name in &variant_names {
        *variant_counts.entry(name.clone()).or_insert(0) += 1;
    }

    let total = variant_names.len();
    let variant_b_count = *variant_counts.get("variant_b").unwrap_or(&0);
    let variant_c_count = *variant_counts.get("variant_c").unwrap_or(&0);

    let variant_b_fraction = variant_b_count as f64 / total as f64;
    let variant_c_fraction = variant_c_count as f64 / total as f64;

    // After removing the original winner, variant_b should now be the winner or at least get most of the pulls
    assert!(variant_b_fraction >= 0.80,
        "Expected variant_b to receive at least 80% of pulls after removing variant_a, but variant_b got {variant_b_fraction}% of pulls"
    );
    assert!(
        variant_c_fraction <= 0.20,
        "Expected variant_c to receive no more than 20% of pulls after removing variant_a, but variant_c got {variant_c_fraction}% of pulls"
    );

    clickhouse_flush_async_insert(&new_clickhouse).await;
}

/// Test that removing a non-winner variant after stopping preserves winner
#[tokio::test(flavor = "multi_thread")]
async fn test_remove_non_winner_variant_after_stopping() {
    let initial_bandit_distribution = vec![
        ("variant_a", 0.05), // Clear winner
        ("variant_b", 0.90),
        ("variant_c", 0.95),
    ];
    let seed = 42;

    let max_batches = 10;
    let inferences_per_batch = 60;

    let initial_config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "boolean",
        optimize: "min",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &[],
        min_samples_per_variant: 20,
        delta: 0.05,
        epsilon: 0.05,
        update_period_s: 1,
        min_prob: None,
    });

    let (client, clickhouse, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&initial_config).await;
    let bandit = BernoulliBandit::new(initial_bandit_distribution.clone(), Some(seed));
    let client = std::sync::Arc::new(client);
    let bandit = std::sync::Arc::new(bandit);
    let mut stopped_at_batch = None;

    // Phase 1: Run until stopping
    for batch in 0..max_batches {
        let inference_results = run_inference_batch(&client, inferences_per_batch).await;

        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

        let variant_names = send_feedback(
            &client,
            &inference_results,
            &Bandit::Bernoulli(bandit.clone()),
            "performance_score",
        )
        .await;

        let mut variant_counts: HashMap<String, usize> = HashMap::new();
        for name in &variant_names {
            *variant_counts.entry(name.clone()).or_insert(0) += 1;
        }

        let max_count = variant_counts.values().max().copied().unwrap_or(0);
        if max_count as f64 / variant_names.len() as f64 >= STOPPED_THRESHOLD
            && stopped_at_batch.is_none()
        {
            stopped_at_batch = Some(batch);
            break;
        }

        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;
    }

    assert!(
        stopped_at_batch.is_some(),
        "Expected stopping to occur in Phase 1"
    );

    // Phase 2: Remove variant_b (a non-winner) and verify winner (variant_a) still selected
    drop(client);
    drop(bandit);
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // Create new config without variant_b
    let new_config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "boolean",
        optimize: "min",
        candidate_variants: &["variant_a", "variant_c"], // Removed variant_b
        fallback_variants: &[],
        min_samples_per_variant: 20,
        delta: 0.05,
        epsilon: 0.05,
        update_period_s: 1,
        min_prob: None,
    });

    let (new_client, new_clickhouse) =
        make_embedded_gateway_with_existing_clickhouse(&new_config, &clickhouse).await;
    let new_client = std::sync::Arc::new(new_client);

    tokio::time::sleep(Duration::from_millis(BACKGROUND_TASK_INIT_DELAY_MS)).await;

    // Run a batch and verify variant_a (winner) is still heavily sampled
    let inference_results = run_inference_batch(&new_client, inferences_per_batch).await;
    let variant_names: Vec<String> = inference_results
        .into_iter()
        .map(|(_, variant_name)| variant_name)
        .collect();

    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for name in &variant_names {
        *variant_counts.entry(name.clone()).or_insert(0) += 1;
    }

    let variant_a_count = *variant_counts.get("variant_a").unwrap_or(&0);
    let variant_a_fraction = variant_a_count as f64 / variant_names.len() as f64;

    // variant_a should still be the winner (100% of pulls)
    assert!(
        variant_a_fraction == 1.0,
        "Expected variant_a (winner) to dominate after non-winner removal, got {:.2}%",
        variant_a_fraction * 100.0
    );

    clickhouse_flush_async_insert(&new_clickhouse).await;
}
