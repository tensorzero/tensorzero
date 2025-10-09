use crate::clickhouse::{get_clean_clickhouse, DeleteDbOnDrop};
use futures::future::join_all;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Mutex;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    FeedbackParams, InferenceOutput, InferenceResponse, Role,
};
use tensorzero_core::db::clickhouse::test_helpers::clickhouse_flush_async_insert;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::FeedbackByVariant;
use tensorzero_core::experimentation::track_and_stop::estimate_optimal_probabilities::{
    estimate_optimal_probabilities, EstimateOptimalProbabilitiesArgs,
};
use tensorzero_core::inference::types::TextKind;
use tokio::time::Duration;
use tracing_test::traced_test;

/// Configuration parameters for generating a Track-and-Stop test config.
#[derive(Debug, Clone)]
pub struct TrackAndStopTestConfig<'a> {
    pub metric_name: &'a str,
    pub metric_type: &'a str,
    pub candidate_variants: &'a [&'a str],
    pub fallback_variants: &'a [&'a str],
    pub min_samples_per_variant: u64,
    pub delta: f64,
    pub epsilon: f64,
    pub update_period_s: u64,
}

impl<'a> Default for TrackAndStopTestConfig<'a> {
    fn default() -> Self {
        Self {
            metric_name: "test_metric",
            metric_type: "boolean",
            candidate_variants: &["variant_a", "variant_b"],
            fallback_variants: &["variant_fallback"],
            min_samples_per_variant: 10,
            delta: 0.05,
            epsilon: 0.1,
            update_period_s: 300,
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
        candidate_variants,
        fallback_variants,
        min_samples_per_variant,
        delta,
        epsilon,
        update_period_s,
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

    format!(
        r#"
[models.test_model]
routing = ["dummy"]

[models.test_model.providers.dummy]
type = "dummy"
model_name = "test"

[metrics.{metric_name}]
type = "{metric_type}"
optimize = "max"
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
"#,
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

/// Compute L2 distance between two probability distributions
fn compute_l2_distance(probs1: &HashMap<String, f64>, probs2: &HashMap<String, f64>) -> f64 {
    let mut sum_squared_diff = 0.0;

    // Get all variant names from both distributions
    let all_variants: std::collections::HashSet<_> = probs1.keys().chain(probs2.keys()).collect();

    for variant in all_variants {
        let p1 = probs1.get(variant).copied().unwrap_or(0.0);
        let p2 = probs2.get(variant).copied().unwrap_or(0.0);
        sum_squared_diff += (p1 - p2).powi(2);
    }

    sum_squared_diff.sqrt()
}

/// Helper function to build a client with a clean database for testing Track-and-Stop.
///
/// Creates an isolated ClickHouse database and builds a client connected to it.
/// The database is automatically cleaned up when the returned guard is dropped.
///
/// # Returns
/// A tuple of (Client, ClickHouseConnectionInfo, DeleteDbOnDrop guard)
///
/// # Panics
/// Panics if TENSORZERO_POSTGRES_URL environment variable is not set
async fn make_client_with_clean_database(
    config: &str,
) -> (tensorzero::Client, ClickHouseConnectionInfo, DeleteDbOnDrop) {
    // Create isolated database for test
    let (clickhouse, guard) = get_clean_clickhouse(false).await;

    // Reconstruct the clickhouse_url string from the base URL and database name
    use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
    let database = clickhouse.database();
    let mut clickhouse_url = url::Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path("");
    clickhouse_url.set_query(Some(&format!("database={database}")));
    let clickhouse_url_string = clickhouse_url.to_string();

    // Build client with isolated database
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for track-and-stop tests");

    let tmp_config = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    let client = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
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

/// Helper function to build an additional client connected to an existing database.
///
/// # Panics
/// Panics if TENSORZERO_POSTGRES_URL environment variable is not set
async fn make_client_with_same_database(
    config: &str,
    clickhouse: &ClickHouseConnectionInfo,
) -> tensorzero::Client {
    use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
    let database = clickhouse.database();
    let mut clickhouse_url = url::Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path("");
    clickhouse_url.set_query(Some(&format!("database={database}")));
    let clickhouse_url_string = clickhouse_url.to_string();

    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for track-and-stop tests");
    let tmp_config = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(clickhouse_url_string),
        postgres_url: Some(postgres_url),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn test_make_track_and_stop_config_generates_valid_toml() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "accuracy",
        metric_type: "float",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &["variant_a", "variant_b"],
        min_samples_per_variant: 10,
        delta: 0.05,
        epsilon: 0.1,
        update_period_s: 300,
    });

    // Verify it's valid TOML
    let parsed: toml::Value =
        toml::from_str(&config).expect("Generated config should be valid TOML");

    // Verify key sections exist
    assert!(parsed.get("models").is_some());
    assert!(parsed.get("metrics").is_some());
    assert!(parsed.get("functions").is_some());

    // Verify experimentation config
    let exp = parsed
        .get("functions")
        .and_then(|f| f.get("test_function"))
        .and_then(|tf| tf.get("experimentation"))
        .expect("Experimentation config should exist");

    assert_eq!(
        exp.get("type").and_then(|v| v.as_str()),
        Some("track_and_stop")
    );
    assert_eq!(exp.get("metric").and_then(|v| v.as_str()), Some("accuracy"));
    assert_eq!(
        exp.get("min_samples_per_variant")
            .and_then(toml::Value::as_integer),
        Some(10)
    );
    assert_eq!(exp.get("delta").and_then(toml::Value::as_float), Some(0.05));
    assert_eq!(
        exp.get("epsilon").and_then(toml::Value::as_float),
        Some(0.1)
    );
    assert_eq!(
        exp.get("update_period_s").and_then(toml::Value::as_integer),
        Some(300)
    );
}

#[test]
fn test_bernoulli_bandit_creation() {
    let bandit = BernoulliBandit::new(
        vec![("variant_a", 0.3), ("variant_b", 0.7), ("variant_c", 0.5)],
        None,
    );

    assert_eq!(bandit.probabilities.len(), 3);
    assert_eq!(bandit.probabilities["variant_a"], 0.3);
    assert_eq!(bandit.probabilities["variant_b"], 0.7);
    assert_eq!(bandit.probabilities["variant_c"], 0.5);
}

#[test]
fn test_bernoulli_bandit_sampling() {
    let bandit = BernoulliBandit::new(vec![("variant_a", 0.5)], None);

    // Sample many times and check distribution is approximately correct
    let n_samples = 10000;
    let successes = (0..n_samples)
        .filter(|_| bandit.sample("variant_a"))
        .count();

    let success_rate = successes as f64 / n_samples as f64;
    // With 10k samples, should be within 1% of 0.5 with high probability
    assert!(
        (success_rate - 0.5).abs() < 0.02,
        "Expected ~0.5, got {success_rate}"
    );
}

#[test]
#[should_panic(expected = "Probability must be in [0, 1]")]
fn test_bernoulli_bandit_invalid_probability() {
    BernoulliBandit::new(vec![("variant_a", 1.5)], None);
}

#[test]
#[should_panic(expected = "Unknown variant")]
fn test_bernoulli_bandit_unknown_variant() {
    let bandit = BernoulliBandit::new(vec![("variant_a", 0.5)], None);
    bandit.sample("variant_b");
}

#[test]
fn test_gaussian_bandit_creation() {
    let bandit = GaussianBandit::new(
        vec![
            ("variant_a", 0.5, 0.1),
            ("variant_b", 0.8, 0.05),
            ("variant_c", 0.3, 0.2),
        ],
        None,
    );

    assert_eq!(bandit.distributions.len(), 3);
    assert_eq!(bandit.distributions["variant_a"], (0.5, 0.1));
    assert_eq!(bandit.distributions["variant_b"], (0.8, 0.05));
    assert_eq!(bandit.distributions["variant_c"], (0.3, 0.2));
}

#[test]
fn test_gaussian_bandit_sampling() {
    let bandit = GaussianBandit::new(vec![("variant_a", 10.0, 1.0)], Some(42));

    // Sample many times and check distribution is approximately correct
    let n_samples = 10000;
    let samples: Vec<f64> = (0..n_samples).map(|_| bandit.sample("variant_a")).collect();

    let mean = samples.iter().sum::<f64>() / n_samples as f64;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;

    // With 10k samples, should be close to true mean and variance
    assert!((mean - 10.0).abs() < 0.1, "Expected mean ~10.0, got {mean}");
    assert!(
        (variance - 1.0).abs() < 0.1,
        "Expected variance ~1.0, got {variance}"
    );
}

#[test]
#[should_panic(expected = "Standard deviation must be non-negative")]
fn test_gaussian_bandit_negative_stddev() {
    GaussianBandit::new(vec![("variant_a", 0.5, -0.1)], None);
}

#[test]
#[should_panic(expected = "Unknown variant")]
fn test_gaussian_bandit_unknown_variant() {
    let bandit = GaussianBandit::new(vec![("variant_a", 0.5, 0.1)], None);
    bandit.sample("variant_b");
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

#[tokio::test]
async fn test_track_and_stop_config_invalid_metric() {
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
  fallback_variants = ["variant_a"]
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

#[tokio::test]
async fn test_track_and_stop_config_invalid_candidate_variant() {
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
candidate_variants = ["variant_a", "nonexistent_variant"]
fallback_variants = ["variant_a"]
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

#[tokio::test]
async fn test_track_and_stop_config_invalid_fallback_variant() {
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

#[tokio::test]
async fn test_track_and_stop_config_invalid_min_samples() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_type: "float",
        candidate_variants: &["variant_a"],
        fallback_variants: &["variant_a"],
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

#[tokio::test]
async fn test_track_and_stop_config_invalid_delta() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_type: "float",
        candidate_variants: &["variant_a"],
        fallback_variants: &["variant_a"],
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

#[tokio::test]
async fn test_track_and_stop_config_invalid_epsilon() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_type: "float",
        candidate_variants: &["variant_a"],
        fallback_variants: &["variant_a"],
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

#[tokio::test]
async fn test_track_and_stop_config_valid() {
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "accuracy",
        metric_type: "float",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &["variant_a", "variant_b"],
        min_samples_per_variant: 10,
        delta: 0.05,
        epsilon: 0.1,
        update_period_s: 300,
    });

    // This should not error
    let _client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

// #[tokio::test(flavor = "multi_thread")]
// async fn test_track_and_stop_bernoulli_bandit_convergence() {
//     // Setup: Create config with short update period for testing
//     let config = make_track_and_stop_config(TrackAndStopTestConfig {
//         metric_name: "success_rate",
//         metric_type: "boolean",
//         candidate_variants: &["variant_a", "variant_b", "variant_c"],
//         fallback_variants: &["variant_fallback"],
//         min_samples_per_variant: 10,
//         delta: 0.05,
//         epsilon: 0.1,
//         update_period_s: 1, // check every 1 second
//     });

//     let client =
//         tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
//     let clickhouse = get_clickhouse().await;

//     // Create Bernoulli bandit environment with clear winner (variant_b)
//     let bandit = BernoulliBandit::new(vec![
//         ("variant_a", 0.3),
//         ("variant_b", 0.9), // Clear winner!
//         ("variant_c", 0.4),
//     ]);

//     // Run for multiple update cycles to observe state transitions
//     let mut variant_counts_per_cycle = Vec::new();
//     let client = std::sync::Arc::new(client);
//     let bandit = std::sync::Arc::new(bandit);

//     for _cycle in 0..5 {
//         // Make inferences and submit feedback in parallel
//         let tasks: Vec<_> = (0..30)
//             .map(|_| {
//                 let client = client.clone();
//                 let bandit = bandit.clone();
//                 async move {
//                     // Make inference - Track-and-Stop selects the variant
//                     let output = client
//                         .inference(ClientInferenceParams {
//                             function_name: Some("test_function".to_string()),
//                             input: ClientInput {
//                                 system: None,
//                                 messages: vec![ClientInputMessage {
//                                     role: Role::User,
//                                     content: vec![ClientInputMessageContent::Text(
//                                         TextKind::Text {
//                                             text: "test input".to_string(),
//                                         },
//                                     )],
//                                 }],
//                             },
//                             ..Default::default()
//                         })
//                         .await
//                         .unwrap();

//                     let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = output
//                     else {
//                         panic!("Expected non-streaming chat response");
//                     };

//                     let variant_name = response.variant_name.clone();

//                     // Get reward from bandit environment for the selected variant
//                     let reward = bandit.sample(&variant_name);

//                     // Submit feedback
//                     client
//                         .feedback(FeedbackParams {
//                             inference_id: Some(response.inference_id),
//                             metric_name: "success_rate".to_string(),
//                             value: json!(reward),
//                             ..Default::default()
//                         })
//                         .await
//                         .unwrap();

//                     variant_name
//                 }
//             })
//             .collect();

//         let variant_names = join_all(tasks).await;

//         // Aggregate results
//         let mut cycle_counts = HashMap::new();
//         for variant_name in variant_names {
//             *cycle_counts.entry(variant_name).or_insert(0) += 1;
//         }

//         variant_counts_per_cycle.push(cycle_counts);

//         // Wait for ClickHouse async inserts to complete
//         clickhouse_flush_async_insert(&clickhouse).await;
//         tokio::time::sleep(Duration::from_millis(500)).await;

//         // Wait for background update task to run (1s period + buffer)
//         tokio::time::sleep(Duration::from_millis(1500)).await;
//     }

//     // Verify convergence: in later cycles, variant_b should dominate
//     let last_cycle_counts = &variant_counts_per_cycle[variant_counts_per_cycle.len() - 1];
//     let variant_b_count = last_cycle_counts.get("variant_b").unwrap_or(&0);

//     // After 5 cycles with clear winner (0.9 vs 0.3 success rate),
//     // Track-and-Stop should heavily favor variant_b
//     // Allow some exploration due to epsilon, but should be >50%
//     assert!(
//         *variant_b_count > 15,
//         "Expected variant_b to be selected >15 times in final cycle, got {variant_b_count}. \
//          Full distribution: {variant_counts_per_cycle:?}"
//     );

//     // Verify trend: variant_b selection should increase over time
//     let first_cycle_b = variant_counts_per_cycle[0].get("variant_b").unwrap_or(&0);
//     let last_cycle_b = variant_counts_per_cycle[4].get("variant_b").unwrap_or(&0);

//     // Should generally increase (allow for some statistical noise)
//     assert!(
//         last_cycle_b >= first_cycle_b,
//         "Expected variant_b selection to increase or stay same, \
//          but went from {first_cycle_b} to {last_cycle_b}"
//     );
// }

// #[tokio::test(flavor = "multi_thread")]
// async fn test_track_and_stop_gaussian_bandit_convergence() {
//     // Setup: Create config with short update period for testing
//     let config = make_track_and_stop_config(TrackAndStopTestConfig {
//         metric_name: "performance_score",
//         metric_type: "float",
//         candidate_variants: &["variant_a", "variant_b", "variant_c"],
//         fallback_variants: &["variant_fallback"],
//         min_samples_per_variant: 10,
//         delta: 0.05,
//         epsilon: 0.1,
//         update_period_s: 1, // check every 1 second
//     });

//     let client =
//         tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
//     let clickhouse = get_clickhouse().await;

//     // Create Gaussian bandit environment with clear winner (variant_b)
//     // variant_b has highest mean (0.85) with low variance
//     let bandit = GaussianBandit::new(vec![
//         ("variant_a", 0.5, 0.1),   // mean=0.5, stddev=0.1
//         ("variant_b", 0.85, 0.08), // mean=0.85, stddev=0.08 - Clear winner!
//         ("variant_c", 0.55, 0.12), // mean=0.55, stddev=0.12
//     ]);

//     // Run for multiple update cycles to observe state transitions
//     let mut variant_counts_per_cycle = Vec::new();
//     let client = std::sync::Arc::new(client);
//     let bandit = std::sync::Arc::new(bandit);

//     for _cycle in 0..5 {
//         // Make inferences and submit feedback in parallel
//         let tasks: Vec<_> = (0..30)
//             .map(|_| {
//                 let client = client.clone();
//                 let bandit = bandit.clone();
//                 async move {
//                     // Make inference - Track-and-Stop selects the variant
//                     let output = client
//                         .inference(ClientInferenceParams {
//                             function_name: Some("test_function".to_string()),
//                             input: ClientInput {
//                                 system: None,
//                                 messages: vec![ClientInputMessage {
//                                     role: Role::User,
//                                     content: vec![ClientInputMessageContent::Text(
//                                         TextKind::Text {
//                                             text: "test input".to_string(),
//                                         },
//                                     )],
//                                 }],
//                             },
//                             ..Default::default()
//                         })
//                         .await
//                         .unwrap();

//                     let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = output
//                     else {
//                         panic!("Expected non-streaming chat response");
//                     };

//                     let variant_name = response.variant_name.clone();

//                     // Get reward from bandit environment for the selected variant
//                     let reward = bandit.sample(&variant_name);

//                     // Submit feedback
//                     client
//                         .feedback(FeedbackParams {
//                             inference_id: Some(response.inference_id),
//                             metric_name: "performance_score".to_string(),
//                             value: json!(reward),
//                             ..Default::default()
//                         })
//                         .await
//                         .unwrap();

//                     variant_name
//                 }
//             })
//             .collect();

//         let variant_names = join_all(tasks).await;

//         // Aggregate results
//         let mut cycle_counts = HashMap::new();
//         for variant_name in variant_names {
//             *cycle_counts.entry(variant_name).or_insert(0) += 1;
//         }

//         variant_counts_per_cycle.push(cycle_counts);

//         // Wait for ClickHouse async inserts to complete
//         clickhouse_flush_async_insert(&clickhouse).await;
//         tokio::time::sleep(Duration::from_millis(500)).await;

//         // Wait for background update task to run (1s period + buffer)
//         tokio::time::sleep(Duration::from_millis(1500)).await;
//     }

//     // Verify convergence: in later cycles, variant_b should dominate
//     let last_cycle_counts = &variant_counts_per_cycle[variant_counts_per_cycle.len() - 1];
//     let variant_b_count = last_cycle_counts.get("variant_b").unwrap_or(&0);

//     // After 5 cycles with clear winner (mean 0.85 vs 0.5/0.55),
//     // Track-and-Stop should heavily favor variant_b
//     // Allow some exploration due to epsilon, but should be >50%
//     assert!(
//         *variant_b_count > 15,
//         "Expected variant_b to be selected >15 times in final cycle, got {variant_b_count}. \
//          Full distribution: {variant_counts_per_cycle:?}"
//     );

//     // Verify trend: variant_b selection should increase over time
//     let first_cycle_b = variant_counts_per_cycle[0].get("variant_b").unwrap_or(&0);
//     let last_cycle_b = variant_counts_per_cycle[4].get("variant_b").unwrap_or(&0);

//     // Should generally increase (allow for some statistical noise)
//     assert!(
//         last_cycle_b >= first_cycle_b,
//         "Expected variant_b selection to increase or stay same, \
//          but went from {first_cycle_b} to {last_cycle_b}"
//     );
// }

// ============================================================================
// Bandit Behavior Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_track_and_stop_min_pulls() {
    // Setup: Create config with specific min_samples_per_variant
    let min_samples = 20;
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &[],
        min_samples_per_variant: min_samples,
        delta: 0.05,
        epsilon: 0.1,
        update_period_s: 1,
    });

    let (client, clickhouse, _guard) = make_client_with_clean_database(&config).await;
    let client = std::sync::Arc::new(client);

    // Run exactly enough inferences to complete the nursery phase
    let num_variants = 3;
    let total_inferences = num_variants * min_samples as usize;

    // Inferences
    let inference_tasks: Vec<_> = (0..total_inferences)
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

    let inference_results = join_all(inference_tasks).await;

    // Wait for ClickHouse to flush
    clickhouse_flush_async_insert(&clickhouse).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Count how many times each variant was selected
    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for (_, variant_name) in &inference_results {
        *variant_counts.entry(variant_name.clone()).or_insert(0) += 1;
    }

    eprintln!("Nursery phase variant counts: {variant_counts:?}");

    // Verify each variant was sampled exactly min_samples times (round-robin)
    for variant in ["variant_a", "variant_b", "variant_c"] {
        let count = variant_counts.get(variant).copied().unwrap_or(0);
        assert_eq!(
            count, min_samples as usize,
            "Expected variant {variant} to be sampled exactly {min_samples} times in nursery phase, got {count}"
        );
    }
}

// #[tokio::test(flavor = "multi_thread")]
// async fn test_track_and_stop_gaussian_convergence_to_optimal_probabilities() {
//     // Setup: Create config with very small delta and epsilon to prevent stopping
//     let use_epsilon: f64 = 0.001;
//     let config = make_track_and_stop_config(TrackAndStopTestConfig {
//         metric_name: "performance_score",
//         metric_type: "float",
//         candidate_variants: &["variant_a", "variant_b", "variant_c"],
//         fallback_variants: &[],
//         min_samples_per_variant: 100,
//         delta: 1e-12, // Very small to prevent stopping
//         epsilon: use_epsilon,
//         update_period_s: 1,
//     });

//     // Bandit distribution
//     let bandit_distribution = [
//         ("variant_a", 0.50, 0.10),
//         ("variant_b", 0.55, 0.10),
//         ("variant_c", 0.60, 0.10),
//     ];

//     // Compute true optimal probabilities using known distributions
//     let true_feedback: Vec<FeedbackByVariant> = bandit_distribution
//         .iter()
//         .map(|(name, mean, stddev)| FeedbackByVariant {
//             variant_name: name.to_string(),
//             mean: *mean as f32,
//             variance: (stddev * stddev) as f32,
//             count: 10000,
//         })
//         .collect();

//     let true_optimal_probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
//         feedback: true_feedback,
//         epsilon: Some(use_epsilon),
//         variance_floor: None,
//         min_prob: None,
//         reg0: None,
//     })
//     .expect("Failed to compute true optimal probabilities");

//     eprintln!("True optimal probabilities: {true_optimal_probs:?}");

//     let num_runs = 5;
//     let num_cycles = 3;
//     let inferences_per_batch = 300;

//     let mut batch2_distances = Vec::new();
//     let mut batch3_distances = Vec::new();

//     for run in 0..num_runs {
//         eprintln!("\n========== RUN {run} ==========");

//         let (client, clickhouse, _guard) = make_client_with_clean_database(&config).await;
//         let bandit = GaussianBandit::new(bandit_distribution.to_vec(), None);

//         let mut l2_distances = Vec::new();
//         let mut cumulative_counts: HashMap<String, u32> = HashMap::new();
//         let mut total_inferences: u32 = 0;
//         let client = std::sync::Arc::new(client);
//         let bandit = std::sync::Arc::new(bandit);

//         for cycle in 0..num_cycles {
//             eprintln!("\n=== Run {run}, Cycle {cycle} ===");

//             // Phase 1: Make all inferences in parallel
//             let inference_tasks: Vec<_> = (0..inferences_per_batch)
//                 .map(|_| {
//                     let client = client.clone();
//                     async move {
//                         let output = client
//                             .inference(ClientInferenceParams {
//                                 function_name: Some("test_function".to_string()),
//                                 input: ClientInput {
//                                     system: None,
//                                     messages: vec![ClientInputMessage {
//                                         role: Role::User,
//                                         content: vec![ClientInputMessageContent::Text(
//                                             TextKind::Text {
//                                                 text: "test input".to_string(),
//                                             },
//                                         )],
//                                     }],
//                                 },
//                                 ..Default::default()
//                             })
//                             .await
//                             .unwrap();

//                         let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) =
//                             output
//                         else {
//                             panic!("Expected non-streaming chat response");
//                         };

//                         (response.inference_id, response.variant_name.clone())
//                     }
//                 })
//                 .collect();

//             let inference_results = join_all(inference_tasks).await;

//             // Wait for ClickHouse to flush inferences before submitting feedback
//             clickhouse_flush_async_insert(&clickhouse).await;
//             tokio::time::sleep(Duration::from_millis(500)).await;

//             // Phase 2: Submit feedback for all inferences in parallel
//             let feedback_tasks: Vec<_> = inference_results
//                 .iter()
//                 .map(|(inference_id, variant_name)| {
//                     let client = client.clone();
//                     let bandit = bandit.clone();
//                     let inference_id = *inference_id;
//                     let variant_name = variant_name.clone();
//                     async move {
//                         let reward = bandit.sample(&variant_name);
//                         client
//                             .feedback(FeedbackParams {
//                                 inference_id: Some(inference_id),
//                                 metric_name: "performance_score".to_string(),
//                                 value: json!(reward),
//                                 ..Default::default()
//                             })
//                             .await
//                             .unwrap();
//                     }
//                 })
//                 .collect();

//             join_all(feedback_tasks).await;

//             // Update cumulative counts
//             for (_, variant_name) in &inference_results {
//                 *cumulative_counts.entry(variant_name.clone()).or_insert(0) += 1;
//                 total_inferences += 1;
//             }

//             // Compute cumulative empirical probabilities
//             let cumulative_empirical_probs: HashMap<String, f64> = cumulative_counts
//                 .iter()
//                 .map(|(name, &count)| (name.clone(), count as f64 / total_inferences as f64))
//                 .collect();

//             // Compute L2 distance
//             let l2_distance = compute_l2_distance(&cumulative_empirical_probs, &true_optimal_probs);
//             l2_distances.push(l2_distance);
//             eprintln!("Run {run}, Cycle {cycle} L2 distance: {l2_distance:.4}");

//             // Wait for ClickHouse and background update
//             clickhouse_flush_async_insert(&clickhouse).await;
//             tokio::time::sleep(Duration::from_millis(1500)).await;
//         }

//         batch2_distances.push(l2_distances[1]);
//         batch3_distances.push(l2_distances[2]);

//         eprintln!(
//             "Run {run} - Batch 2 distance: {:.4}, Batch 3 distance: {:.4}",
//             l2_distances[1], l2_distances[2]
//         );
//     }

//     // Compute averages
//     let avg_batch2 = batch2_distances.iter().sum::<f64>() / num_runs as f64;
//     let avg_batch3 = batch3_distances.iter().sum::<f64>() / num_runs as f64;

//     eprintln!("\n========== Final Results ==========");
//     eprintln!("Batch 2 distances: {batch2_distances:?}");
//     eprintln!("Batch 3 distances: {batch3_distances:?}");
//     eprintln!("Average Batch 2 L2 distance: {avg_batch2:.4}");
//     eprintln!("Average Batch 3 L2 distance: {avg_batch3:.4}");

//     assert!(
//         avg_batch3 < avg_batch2,
//         "Expected average L2 distance to decrease from batch 2 to batch 3. \
//          Avg Batch 2: {avg_batch2:.4}, Avg Batch 3: {avg_batch3:.4}"
//     );
//     assert_eq!(1, 0);
// }

// // #[traced_test]
// #[tokio::test(flavor = "multi_thread")]
// async fn test_track_and_stop_bernoulli_convergence_to_optimal_probabilities() {
//     // Setup: Create config with very small delta and epsilon to prevent stopping
//     let use_epsilon: f64 = 0.001;
//     let config = make_track_and_stop_config(TrackAndStopTestConfig {
//         metric_name: "success_rate",
//         metric_type: "boolean",
//         candidate_variants: &["variant_a", "variant_b", "variant_c"],
//         fallback_variants: &[],
//         min_samples_per_variant: 100,
//         delta: 1e-9, // Very small to prevent stopping
//         epsilon: use_epsilon,
//         update_period_s: 1,
//     });

//     // Create Bernoulli bandit environment with close success probabilities to prevent early stopping
//     let bandit_distribution = [
//         ("variant_a", 0.54),
//         ("variant_b", 0.55),
//         ("variant_c", 0.56),
//     ];

//     // Compute true optimal probabilities using known distributions
//     // For Bernoulli: mean = p, variance = p(1-p)
//     let true_feedback: Vec<FeedbackByVariant> = bandit_distribution
//         .iter()
//         .map(|(name, prob)| FeedbackByVariant {
//             variant_name: name.to_string(),
//             mean: *prob as f32,
//             variance: (*prob * (1.0 - *prob)) as f32,
//             count: 10000, // Large count for stable estimates
//         })
//         .collect();

//     let true_optimal_probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
//         feedback: true_feedback,
//         epsilon: Some(use_epsilon),
//         variance_floor: None,
//         min_prob: None,
//         reg0: None,
//     })
//     .expect("Failed to compute true optimal probabilities");

//     eprintln!("True optimal probabilities: {true_optimal_probs:?}");

//     let num_runs = 5;
//     let num_cycles = 3;
//     let inferences_per_batch = 300;

//     let mut batch2_distances = Vec::new();
//     let mut batch3_distances = Vec::new();

//     for run in 0..num_runs {
//         eprintln!("\n========== RUN {run} ==========");

//         let (client, clickhouse, _guard) = make_client_with_clean_database(&config).await;
//         let bandit = BernoulliBandit::new(bandit_distribution.to_vec(), None);

//         let mut l2_distances = Vec::new();
//         let mut cumulative_counts: HashMap<String, u32> = HashMap::new();
//         let mut total_inferences: u32 = 0;
//         let client = std::sync::Arc::new(client);
//         let bandit = std::sync::Arc::new(bandit);

//         for cycle in 0..num_cycles {
//             eprintln!("\n=== Run {run}, Cycle {cycle} ===");

//             // Phase 1: Make all inferences in parallel
//             let inference_tasks: Vec<_> = (0..inferences_per_batch)
//                 .map(|_| {
//                     let client = client.clone();
//                     async move {
//                         let output = client
//                             .inference(ClientInferenceParams {
//                                 function_name: Some("test_function".to_string()),
//                                 input: ClientInput {
//                                     system: None,
//                                     messages: vec![ClientInputMessage {
//                                         role: Role::User,
//                                         content: vec![ClientInputMessageContent::Text(
//                                             TextKind::Text {
//                                                 text: "test input".to_string(),
//                                             },
//                                         )],
//                                     }],
//                                 },
//                                 ..Default::default()
//                             })
//                             .await
//                             .unwrap();

//                         let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) =
//                             output
//                         else {
//                             panic!("Expected non-streaming chat response");
//                         };

//                         (response.inference_id, response.variant_name.clone())
//                     }
//                 })
//                 .collect();

//             let inference_results = join_all(inference_tasks).await;

//             // Wait for ClickHouse to flush inferences before submitting feedback
//             clickhouse_flush_async_insert(&clickhouse).await;
//             tokio::time::sleep(Duration::from_millis(500)).await;

//             // Phase 2: Submit feedback for all inferences in parallel
//             let feedback_tasks: Vec<_> = inference_results
//                 .iter()
//                 .map(|(inference_id, variant_name)| {
//                     let client = client.clone();
//                     let bandit = bandit.clone();
//                     let inference_id = *inference_id;
//                     let variant_name = variant_name.clone();
//                     async move {
//                         let reward = bandit.sample(&variant_name);
//                         client
//                             .feedback(FeedbackParams {
//                                 inference_id: Some(inference_id),
//                                 metric_name: "success_rate".to_string(),
//                                 value: json!(reward),
//                                 ..Default::default()
//                             })
//                             .await
//                             .unwrap();
//                     }
//                 })
//                 .collect();

//             join_all(feedback_tasks).await;

//             // Update cumulative counts
//             for (_, variant_name) in &inference_results {
//                 *cumulative_counts.entry(variant_name.clone()).or_insert(0) += 1;
//                 total_inferences += 1;
//             }

//             // Compute cumulative empirical probabilities
//             let cumulative_empirical_probs: HashMap<String, f64> = cumulative_counts
//                 .iter()
//                 .map(|(name, &count)| (name.clone(), count as f64 / total_inferences as f64))
//                 .collect();

//             // Compute L2 distance
//             let l2_distance = compute_l2_distance(&cumulative_empirical_probs, &true_optimal_probs);
//             l2_distances.push(l2_distance);
//             eprintln!("Run {run}, Cycle {cycle} L2 distance: {l2_distance:.4}");

//             // Wait for ClickHouse and background update
//             clickhouse_flush_async_insert(&clickhouse).await;
//             tokio::time::sleep(Duration::from_millis(500)).await;
//             tokio::time::sleep(Duration::from_millis(1500)).await;

//             // Check if the algorithm has stopped
//             // if logs_contain("Track-and-Stop experiment stopped") {
//             //     panic!(
//             //         "Track-and-Stop algorithm stopped early at run {run}, cycle {cycle}. \
//             //          This prevents testing the convergence of sampling proportions to optimal probabilities. \
//             //          Consider adjusting bandit parameters (closer probabilities) or delta to prevent early stopping."
//             //     );
//             // }
//         }

//         batch2_distances.push(l2_distances[1]);
//         batch3_distances.push(l2_distances[2]);

//         eprintln!(
//             "Run {run} - Batch 2 distance: {:.4}, Batch 3 distance: {:.4}",
//             l2_distances[1], l2_distances[2]
//         );
//     }

//     // Compute averages
//     let avg_batch2 = batch2_distances.iter().sum::<f64>() / num_runs as f64;
//     let avg_batch3 = batch3_distances.iter().sum::<f64>() / num_runs as f64;

//     eprintln!("\n========== Final Results ==========");
//     eprintln!("Batch 2 distances: {batch2_distances:?}");
//     eprintln!("Batch 3 distances: {batch3_distances:?}");
//     eprintln!("Average Batch 2 L2 distance: {avg_batch2:.4}");
//     eprintln!("Average Batch 3 L2 distance: {avg_batch3:.4}");

//     assert!(
//         avg_batch3 < avg_batch2,
//         "Expected average L2 distance to decrease from batch 2 to batch 3. \
//          Avg Batch 2: {avg_batch2:.4}, Avg Batch 3: {avg_batch3:.4}"
//     );
//     assert_eq!(1, 0);
// }

// #[traced_test]
// #[tokio::test(flavor = "multi_thread")]
// async fn test_track_and_stop_gaussian_convergence_parallel_clients() {
//     // Setup: Create config with very small delta and epsilon to prevent stopping
//     let use_epsilon: f64 = 0.001;
//     let config = make_track_and_stop_config(TrackAndStopTestConfig {
//         metric_name: "performance_score",
//         metric_type: "float",
//         candidate_variants: &["variant_a", "variant_b", "variant_c", "variant_d"],
//         fallback_variants: &[],
//         min_samples_per_variant: 20,
//         delta: 1e-9, // Very small to prevent stopping
//         epsilon: use_epsilon,
//         update_period_s: 1,
//     });

//     let (client, clickhouse, _guard) = make_client_with_clean_database(&config).await;

//     // Create Gaussian bandit environment
//     let bandit_distribution = [
//         ("variant_a", 0.54, 0.10),
//         ("variant_b", 0.55, 0.15),
//         ("variant_c", 0.56, 0.10),
//         ("variant_d", 0.57, 0.05),
//     ];
//     let bandit = GaussianBandit::new(bandit_distribution.to_vec(), None);

//     // Compute true optimal probabilities
//     let true_feedback: Vec<FeedbackByVariant> = bandit_distribution
//         .iter()
//         .map(|(name, mean, stddev)| FeedbackByVariant {
//             variant_name: name.to_string(),
//             mean: *mean as f32,
//             variance: (stddev * stddev) as f32,
//             count: 10000,
//         })
//         .collect();

//     let true_optimal_probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
//         feedback: true_feedback,
//         epsilon: Some(use_epsilon),
//         variance_floor: None,
//         min_prob: None,
//         reg0: None,
//     })
//     .expect("Failed to compute true optimal probabilities");

//     // Create 5 clients all connected to the same database
//     let num_clients = 5;
//     let mut clients = vec![std::sync::Arc::new(client)];
//     for _ in 1..num_clients {
//         let new_client = make_client_with_same_database(&config, &clickhouse).await;
//         clients.push(std::sync::Arc::new(new_client));
//     }

//     let bandit = std::sync::Arc::new(bandit);
//     eprintln!("True optimal probabilities: {true_optimal_probs:?}");

//     // Track cumulative counts across all clients
//     let mut l2_distances = Vec::new();
//     let mut cumulative_counts: HashMap<String, u32> = HashMap::new();
//     let mut total_inferences: u32 = 0;

//     for cycle in 0..10 {
//         eprintln!("\n=== Starting cycle {cycle} with {num_clients} parallel clients ===");

//         // Each client makes inferences in parallel
//         let inferences_per_client: usize = 16;
//         let mut all_tasks = Vec::new();

//         for client in &clients {
//             let client_tasks: Vec<_> = (0..inferences_per_client)
//                 .map(|_| {
//                     let client = client.clone();
//                     let bandit = bandit.clone();
//                     async move {
//                         let output = client
//                             .inference(ClientInferenceParams {
//                                 function_name: Some("test_function".to_string()),
//                                 input: ClientInput {
//                                     system: None,
//                                     messages: vec![ClientInputMessage {
//                                         role: Role::User,
//                                         content: vec![ClientInputMessageContent::Text(
//                                             TextKind::Text {
//                                                 text: "test input".to_string(),
//                                             },
//                                         )],
//                                     }],
//                                 },
//                                 ..Default::default()
//                             })
//                             .await
//                             .unwrap();

//                         let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) =
//                             output
//                         else {
//                             panic!("Expected non-streaming chat response");
//                         };

//                         let variant_name = response.variant_name.clone();
//                         let reward = bandit.sample(&variant_name);

//                         client
//                             .feedback(FeedbackParams {
//                                 inference_id: Some(response.inference_id),
//                                 metric_name: "performance_score".to_string(),
//                                 value: json!(reward),
//                                 ..Default::default()
//                             })
//                             .await
//                             .unwrap();

//                         variant_name
//                     }
//                 })
//                 .collect();

//             all_tasks.extend(client_tasks);
//         }

//         // Wait for all clients to complete their inferences
//         let variant_names = join_all(all_tasks).await;

//         // Calculate current cycle counts
//         let mut cycle_counts: HashMap<String, usize> = HashMap::new();
//         for variant_name in &variant_names {
//             *cycle_counts.entry(variant_name.clone()).or_insert(0) += 1;
//         }

//         // Update cumulative counts
//         for variant_name in variant_names {
//             *cumulative_counts.entry(variant_name).or_insert(0) += 1;
//             total_inferences += 1;
//         }

//         // Compute cumulative empirical probabilities
//         let cumulative_empirical_probs: HashMap<String, f64> = cumulative_counts
//             .iter()
//             .map(|(name, &count)| (name.clone(), count as f64 / total_inferences as f64))
//             .collect();

//         eprintln!("Cycle {cycle} counts: {cycle_counts:?}");
//         eprintln!("Cycle {cycle} cumulative counts: {cumulative_counts:?}");
//         eprintln!("Cycle {cycle} cumulative empirical probs: {cumulative_empirical_probs:?}");

//         // Compute L2 distance
//         let l2_distance = compute_l2_distance(&cumulative_empirical_probs, &true_optimal_probs);
//         l2_distances.push(l2_distance);
//         eprintln!("Cycle {cycle} L2 distance: {l2_distance:.4}");

//         // Wait for ClickHouse and background update
//         clickhouse_flush_async_insert(&clickhouse).await;
//         tokio::time::sleep(Duration::from_millis(500)).await;
//         tokio::time::sleep(Duration::from_millis(1500)).await;

//         // Check if the algorithm has stopped
//         if logs_contain("Track-and-Stop experiment stopped") {
//             panic!(
//                 "Track-and-Stop algorithm stopped early at cycle {cycle}. \
//                  This prevents testing the convergence of sampling proportions to optimal probabilities. \
//                  Consider adjusting bandit parameters (closer means, larger stddevs) or delta to prevent early stopping."
//             );
//         }
//     }

//     // Verify convergence
//     eprintln!("\nFinal results:");
//     eprintln!("True optimal probabilities: {true_optimal_probs:?}");
//     eprintln!("Final cumulative counts: {cumulative_counts:?}");
//     eprintln!("L2 distances over time: {l2_distances:?}");

//     let second_cycle_distance = l2_distances[1];
//     let last_cycle_distance = l2_distances[l2_distances.len() - 1];

//     eprintln!("\nConvergence check:");
//     eprintln!("Second cycle L2 distance: {second_cycle_distance:.4}");
//     eprintln!("Last cycle L2 distance: {last_cycle_distance:.4}");

//     assert!(
//         last_cycle_distance < second_cycle_distance,
//         "Expected L2 distance to decrease from second cycle to last cycle. \
//          Second cycle: {second_cycle_distance:.4}, Last cycle: {last_cycle_distance:.4}"
//     );
// }

// #[traced_test]
// #[tokio::test(flavor = "multi_thread")]
// async fn test_track_and_stop_bernoulli_convergence_parallel_clients() {
// Setup: Create config with very small delta and epsilon to prevent stopping
//     let use_epsilon: f64 = 0.001;
//     let config = make_track_and_stop_config(TrackAndStopTestConfig {
//         metric_name: "success_rate",
//         metric_type: "boolean",
//         candidate_variants: &["variant_a", "variant_b", "variant_c", "variant_d"],
//         fallback_variants: &[],
//         min_samples_per_variant: 20,
//         delta: 1e-9, // Very small to prevent stopping
//         epsilon: use_epsilon,
//         update_period_s: 1,
//     });

//     let (client, clickhouse, _guard) = make_client_with_clean_database(&config).await;

//     // Create Bernoulli bandit environment
//     let bandit_distribution = [
//         ("variant_a", 0.54),
//         ("variant_b", 0.55),
//         ("variant_c", 0.56),
//         ("variant_d", 0.57),
//     ];
//     let bandit = BernoulliBandit::new(bandit_distribution.to_vec(), None);

//     // Compute true optimal probabilities
//     let true_feedback: Vec<FeedbackByVariant> = bandit_distribution
//         .iter()
//         .map(|(name, prob)| FeedbackByVariant {
//             variant_name: name.to_string(),
//             mean: *prob as f32,
//             variance: (*prob * (1.0 - *prob)) as f32,
//             count: 10000,
//         })
//         .collect();

//     let true_optimal_probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
//         feedback: true_feedback,
//         epsilon: Some(use_epsilon),
//         variance_floor: None,
//         min_prob: None,
//         reg0: None,
//     })
//     .expect("Failed to compute true optimal probabilities");

//     // Create 5 clients all connected to the same database
//     let num_clients = 5;
//     let mut clients = vec![std::sync::Arc::new(client)];
//     for _ in 1..num_clients {
//         let new_client = make_client_with_same_database(&config, &clickhouse).await;
//         clients.push(std::sync::Arc::new(new_client));
//     }

//     let bandit = std::sync::Arc::new(bandit);
//     eprintln!("True optimal probabilities: {true_optimal_probs:?}");

//     // Track cumulative counts across all clients
//     let mut l2_distances = Vec::new();
//     let mut cumulative_counts: HashMap<String, u32> = HashMap::new();
//     let mut total_inferences: u32 = 0;

//     for cycle in 0..10 {
//         eprintln!("\n=== Starting cycle {cycle} with {num_clients} parallel clients ===");

//         // Each client makes inferences in parallel
//         let inferences_per_client: usize = 16;
//         let mut all_tasks = Vec::new();

//         for client in &clients {
//             let client_tasks: Vec<_> = (0..inferences_per_client)
//                 .map(|_| {
//                     let client = client.clone();
//                     let bandit = bandit.clone();
//                     async move {
//                         let output = client
//                             .inference(ClientInferenceParams {
//                                 function_name: Some("test_function".to_string()),
//                                 input: ClientInput {
//                                     system: None,
//                                     messages: vec![ClientInputMessage {
//                                         role: Role::User,
//                                         content: vec![ClientInputMessageContent::Text(
//                                             TextKind::Text {
//                                                 text: "test input".to_string(),
//                                             },
//                                         )],
//                                     }],
//                                 },
//                                 ..Default::default()
//                             })
//                             .await
//                             .unwrap();

//                         let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) =
//                             output
//                         else {
//                             panic!("Expected non-streaming chat response");
//                         };

//                         let variant_name = response.variant_name.clone();
//                         let reward = bandit.sample(&variant_name);

//                         client
//                             .feedback(FeedbackParams {
//                                 inference_id: Some(response.inference_id),
//                                 metric_name: "success_rate".to_string(),
//                                 value: json!(reward),
//                                 ..Default::default()
//                             })
//                             .await
//                             .unwrap();

//                         variant_name
//                     }
//                 })
//                 .collect();

//             all_tasks.extend(client_tasks);
//         }

//         // Wait for all clients to complete their inferences
//         let variant_names = join_all(all_tasks).await;

//         // Calculate current cycle counts
//         let mut cycle_counts: HashMap<String, usize> = HashMap::new();
//         for variant_name in &variant_names {
//             *cycle_counts.entry(variant_name.clone()).or_insert(0) += 1;
//         }

//         // Update cumulative counts
//         for variant_name in variant_names {
//             *cumulative_counts.entry(variant_name).or_insert(0) += 1;
//             total_inferences += 1;
//         }

//         // Compute cumulative empirical probabilities
//         let cumulative_empirical_probs: HashMap<String, f64> = cumulative_counts
//             .iter()
//             .map(|(name, &count)| (name.clone(), count as f64 / total_inferences as f64))
//             .collect();

//         eprintln!("Cycle {cycle} counts: {cycle_counts:?}");
//         eprintln!("Cycle {cycle} cumulative counts: {cumulative_counts:?}");
//         eprintln!("Cycle {cycle} cumulative empirical probs: {cumulative_empirical_probs:?}");

//         // Compute L2 distance
//         let l2_distance = compute_l2_distance(&cumulative_empirical_probs, &true_optimal_probs);
//         l2_distances.push(l2_distance);
//         eprintln!("Cycle {cycle} L2 distance: {l2_distance:.4}");

//         // Wait for ClickHouse and background update
//         clickhouse_flush_async_insert(&clickhouse).await;
//         tokio::time::sleep(Duration::from_millis(500)).await;
//         tokio::time::sleep(Duration::from_millis(1500)).await;

//         // Check if the algorithm has stopped
//         if logs_contain("Track-and-Stop experiment stopped") {
//             panic!(
//                 "Track-and-Stop algorithm stopped early at cycle {cycle}. \
//                  This prevents testing the convergence of sampling proportions to optimal probabilities. \
//                  Consider adjusting bandit parameters (closer probabilities) or delta to prevent early stopping."
//             );
//         }
//     }

//     // Verify convergence
//     eprintln!("\nFinal results:");
//     eprintln!("True optimal probabilities: {true_optimal_probs:?}");
//     eprintln!("Final cumulative counts: {cumulative_counts:?}");
//     eprintln!("L2 distances over time: {l2_distances:?}");

//     let second_cycle_distance = l2_distances[1];
//     let last_cycle_distance = l2_distances[l2_distances.len() - 1];

//     eprintln!("\nConvergence check:");
//     eprintln!("Second cycle L2 distance: {second_cycle_distance:.4}");
//     eprintln!("Last cycle L2 distance: {last_cycle_distance:.4}");

//     assert!(
//         last_cycle_distance < second_cycle_distance,
//         "Expected L2 distance to decrease from second cycle to last cycle. \
//          Second cycle: {second_cycle_distance:.4}, Last cycle: {last_cycle_distance:.4}"
//     );
// }
