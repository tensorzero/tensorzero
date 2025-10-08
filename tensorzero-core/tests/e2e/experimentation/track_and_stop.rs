use rand_distr::Distribution;
use serde_json::json;
use std::collections::HashMap;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    FeedbackParams, InferenceOutput, InferenceResponse, Role,
};
use tensorzero_core::db::clickhouse::test_helpers::{
    clickhouse_flush_async_insert, get_clickhouse,
};
use tensorzero_core::inference::types::TextKind;
use tokio::time::Duration;

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
}

impl BernoulliBandit {
    /// Create a new Bernoulli bandit environment.
    ///
    /// # Arguments
    /// * `variant_probs` - List of (variant_name, success_probability) tuples
    ///
    /// # Panics
    /// Panics if any probability is outside [0, 1]
    pub fn new(variant_probs: Vec<(&str, f64)>) -> Self {
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
        Self { probabilities }
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
        rand::Rng::random_bool(&mut rand::rng(), *prob)
    }
}

/// Gaussian bandit environment for testing.
///
/// Each arm (variant) returns a sample from N(μ, σ²).
/// Use this to generate rewards for float metrics.
pub struct GaussianBandit {
    distributions: std::collections::HashMap<String, (f64, f64)>, // (mean, stddev)
}

impl GaussianBandit {
    /// Create a new Gaussian bandit environment.
    ///
    /// # Arguments
    /// * `variant_distributions` - List of (variant_name, mean, stddev) tuples
    ///
    /// # Panics
    /// Panics if any stddev is negative
    pub fn new(variant_distributions: Vec<(&str, f64, f64)>) -> Self {
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
        Self { distributions }
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
        normal.sample(&mut rand::rng())
    }
}

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
    let bandit = BernoulliBandit::new(vec![
        ("variant_a", 0.3),
        ("variant_b", 0.7),
        ("variant_c", 0.5),
    ]);

    assert_eq!(bandit.probabilities.len(), 3);
    assert_eq!(bandit.probabilities["variant_a"], 0.3);
    assert_eq!(bandit.probabilities["variant_b"], 0.7);
    assert_eq!(bandit.probabilities["variant_c"], 0.5);
}

#[test]
fn test_bernoulli_bandit_sampling() {
    let bandit = BernoulliBandit::new(vec![("variant_a", 0.5)]);

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
    BernoulliBandit::new(vec![("variant_a", 1.5)]);
}

#[test]
#[should_panic(expected = "Unknown variant")]
fn test_bernoulli_bandit_unknown_variant() {
    let bandit = BernoulliBandit::new(vec![("variant_a", 0.5)]);
    bandit.sample("variant_b");
}

#[test]
fn test_gaussian_bandit_creation() {
    let bandit = GaussianBandit::new(vec![
        ("variant_a", 0.5, 0.1),
        ("variant_b", 0.8, 0.05),
        ("variant_c", 0.3, 0.2),
    ]);

    assert_eq!(bandit.distributions.len(), 3);
    assert_eq!(bandit.distributions["variant_a"], (0.5, 0.1));
    assert_eq!(bandit.distributions["variant_b"], (0.8, 0.05));
    assert_eq!(bandit.distributions["variant_c"], (0.3, 0.2));
}

#[test]
fn test_gaussian_bandit_sampling() {
    let bandit = GaussianBandit::new(vec![("variant_a", 10.0, 1.0)]);

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
    GaussianBandit::new(vec![("variant_a", 0.5, -0.1)]);
}

#[test]
#[should_panic(expected = "Unknown variant")]
fn test_gaussian_bandit_unknown_variant() {
    let bandit = GaussianBandit::new(vec![("variant_a", 0.5, 0.1)]);
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

#[tokio::test(flavor = "multi_thread")]
async fn test_track_and_stop_bernoulli_bandit_convergence() {
    // Setup: Create config with short update period for testing
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "success_rate",
        metric_type: "boolean",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &["variant_fallback"],
        min_samples_per_variant: 10,
        delta: 0.05,
        epsilon: 0.1,
        update_period_s: 1, // check every 1 second
    });

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let clickhouse = get_clickhouse().await;

    // Create Bernoulli bandit environment with clear winner (variant_b)
    let bandit = BernoulliBandit::new(vec![
        ("variant_a", 0.3),
        ("variant_b", 0.9), // Clear winner!
        ("variant_c", 0.4),
    ]);

    // Run for multiple update cycles to observe state transitions
    let mut variant_counts_per_cycle = Vec::new();

    for _cycle in 0..5 {
        let mut cycle_counts = HashMap::new();

        // Make inferences and submit feedback for one update period
        for _ in 0..30 {
            // Make inference - Track-and-Stop selects the variant
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

            let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = output else {
                panic!("Expected non-streaming chat response");
            };

            // Track which variant was selected
            *cycle_counts
                .entry(response.variant_name.clone())
                .or_insert(0) += 1;

            // Get reward from bandit environment for the selected variant
            let reward = bandit.sample(&response.variant_name);

            // Submit feedback
            client
                .feedback(FeedbackParams {
                    inference_id: Some(response.inference_id),
                    metric_name: "success_rate".to_string(),
                    value: json!(reward),
                    ..Default::default()
                })
                .await
                .unwrap();
        }

        variant_counts_per_cycle.push(cycle_counts);

        // Wait for ClickHouse async inserts to complete
        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Wait for background update task to run (1s period + buffer)
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }

    // Verify convergence: in later cycles, variant_b should dominate
    let last_cycle_counts = &variant_counts_per_cycle[variant_counts_per_cycle.len() - 1];
    let variant_b_count = last_cycle_counts.get("variant_b").unwrap_or(&0);

    // After 5 cycles with clear winner (0.9 vs 0.3 success rate),
    // Track-and-Stop should heavily favor variant_b
    // Allow some exploration due to epsilon, but should be >50%
    assert!(
        *variant_b_count > 15,
        "Expected variant_b to be selected >15 times in final cycle, got {variant_b_count}. \
         Full distribution: {variant_counts_per_cycle:?}"
    );

    // Verify trend: variant_b selection should increase over time
    let first_cycle_b = variant_counts_per_cycle[0].get("variant_b").unwrap_or(&0);
    let last_cycle_b = variant_counts_per_cycle[4].get("variant_b").unwrap_or(&0);

    // Should generally increase (allow for some statistical noise)
    assert!(
        last_cycle_b >= first_cycle_b,
        "Expected variant_b selection to increase or stay same, \
         but went from {first_cycle_b} to {last_cycle_b}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_track_and_stop_gaussian_bandit_convergence() {
    // Setup: Create config with short update period for testing
    let config = make_track_and_stop_config(TrackAndStopTestConfig {
        metric_name: "performance_score",
        metric_type: "float",
        candidate_variants: &["variant_a", "variant_b", "variant_c"],
        fallback_variants: &["variant_fallback"],
        min_samples_per_variant: 10,
        delta: 0.05,
        epsilon: 0.1,
        update_period_s: 1, // check every 1 second
    });

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let clickhouse = get_clickhouse().await;

    // Create Gaussian bandit environment with clear winner (variant_b)
    // variant_b has highest mean (0.85) with low variance
    let bandit = GaussianBandit::new(vec![
        ("variant_a", 0.5, 0.1),   // mean=0.5, stddev=0.1
        ("variant_b", 0.85, 0.08), // mean=0.85, stddev=0.08 - Clear winner!
        ("variant_c", 0.55, 0.12), // mean=0.55, stddev=0.12
    ]);

    // Run for multiple update cycles to observe state transitions
    let mut variant_counts_per_cycle = Vec::new();

    for _cycle in 0..5 {
        let mut cycle_counts = HashMap::new();

        // Make inferences and submit feedback for one update period
        for _ in 0..30 {
            // TODO: run in parallel
            // Make inference - Track-and-Stop selects the variant
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

            let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = output else {
                panic!("Expected non-streaming chat response");
            };

            // Track which variant was selected
            *cycle_counts
                .entry(response.variant_name.clone())
                .or_insert(0) += 1;

            // Get reward from bandit environment for the selected variant
            let reward = bandit.sample(&response.variant_name);

            // Submit feedback
            client
                .feedback(FeedbackParams {
                    inference_id: Some(response.inference_id),
                    metric_name: "performance_score".to_string(),
                    value: json!(reward),
                    ..Default::default()
                })
                .await
                .unwrap();
        }

        variant_counts_per_cycle.push(cycle_counts);

        // Wait for ClickHouse async inserts to complete
        clickhouse_flush_async_insert(&clickhouse).await;
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Wait for background update task to run (1s period + buffer)
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }

    // Verify convergence: in later cycles, variant_b should dominate
    let last_cycle_counts = &variant_counts_per_cycle[variant_counts_per_cycle.len() - 1];
    let variant_b_count = last_cycle_counts.get("variant_b").unwrap_or(&0);

    // After 5 cycles with clear winner (mean 0.85 vs 0.5/0.55),
    // Track-and-Stop should heavily favor variant_b
    // Allow some exploration due to epsilon, but should be >50%
    assert!(
        *variant_b_count > 15,
        "Expected variant_b to be selected >15 times in final cycle, got {variant_b_count}. \
         Full distribution: {variant_counts_per_cycle:?}"
    );

    // Verify trend: variant_b selection should increase over time
    let first_cycle_b = variant_counts_per_cycle[0].get("variant_b").unwrap_or(&0);
    let last_cycle_b = variant_counts_per_cycle[4].get("variant_b").unwrap_or(&0);

    // Should generally increase (allow for some statistical noise)
    assert!(
        last_cycle_b >= first_cycle_b,
        "Expected variant_b selection to increase or stay same, \
         but went from {first_cycle_b} to {last_cycle_b}"
    );
}
