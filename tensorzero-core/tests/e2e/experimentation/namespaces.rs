use futures::future::join_all;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::NamedTempFile;
use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams, FeedbackParams,
    InferenceOutput, InferenceResponse, Input, InputMessage, InputMessageContent, PostgresConfig,
    Role,
};
use tensorzero_core::config::Namespace;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use tensorzero_core::db::feedback::FeedbackQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::Text;
use tokio::time::Duration;
use url::Url;
use uuid::Uuid;

use crate::clickhouse::{DeleteDbOnDrop, get_clean_clickhouse};
use crate::experimentation::track_and_stop::BernoulliBandit;

// ============================================================================
// Helpers
// ============================================================================

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

    let database = clickhouse.database();
    let mut clickhouse_url = Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path(database);
    let clickhouse_url_string = clickhouse_url.to_string();

    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(clickhouse_url_string),
        postgres_config: Some(PostgresConfig::Url(postgres_url)),
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();

    (client, clickhouse, guard)
}

fn make_namespace_test_config() -> String {
    r#"
[models.test_model]
routing = ["test"]

[models.test_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_function]
type = "chat"

[functions.test_function.variants.variant_a]
type = "chat_completion"
model = "test_model"

[functions.test_function.variants.variant_b]
type = "chat_completion"
model = "test_model"

[functions.test_function.variants.variant_c]
type = "chat_completion"
model = "test_model"

[functions.test_function.experimentation]
type = "uniform"

[functions.test_function.experimentation.namespaces.mobile]
type = "static_weights"
candidate_variants = {"variant_a" = 1.0}

[functions.test_function.experimentation.namespaces.web]
type = "static_weights"
candidate_variants = {"variant_b" = 1.0}
"#
    .to_string()
}

async fn do_inference(client: &Client, namespace: Option<&str>) -> (uuid::Uuid, String) {
    let output = client
        .inference(ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            namespace: namespace.map(|ns| Namespace::new(ns).unwrap()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "test".to_string(),
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

    (response.inference_id, response.variant_name.clone())
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_base_config_used_without_namespace() {
    let config = make_namespace_test_config();
    let (client, _clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;

    let sample_size = 100;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..sample_size {
        let (_, variant_name) = do_inference(&client, None).await;
        *counts.entry(variant_name).or_insert(0) += 1;
    }

    // With uniform sampling over 3 variants, we expect all three to appear
    assert!(
        counts.len() >= 2,
        "Uniform sampling without namespace should produce multiple variants, got: {counts:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_specific_config_used() {
    let config = make_namespace_test_config();
    let (client, _clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;

    let sample_size = 20;
    for _ in 0..sample_size {
        let (_, variant_name) = do_inference(&client, Some("mobile")).await;
        assert_eq!(
            variant_name, "variant_a",
            "With namespace `mobile` (static_weights A=1.0), all inferences should use variant_a"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_different_configs() {
    let config = make_namespace_test_config();
    let (client, _clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;

    let sample_size = 20;
    for _ in 0..sample_size {
        let (_, variant_name) = do_inference(&client, Some("web")).await;
        assert_eq!(
            variant_name, "variant_b",
            "With namespace `web` (static_weights B=1.0), all inferences should use variant_b"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_unknown_falls_back_to_base() {
    let config = make_namespace_test_config();
    let (client, _clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;

    let sample_size = 100;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..sample_size {
        let (_, variant_name) = do_inference(&client, Some("unknown_ns")).await;
        *counts.entry(variant_name).or_insert(0) += 1;
    }

    // Should fall back to base uniform config, producing multiple variants
    assert!(
        counts.len() >= 2,
        "Unknown namespace should fall back to uniform base config, got: {counts:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_stored_as_tag() {
    let config = make_namespace_test_config();
    let (client, clickhouse, _guard) = make_embedded_gateway_with_clean_clickhouse(&config).await;

    let (inference_id, _) = do_inference(&client, Some("mobile")).await;

    // Flush ClickHouse writes and wait for them to be visible
    clickhouse.flush_pending_writes().await;
    clickhouse.sleep_for_writes_to_be_visible().await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Query ClickHouse for the tag
    let query = format!(
        "SELECT tags['tensorzero::namespace'] AS ns FROM ChatInference WHERE id = '{inference_id}' FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .expect("ClickHouse query should succeed");
    let rows: Vec<serde_json::Value> = response
        .response
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "Should find exactly one row for the inference"
    );
    assert_eq!(
        rows[0]["ns"].as_str().unwrap(),
        "mobile",
        "The `tensorzero::namespace` tag should be stored as `mobile`"
    );
}

// ============================================================================
// Track-and-Stop Namespace Tests
// ============================================================================

/// Delay after ClickHouse flush for inferences (milliseconds).
const CLICKHOUSE_FLUSH_DELAY_MS: u64 = 1000;

/// Delay after ClickHouse flush for feedback to allow background update (milliseconds).
const BACKGROUND_UPDATE_DELAY_MS: u64 = 1000;

fn make_namespace_track_and_stop_config() -> String {
    r#"
gateway.unstable_disable_feedback_target_validation = true

[models.test_model]
routing = ["test"]

[models.test_model.providers.test]
type = "dummy"
model_name = "test"

[metrics.test_metric]
type = "boolean"
optimize = "max"
level = "inference"

[functions.test_function]
type = "chat"

[functions.test_function.variants.variant_a]
type = "chat_completion"
model = "test_model"

[functions.test_function.variants.variant_b]
type = "chat_completion"
model = "test_model"

[functions.test_function.experimentation]
type = "uniform"

[functions.test_function.experimentation.namespaces.mobile]
type = "track_and_stop"
metric = "test_metric"
candidate_variants = ["variant_a", "variant_b"]
min_samples_per_variant = 10
delta = 0.05
epsilon = 0.01
update_period_s = 1
"#
    .to_string()
}

/// Run a batch of inferences with a namespace and return (inference_id, variant_name) pairs.
async fn run_namespace_inference_batch(
    client: &Arc<Client>,
    count: usize,
    namespace: Option<&str>,
) -> Vec<(Uuid, String)> {
    let tasks: Vec<_> = (0..count)
        .map(|_| {
            let client = client.clone();
            let ns = namespace.map(|s| s.to_string());
            async move { do_inference(&client, ns.as_deref()).await }
        })
        .collect();
    join_all(tasks).await
}

/// Send boolean feedback for a batch of inferences using a Bernoulli bandit.
async fn send_namespace_feedback(
    client: &Arc<Client>,
    inference_results: &[(Uuid, String)],
    bandit: &BernoulliBandit,
    metric_name: &str,
) {
    for (inference_id, variant_name) in inference_results {
        let reward = bandit.sample(variant_name);
        client
            .feedback(FeedbackParams {
                inference_id: Some(*inference_id),
                metric_name: metric_name.to_string(),
                value: serde_json::json!(reward),
                ..Default::default()
            })
            .await
            .unwrap();
    }
}

/// Test that track_and_stop within a namespace converges to the winning variant.
///
/// This is the most important e2e test for the namespace track_and_stop feature.
/// It verifies the full pipeline: config loading -> inference routing -> namespace tag ->
/// feedback -> background task -> namespace-filtered ClickHouse query -> probability
/// update -> convergence.
#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_track_and_stop_convergence() {
    let config = make_namespace_track_and_stop_config();
    let (client, clickhouse, _guard) =
        Box::pin(make_embedded_gateway_with_clean_clickhouse(&config)).await;
    let client = Arc::new(client);

    // Set up bandit with a very clear winner: variant_a = 0.95, variant_b = 0.10
    let bandit = BernoulliBandit::new(vec![("variant_a", 0.95), ("variant_b", 0.10)], Some(42));

    let num_initial_batches = 2;
    let inferences_per_batch = 300;

    // Phase 1: Run inference + feedback batches with namespace="mobile" to train the model
    for _batch in 0..num_initial_batches {
        let inference_results =
            run_namespace_inference_batch(&client, inferences_per_batch, Some("mobile")).await;

        clickhouse.flush_pending_writes().await;
        tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

        send_namespace_feedback(&client, &inference_results, &bandit, "test_metric").await;

        clickhouse.flush_pending_writes().await;
        tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;
    }

    // Phase 2: Run inferences with namespace="mobile" and verify convergence to variant_a
    let verification_count = 50;
    let verification_results =
        run_namespace_inference_batch(&client, verification_count, Some("mobile")).await;
    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for (_, variant_name) in &verification_results {
        *variant_counts.entry(variant_name.clone()).or_insert(0) += 1;
    }

    let variant_a_count = variant_counts.get("variant_a").copied().unwrap_or(0);
    assert_eq!(
        variant_a_count, verification_count,
        "Expected 100% of namespace `mobile` inferences to converge to variant_a (the winner), \
         but got distribution: {variant_counts:?}"
    );

    // Phase 3: Verify that non-namespaced inferences still use uniform distribution
    let base_results = run_namespace_inference_batch(&client, 50, None).await;
    let mut base_counts: HashMap<String, usize> = HashMap::new();
    for (_, variant_name) in &base_results {
        *base_counts.entry(variant_name.clone()).or_insert(0) += 1;
    }

    assert!(
        base_counts.len() >= 2,
        "Base config (no namespace) should still use uniform distribution, got: {base_counts:?}"
    );
}

/// Test that the namespace-filtered `get_feedback_by_variant` query correctly
/// returns only feedback for inferences tagged with the specified namespace.
#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_feedback_query_filters_correctly() {
    let config = make_namespace_track_and_stop_config();
    let (client, clickhouse, _guard) =
        Box::pin(make_embedded_gateway_with_clean_clickhouse(&config)).await;
    let client = Arc::new(client);

    // Send inferences + feedback for namespace="mobile"
    let mobile_results = run_namespace_inference_batch(&client, 30, Some("mobile")).await;
    clickhouse.flush_pending_writes().await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // All mobile feedback: variant_a = true (1.0), variant_b = false (0.0)
    for (inference_id, variant_name) in &mobile_results {
        let value = if variant_name == "variant_a" {
            serde_json::json!(true)
        } else {
            serde_json::json!(false)
        };
        client
            .feedback(FeedbackParams {
                inference_id: Some(*inference_id),
                metric_name: "test_metric".to_string(),
                value,
                ..Default::default()
            })
            .await
            .unwrap();
    }

    // Send inferences + feedback with NO namespace (should not appear in namespace queries)
    let base_results = run_namespace_inference_batch(&client, 30, None).await;
    clickhouse.flush_pending_writes().await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // All base feedback: all true (different from mobile variant_b feedback)
    for (inference_id, _) in &base_results {
        client
            .feedback(FeedbackParams {
                inference_id: Some(*inference_id),
                metric_name: "test_metric".to_string(),
                value: serde_json::json!(true),
                ..Default::default()
            })
            .await
            .unwrap();
    }

    clickhouse.flush_pending_writes().await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // Query feedback filtered by namespace="mobile"
    let mobile_feedback = clickhouse
        .get_feedback_by_variant("test_metric", "test_function", None, Some("mobile"), None)
        .await
        .expect("Namespace-filtered feedback query should succeed");

    // We should only get feedback from mobile-tagged inferences
    let total_mobile_count: u64 = mobile_feedback.iter().map(|f| f.count).sum();
    let total_mobile_inferences = mobile_results.len() as u64;
    assert_eq!(
        total_mobile_count, total_mobile_inferences,
        "Namespace-filtered query should return exactly the count of mobile inferences ({total_mobile_inferences}), \
         but got {total_mobile_count}. Feedback: {mobile_feedback:?}"
    );

    // Query feedback WITHOUT namespace filter (should include all inferences)
    let all_feedback = clickhouse
        .get_feedback_by_variant("test_metric", "test_function", None, None, None)
        .await
        .expect("Unfiltered feedback query should succeed");

    let total_all_count: u64 = all_feedback.iter().map(|f| f.count).sum();
    let total_all_inferences = (mobile_results.len() + base_results.len()) as u64;
    assert_eq!(
        total_all_count, total_all_inferences,
        "Unfiltered query should return all inferences ({total_all_inferences}), \
         but got {total_all_count}. Feedback: {all_feedback:?}"
    );
}
