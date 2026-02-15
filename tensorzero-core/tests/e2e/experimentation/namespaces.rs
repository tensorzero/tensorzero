use std::collections::HashMap;
use tempfile::NamedTempFile;
use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams, InferenceOutput,
    InferenceResponse, Input, InputMessage, InputMessageContent, PostgresConfig, Role,
};
use tensorzero_core::config::Namespace;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::Text;
use tokio::time::Duration;
use url::Url;

use crate::clickhouse::{DeleteDbOnDrop, get_clean_clickhouse};

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
        valkey_rate_limiting_url: None,
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
