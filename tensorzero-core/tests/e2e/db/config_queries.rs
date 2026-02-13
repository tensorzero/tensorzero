//! E2E tests for ConfigQueries implementations.
//!
//! Tests that use only the `ConfigQueries` trait run against both ClickHouse and Postgres
//! via `make_db_test!`. Tests that require ClickHouse-specific APIs (embedded gateway,
//! raw queries) run against ClickHouse only.

use crate::db::get_test_postgres;

use sqlx::Row;
use std::collections::HashMap;
use std::time::Duration;
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    ClientBuilder, ClientInferenceParams, InferenceOutput, Input, InputMessage,
    InputMessageContent, Role,
};
use tensorzero_core::config::snapshot::{ConfigSnapshot, SnapshotHash};
use tensorzero_core::config::{Config, ConfigFileGlob};
use tensorzero_core::db::ConfigQueries;
use tensorzero_core::db::clickhouse::test_helpers::{
    CLICKHOUSE_URL, get_clickhouse, select_chat_inference_clickhouse,
};
use tensorzero_core::db::test_helpers::{TestDatabaseHelpers, poll_result_until_some};
use tensorzero_core::error::ErrorDetails;
use tensorzero_core::inference::types::Text;
use tensorzero_core::poll_clickhouse_for_result;
use uuid::Uuid;

// ===== DUAL-BACKEND TESTS (ClickHouse + Postgres) =====

async fn test_config_snapshot_write_and_read(conn: impl ConfigQueries + TestDatabaseHelpers) {
    let random_id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.test_metric_{random_id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let mut extra_templates = HashMap::new();
    extra_templates.insert("test_template".to_string(), "Hello {{name}}!".to_string());

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, extra_templates.clone()).unwrap();

    let hash = snapshot.hash.clone();

    conn.write_config_snapshot(&snapshot).await.unwrap();
    conn.flush_pending_writes().await;

    let retrieved_snapshot = conn.get_config_snapshot(hash).await.unwrap();

    let serialized_config = toml::to_string(&retrieved_snapshot.config).unwrap();
    assert!(
        serialized_config.contains(&format!("test_metric_{random_id}")),
        "Config should contain our test metric"
    );
    assert_eq!(
        retrieved_snapshot.extra_templates, extra_templates,
        "Extra templates should match"
    );
}
make_db_test!(test_config_snapshot_write_and_read);

async fn test_config_snapshot_not_found(conn: impl ConfigQueries + TestDatabaseHelpers) {
    let nonexistent_hash = SnapshotHash::new_test();

    let result = conn.get_config_snapshot(nonexistent_hash).await;

    let err = result.unwrap_err();
    assert!(
        matches!(
            err.get_details(),
            ErrorDetails::ConfigSnapshotNotFound { .. }
        ),
        "Expected ConfigSnapshotNotFound error, got: {err:?}"
    );
}
make_db_test!(test_config_snapshot_not_found);

async fn test_config_snapshot_with_extra_templates(conn: impl ConfigQueries + TestDatabaseHelpers) {
    let random_id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.test_metric_{random_id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let mut extra_templates = HashMap::new();
    extra_templates.insert(
        "system_template".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    extra_templates.insert(
        "user_template".to_string(),
        "User said: {{message}}".to_string(),
    );
    extra_templates.insert(
        "assistant_template".to_string(),
        "Assistant responds: {{response}}".to_string(),
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, extra_templates.clone()).unwrap();

    let hash = snapshot.hash.clone();

    conn.write_config_snapshot(&snapshot).await.unwrap();
    conn.flush_pending_writes().await;

    let retrieved_snapshot = conn.get_config_snapshot(hash).await.unwrap();

    let serialized_config = toml::to_string(&retrieved_snapshot.config).unwrap();
    assert!(
        serialized_config.contains(&format!("test_metric_{random_id}")),
        "Config should contain our test metric"
    );
    assert_eq!(
        retrieved_snapshot.extra_templates.len(),
        3,
        "Should have 3 extra templates"
    );
    assert_eq!(
        retrieved_snapshot.extra_templates.get("system_template"),
        Some(&"You are a helpful assistant.".to_string()),
        "system_template should match"
    );
    assert_eq!(
        retrieved_snapshot.extra_templates.get("user_template"),
        Some(&"User said: {{message}}".to_string()),
        "user_template should match"
    );
    assert_eq!(
        retrieved_snapshot.extra_templates.get("assistant_template"),
        Some(&"Assistant responds: {{response}}".to_string()),
        "assistant_template should match"
    );
}
make_db_test!(test_config_snapshot_with_extra_templates);

async fn test_config_snapshot_includes_built_in_functions(
    conn: impl ConfigQueries + TestDatabaseHelpers,
) {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("config.toml");

    // Write a minimal config (no user-defined functions)
    std::fs::write(&config_path, "[gateway]").unwrap();

    // Load the config - this injects built-in functions in process_config_input
    let loaded = Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new(config_path.to_string_lossy().to_string()).unwrap(),
        false,
    )
    .await
    .unwrap();

    // Write snapshot to the database and get the config with its hash
    let config = Box::pin(loaded.into_config(&conn)).await.unwrap();
    conn.flush_pending_writes().await;

    // Read back the snapshot via ConfigQueries
    let retrieved = conn.get_config_snapshot(config.hash.clone()).await.unwrap();
    let stored_config = toml::to_string(&retrieved.config).unwrap();

    // Verify built-in functions are in the stored config
    assert!(
        stored_config.contains("tensorzero::optimization::gepa::analyze"),
        "Snapshot should contain GEPA analyze function. Config:\n{stored_config}"
    );
    assert!(
        stored_config.contains("tensorzero::hello_chat"),
        "Snapshot should contain hello_chat function. Config:\n{stored_config}"
    );
    assert!(
        stored_config.contains("tensorzero::hello_json"),
        "Snapshot should contain hello_json function. Config:\n{stored_config}"
    );
}
make_db_test!(test_config_snapshot_includes_built_in_functions);

async fn test_config_snapshot_tag_merging(conn: impl ConfigQueries + TestDatabaseHelpers) {
    use tensorzero_core::config::stored::StoredConfig;

    let random_id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.tag_test_metric_{random_id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let stored_config: StoredConfig = toml::from_str(&config_toml).unwrap();
    let mut tags1 = HashMap::new();
    tags1.insert("key1".to_string(), "value1".to_string());
    tags1.insert("key2".to_string(), "original".to_string());

    let snapshot1 =
        ConfigSnapshot::from_stored_config(stored_config.clone(), HashMap::new(), tags1).unwrap();

    let hash = snapshot1.hash.clone();

    conn.write_config_snapshot(&snapshot1).await.unwrap();
    conn.flush_pending_writes().await;

    // Verify initial tags
    let retrieved1 = conn.get_config_snapshot(hash.clone()).await.unwrap();
    assert_eq!(
        retrieved1.tags.get("key1"),
        Some(&"value1".to_string()),
        "key1 should have initial value"
    );
    assert_eq!(
        retrieved1.tags.get("key2"),
        Some(&"original".to_string()),
        "key2 should have initial value"
    );

    // Write the same config with different tags
    let mut tags2 = HashMap::new();
    tags2.insert("key2".to_string(), "updated".to_string());
    tags2.insert("key3".to_string(), "new".to_string());

    let snapshot2 =
        ConfigSnapshot::from_stored_config(stored_config.clone(), HashMap::new(), tags2).unwrap();

    assert_eq!(snapshot2.hash, hash, "Same config should produce same hash");

    conn.write_config_snapshot(&snapshot2).await.unwrap();
    conn.flush_pending_writes().await;

    // Verify tags were merged
    let retrieved2 = conn.get_config_snapshot(hash).await.unwrap();

    assert_eq!(
        retrieved2.tags.get("key1"),
        Some(&"value1".to_string()),
        "key1 should be preserved from first write"
    );
    assert_eq!(
        retrieved2.tags.get("key2"),
        Some(&"updated".to_string()),
        "key2 should be updated from second write"
    );
    assert_eq!(
        retrieved2.tags.get("key3"),
        Some(&"new".to_string()),
        "key3 should be added from second write"
    );
}
make_db_test!(test_config_snapshot_tag_merging);

/// Verifies ClickHouse-specific upsert behavior: `created_at` is preserved and
/// `last_used` is updated when writing the same config snapshot twice.
#[tokio::test(flavor = "multi_thread")]
async fn test_write_config_snapshot_upsert_clickhouse() {
    let clickhouse = get_clickhouse().await;

    let random_id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.test_metric_{random_id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let mut extra_templates = HashMap::new();
    extra_templates.insert("test_template".to_string(), "Hello {{name}}!".to_string());

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, extra_templates.clone()).unwrap();

    let hash = snapshot.hash.clone();
    let hash_number = hash.to_string();

    clickhouse.write_config_snapshot(&snapshot).await.unwrap();

    // Query the ConfigSnapshot table to verify the data was written
    let query = format!(
        "SELECT config, tensorzero_version, hash, created_at, last_used FROM ConfigSnapshot FINAL WHERE hash = toUInt256('{hash_number}') FORMAT JSONEachRow"
    );
    let response = poll_result_until_some(async || {
        clickhouse.flush_pending_writes().await;
        let response = clickhouse
            .run_query_synchronous_no_params(query.clone())
            .await
            .ok()?;
        (!response.response.is_empty()).then_some(response)
    })
    .await;

    let snapshot_row: serde_json::Value = serde_json::from_str(&response.response).unwrap();

    let stored_config = snapshot_row["config"].as_str().unwrap();
    assert!(
        stored_config.contains(&format!("test_metric_{random_id}")),
        "Config should contain our test metric"
    );
    assert!(
        !snapshot_row["tensorzero_version"]
            .as_str()
            .unwrap()
            .is_empty(),
        "tensorzero_version should not be empty"
    );
    assert_eq!(
        snapshot_row["hash"].as_str().unwrap().to_lowercase(),
        hash_number,
        "Hash should match"
    );

    let created_at = snapshot_row["created_at"].as_str().unwrap();
    let last_used_1 = snapshot_row["last_used"].as_str().unwrap();

    // Write the same config again to test upsert
    let snapshot2 =
        ConfigSnapshot::new_from_toml_string(&config_toml, extra_templates.clone()).unwrap();

    clickhouse.write_config_snapshot(&snapshot2).await.unwrap();

    let response2 = poll_result_until_some(async || {
        clickhouse.flush_pending_writes().await;
        let response = clickhouse
            .run_query_synchronous_no_params(query.clone())
            .await
            .ok()?;
        (!response.response.is_empty()).then_some(response)
    })
    .await;

    let snapshot_row2: serde_json::Value = serde_json::from_str(&response2.response).unwrap();

    assert_eq!(
        snapshot_row2["created_at"].as_str().unwrap(),
        created_at,
        "created_at should be preserved on upsert"
    );

    let last_used_2 = snapshot_row2["last_used"].as_str().unwrap();
    assert!(
        last_used_2 >= last_used_1,
        "last_used should be updated on upsert"
    );

    let stored_config2 = snapshot_row2["config"].as_str().unwrap();
    assert!(
        stored_config2.contains(&format!("test_metric_{random_id}")),
        "Config should still contain our test metric after upsert"
    );
    assert_eq!(
        snapshot_row2["hash"].as_str().unwrap().to_lowercase(),
        hash_number,
        "Hash should still match after upsert"
    );
}

/// Verifies Postgres-specific upsert behavior: `created_at` is preserved and
/// `last_used` is updated when writing the same config snapshot twice.
#[tokio::test]
async fn test_write_config_snapshot_upsert_postgres() {
    let postgres = get_test_postgres().await;

    let random_id = Uuid::now_v7();

    let config_toml = format!(
        r#"
[metrics.test_metric_{random_id}]
type = "boolean"
level = "inference"
optimize = "max"
"#
    );

    let mut extra_templates = HashMap::new();
    extra_templates.insert("test_template".to_string(), "Hello {{name}}!".to_string());

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, extra_templates.clone()).unwrap();

    let hash = snapshot.hash.clone();

    postgres.write_config_snapshot(&snapshot).await.unwrap();

    // Query the config_snapshots table directly to verify the data was written
    let pool = postgres.get_pool().unwrap();
    let row = sqlx::query(
        "SELECT config, tensorzero_version, created_at, last_used FROM tensorzero.config_snapshots WHERE hash = $1",
    )
    .bind(hash.as_bytes())
    .fetch_one(pool)
    .await
    .unwrap();

    let stored_config: &str = row.get("config");
    assert!(
        stored_config.contains(&format!("test_metric_{random_id}")),
        "Config should contain our test metric"
    );
    let tensorzero_version: &str = row.get("tensorzero_version");
    assert!(
        !tensorzero_version.is_empty(),
        "tensorzero_version should not be empty"
    );

    let created_at: chrono::DateTime<chrono::Utc> = row.get("created_at");
    let last_used_1: chrono::DateTime<chrono::Utc> = row.get("last_used");

    // Write the same config again to test upsert
    let snapshot2 =
        ConfigSnapshot::new_from_toml_string(&config_toml, extra_templates.clone()).unwrap();

    postgres.write_config_snapshot(&snapshot2).await.unwrap();

    let row2 = sqlx::query(
        "SELECT config, created_at, last_used FROM tensorzero.config_snapshots WHERE hash = $1",
    )
    .bind(hash.as_bytes())
    .fetch_one(pool)
    .await
    .unwrap();

    let created_at_2: chrono::DateTime<chrono::Utc> = row2.get("created_at");
    assert_eq!(
        created_at_2, created_at,
        "created_at should be preserved on upsert"
    );

    let last_used_2: chrono::DateTime<chrono::Utc> = row2.get("last_used");
    assert!(
        last_used_2 >= last_used_1,
        "last_used should be updated on upsert"
    );

    let stored_config2: &str = row2.get("config");
    assert!(
        stored_config2.contains(&format!("test_metric_{random_id}")),
        "Config should still contain our test metric after upsert"
    );
}

// ===== Embedded Gateway E2E Tests =====
// These tests use ClickHouse-specific APIs (embedded gateway, raw queries,
// select_chat_inference_clickhouse) that don't have Postgres equivalents.
// TODO(#5691): Change these to work with Postgres once we have e2e writes working.

/// Test the config snapshot lifecycle when runtime-overlaid fields are NOT explicitly set.
///
/// # Test Flow
/// 1. Build a client from a config (without explicit runtime gateway fields)
/// 2. Do an inference
/// 3. Drop the client
/// 4. Load the snapshot from ClickHouse
/// 5. Build a new client from the snapshot
/// 6. Do another inference
/// 7. Assert the hashes are DIFFERENT (expected behavior - see below)
///
/// # Why the hashes are different
///
/// When a config is rehydrated from a snapshot, `RuntimeOverlay::from_config()` takes the
/// live `Config` (where defaults have been applied) and copies those concrete values back
/// to `UninitializedGatewayConfig` with explicit `Some(value)`. This causes a difference
/// in TOML serialization:
///
/// - **Original config**: Fields like `fetch_and_encode_input_files_before_inference` and
///   `global_outbound_http_timeout_ms` are `None`, so they're omitted from the serialized TOML.
///
/// - **Rehydrated config**: These fields become `Some(default_value)` via the RuntimeOverlay,
///   so they're explicitly present in the serialized TOML.
///
/// This different serialization produces different hashes, which is **expected behavior**.
/// The hash represents the exact TOML content, and the content IS different even though
/// the semantic meaning is the same.
///
/// # How to get stable hashes
///
/// If you need hash stability across rehydration, explicitly set all runtime-overlaid
/// gateway fields in your original config. See `test_config_snapshot_hash_stable_with_explicit_runtime_fields`
/// for an example of this.
#[tokio::test(flavor = "multi_thread")]
async fn test_config_snapshot_inference_roundtrip() {
    let random_id = Uuid::now_v7();
    let config = format!(
        r#"
[models.test_model_{random_id}]
routing = ["good"]

[models.test_model_{random_id}.providers.good]
type = "dummy"
model_name = "good"

[functions.basic_test_{random_id}]
type = "chat"

[functions.basic_test_{random_id}.variants.test_variant]
type = "chat_completion"
model = "test_model_{random_id}"
"#
    );

    let client = make_embedded_gateway_with_config(&config).await;

    let params = ClientInferenceParams {
        function_name: Some(format!("basic_test_{random_id}")),
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };
    let InferenceOutput::NonStreaming(response1) = client.inference(params).await.unwrap() else {
        panic!("Expected a non-streaming response");
    };
    let inference_id_1 = response1.inference_id();

    drop(client);

    let clickhouse = get_clickhouse().await;
    let inference_row = poll_clickhouse_for_result!(
        select_chat_inference_clickhouse(&clickhouse, inference_id_1).await
    );
    let snapshot_hash_str = inference_row
        .get("snapshot_hash")
        .unwrap()
        .as_str()
        .unwrap();
    let snapshot_hash: SnapshotHash = snapshot_hash_str.parse().unwrap();

    let retrieved_snapshot = clickhouse
        .get_config_snapshot(snapshot_hash.clone())
        .await
        .unwrap();

    let live_config = Config::default();
    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        &live_config,
        Some(CLICKHOUSE_URL.clone()),
        None,
        None,
        false,
        Some(Duration::from_secs(60)),
    ))
    .await
    .unwrap();

    let params2 = ClientInferenceParams {
        function_name: Some(format!("basic_test_{random_id}")),
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello again!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };
    let InferenceOutput::NonStreaming(response2) = new_client.inference(params2).await.unwrap()
    else {
        panic!("Expected a non-streaming response");
    };
    let inference_id_2 = response2.inference_id();

    let inference_row2 = poll_clickhouse_for_result!(
        select_chat_inference_clickhouse(&clickhouse, inference_id_2).await
    );
    let stored_hash2 = inference_row2
        .get("snapshot_hash")
        .unwrap()
        .as_str()
        .unwrap();

    // The hashes are DIFFERENT because the original config didn't explicitly set
    // runtime-overlaid gateway fields. When rehydrated, RuntimeOverlay adds these
    // fields with explicit default values, changing the serialized TOML and thus the hash.
    assert_ne!(
        snapshot_hash_str, stored_hash2,
        "Hashes should be DIFFERENT because runtime-overlaid fields were not explicitly set. \
         Original: {snapshot_hash_str}, Rehydrated: {stored_hash2}",
    );

    assert!(
        !snapshot_hash_str.is_empty(),
        "Original hash should not be empty"
    );
    assert!(
        !stored_hash2.is_empty(),
        "Rehydrated hash should not be empty"
    );
}

/// Test that config snapshot hash is stable through a roundtrip when all runtime-overlaid
/// fields are explicitly set to their default values in the original config.
///
/// See `test_config_snapshot_inference_roundtrip` for details on why hashes differ
/// when runtime fields are not explicitly set.
#[tokio::test(flavor = "multi_thread")]
async fn test_config_snapshot_hash_stable_with_explicit_runtime_fields() {
    let random_id = Uuid::now_v7();
    let config = format!(
        r#"
[gateway]
fetch_and_encode_input_files_before_inference = false
global_outbound_http_timeout_ms = 900000

[gateway.template_filesystem_access]
enabled = false

[models.test_model_{random_id}]
routing = ["good"]

[models.test_model_{random_id}.providers.good]
type = "dummy"
model_name = "good"

[functions.basic_test_{random_id}]
type = "chat"

[functions.basic_test_{random_id}.variants.test_variant]
type = "chat_completion"
model = "test_model_{random_id}"
"#
    );

    let client = make_embedded_gateway_with_config(&config).await;

    let params = ClientInferenceParams {
        function_name: Some(format!("basic_test_{random_id}")),
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };
    let InferenceOutput::NonStreaming(response1) = client.inference(params).await.unwrap() else {
        panic!("Expected a non-streaming response");
    };
    let inference_id_1 = response1.inference_id();

    drop(client);

    let clickhouse = get_clickhouse().await;
    let inference_row = poll_clickhouse_for_result!(
        select_chat_inference_clickhouse(&clickhouse, inference_id_1).await
    );
    let snapshot_hash_str = inference_row
        .get("snapshot_hash")
        .unwrap()
        .as_str()
        .unwrap();
    let snapshot_hash: SnapshotHash = snapshot_hash_str.parse().unwrap();

    let retrieved_snapshot = clickhouse
        .get_config_snapshot(snapshot_hash.clone())
        .await
        .unwrap();

    let live_config = Config::default();
    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        &live_config,
        Some(CLICKHOUSE_URL.clone()),
        None,
        None,
        false,
        Some(Duration::from_secs(60)),
    ))
    .await
    .unwrap();

    let params2 = ClientInferenceParams {
        function_name: Some(format!("basic_test_{random_id}")),
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello again!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };
    let InferenceOutput::NonStreaming(response2) = new_client.inference(params2).await.unwrap()
    else {
        panic!("Expected a non-streaming response");
    };
    let inference_id_2 = response2.inference_id();

    let inference_row2 = poll_clickhouse_for_result!(
        select_chat_inference_clickhouse(&clickhouse, inference_id_2).await
    );
    let stored_hash2 = inference_row2
        .get("snapshot_hash")
        .unwrap()
        .as_str()
        .unwrap();

    assert_eq!(
        snapshot_hash_str, stored_hash2,
        "Hash mismatch! Original: {snapshot_hash_str}, Rehydrated: {stored_hash2}. \
         This indicates that RuntimeOverlay is adding fields not present in the original config. \
         Ensure all runtime-overlaid gateway fields are explicitly set.",
    );
}

/// Test that from_config_snapshot correctly overlays runtime config from live_config
#[tokio::test(flavor = "multi_thread")]
async fn test_from_config_snapshot_overlays_runtime_config() {
    use tensorzero::ClientExt;

    let random_id = Uuid::now_v7();
    let config = format!(
        r#"
[gateway]
debug = false

[models.test_model_{random_id}]
routing = ["good"]

[models.test_model_{random_id}.providers.good]
type = "dummy"
model_name = "good"

[functions.basic_test_{random_id}]
type = "chat"

[functions.basic_test_{random_id}.variants.test_variant]
type = "chat_completion"
model = "test_model_{random_id}"
"#
    );

    let client = make_embedded_gateway_with_config(&config).await;

    let original_config = client.get_config().unwrap();
    assert!(
        !original_config.gateway.debug,
        "Original config should have debug = false"
    );

    let params = ClientInferenceParams {
        function_name: Some(format!("basic_test_{random_id}")),
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };
    let InferenceOutput::NonStreaming(response) = client.inference(params).await.unwrap() else {
        panic!("Expected a non-streaming response");
    };
    let inference_id = response.inference_id();

    drop(client);

    let clickhouse = get_clickhouse().await;
    let inference_row = poll_clickhouse_for_result!(
        select_chat_inference_clickhouse(&clickhouse, inference_id).await
    );
    let snapshot_hash_str = inference_row
        .get("snapshot_hash")
        .unwrap()
        .as_str()
        .unwrap();
    let snapshot_hash: SnapshotHash = snapshot_hash_str.parse().unwrap();

    let retrieved_snapshot = clickhouse
        .get_config_snapshot(snapshot_hash.clone())
        .await
        .unwrap();

    let mut live_config = Config::default();
    live_config.gateway.debug = true;
    live_config.postgres.connection_pool_size = 99;

    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        &live_config,
        Some(CLICKHOUSE_URL.clone()),
        None,
        None,
        false,
        Some(Duration::from_secs(60)),
    ))
    .await
    .unwrap();

    let new_config = new_client.get_config().unwrap();

    assert!(
        new_config.gateway.debug,
        "Gateway should be overlaid from live_config (debug = true)"
    );
    assert_eq!(
        new_config.postgres.connection_pool_size, 99,
        "Postgres should be overlaid from live_config"
    );
    assert!(
        new_config
            .functions
            .contains_key(&format!("basic_test_{random_id}")),
        "Function from snapshot should still be present"
    );
}
