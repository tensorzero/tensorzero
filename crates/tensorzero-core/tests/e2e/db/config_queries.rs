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
    ClientBuilder, ClientExt, ClientInferenceParams, InferenceOutput, Input, InputMessage,
    InputMessageContent, Role,
};
use tensorzero_core::config::gateway::UninitializedGatewayConfig;
use tensorzero_core::config::snapshot::{ConfigSnapshot, SnapshotHash};
use tensorzero_core::config::{
    Config, ConfigFileGlob, ObjectStoreInfo, PostgresConfig, RuntimeOverlay, UninitializedConfig,
};
use tensorzero_core::db::clickhouse::test_helpers::{
    CLICKHOUSE_URL, get_clickhouse, select_chat_inference_clickhouse,
};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::db::{ConfigQueries, ConfigSnapshotSearch};
use tensorzero_core::error::ErrorDetails;
use tensorzero_core::inference::types::Text;
use tensorzero_types::SnapshotHashScheme;
use uuid::Uuid;

// ===== DUAL-BACKEND TESTS (ClickHouse + Postgres) =====

fn runtime_overlay_from_toml(config_toml: &str) -> RuntimeOverlay {
    let table: toml::Table = toml::from_str(config_toml).unwrap();
    let config = UninitializedConfig::try_from(table).unwrap();
    let object_store_info = ObjectStoreInfo::new(config.object_storage.clone()).unwrap();
    RuntimeOverlay::from_uninitialized_config(&config, object_store_info)
}

#[expect(clippy::disallowed_methods)]
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
    conn.sleep_for_writes_to_be_visible().await;

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

#[expect(clippy::disallowed_methods)]
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
    conn.sleep_for_writes_to_be_visible().await;

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
    let (config, _) = Box::pin(loaded.into_config(&conn)).await.unwrap();
    conn.sleep_for_writes_to_be_visible().await;

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

#[expect(clippy::disallowed_methods)]
async fn test_config_snapshot_tag_merging(conn: impl ConfigQueries + TestDatabaseHelpers) {
    use tensorzero_core::config::snapshot::StoredConfig;

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
    conn.sleep_for_writes_to_be_visible().await;

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
    conn.sleep_for_writes_to_be_visible().await;

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
#[expect(clippy::disallowed_methods)]
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
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Query the ConfigSnapshot table to verify the data was written
    let query = format!(
        "SELECT config, tensorzero_version, hash, created_at, last_used FROM ConfigSnapshot FINAL WHERE hash = toUInt256('{hash_number}') FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query.clone())
        .await
        .unwrap();

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
    tokio::time::sleep(Duration::from_millis(500)).await;

    let response2 = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();

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
#[expect(clippy::disallowed_methods)]
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

/// Test the config snapshot lifecycle when runtime-overlaid fields are omitted.
///
/// # Test Flow
/// 1. Build a client from a config (without explicit runtime gateway fields)
/// 2. Do an inference
/// 3. Drop the client
/// 4. Load the snapshot from ClickHouse
/// 5. Build a new client from the snapshot
/// 6. Do another inference
/// 7. Assert the hashes are the SAME
///
/// # Why the hashes are stable
///
/// The runtime overlay is captured from the original `UninitializedConfig` before defaults
/// are resolved, so omitted runtime fields stay omitted when a snapshot is rehydrated.
/// This preserves the serialized config and keeps the snapshot hash stable.
///
/// This test exercises the regression directly with omitted runtime sections.
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
    let runtime_overlay = runtime_overlay_from_toml(&config);

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
    tokio::time::sleep(Duration::from_millis(500)).await;

    let clickhouse = get_clickhouse().await;
    let inference_row = select_chat_inference_clickhouse(&clickhouse, inference_id_1)
        .await
        .unwrap();
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

    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        runtime_overlay,
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

    tokio::time::sleep(Duration::from_millis(500)).await;

    let inference_row2 = select_chat_inference_clickhouse(&clickhouse, inference_id_2)
        .await
        .unwrap();
    let stored_hash2 = inference_row2
        .get("snapshot_hash")
        .unwrap()
        .as_str()
        .unwrap();

    assert_eq!(
        snapshot_hash_str, stored_hash2,
        "Hashes should be stable even when runtime-overlaid fields were omitted. \
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
    let runtime_overlay = runtime_overlay_from_toml(&config);

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
    tokio::time::sleep(Duration::from_millis(500)).await;

    let clickhouse = get_clickhouse().await;
    let inference_row = select_chat_inference_clickhouse(&clickhouse, inference_id_1)
        .await
        .unwrap();
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

    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        runtime_overlay,
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

    tokio::time::sleep(Duration::from_millis(500)).await;

    let inference_row2 = select_chat_inference_clickhouse(&clickhouse, inference_id_2)
        .await
        .unwrap();
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
    tokio::time::sleep(Duration::from_millis(500)).await;

    let clickhouse = get_clickhouse().await;
    let inference_row = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
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

    // These runtime overlay items should not affect snapshot hash.
    let live_runtime_overlay = RuntimeOverlay {
        gateway: Some(UninitializedGatewayConfig {
            debug: Some(true),
            ..Default::default()
        }),
        postgres: Some(PostgresConfig {
            connection_pool_size: Some(99),
            ..Default::default()
        }),
        ..Default::default()
    };

    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        live_runtime_overlay,
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
        new_config.postgres.connection_pool_size,
        Some(99),
        "Postgres should be overlaid from live_config"
    );
    assert!(
        new_config
            .functions
            .contains_key(&format!("basic_test_{random_id}")),
        "Function from snapshot should still be present"
    );
}

// ===== Postgres `config_jsonb` JSONB tests =====
//
// These cover the new column added alongside `config TEXT`:
// - `write_config_snapshot` populates `config_jsonb` with the JSON form of
//   the same `StoredConfig` that's serialized to TOML in `config`.
// - The `ConfigSnapshotSearch` trait queries the JSONB column via `@>`
//   containment, which is what the GIN index
//   `config_snapshots_config_jsonb_gin` is built for.
// - `backfill_config_snapshot_jsonb` populates rows that predate the column.
// - `get_config_snapshot` continues to read from `config TEXT` (TOML), NOT
//   from `config_jsonb`. This is the V0 invariant: the TOML column stays
//   the source of truth for hashing until the planned `hash_v2` cutover
//   (out of V0). Switching the read path to JSONB right now would drift
//   hashes for snapshots persisted before the cutover migration.

/// Helper: insert a row with `config_jsonb = NULL` to simulate a snapshot
/// written before the JSONB column existed. Mirrors the on-disk shape of
/// rows from earlier gateway versions.
async fn insert_legacy_snapshot_row(
    postgres: &tensorzero_core::db::postgres::PostgresConnectionInfo,
    snapshot: &ConfigSnapshot,
) {
    let pool = postgres.get_pool().unwrap();
    let config_string = toml::to_string(&snapshot.config).expect("serialize config to TOML");
    sqlx::query(
        r"INSERT INTO tensorzero.config_snapshots (hash, config, extra_templates, tensorzero_version, tags, config_jsonb)
           VALUES ($1, $2, '{}'::jsonb, $3, '{}'::jsonb, NULL)
           ON CONFLICT (hash) DO UPDATE SET config_jsonb = NULL",
    )
    .bind(snapshot.hash.as_bytes())
    .bind(&config_string)
    .bind(tensorzero_core::endpoints::status::TENSORZERO_VERSION)
    .execute(pool)
    .await
    .expect("legacy snapshot insert should succeed");
}

#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn config_jsonb_column_is_populated_on_write_postgres() {
    let postgres = get_test_postgres().await;
    let random_id = Uuid::now_v7();

    // Function-level discriminator we can later search for via JSONB
    // containment. (We use `description` rather than the planned
    // `version` field because the latter doesn't exist on
    // `UninitializedFunctionConfig` until the per-object metadata PR.)
    let config_toml = format!(
        r#"
[models.dummy_{random_id}]
routing = ["dummy"]

[models.dummy_{random_id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.f_{random_id}]
type = "chat"
description = "v4"

[functions.f_{random_id}.variants.v1]
type = "chat_completion"
model = "dummy_{random_id}"
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new()).expect("parse fixture");
    let hash = snapshot.hash.clone();
    postgres.write_config_snapshot(&snapshot).await.unwrap();

    let pool = postgres.get_pool().unwrap();
    let row = sqlx::query(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
        .bind(hash.as_bytes())
        .fetch_one(pool)
        .await
        .unwrap();

    let stored_json: serde_json::Value = row.try_get("config_jsonb").unwrap();
    let in_memory_json = serde_json::to_value(&snapshot.config).unwrap();

    // JSONB is content-equal even though Postgres normalizes key order
    // (serde_json::Value compares by content, not by string).
    assert_eq!(
        stored_json, in_memory_json,
        "config_jsonb column must equal serde_json::to_value(&snapshot.config)",
    );

    // The discriminator field landed at the expected path — same shape
    // the GIN queries target.
    assert_eq!(
        stored_json
            .pointer(&format!("/functions/f_{random_id}/description"))
            .and_then(|v| v.as_str()),
        Some("v4"),
    );
}

#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn snapshots_with_path_value_finds_matching_function_description() {
    let postgres = get_test_postgres().await;
    let target_id = Uuid::now_v7();

    // Three snapshots: two with the SAME function name carrying
    // `description = "v7"`, one with `description = "v8"`. A decoy uses
    // a different function name. Only the rows whose description
    // matches the query should come back. (Using `description` rather
    // than the planned `version` field — the latter doesn't exist on
    // `UninitializedFunctionConfig` until per-object metadata lands.)
    for (description, marker) in [("v7", "alpha"), ("v7", "beta"), ("v8", "gamma")] {
        let config_toml = format!(
            r#"
[models.dummy_{target_id}_{marker}]
routing = ["dummy"]

[models.dummy_{target_id}_{marker}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.target_{target_id}]
type = "chat"
description = "{description}"

[functions.target_{target_id}.variants.{marker}]
type = "chat_completion"
model = "dummy_{target_id}_{marker}"
"#
        );
        let snap = ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new())
            .expect("fixture parse");
        postgres.write_config_snapshot(&snap).await.unwrap();
    }
    // Decoy: same description value but on a DIFFERENT function name —
    // must not match the target-name query because the JSONPath nests
    // the function name above the description.
    let decoy = format!(
        r#"
[models.decoy_{target_id}]
routing = ["dummy"]

[models.decoy_{target_id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.unrelated_{target_id}]
type = "chat"
description = "v7"

[functions.unrelated_{target_id}.variants.only]
type = "chat_completion"
model = "decoy_{target_id}"
"#
    );
    let decoy_snap =
        ConfigSnapshot::new_from_toml_string(&decoy, HashMap::new()).expect("decoy parse");
    postgres.write_config_snapshot(&decoy_snap).await.unwrap();

    // Query for description = "v7" on the target function name.
    let hits = postgres
        .snapshots_with_path_value(
            &format!("functions/target_{target_id}/description"),
            &serde_json::json!("v7"),
        )
        .await
        .unwrap();
    assert_eq!(
        hits.len(),
        2,
        "expected 2 snapshots with target function description=v7, got {}",
        hits.len(),
    );

    // Query for description = "v8" — only one match.
    let hits_v8 = postgres
        .snapshots_with_path_value(
            &format!("functions/target_{target_id}/description"),
            &serde_json::json!("v8"),
        )
        .await
        .unwrap();
    assert_eq!(hits_v8.len(), 1, "expected 1 snapshot at description=v8");

    // Query for a non-existent description — zero matches.
    let hits_missing = postgres
        .snapshots_with_path_value(
            &format!("functions/target_{target_id}/description"),
            &serde_json::json!("v999"),
        )
        .await
        .unwrap();
    assert_eq!(hits_missing.len(), 0, "expected no matches for v999");
}

#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn snapshots_containing_finds_variant_version_via_explicit_fragment() {
    let postgres = get_test_postgres().await;
    let id = Uuid::now_v7();

    // Two variants with different `temperature` values used as the
    // discriminator. (Using `temperature` rather than `version` since
    // the latter doesn't exist on `UninitializedVariantConfig` until
    // per-object metadata lands; `temperature` round-trips into JSONB
    // as a number, so containment queries work the same way.) We pick
    // values that are exactly representable in `f32` — `temperature`
    // is `f32` in `StoredConfig`, so a non-exact value like `0.3`
    // widens to `0.30000001192092896` in JSONB and never matches a
    // numerically-typed needle of `0.3`.
    let config_toml = format!(
        r#"
[models.m_{id}]
routing = ["dummy"]

[models.m_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.fn_{id}]
type = "chat"

[functions.fn_{id}.variants.a]
type = "chat_completion"
model = "m_{id}"
temperature = 0.5

[functions.fn_{id}.variants.b]
type = "chat_completion"
model = "m_{id}"
temperature = 0.25
"#
    );
    let snap = ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new()).unwrap();
    postgres.write_config_snapshot(&snap).await.unwrap();

    // Use `snapshots_containing` with an explicit nested fragment.
    let hits_a05 = postgres
        .snapshots_containing(serde_json::json!({
            "functions": {format!("fn_{id}"): {"variants": {"a": {"temperature": 0.5}}}}
        }))
        .await
        .unwrap();
    assert_eq!(
        hits_a05.len(),
        1,
        "variant a at temperature 0.5 should match"
    );
    assert_eq!(hits_a05[0].as_bytes(), snap.hash.as_bytes());

    let hits_a025 = postgres
        .snapshots_containing(serde_json::json!({
            "functions": {format!("fn_{id}"): {"variants": {"a": {"temperature": 0.25}}}}
        }))
        .await
        .unwrap();
    assert_eq!(
        hits_a025.len(),
        0,
        "variant a is at temperature 0.5, not 0.25; should not match",
    );

    // Equivalent via the path-value convenience.
    let hits_b025 = postgres
        .snapshots_with_path_value(
            &format!("functions/fn_{id}/variants/b/temperature"),
            &serde_json::json!(0.25),
        )
        .await
        .unwrap();
    assert_eq!(
        hits_b025.len(),
        1,
        "variant b at temperature 0.25 should match"
    );
}

#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn config_jsonb_gin_index_exists_and_serves_containment_queries() {
    // We test two production-faithful invariants:
    //
    // 1. `config_snapshots_config_jsonb_gin` exists with the
    //    `jsonb_path_ops` operator class — without it, no plan the
    //    planner could ever pick would use index scan.
    // 2. A `@>` containment query returns the right row.
    //
    // We deliberately do NOT assert "the planner picks GIN here". On a
    // small test table (`SELECT count(*) FROM config_snapshots ≪ 1000`)
    // Postgres correctly prefers Seq Scan even with the GIN index
    // available — that's the same logic production uses; the planner
    // flips to GIN once the table is big enough that walking it
    // beats the index. Forcing the choice with `enable_seqscan = off`
    // would make the test pass but diverge from production behavior.
    let postgres = get_test_postgres().await;
    let pool = postgres.get_pool().unwrap();

    // Invariant 1: index exists with the right operator class.
    let index_def: Option<String> = sqlx::query_scalar(
        r"SELECT indexdef FROM pg_indexes
           WHERE schemaname = 'tensorzero'
             AND tablename  = 'config_snapshots'
             AND indexname  = 'config_snapshots_config_jsonb_gin'",
    )
    .fetch_optional(pool)
    .await
    .unwrap();
    let index_def = index_def.expect(
        "GIN index `config_snapshots_config_jsonb_gin` must exist; \
         the migration that defines it is the whole reason for this test",
    );
    assert!(
        index_def.contains("USING gin") && index_def.contains("jsonb_path_ops"),
        "GIN index must use the `jsonb_path_ops` operator class for `@>` \
         containment to be supported. Got: {index_def}",
    );

    // Invariant 2: `@>` containment query against `config_jsonb`
    // returns the correct row. This exercises the full code path the
    // `ConfigSnapshotSearch` helpers use — whether the planner picks
    // Seq Scan or Bitmap Index Scan, the result is the same row.
    let id = Uuid::now_v7();
    let toml = format!(
        r#"
[models.m_{id}]
routing = ["dummy"]

[models.m_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.f_{id}]
type = "chat"

[functions.f_{id}.variants.only]
type = "chat_completion"
model = "m_{id}"
"#
    );
    let snap = ConfigSnapshot::new_from_toml_string(&toml, HashMap::new()).unwrap();
    postgres.write_config_snapshot(&snap).await.unwrap();

    // The function name `f_{id}` is unique to this run, so any field
    // present on the function in the canonical form makes a sufficient
    // discriminator. We use `type: "chat"` since it's part of the
    // internally-tagged enum's serialization and is guaranteed to be
    // in the JSONB. (Avoid using `version` here — that field doesn't
    // exist on `UninitializedFunctionConfig` yet, and earlier revisions
    // of this test inherited a `version = 5` TOML line that no longer
    // parses on this branch.)
    let needle = serde_json::json!({
        "functions": {
            format!("f_{id}"): { "type": "chat" }
        }
    });
    let hits: Vec<Vec<u8>> = sqlx::query_scalar(
        r"SELECT hash FROM tensorzero.config_snapshots WHERE config_jsonb @> $1",
    )
    .bind(&needle)
    .fetch_all(pool)
    .await
    .unwrap();
    assert_eq!(
        hits.len(),
        1,
        "containment query should match exactly the row we just wrote",
    );
    assert_eq!(hits[0], snap.hash.as_bytes());
}

#[tokio::test]
async fn backfill_populates_config_jsonb_for_legacy_rows() {
    use tensorzero_core::db::postgres::config_queries::backfill_config_snapshot_jsonb;
    let postgres = get_test_postgres().await;
    let pool = postgres.get_pool().unwrap();

    let id = Uuid::now_v7();
    let config_toml = format!(
        r#"
[models.bf_{id}]
routing = ["dummy"]

[models.bf_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.bf_{id}]
type = "chat"

[functions.bf_{id}.variants.only]
type = "chat_completion"
model = "bf_{id}"
"#
    );
    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new()).expect("parse");
    let hash = snapshot.hash.clone();

    // Insert as if from a pre-jsonb gateway (config_jsonb = NULL).
    insert_legacy_snapshot_row(&postgres, &snapshot).await;

    // Sanity: the row really has NULL config_jsonb before backfill.
    let nullness: Option<serde_json::Value> =
        sqlx::query_scalar(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
            .bind(hash.as_bytes())
            .fetch_one(pool)
            .await
            .unwrap();
    assert!(
        nullness.is_none(),
        "test setup should leave config_jsonb NULL pre-backfill",
    );

    // Run the backfill.
    backfill_config_snapshot_jsonb(pool).await.unwrap();

    let after: serde_json::Value =
        sqlx::query_scalar(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
            .bind(hash.as_bytes())
            .fetch_one(pool)
            .await
            .unwrap();
    let expected = serde_json::to_value(&snapshot.config).unwrap();
    assert_eq!(
        after, expected,
        "backfilled config_jsonb must equal serde_json::to_value(&snapshot.config)",
    );

    // The function shape is reachable via the same JSONB path the search
    // queries target. (Direct DB-side query rather than the helper, so this
    // remains a tight test of the column contents.)
    let cnt: i64 = sqlx::query_scalar(
        r"SELECT COUNT(*) FROM tensorzero.config_snapshots WHERE hash = $1 AND config_jsonb @> $2",
    )
    .bind(hash.as_bytes())
    .bind(serde_json::json!({"functions": {format!("bf_{id}"): {"type": "chat"}}}))
    .fetch_one(pool)
    .await
    .unwrap();
    assert_eq!(
        cnt, 1,
        "post-backfill row must match the function-shape containment query"
    );

    // Re-running backfill is a no-op. Run it again and ensure the value
    // didn't change.
    backfill_config_snapshot_jsonb(pool).await.unwrap();
    let after_again: serde_json::Value =
        sqlx::query_scalar(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
            .bind(hash.as_bytes())
            .fetch_one(pool)
            .await
            .unwrap();
    assert_eq!(after_again, expected, "backfill must be idempotent");
}

/// Backfill must skip rows whose `config TEXT` no longer parses as a
/// `StoredConfig` — and continue processing the rest of the table. We
/// fail-soft per row (log + skip) so a single broken legacy row does
/// not block startup. Verifies the contract end-to-end: insert one
/// backfillable row alongside one whose `config TEXT` is intentional
/// garbage, run the backfill, assert the good row got populated and
/// the bad row stayed `NULL`.
#[tokio::test]
async fn backfill_skips_unparseable_legacy_rows_postgres() {
    use tensorzero_core::db::postgres::config_queries::backfill_config_snapshot_jsonb;
    let postgres = get_test_postgres().await;
    let pool = postgres.get_pool().unwrap();

    // Good row: standard backfillable shape.
    let good_id = Uuid::now_v7();
    let good_toml = format!(
        r#"
[models.good_{good_id}]
routing = ["dummy"]

[models.good_{good_id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.good_{good_id}]
type = "chat"

[functions.good_{good_id}.variants.only]
type = "chat_completion"
model = "good_{good_id}"
"#
    );
    let good_snapshot =
        ConfigSnapshot::new_from_toml_string(&good_toml, HashMap::new()).expect("parse good");
    let good_hash = good_snapshot.hash.clone();
    insert_legacy_snapshot_row(&postgres, &good_snapshot).await;

    // Bad row: synthesize a fake hash and insert garbage TOML directly.
    // The hash bytes don't have to match the (un-)content; the backfill
    // looks up by `hash` and tries to reparse `config`.
    let bad_hash_bytes = blake3::hash(format!("bad-row-{}", Uuid::now_v7()).as_bytes());
    let bad_hash = bad_hash_bytes.as_bytes().to_vec();
    sqlx::query(
        r"INSERT INTO tensorzero.config_snapshots (hash, config, extra_templates, tensorzero_version, tags, config_jsonb)
           VALUES ($1, $2, '{}'::jsonb, $3, '{}'::jsonb, NULL)
           ON CONFLICT (hash) DO UPDATE SET config_jsonb = NULL, config = EXCLUDED.config",
    )
    .bind(&bad_hash)
    .bind("this is not valid toml :::: !! [malformed garbage")
    .bind(tensorzero_core::endpoints::status::TENSORZERO_VERSION)
    .execute(pool)
    .await
    .expect("bad row insert should succeed");

    // Run the backfill. Must NOT panic despite the bad row.
    backfill_config_snapshot_jsonb(pool)
        .await
        .expect("backfill should succeed even with unparseable rows present");

    // Good row got backfilled.
    let good_jsonb: Option<serde_json::Value> =
        sqlx::query_scalar(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
            .bind(good_hash.as_bytes())
            .fetch_one(pool)
            .await
            .unwrap();
    assert!(
        good_jsonb.is_some(),
        "well-formed legacy row must be backfilled despite a malformed sibling row"
    );

    // Bad row stayed NULL — backfill skipped it.
    let bad_jsonb: Option<serde_json::Value> =
        sqlx::query_scalar(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
            .bind(&bad_hash)
            .fetch_one(pool)
            .await
            .unwrap();
    assert!(
        bad_jsonb.is_none(),
        "unparseable row must remain NULL — backfill is best-effort, no fabricated content",
    );

    // Bad row's canonical_hash also still NULL (skip applies to BOTH columns).
    let bad_canonical: Option<Vec<u8>> = sqlx::query_scalar(
        r"SELECT canonical_hash FROM tensorzero.config_snapshots WHERE hash = $1",
    )
    .bind(&bad_hash)
    .fetch_one(pool)
    .await
    .unwrap();
    assert!(
        bad_canonical.is_none(),
        "canonical_hash must also stay NULL on a skipped row",
    );
}

/// Backfill must leave already-backfilled rows untouched. `WHERE
/// config_jsonb IS NULL OR canonical_hash IS NULL` filters them out at
/// the SQL level; the COALESCE guards the UPDATE itself. Together they
/// promise: if a row had a canonical_hash before, it has the same
/// canonical_hash after, even if a backfill rerun would compute
/// something different (which it shouldn't, but the guard is
/// defensive).
#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn backfill_leaves_already_populated_rows_unchanged_postgres() {
    use tensorzero_core::db::postgres::config_queries::backfill_config_snapshot_jsonb;
    let postgres = get_test_postgres().await;
    let pool = postgres.get_pool().unwrap();

    let id = Uuid::now_v7();
    let toml = format!(
        r#"
[models.preserved_{id}]
routing = ["dummy"]

[models.preserved_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.preserved_{id}]
type = "chat"

[functions.preserved_{id}.variants.only]
type = "chat_completion"
model = "preserved_{id}"
"#
    );
    let snapshot = ConfigSnapshot::new_from_toml_string(&toml, HashMap::new()).expect("parse");

    // Write through the normal path — both columns populated.
    postgres.write_config_snapshot(&snapshot).await.unwrap();

    // Capture before-state.
    let before_jsonb: serde_json::Value =
        sqlx::query_scalar(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
            .bind(snapshot.hash.as_bytes())
            .fetch_one(pool)
            .await
            .unwrap();
    let before_canonical: Vec<u8> = sqlx::query_scalar(
        r"SELECT canonical_hash FROM tensorzero.config_snapshots WHERE hash = $1",
    )
    .bind(snapshot.hash.as_bytes())
    .fetch_one(pool)
    .await
    .unwrap();

    // Run backfill — should NOT touch this row.
    backfill_config_snapshot_jsonb(pool).await.unwrap();

    // After-state matches before-state byte-for-byte.
    let after_jsonb: serde_json::Value =
        sqlx::query_scalar(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
            .bind(snapshot.hash.as_bytes())
            .fetch_one(pool)
            .await
            .unwrap();
    let after_canonical: Vec<u8> = sqlx::query_scalar(
        r"SELECT canonical_hash FROM tensorzero.config_snapshots WHERE hash = $1",
    )
    .bind(snapshot.hash.as_bytes())
    .fetch_one(pool)
    .await
    .unwrap();
    assert_eq!(
        after_jsonb, before_jsonb,
        "backfill must not rewrite already-populated config_jsonb",
    );
    assert_eq!(
        after_canonical, before_canonical,
        "backfill must not rewrite already-populated canonical_hash",
    );
}

/// Backfill on an empty table is a no-op — no panic, no error, no
/// log spam. Cheap regression for the trivial case.
#[tokio::test]
async fn backfill_no_op_on_empty_state_postgres() {
    use tensorzero_core::db::postgres::config_queries::backfill_config_snapshot_jsonb;
    let postgres = get_test_postgres().await;
    let pool = postgres.get_pool().unwrap();
    // We can't really nuke the table (other tests use it), but running
    // the backfill should be safe regardless of whether there are
    // unbackfilled rows or not. The contract is: it never panics, never
    // errors, returns Ok.
    backfill_config_snapshot_jsonb(pool)
        .await
        .expect("backfill must succeed regardless of table state");
}

/// Backfill must populate `canonical_hash` even on a row whose
/// `config_jsonb` has *already* been backfilled but whose
/// `canonical_hash` was never written. Exercises the partial-state
/// case — relevant if a previous backfill attempt completed JSONB but
/// crashed before getting to canonical_hash, or if we ever ship
/// the columns in two separate migrations. The `WHERE … IS NULL OR …
/// IS NULL` filter handles both flavors of incompletion.
#[tokio::test]
async fn backfill_populates_canonical_hash_when_config_jsonb_already_present_postgres() {
    use tensorzero_core::db::postgres::config_queries::backfill_config_snapshot_jsonb;
    let postgres = get_test_postgres().await;
    let pool = postgres.get_pool().unwrap();

    let id = Uuid::now_v7();
    let toml = format!(
        r#"
[models.partial_{id}]
routing = ["dummy"]

[models.partial_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.partial_{id}]
type = "chat"

[functions.partial_{id}.variants.only]
type = "chat_completion"
model = "partial_{id}"
"#
    );
    let snapshot = ConfigSnapshot::new_from_toml_string(&toml, HashMap::new()).expect("parse");
    let pre_jsonb = serde_json::to_value(&snapshot.config).expect("to_value");

    // Insert with config_jsonb populated but canonical_hash NULL.
    let config_string = toml::to_string(&snapshot.config).expect("to TOML");
    sqlx::query(
        r"INSERT INTO tensorzero.config_snapshots (hash, config, extra_templates, tensorzero_version, tags, config_jsonb, canonical_hash)
           VALUES ($1, $2, '{}'::jsonb, $3, '{}'::jsonb, $4, NULL)
           ON CONFLICT (hash) DO UPDATE SET canonical_hash = NULL, config_jsonb = EXCLUDED.config_jsonb",
    )
    .bind(snapshot.hash.as_bytes())
    .bind(&config_string)
    .bind(tensorzero_core::endpoints::status::TENSORZERO_VERSION)
    .bind(&pre_jsonb)
    .execute(pool)
    .await
    .expect("partial-state insert should succeed");

    // Run backfill.
    backfill_config_snapshot_jsonb(pool).await.unwrap();

    // canonical_hash now equals StoredConfig::canonical_hash().
    let after: Vec<u8> = sqlx::query_scalar(
        r"SELECT canonical_hash FROM tensorzero.config_snapshots WHERE hash = $1",
    )
    .bind(snapshot.hash.as_bytes())
    .fetch_one(pool)
    .await
    .unwrap();
    let expected = snapshot.config.canonical_hash().expect("canonical_hash");
    assert_eq!(
        after.as_slice(),
        expected.as_bytes(),
        "partial-state row's canonical_hash must equal StoredConfig::canonical_hash",
    );

    // config_jsonb stays put — wasn't touched.
    let after_jsonb: serde_json::Value =
        sqlx::query_scalar(r"SELECT config_jsonb FROM tensorzero.config_snapshots WHERE hash = $1")
            .bind(snapshot.hash.as_bytes())
            .fetch_one(pool)
            .await
            .unwrap();
    assert_eq!(
        after_jsonb, pre_jsonb,
        "config_jsonb must remain the value the partial-state insert wrote",
    );
}

/// Canonical invariant: `get_config_snapshot` reads `config_jsonb` (JSON
/// is the source of truth). `config TEXT` (TOML) is the deprecated
/// migration column; reading it on the canonical path would force the
/// system to keep TOML consistent with JSON forever, defeating the goal of
/// the migration.
///
/// Test: write a snapshot, tamper with the deprecated `config TEXT` column
/// (set it to garbage TOML that wouldn't parse), confirm the read still
/// succeeds and returns the original content — because we never look at
/// the TOML column on this path.
#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn get_config_snapshot_reads_jsonb_not_toml() {
    let postgres = get_test_postgres().await;
    let id = Uuid::now_v7();
    let config_toml = format!(
        r#"
[models.j_{id}]
routing = ["dummy"]

[models.j_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.j_fn_{id}]
type = "chat"

[functions.j_fn_{id}.variants.only]
type = "chat_completion"
model = "j_{id}"
"#
    );
    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new()).expect("parse");
    let hash = snapshot.hash.clone();
    postgres.write_config_snapshot(&snapshot).await.unwrap();

    // Corrupt the deprecated TOML column. If the read path ever falls back
    // to TOML when JSONB is present, this corruption surfaces as a parse
    // error or content mismatch.
    let pool = postgres.get_pool().unwrap();
    sqlx::query(
        r"UPDATE tensorzero.config_snapshots
           SET config = 'this is not valid TOML // <<< corrupted by test'
           WHERE hash = $1",
    )
    .bind(hash.as_bytes())
    .execute(pool)
    .await
    .unwrap();

    // Read via the public API. The corrupted TOML must NOT be observed.
    let retrieved = postgres.get_config_snapshot(hash.clone()).await.unwrap();
    assert_eq!(
        retrieved.hash.to_hex_string(),
        hash.to_hex_string(),
        "get_config_snapshot must return the original hash verbatim",
    );
    let original_json = serde_json::to_value(&snapshot.config).unwrap();
    let retrieved_json = serde_json::to_value(&retrieved.config).unwrap();
    assert_eq!(
        retrieved_json, original_json,
        "get_config_snapshot must return content-equal config sourced from JSONB",
    );
}

/// `canonical_hash` column is populated on every new write and equals
/// `StoredConfig::canonical_hash` of the same in-memory config.
#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn canonical_hash_column_is_populated_on_write_postgres() {
    let postgres = get_test_postgres().await;
    let id = Uuid::now_v7();
    let config_toml = format!(
        r#"
[models.dummy_{id}]
routing = ["dummy"]

[models.dummy_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.f_{id}]
type = "chat"

[functions.f_{id}.variants.v]
type = "chat_completion"
model = "dummy_{id}"
"#
    );
    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new()).expect("parse");
    let legacy_hash = snapshot.hash.clone();
    let expected_canonical = snapshot.config.canonical_hash().expect("canonical hash");
    postgres.write_config_snapshot(&snapshot).await.unwrap();

    let pool = postgres.get_pool().unwrap();
    let stored_canonical: Vec<u8> = sqlx::query_scalar(
        r"SELECT canonical_hash FROM tensorzero.config_snapshots WHERE hash = $1",
    )
    .bind(legacy_hash.as_bytes())
    .fetch_one(pool)
    .await
    .unwrap();
    assert_eq!(
        stored_canonical.as_slice(),
        expected_canonical.as_bytes(),
        "canonical_hash column must equal StoredConfig::canonical_hash",
    );
}

/// `get_config_snapshot` resolves the same row whether passed the legacy
/// hash (queries `hash` column) or the canonical hash (queries
/// `canonical_hash` column). Both lookups return the same content.
#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn get_config_snapshot_dispatches_on_scheme() {
    let postgres = get_test_postgres().await;
    let id = Uuid::now_v7();
    let config_toml = format!(
        r#"
[models.disp_{id}]
routing = ["dummy"]

[models.disp_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.disp_fn_{id}]
type = "chat"

[functions.disp_fn_{id}.variants.v]
type = "chat_completion"
model = "disp_{id}"
"#
    );
    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new()).expect("parse");
    let legacy_hash = snapshot.hash.clone();
    let canonical_hash = snapshot.config.canonical_hash().expect("canonical");
    assert_eq!(legacy_hash.scheme(), SnapshotHashScheme::LegacyToml);
    assert_eq!(canonical_hash.scheme(), SnapshotHashScheme::Canonical);
    postgres.write_config_snapshot(&snapshot).await.unwrap();

    // Lookup via legacy scheme.
    let via_legacy = postgres
        .get_config_snapshot(legacy_hash.clone())
        .await
        .expect("legacy lookup");
    // Lookup via canonical scheme.
    let via_canonical = postgres
        .get_config_snapshot(canonical_hash)
        .await
        .expect("canonical lookup");

    // Both lookups returned the same row — same hash returned (the legacy
    // one, because the row's primary key is legacy).
    assert_eq!(via_legacy.hash.as_bytes(), legacy_hash.as_bytes());
    let via_legacy_json = serde_json::to_value(&via_legacy.config).unwrap();
    let via_canonical_json = serde_json::to_value(&via_canonical.config).unwrap();
    assert_eq!(
        via_legacy_json, via_canonical_json,
        "two scheme lookups must resolve to content-equal config",
    );
}

/// Lookup via canonical hash on a non-existent row returns the
/// `ConfigSnapshotNotFound` error (and not, say, a generic SQL error).
#[tokio::test]
async fn get_config_snapshot_via_canonical_returns_not_found() {
    let postgres = get_test_postgres().await;
    // Construct a canonical-scheme hash from random bytes that won't
    // appear in the DB.
    let synthetic = SnapshotHash::from_canonical_bytes(&[0xFFu8; 32]);
    let result = postgres.get_config_snapshot(synthetic).await;
    let err = result.expect_err("non-existent canonical hash must error");
    assert!(matches!(
        err.get_details(),
        tensorzero_core::error::ErrorDetails::ConfigSnapshotNotFound { .. }
    ));
}

/// `SnapshotHash` round-trips through Postgres and serde-JSON: a
/// canonical-scheme hash, serialized to JSON (bare decimal — matches
/// the DB column form), deserialized, and used for a DB lookup,
/// resolves the same row. The scheme tag is metadata-only: bytes are
/// the lookup key, and `Display` (not `Serialize`) is what carries the
/// `v2:` prefix on the URL/log surface.
#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn canonical_hash_serde_round_trip_through_db() {
    let postgres = get_test_postgres().await;
    let id = Uuid::now_v7();
    let config_toml = format!(
        r#"
[models.rt_{id}]
routing = ["dummy"]

[models.rt_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.rt_fn_{id}]
type = "chat"

[functions.rt_fn_{id}.variants.only]
type = "chat_completion"
model = "rt_{id}"
"#
    );
    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new()).expect("parse");
    postgres.write_config_snapshot(&snapshot).await.unwrap();

    let canonical = snapshot.config.canonical_hash().expect("canonical");

    // Wire form: canonical hashes carry the `can:` prefix on the wire
    // (Display/Serialize). Legacy hashes are bare decimal. DB-row
    // structs that hit numeric columns (`UInt256` / `NUMERIC(78,0)`)
    // opt out via the `serialize_optional_hash_bare_decimal` helper —
    // tested separately at the type level.
    let wire = serde_json::to_string(&canonical).expect("serialize");
    assert!(
        wire.contains("can:"),
        "canonical wire form must carry the `can:` prefix; got: {wire}"
    );

    // The display form carries the same prefix for URL routing.
    let displayed = canonical.to_string();
    assert!(
        displayed.starts_with("can:"),
        "Display must carry the `can:` prefix; got: {displayed}",
    );

    // Round-trip via the wire form recovers the scheme tag and the
    // bytes — `from_str` of `can:DECIMAL` re-derives `Canonical`.
    let parsed_back: SnapshotHash = wire.trim_matches('"').parse().expect("FromStr can:DECIMAL");
    assert_eq!(parsed_back.scheme(), SnapshotHashScheme::Canonical);

    let retrieved = postgres
        .get_config_snapshot(parsed_back)
        .await
        .expect("lookup after round-trip");
    let original_json = serde_json::to_value(&snapshot.config).unwrap();
    let retrieved_json = serde_json::to_value(&retrieved.config).unwrap();
    assert_eq!(original_json, retrieved_json);
}

/// Migration safety net: legacy rows that predate the `config_jsonb`
/// column (or whose backfill skipped them due to a TOML parse error)
/// still load via the deprecated TOML column. This is the only path that
/// reads `config TEXT`; it's expected to disappear when the TOML column
/// is dropped post-`hash_v2`.
#[tokio::test]
async fn get_config_snapshot_falls_back_to_toml_for_legacy_null_rows() {
    let postgres = get_test_postgres().await;
    let id = Uuid::now_v7();
    let config_toml = format!(
        r#"
[models.l_{id}]
routing = ["dummy"]

[models.l_{id}.providers.dummy]
type = "dummy"
model_name = "test"

[functions.l_fn_{id}]
type = "chat"

[functions.l_fn_{id}.variants.only]
type = "chat_completion"
model = "l_{id}"
"#
    );
    let snapshot =
        ConfigSnapshot::new_from_toml_string(&config_toml, HashMap::new()).expect("parse");
    let hash = snapshot.hash.clone();

    // Insert as a pre-backfill legacy row: TOML present, JSONB NULL.
    // This is exactly the shape of rows written by gateways before the
    // `config_jsonb` column existed.
    insert_legacy_snapshot_row(&postgres, &snapshot).await;

    // Read via the public API. Should succeed via the TOML fallback.
    let retrieved = postgres.get_config_snapshot(hash.clone()).await.unwrap();
    assert_eq!(
        retrieved.hash.to_hex_string(),
        hash.to_hex_string(),
        "fallback path must preserve the original hash",
    );
    let original_json = serde_json::to_value(&snapshot.config).unwrap();
    let retrieved_json = serde_json::to_value(&retrieved.config).unwrap();
    assert_eq!(
        retrieved_json, original_json,
        "fallback path must return content-equal config",
    );
}
