use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    ClientBuilder, ClientInferenceParams, InferenceOutput, Input, InputMessage,
    InputMessageContent, Role,
};
use tensorzero_core::config::snapshot::{ConfigSnapshot, SnapshotHash};
use tensorzero_core::config::{Config, ConfigFileGlob, write_config_snapshot};
use tensorzero_core::db::ConfigQueries;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::clickhouse::migration_manager::{self, RunMigrationManagerArgs};
use tensorzero_core::db::clickhouse::test_helpers::{
    CLICKHOUSE_URL, get_clickhouse, select_chat_inference_clickhouse,
};
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::error::ErrorDetails;
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::inference::types::Text;
use uuid::Uuid;

#[tokio::test]
async fn test_embedded_invalid_glob() {
    let err = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some("/invalid/tensorzero-e2e/glob/**/*.toml".into()),
        clickhouse_url: None,
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap_err();

    assert_eq!(
        err.to_string(),
        "Failed to parse config: Internal TensorZero Error: Error using glob: `/invalid/tensorzero-e2e/glob/**/*.toml`: No files matched the glob pattern. Ensure that the path exists, and contains at least one file."
    );
}

#[tokio::test]
async fn test_embedded_duplicate_key() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_a_path = temp_dir.path().join("config_a.toml");
    let config_b_path = temp_dir.path().join("config_b.toml");
    std::fs::write(
        &config_a_path,
        r#"
        [functions.first]
        type = "chat"
        "#,
    )
    .unwrap();
    std::fs::write(
        &config_b_path,
        r#"
        functions.first.type = "json"
        "#,
    )
    .unwrap();

    let glob = temp_dir.path().join("*.toml").to_owned();

    let err = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(glob.clone()),
        clickhouse_url: None,
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap_err();

    let config_a_path = config_a_path.display();
    let config_b_path = config_b_path.display();
    let glob = glob.display();

    assert_eq!(
        err.to_string(),
        format!(
            "Failed to parse config: Internal TensorZero Error: `functions.first.type`: Found duplicate values in globbed TOML config files `{config_a_path}` and `{config_b_path}`. Config file glob `{glob}` resolved to the following files:\n{config_a_path}\n{config_b_path}"
        )
    );
}

#[tokio::test]
async fn test_from_components_basic() {
    // Create a minimal config
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        r"
        [gateway]
        # Auth is disabled by default
        ",
    )
    .unwrap();

    // Load config
    let config = Arc::new(
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new(config_path.to_string_lossy().to_string()).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests(),
    );
    // Create components
    let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
    let postgres_connection_info = PostgresConnectionInfo::Disabled;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();

    // Build client using FromComponents mode
    let client = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::FromComponents {
        config,
        clickhouse_connection_info,
        postgres_connection_info,
        http_client,
        timeout: Some(Duration::from_secs(60)),
    })
    .build()
    .await
    .expect("Failed to build client from components");

    // Verify client was created successfully (basic smoke test)
    // The fact that it built without error is the main validation
    assert!(!client.verbose_errors); // Default value
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_config_snapshot() {
    // Get a clean ClickHouse instance with automatic cleanup
    let clickhouse = get_clickhouse().await;

    // Run migrations to set up the ConfigSnapshot table
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();
    let random_id = Uuid::now_v7();

    // Create a test config snapshot with minimal valid config
    // Using a unique metric name to avoid conflicts between test runs
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

    // Write the config snapshot
    write_config_snapshot(&clickhouse, snapshot).await.unwrap();

    // Wait a bit for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Query the ConfigSnapshot table to verify the data was written
    let query = format!(
        "SELECT config, tensorzero_version, hash, created_at, last_used FROM ConfigSnapshot FINAL WHERE hash = toUInt256('{hash_number}') FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query.clone())
        .await
        .unwrap();

    println!("response: {}", &response.response);
    // Parse and verify the result
    let snapshot_row: serde_json::Value = serde_json::from_str(&response.response).unwrap();

    // Config is serialized from StoredConfig, so format may differ from original.
    // Just verify it contains our metric definition.
    let stored_config = snapshot_row["config"].as_str().unwrap();
    assert!(
        stored_config.contains(&format!("test_metric_{random_id}")),
        "Config should contain our test metric"
    );
    assert!(
        !snapshot_row["tensorzero_version"]
            .as_str()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        snapshot_row["hash"].as_str().unwrap().to_lowercase(),
        hash_number
    );

    let created_at = snapshot_row["created_at"].as_str().unwrap();
    let last_used_1 = snapshot_row["last_used"].as_str().unwrap();

    // Test upsert behavior: write the same config again
    let snapshot2 =
        ConfigSnapshot::new_from_toml_string(&config_toml, extra_templates.clone()).unwrap();

    write_config_snapshot(&clickhouse, snapshot2).await.unwrap();

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Query again to verify upsert behavior
    let response2 = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();

    let snapshot_row2: serde_json::Value = serde_json::from_str(&response2.response).unwrap();

    // Verify created_at is preserved
    assert_eq!(
        snapshot_row2["created_at"].as_str().unwrap(),
        created_at,
        "created_at should be preserved on upsert"
    );

    // Verify last_used is updated (should be different from the first insert)
    let last_used_2 = snapshot_row2["last_used"].as_str().unwrap();
    assert!(
        last_used_2 >= last_used_1,
        "last_used should be updated on upsert"
    );

    // Verify the data is still correct (config contains our metric)
    let stored_config2 = snapshot_row2["config"].as_str().unwrap();
    assert!(
        stored_config2.contains(&format!("test_metric_{random_id}")),
        "Config should still contain our test metric after upsert"
    );
    assert_eq!(
        snapshot_row2["hash"].as_str().unwrap().to_lowercase(),
        hash_number
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_config_snapshot_success() {
    // Get a clean ClickHouse instance
    let clickhouse = get_clickhouse().await;

    // Run migrations to set up the ConfigSnapshot table
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    let random_id = Uuid::now_v7();

    // Create a test config snapshot
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

    // Write the config snapshot
    write_config_snapshot(&clickhouse, snapshot).await.unwrap();

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read the config snapshot using get_config_snapshot
    let retrieved_snapshot = clickhouse.get_config_snapshot(hash).await.unwrap();

    // Verify the retrieved snapshot matches what we wrote
    // Compare by serializing to TOML and checking it contains our model definition
    let serialized_config = toml::to_string(&retrieved_snapshot.config).unwrap();
    assert!(
        serialized_config.contains(&format!("test_metric_{random_id}")),
        "Config should contain our test metric"
    );
    assert_eq!(retrieved_snapshot.extra_templates, extra_templates);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_config_snapshot_not_found() {
    // Get a clean ClickHouse instance
    let clickhouse = get_clickhouse().await;

    // Run migrations to set up the ConfigSnapshot table
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    // Create a test hash that doesn't exist in the database
    let nonexistent_hash = SnapshotHash::new_test();

    // Try to get a config snapshot that doesn't exist
    let result = clickhouse.get_config_snapshot(nonexistent_hash).await;

    // Verify we get a ConfigSnapshotNotFound error
    let err = result.unwrap_err();
    assert!(
        matches!(
            err.get_details(),
            ErrorDetails::ConfigSnapshotNotFound { .. }
        ),
        "Expected ConfigSnapshotNotFound error, got: {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_config_snapshot_with_extra_templates() {
    // Get a clean ClickHouse instance
    let clickhouse = get_clickhouse().await;

    // Run migrations to set up the ConfigSnapshot table
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    let random_id = Uuid::now_v7();

    // Create a config snapshot with multiple extra templates
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

    // Write the config snapshot
    write_config_snapshot(&clickhouse, snapshot).await.unwrap();

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read the config snapshot
    let retrieved_snapshot = clickhouse.get_config_snapshot(hash).await.unwrap();

    // Verify all extra templates are correctly stored and retrieved
    // Compare by serializing to TOML and checking it contains our model definition
    let serialized_config = toml::to_string(&retrieved_snapshot.config).unwrap();
    assert!(
        serialized_config.contains(&format!("test_metric_{random_id}")),
        "Config should contain our test metric"
    );
    assert_eq!(retrieved_snapshot.extra_templates.len(), 3);
    assert_eq!(
        retrieved_snapshot.extra_templates.get("system_template"),
        Some(&"You are a helpful assistant.".to_string())
    );
    assert_eq!(
        retrieved_snapshot.extra_templates.get("user_template"),
        Some(&"User said: {{message}}".to_string())
    );
    assert_eq!(
        retrieved_snapshot.extra_templates.get("assistant_template"),
        Some(&"Assistant responds: {{response}}".to_string())
    );
}

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
    // Create a unique config with randomness.
    // NOTE: This config does NOT explicitly set runtime-overlaid gateway fields like
    // `fetch_and_encode_input_files_before_inference` or `global_outbound_http_timeout_ms`.
    // This means the hash will change after rehydration (see docstring above).
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

    // Build first client using make_embedded_gateway_with_config
    // This automatically writes the snapshot to ClickHouse
    let client = make_embedded_gateway_with_config(&config).await;

    // Perform first inference
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

    // Drop the first client
    drop(client);

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Get snapshot hash from ChatInference table
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

    // Load snapshot from ClickHouse
    let retrieved_snapshot = clickhouse
        .get_config_snapshot(snapshot_hash.clone())
        .await
        .unwrap();

    // Build new client from snapshot with a default live config for runtime settings.
    // The RuntimeOverlay will add explicit values for gateway fields that were None
    // in the original config, causing a different hash.
    let live_config = Config::default();
    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        &live_config,
        Some(CLICKHOUSE_URL.clone()),
        None,  // No Postgres
        false, // Don't verify credentials
        Some(Duration::from_secs(60)),
    ))
    .await
    .unwrap();

    // Perform second inference with new client
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

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify second inference completed and has a snapshot hash
    let inference_row2 = select_chat_inference_clickhouse(&clickhouse, inference_id_2)
        .await
        .unwrap();
    let stored_hash2 = inference_row2
        .get("snapshot_hash")
        .unwrap()
        .as_str()
        .unwrap();

    // The hashes are DIFFERENT because the original config didn't explicitly set
    // runtime-overlaid gateway fields. When rehydrated, RuntimeOverlay adds these
    // fields with explicit default values, changing the serialized TOML and thus the hash.
    //
    // This is expected behavior. For hash stability, see
    // `test_config_snapshot_hash_stable_with_explicit_runtime_fields`.
    assert_ne!(
        snapshot_hash_str, stored_hash2,
        "Hashes should be DIFFERENT because runtime-overlaid fields were not explicitly set. \
         Original: {snapshot_hash_str}, Rehydrated: {stored_hash2}",
    );

    // Both hashes should be valid non-empty strings
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
/// # Background
///
/// When a config is loaded from a snapshot, `RuntimeOverlay::from_config()` takes a live
/// `Config` (where defaults have been applied) and copies those values back to
/// `UninitializedGatewayConfig` with explicit `Some(value)`. This means:
///
/// - **Original config without explicit fields**: Fields like `fetch_and_encode_input_files_before_inference`
///   and `global_outbound_http_timeout_ms` are `None` in `UninitializedGatewayConfig`, so they
///   serialize to TOML without those keys.
///
/// - **Rehydrated config with RuntimeOverlay**: These same fields become `Some(default_value)`,
///   so they serialize to TOML WITH those keys explicitly present.
///
/// This difference in serialization causes different hashes, which is **expected behavior**
/// when the original config doesn't explicitly set runtime fields.
///
/// # What This Test Validates
///
/// This test confirms that when all runtime-overlaid gateway fields ARE explicitly set to
/// their default values in the original config, the hash remains stable through:
/// 1. Fresh config load → inference → snapshot stored
/// 2. Snapshot load with RuntimeOverlay → inference → hash comparison
///
/// The fields that must be explicitly set for hash stability are:
/// - `gateway.fetch_and_encode_input_files_before_inference = false`
/// - `gateway.global_outbound_http_timeout_ms = 300000` (5 minutes in ms)
/// - `gateway.template_filesystem_access.enabled = false` (and no base_path)
///
/// # Why This Matters
///
/// This test documents and validates the expected behavior: configs that don't explicitly
/// set runtime fields will have different hashes when rehydrated. Users who want hash
/// stability across rehydration must explicitly set these fields.
#[tokio::test(flavor = "multi_thread")]
async fn test_config_snapshot_hash_stable_with_explicit_runtime_fields() {
    // Create a unique config with randomness, explicitly setting ALL runtime-overlaid fields
    // to their default values. This ensures the hash will be stable through rehydration.
    let random_id = Uuid::now_v7();
    let config = format!(
        r#"
# Gateway configuration with ALL runtime-overlaid fields explicitly set to defaults.
# This is necessary for hash stability when rehydrating from a snapshot.
[gateway]
# Default: false - whether to fetch and encode input files before inference
fetch_and_encode_input_files_before_inference = false
# Default: 300000 (5 minutes) - global HTTP timeout in milliseconds
global_outbound_http_timeout_ms = 300000

# Template filesystem access must be explicitly configured (defaults to disabled)
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

    // Build first client using make_embedded_gateway_with_config
    // This automatically writes the snapshot to ClickHouse
    let client = make_embedded_gateway_with_config(&config).await;

    // Perform first inference
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

    // Drop the first client
    drop(client);

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Get snapshot hash from ChatInference table
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

    // Load snapshot from ClickHouse
    let retrieved_snapshot = clickhouse
        .get_config_snapshot(snapshot_hash.clone())
        .await
        .unwrap();

    // Build new client from snapshot with a default live config for runtime settings.
    // Because the original config explicitly set all runtime fields to their defaults,
    // the RuntimeOverlay should produce identical serialization.
    let live_config = Config::default();
    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        &live_config,
        Some(CLICKHOUSE_URL.clone()),
        None,  // No Postgres
        false, // Don't verify credentials
        Some(Duration::from_secs(60)),
    ))
    .await
    .unwrap();

    // Perform second inference with new client
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

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify both inferences have the same snapshot hash
    let inference_row2 = select_chat_inference_clickhouse(&clickhouse, inference_id_2)
        .await
        .unwrap();
    let stored_hash2 = inference_row2
        .get("snapshot_hash")
        .unwrap()
        .as_str()
        .unwrap();

    // Both inferences should have the same snapshot hash because all runtime-overlaid
    // fields were explicitly set in the original config
    assert_eq!(
        snapshot_hash_str, stored_hash2,
        "Hash mismatch! Original: {snapshot_hash_str}, Rehydrated: {stored_hash2}. \
         This indicates that RuntimeOverlay is adding fields not present in the original config. \
         Ensure all runtime-overlaid gateway fields are explicitly set.",
    );
}

/// Test that fresh configs REJECT the deprecated timeouts field for embedding models.
/// This ensures users are forced to migrate to the new `timeout_ms` field.
#[tokio::test]
async fn test_fresh_config_rejects_deprecated_embedding_timeouts() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("config.toml");

    // Config with deprecated `timeouts` field
    std::fs::write(
        &config_path,
        r#"
[embedding_models.test_model]
routing = ["provider"]
timeouts.non_streaming.total_ms = 5000

[embedding_models.test_model.providers.provider]
type = "dummy"
model_name = "test"
"#,
    )
    .unwrap();

    let result = Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new(config_path.to_string_lossy().to_string()).unwrap(),
        false,
    )
    .await;

    // Should fail with "unknown field" error for the deprecated timeouts field
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("unknown field"),
        "Expected 'unknown field' error for deprecated timeouts, got: {err}"
    );
}

/// Test that built-in functions are included in config snapshots.
/// This verifies that when a config is loaded and written to ClickHouse,
/// built-in functions are serialized into the snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn test_config_snapshot_includes_built_in_functions() {
    let clickhouse = get_clickhouse().await;

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

    // Write snapshot to ClickHouse and get the config with its hash
    let config = loaded.into_config(&clickhouse).await.unwrap();

    // Wait for data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Query the snapshot from ClickHouse using config.hash
    let hash = &config.hash;
    let query = format!(
        "SELECT config FROM ConfigSnapshot FINAL WHERE hash = toUInt256('{hash}') FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let row: serde_json::Value = serde_json::from_str(&response.response).unwrap();
    let stored_config = row["config"].as_str().unwrap();

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

/// Test that from_config_snapshot correctly overlays runtime config from live_config
#[tokio::test(flavor = "multi_thread")]
async fn test_from_config_snapshot_overlays_runtime_config() {
    use tensorzero::ClientExt;

    // Create a unique config with randomness
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

    // Build first client - this writes the snapshot to ClickHouse
    let client = make_embedded_gateway_with_config(&config).await;

    // Verify the original config has debug = false
    let original_config = client.get_config().unwrap();
    assert!(
        !original_config.gateway.debug,
        "Original config should have debug = false"
    );

    // Perform an inference to ensure snapshot is written
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

    // Get snapshot hash from ChatInference table
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

    // Load snapshot from ClickHouse
    let retrieved_snapshot = clickhouse
        .get_config_snapshot(snapshot_hash.clone())
        .await
        .unwrap();

    // Create a live config with DIFFERENT runtime values
    // Specifically, set debug = true (different from snapshot's debug = false)
    let mut live_config = Config::default();
    live_config.gateway.debug = true;
    live_config.postgres.connection_pool_size = 99; // Different from default of 20

    // Build new client from snapshot with our modified live config
    let new_client = Box::pin(ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        &live_config,
        Some(CLICKHOUSE_URL.clone()),
        None,
        false,
        Some(Duration::from_secs(60)),
    ))
    .await
    .unwrap();

    // Verify that the new client's config has the LIVE config's runtime values,
    // not the snapshot's values
    let new_config = new_client.get_config().unwrap();

    // Gateway should come from live_config (debug = true), not snapshot (debug = false)
    assert!(
        new_config.gateway.debug,
        "Gateway should be overlaid from live_config (debug = true)"
    );

    // Postgres should come from live_config (connection_pool_size = 99)
    assert_eq!(
        new_config.postgres.connection_pool_size, 99,
        "Postgres should be overlaid from live_config"
    );

    // But the function should still come from the snapshot
    assert!(
        new_config
            .functions
            .contains_key(&format!("basic_test_{random_id}")),
        "Function from snapshot should still be present"
    );
}
