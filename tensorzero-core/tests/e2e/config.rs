use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::test_helpers::make_embedded_gateway_with_config;
use tensorzero::{
    ClientBuilder, ClientInferenceParams, ClientInput, ClientInputMessage,
    ClientInputMessageContent, InferenceOutput, Role,
};
use tensorzero_core::config::snapshot::{ConfigSnapshot, SnapshotHash};
use tensorzero_core::config::{write_config_snapshot, Config, ConfigFileGlob};
use tensorzero_core::db::clickhouse::migration_manager::{self, RunMigrationManagerArgs};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, CLICKHOUSE_URL,
};
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::ConfigQueries;
use tensorzero_core::error::ErrorDetails;
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::inference::types::TextKind;
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
        .dangerous_into_config_without_writing(),
    );
    let snapshot_hash = SnapshotHash::new_test();

    // Create components
    let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
    let postgres_connection_info = PostgresConnectionInfo::Disabled;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();

    // Build client using FromComponents mode
    let client = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::FromComponents {
        config,
        snapshot_hash,
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

    // Create a test config snapshot
    let config_toml = format!(
        r#"
[gateway]
bind = "0.0.0.0:3000"

[models.test_model{random_id}]
routing = ["test_provider::gpt-4"]
"#
    );

    let mut extra_templates = HashMap::new();
    extra_templates.insert("test_template".to_string(), "Hello {{name}}!".to_string());

    let snapshot = ConfigSnapshot {
        config: config_toml.to_string(),
        extra_templates: extra_templates.clone(),
    };

    let hash = snapshot.hash();
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

    assert_eq!(snapshot_row["config"].as_str().unwrap(), config_toml);
    assert!(!snapshot_row["tensorzero_version"]
        .as_str()
        .unwrap()
        .is_empty());
    assert_eq!(
        snapshot_row["hash"].as_str().unwrap().to_lowercase(),
        hash_number
    );

    let created_at = snapshot_row["created_at"].as_str().unwrap();
    let last_used_1 = snapshot_row["last_used"].as_str().unwrap();

    // Test upsert behavior: write the same config again
    let snapshot2 = ConfigSnapshot {
        config: config_toml.to_string(),
        extra_templates: extra_templates.clone(),
    };

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

    // Verify the data is still correct
    assert_eq!(snapshot_row2["config"].as_str().unwrap(), config_toml);
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
[gateway]
bind = "0.0.0.0:3000"

[models.test_model_{random_id}]
routing = ["test_provider::gpt-4"]
"#
    );

    let mut extra_templates = HashMap::new();
    extra_templates.insert("test_template".to_string(), "Hello {{name}}!".to_string());

    let snapshot = ConfigSnapshot {
        config: config_toml.clone(),
        extra_templates: extra_templates.clone(),
    };

    let hash = snapshot.hash();

    // Write the config snapshot
    write_config_snapshot(&clickhouse, snapshot).await.unwrap();

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read the config snapshot using get_config_snapshot
    let retrieved_snapshot = clickhouse.get_config_snapshot(hash).await.unwrap();

    // Verify the retrieved snapshot matches what we wrote
    assert_eq!(retrieved_snapshot.config, config_toml);
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
[gateway]
bind = "0.0.0.0:3000"

[models.test_model_{random_id}]
routing = ["test_provider::gpt-4"]
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

    let snapshot = ConfigSnapshot {
        config: config_toml.clone(),
        extra_templates: extra_templates.clone(),
    };

    let hash = snapshot.hash();

    // Write the config snapshot
    write_config_snapshot(&clickhouse, snapshot).await.unwrap();

    // Wait for the data to be committed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Read the config snapshot
    let retrieved_snapshot = clickhouse.get_config_snapshot(hash).await.unwrap();

    // Verify all extra templates are correctly stored and retrieved
    assert_eq!(retrieved_snapshot.config, config_toml);
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

/// Test that we can:
/// 1. Build a client from a config
/// 2. Do an inference
/// 3. Drop the client
/// 4. Load the snapshot from ClickHouse
/// 5. Build a new client from the snapshot
/// 6. Do another inference
/// 7. Assert both inferences have the same snapshot hash
#[tokio::test(flavor = "multi_thread")]
async fn test_config_snapshot_inference_roundtrip() {
    // Create a unique config with randomness
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
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
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
    let snapshot_hash = SnapshotHash::from_str(snapshot_hash_str).unwrap();

    // Load snapshot from ClickHouse
    let retrieved_snapshot = clickhouse
        .get_config_snapshot(snapshot_hash.clone())
        .await
        .unwrap();

    // Build new client from snapshot
    let new_client = ClientBuilder::from_config_snapshot(
        retrieved_snapshot,
        Some(CLICKHOUSE_URL.clone()),
        None,  // No Postgres
        false, // Don't verify credentials
        Some(Duration::from_secs(60)),
    )
    .await
    .unwrap();

    // Perform second inference with new client
    let params2 = ClientInferenceParams {
        function_name: Some(format!("basic_test_{random_id}")),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
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

    // Both inferences should have the same snapshot hash
    assert_eq!(snapshot_hash_str, stored_hash2);
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

/// Test that fresh configs REJECT the deprecated timeouts field for embedding providers.
#[tokio::test]
async fn test_fresh_config_rejects_deprecated_embedding_provider_timeouts() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("config.toml");

    // Config with deprecated `timeouts` field on provider
    std::fs::write(
        &config_path,
        r#"
[embedding_models.test_model]
routing = ["provider"]

[embedding_models.test_model.providers.provider]
type = "dummy"
model_name = "test"
timeouts.non_streaming.total_ms = 5000
"#,
    )
    .unwrap();

    let result = Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new(config_path.to_string_lossy().to_string()).unwrap(),
        false,
    )
    .await;

    // Should fail with "unknown field" error
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("unknown field"),
        "Expected 'unknown field' error for deprecated provider timeouts, got: {err}"
    );
}

/// Test that snapshot loading ACCEPTS the deprecated timeouts field for embedding models
/// and correctly migrates it to timeout_ms.
#[tokio::test]
async fn test_snapshot_accepts_deprecated_embedding_timeouts() {
    let config_toml = r#"
[embedding_models.test_model]
routing = ["provider"]
timeouts.non_streaming.total_ms = 5000

[embedding_models.test_model.providers.provider]
type = "dummy"
model_name = "test"
"#;

    let snapshot = ConfigSnapshot {
        config: config_toml.to_string(),
        extra_templates: HashMap::new(),
    };

    // Loading from snapshot should succeed and migrate the timeout
    let config_load_info = Config::load_from_snapshot(snapshot, false).await.unwrap();
    let config = config_load_info.dangerous_into_config_without_writing();

    // Verify the timeout was migrated correctly
    let embedding_model = config
        .embedding_models
        .get("test_model")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        embedding_model.timeout_ms,
        Some(5000),
        "Deprecated timeouts.non_streaming.total_ms should be migrated to timeout_ms"
    );
}

/// Test that snapshot loading ACCEPTS the deprecated timeouts field for embedding providers.
#[tokio::test]
async fn test_snapshot_accepts_deprecated_embedding_provider_timeouts() {
    let config_toml = r#"
[embedding_models.test_model]
routing = ["provider"]

[embedding_models.test_model.providers.provider]
type = "dummy"
model_name = "test"
timeouts.non_streaming.total_ms = 3000
"#;

    let snapshot = ConfigSnapshot {
        config: config_toml.to_string(),
        extra_templates: HashMap::new(),
    };

    // Loading from snapshot should succeed and migrate the timeout
    let config_load_info = Config::load_from_snapshot(snapshot, false).await.unwrap();
    let config = config_load_info.dangerous_into_config_without_writing();

    // Verify the provider timeout was migrated correctly
    let embedding_model = config
        .embedding_models
        .get("test_model")
        .await
        .unwrap()
        .unwrap();
    let provider = embedding_model.providers.get("provider").unwrap();
    assert_eq!(
        provider.timeout_ms,
        Some(3000),
        "Deprecated timeouts.non_streaming.total_ms should be migrated to timeout_ms on provider"
    );
}

/// Test that new timeout_ms field takes precedence over deprecated timeouts field
/// when both are present in a snapshot (edge case for backward compatibility).
#[tokio::test]
async fn test_snapshot_timeout_ms_takes_precedence() {
    let config_toml = r#"
[embedding_models.test_model]
routing = ["provider"]
timeout_ms = 10000
timeouts.non_streaming.total_ms = 5000

[embedding_models.test_model.providers.provider]
type = "dummy"
model_name = "test"
"#;

    let snapshot = ConfigSnapshot {
        config: config_toml.to_string(),
        extra_templates: HashMap::new(),
    };

    // Loading from snapshot should succeed
    let config_load_info = Config::load_from_snapshot(snapshot, false).await.unwrap();
    let config = config_load_info.dangerous_into_config_without_writing();

    // New timeout_ms field should take precedence
    let embedding_model = config
        .embedding_models
        .get("test_model")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        embedding_model.timeout_ms,
        Some(10000),
        "timeout_ms should take precedence over deprecated timeouts"
    );
}
