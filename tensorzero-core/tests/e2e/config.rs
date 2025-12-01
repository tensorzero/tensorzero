use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tensorzero_core::config::snapshot::ConfigSnapshot;
use tensorzero_core::config::{write_config_snapshot, Config, ConfigFileGlob};
use tensorzero_core::db::clickhouse::migration_manager;
use tensorzero_core::db::clickhouse::migration_manager::RunMigrationManagerArgs;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::http::TensorzeroHttpClient;
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
