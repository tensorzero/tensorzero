use std::sync::Arc;
use std::time::Duration;
use tensorzero_core::config::{Config, ConfigFileGlob};
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::http::TensorzeroHttpClient;

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
        .config,
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
