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
