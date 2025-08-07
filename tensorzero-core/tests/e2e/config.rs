#[tokio::test]
async fn test_embedded_invalid_glob() {
    let err = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some("/invalid/tensorzero-e2e/glob/**/*.toml".into()),
        clickhouse_url: None,
        timeout: None,
        verify_credentials: true,
    })
    .build()
    .await
    .unwrap_err();

    assert_eq!(
        err.to_string(),
        "Failed to parse config: Internal TensorZero Error: Error using glob: `/invalid/tensorzero-e2e/glob/**/*.toml`: No files matched the glob pattern. Ensure that the path exists, and contains at least one file."
    );
}
