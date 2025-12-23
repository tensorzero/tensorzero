//! Tests for rate limiting startup validation

use tempfile::NamedTempFile;
use tensorzero::ClientBuilder;
use tensorzero::ClientBuilderMode;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_requires_postgres() {
    // Configuration with rate limiting but no Postgres
    let config = r#"
[rate_limiting]
enabled = true

[[rate_limiting.rules]]
model_inferences_per_minute = 10
always = true

[models."dummy"]
routing = ["dummy"]

[models."dummy".providers.dummy]
type = "dummy"
model_name = "dummy"

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "dummy"
"#;

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    // Try to create a gateway without Postgres - this should fail
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None, // No Postgres URL
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    // Assert that the gateway failed to start
    assert!(
        result.is_err(),
        "Gateway should fail to start when rate limiting is configured but Postgres is missing"
    );

    // Check that the error message is helpful
    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("Rate limiting") || error_message.contains("rate limiting"),
        "Error message should mention rate limiting, got: {error_message}"
    );
    assert!(
        error_message.contains("PostgreSQL") || error_message.contains("Postgres"),
        "Error message should mention PostgreSQL, got: {error_message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_no_rate_limiting_no_postgres_ok() {
    // Configuration without rate limiting and no Postgres - should work fine
    let config = r#"
[models."dummy"]
routing = ["dummy"]

[models."dummy".providers.dummy]
type = "dummy"
model_name = "dummy"

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "dummy"
"#;

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    // Try to create a gateway without Postgres or rate limiting - this should succeed
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None, // No Postgres URL
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_ok(),
        "Gateway should start successfully without rate limiting and without Postgres, got: {result:?}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_disabled_rate_limiting_no_postgres_ok() {
    // Configuration with disabled rate limiting and no Postgres - should work fine
    let config = r#"
[rate_limiting]
enabled = false

[[rate_limiting.rules]]
model_inferences_per_minute = 10
always = true

[models."dummy"]
routing = ["dummy"]

[models."dummy".providers.dummy]
type = "dummy"
model_name = "dummy"

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "dummy"
"#;

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    // Try to create a gateway with disabled rate limiting and no Postgres - this should succeed
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None, // No Postgres URL
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_ok(),
        "Gateway should start successfully with disabled rate limiting and without Postgres, got: {result:?}",
    );
}
