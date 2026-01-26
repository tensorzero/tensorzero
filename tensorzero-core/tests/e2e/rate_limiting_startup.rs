//! Tests for rate limiting startup validation
//!
//! These tests verify the startup behavior for different rate limiting backend configurations:
//! - Starting with no backend (should fail if rate limiting is enabled)
//! - Starting with Postgres-only
//! - Starting with Valkey-only
//! - Starting with both backends available
//! - Backend selection via config

use tempfile::NamedTempFile;
use tensorzero::ClientBuilder;
use tensorzero::ClientBuilderMode;
use tensorzero::PostgresConfig;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;

/// Helper to get the test Postgres URL from environment
fn postgres_url() -> Option<PostgresConfig> {
    std::env::var("TENSORZERO_POSTGRES_URL")
        .ok()
        .map(PostgresConfig::Url)
}

/// Helper to get the test Valkey URL from environment
fn valkey_url() -> Option<String> {
    std::env::var("TENSORZERO_VALKEY_URL")
        .ok()
        .or_else(|| Some("redis://localhost:6379".to_string()))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_requires_backend() {
    // Configuration with rate limiting but no backend (no Postgres, no Valkey)
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

    // Try to create a gateway without any backend - this should fail
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    // Assert that the gateway failed to start
    assert!(
        result.is_err(),
        "Gateway should fail to start when rate limiting is configured but no backend is available"
    );

    // Check that the error message is helpful
    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("rate limiting") || error_message.contains("Rate limiting"),
        "Error message should mention rate limiting, got: {error_message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_with_valkey_only() {
    // Configuration with rate limiting
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

    // Create a gateway with Valkey only - this should succeed
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: valkey_url(),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_ok(),
        "Gateway should start successfully with Valkey backend, got: {result:?}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_with_postgres_only() {
    // Configuration with rate limiting
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

    // Create a gateway with Postgres only - this should succeed
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: postgres_url(),
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_ok(),
        "Gateway should start successfully with Postgres backend, got: {result:?}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_with_both_backends_auto() {
    // Configuration with rate limiting
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

    // Create a gateway with both backends - this should succeed
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: postgres_url(),
        valkey_url: valkey_url(),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_ok(),
        "Gateway should start successfully with both backends (auto mode), got: {result:?}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_valkey_backend_requires_valkey() {
    // Configuration with rate limiting explicitly requesting Valkey backend
    let config = r#"
[rate_limiting]
enabled = true
backend = "valkey"

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

    // Try to create a gateway with Valkey backend but no Valkey URL - this should fail
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: postgres_url(), // Has Postgres but not Valkey
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_err(),
        "Gateway should fail to start when Valkey backend is configured but Valkey URL is missing"
    );

    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("valkey") || error_message.contains("Valkey"),
        "Error message should mention Valkey, got: {error_message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_postgres_backend_requires_postgres() {
    // Configuration with rate limiting explicitly requesting Postgres backend
    let config = r#"
[rate_limiting]
enabled = true
backend = "postgres"

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

    // Try to create a gateway with Postgres backend but no Postgres URL - this should fail
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: valkey_url(), // Has Valkey but not Postgres
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_err(),
        "Gateway should fail to start when Postgres backend is configured but Postgres URL is missing"
    );

    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("postgres") || error_message.contains("Postgres"),
        "Error message should mention Postgres, got: {error_message}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_no_rate_limiting_no_backends_ok() {
    // Configuration without rate limiting and no backends - should work fine
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

    // Try to create a gateway without backends or rate limiting - this should succeed
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None, // No Postgres URL
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_ok(),
        "Gateway should start successfully without rate limiting and without backends, got: {result:?}",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_disabled_rate_limiting_no_backends_ok() {
    // Configuration with disabled rate limiting and no backends - should work fine
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

    // Try to create a gateway with disabled rate limiting and no backends - this should succeed
    let result = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await;

    assert!(
        result.is_ok(),
        "Gateway should start successfully with disabled rate limiting and without backends, got: {result:?}",
    );
}
