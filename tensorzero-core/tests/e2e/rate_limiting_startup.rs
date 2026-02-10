//! Tests for rate limiting startup validation
//!
//! These tests verify the startup behavior for different rate limiting backend configurations:
//! - Starting with no backend (should fail if rate limiting is enabled)
//! - Starting with Postgres-only
//! - Starting with Valkey-only
//! - Starting with both backends available
//! - Backend selection via config

use redis::AsyncCommands;
use tempfile::NamedTempFile;
use tensorzero::ClientBuilder;
use tensorzero::ClientBuilderMode;
use tensorzero::PostgresConfig;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use tensorzero_core::db::valkey::ValkeyConnectionInfo;
use uuid::Uuid;

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

/// Test that old rate limit keys (`ratelimit:*`) are migrated to new keys (`tensorzero_ratelimit:*`)
/// when creating a Valkey connection.
#[tokio::test]
async fn test_valkey_migration_old_ratelimit_keys() {
    let valkey_url =
        valkey_url().expect("Valkey tests should have a TENSORZERO_VALKEY_URL env variable");

    // Create a raw Redis client to set up the old key
    let client = redis::Client::open(valkey_url.as_str()).expect("Failed to create Redis client");
    let mut raw_conn = client
        .get_multiplexed_async_connection()
        .await
        .expect("Failed to connect to Valkey");

    // Create a unique test key to avoid interference with other tests
    let test_id = Uuid::now_v7().to_string();
    let old_key = format!("ratelimit:{test_id}");
    let new_key = format!("tensorzero_ratelimit:{test_id}");

    // Set up the old key with some rate limit state
    let _: () = raw_conn
        .hset_multiple(
            &old_key,
            &[("balance", "42"), ("last_refilled", "1234567890")],
        )
        .await
        .expect("Failed to set old key");

    // Set a TTL on the old key
    let _: () = raw_conn
        .expire(&old_key, 3600)
        .await
        .expect("Failed to set TTL on old key");

    // Verify the old key exists and new key doesn't
    let old_exists: bool = raw_conn
        .exists(&old_key)
        .await
        .expect("Failed to check old key existence");
    assert!(old_exists, "Old key should exist before migration");

    let new_exists_before: bool = raw_conn
        .exists(&new_key)
        .await
        .expect("Failed to check new key existence");
    assert!(
        !new_exists_before,
        "New key should not exist before migration"
    );

    // Create the ValkeyConnectionInfo, which triggers the migration
    let _valkey_conn = ValkeyConnectionInfo::new(&valkey_url)
        .await
        .expect("Failed to create ValkeyConnectionInfo");

    // Verify the new key now exists with the same data
    let new_exists_after: bool = raw_conn
        .exists(&new_key)
        .await
        .expect("Failed to check new key existence after migration");
    assert!(new_exists_after, "New key should exist after migration");

    // Verify the data was copied correctly
    let new_data: Vec<(String, String)> = raw_conn
        .hgetall(&new_key)
        .await
        .expect("Failed to get new key data");

    let balance = new_data
        .iter()
        .find(|(k, _)| k == "balance")
        .map(|(_, v)| v.as_str());
    let last_refilled = new_data
        .iter()
        .find(|(k, _)| k == "last_refilled")
        .map(|(_, v)| v.as_str());

    assert_eq!(
        balance,
        Some("42"),
        "Balance should be copied from old key to new key"
    );
    assert_eq!(
        last_refilled,
        Some("1234567890"),
        "last_refilled should be copied from old key to new key"
    );

    // Verify TTL was copied (should be close to 3600, accounting for test execution time)
    let new_ttl: i64 = raw_conn
        .ttl(&new_key)
        .await
        .expect("Failed to get new key TTL");
    assert!(
        new_ttl > 3500 && new_ttl <= 3600,
        "TTL should be copied from old key, got {new_ttl}"
    );
}

/// Test that migration is idempotent - existing new keys are not overwritten
#[tokio::test]
async fn test_valkey_migration_does_not_overwrite_existing_new_keys() {
    let valkey_url =
        valkey_url().expect("Valkey tests should have a TENSORZERO_VALKEY_URL env variable");

    let client = redis::Client::open(valkey_url.as_str()).expect("Failed to create Redis client");
    let mut raw_conn = client
        .get_multiplexed_async_connection()
        .await
        .expect("Failed to connect to Valkey");

    let test_id = Uuid::now_v7().to_string();
    let old_key = format!("ratelimit:{test_id}");
    let new_key = format!("tensorzero_ratelimit:{test_id}");

    // Set up both old and new keys with different values
    let _: () = raw_conn
        .hset_multiple(&old_key, &[("balance", "100"), ("last_refilled", "111")])
        .await
        .expect("Failed to set old key");

    let _: () = raw_conn
        .hset_multiple(&new_key, &[("balance", "50"), ("last_refilled", "222")])
        .await
        .expect("Failed to set new key");

    // Create the ValkeyConnectionInfo, which triggers the migration
    let _valkey_conn = ValkeyConnectionInfo::new(&valkey_url)
        .await
        .expect("Failed to create ValkeyConnectionInfo");

    // Verify the new key still has its original data (not overwritten)
    let new_data: Vec<(String, String)> = raw_conn
        .hgetall(&new_key)
        .await
        .expect("Failed to get new key data");

    let balance = new_data
        .iter()
        .find(|(k, _)| k == "balance")
        .map(|(_, v)| v.as_str());

    assert_eq!(
        balance,
        Some("50"),
        "New key should not be overwritten by migration"
    );
}
