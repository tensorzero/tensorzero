#![expect(clippy::expect_used, clippy::unwrap_used, clippy::missing_panics_doc)]

use std::collections::HashMap;

use crate::{Client, ClientBuilder, ClientBuilderMode, PostgresConfig};
use tempfile::NamedTempFile;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use url::Url;
use uuid::Uuid;

// Re-export e2e test helpers from tensorzero-core
pub use tensorzero_core::test_helpers::{get_e2e_config, get_e2e_config_path};

pub async fn make_http_gateway() -> Client {
    ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: Url::parse(&get_gateway_endpoint("/")).unwrap(),
    })
    .build()
    .await
    .unwrap()
}

pub async fn make_embedded_gateway() -> Client {
    let config_path = get_e2e_config_path();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

pub async fn make_embedded_gateway_no_config() -> Client {
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: None,
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

pub async fn make_embedded_gateway_with_config(config: &str) -> Client {
    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

pub async fn make_embedded_gateway_with_config_and_postgres(config: &str) -> Client {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for tests that require Postgres");

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: Some(PostgresConfig::Url(postgres_url)),
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

/// Creates an embedded gateway with rate limiting backend support.
/// Reads both TENSORZERO_POSTGRES_URL and TENSORZERO_VALKEY_URL from env vars.
/// The rate limiting backend selection is determined by the config's `[rate_limiting].backend` field:
/// - `auto` (default): Valkey if available, otherwise Postgres
/// - `postgres`: Force Postgres backend
/// - `valkey`: Force Valkey backend
pub async fn make_embedded_gateway_with_rate_limiting(config: &str) -> Client {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL").ok();
    let valkey_url = std::env::var("TENSORZERO_VALKEY_URL").ok();

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: postgres_url.map(PostgresConfig::Url),
        valkey_url,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

// We use a multi-threaded runtime so that the embedded gateway can use 'block_on'.
// For consistency, we also use a multi-threaded runtime for the http gateway test.

#[macro_export]
macro_rules! make_gateway_test_functions {
    ($prefix:ident) => {
        paste::paste! {

            #[tokio::test(flavor = "multi_thread")]
            async fn [<$prefix _embedded_gateway>]() {
                $prefix (tensorzero::test_helpers::make_embedded_gateway().await).await;
            }


            #[tokio::test(flavor = "multi_thread")]
            async fn [<$prefix _http_gateway>]() {
                $prefix (tensorzero::test_helpers::make_http_gateway().await).await;
            }
        }
    };
}

pub fn get_gateway_endpoint(path: &str) -> String {
    let gateway_host =
        std::env::var("TENSORZERO_GATEWAY_HOST").unwrap_or_else(|_| "localhost".to_string());
    let gateway_port =
        std::env::var("TENSORZERO_GATEWAY_PORT").unwrap_or_else(|_| "3000".to_string());
    format!("http://{gateway_host}:{gateway_port}{path}")
}

pub async fn get_metrics(client: &reqwest::Client, url: &str) -> HashMap<String, String> {
    let response = client.get(url).send().await.unwrap().text().await.unwrap();
    let metrics: HashMap<String, String> = response
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            // Split on the last space, since the metric name may itself have spaces
            let mut parts = line.rsplitn(2, ' ');
            match (parts.next(), parts.next()) {
                (Some(value), Some(key)) => Some((key.to_string(), value.to_string())),
                _ => None,
            }
        })
        .collect();

    metrics
}

/// Creates a ClickHouse URL with a unique database name.
/// The migration manager will create the database on gateway startup.
pub fn create_unique_clickhouse_url(prefix: &str) -> String {
    let mut url = Url::parse(&CLICKHOUSE_URL).unwrap();
    let db_name = format!("{}_{}", prefix, Uuid::now_v7().simple());
    url.set_path(&db_name);
    url.to_string()
}

/// Creates an embedded gateway with a unique ClickHouse database.
/// This provides test isolation - each test gets its own database with fresh migrations.
pub async fn make_embedded_gateway_with_unique_db(config: &str, db_prefix: &str) -> Client {
    let clickhouse_url = create_unique_clickhouse_url(db_prefix);

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(clickhouse_url),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

/// Creates an embedded gateway using the e2e config with a unique ClickHouse database.
/// This provides test isolation while using the full e2e test configuration.
pub async fn make_embedded_gateway_e2e_with_unique_db(db_prefix: &str) -> Client {
    let clickhouse_url = create_unique_clickhouse_url(db_prefix);
    let config_path = get_e2e_config_path();

    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path),
        clickhouse_url: Some(clickhouse_url),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

/// Starts an HTTP gateway with a unique ClickHouse database.
/// Returns the base URL (e.g., "http://127.0.0.1:12345") and a shutdown handle.
/// The gateway shuts down when the handle is dropped.
pub async fn make_http_gateway_with_unique_db(
    db_prefix: &str,
) -> (String, tensorzero_core::utils::gateway::ShutdownHandle) {
    let clickhouse_url = create_unique_clickhouse_url(db_prefix);
    let config_path = get_e2e_config_path();

    let (addr, shutdown_handle) = tensorzero_core::utils::gateway::start_openai_compatible_gateway(
        Some(config_path.to_string_lossy().to_string()),
        Some(clickhouse_url),
        None, // postgres_url
        None, // valkey_url
    )
    .await
    .unwrap();

    (format!("http://{addr}"), shutdown_handle)
}
