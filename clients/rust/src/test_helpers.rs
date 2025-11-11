#![expect(clippy::expect_used, clippy::unwrap_used, clippy::missing_panics_doc)]

use crate::{Client, ClientBuilder, ClientBuilderMode};
use tempfile::NamedTempFile;
use tensorzero_core::config::{Config, ConfigFileGlob};
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use url::Url;

pub async fn make_http_gateway() -> Client {
    ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: Url::parse(&get_gateway_endpoint("/")).unwrap(),
    })
    .build()
    .await
    .unwrap()
}

pub fn get_e2e_config_path() -> std::path::PathBuf {
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("../../tensorzero-core/tests/e2e/tensorzero.toml");
    config_path
}

pub async fn get_e2e_config() -> Config {
    let config_path = get_e2e_config_path();
    let config_glob = ConfigFileGlob::new_from_path(&config_path).unwrap();
    Config::load_from_path_optional_verify_credentials(&config_glob, false)
        .await
        .unwrap()
}

pub async fn make_embedded_gateway() -> Client {
    let config_path = get_e2e_config_path();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None,
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
        postgres_url: None,
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
        postgres_url: None,
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
        .expect("TENSORZERO_POSTGRES_URL must be set for rate limiting tests");

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: Some(postgres_url),
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

#[cfg(any(test, feature = "e2e_tests"))]
pub fn get_gateway_endpoint(path: &str) -> String {
    let gateway_host =
        std::env::var("TENSORZERO_GATEWAY_HOST").unwrap_or_else(|_| "localhost".to_string());
    let gateway_port =
        std::env::var("TENSORZERO_GATEWAY_PORT").unwrap_or_else(|_| "3000".to_string());
    format!("http://{gateway_host}:{gateway_port}{path}")
}
