#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
use std::sync::Arc;

pub fn init_tracing_for_tests() {
    let _ = tracing_subscriber::fmt().try_init();
}

use tensorzero_core::client::{Client, ClientBuilder, ClientBuilderMode};
use tensorzero_core::config::Config;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;

// Re-export test helpers from tensorzero-core
pub use tensorzero_core::test_helpers::get_e2e_config_path;

pub async fn get_tensorzero_client() -> Client {
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(get_e2e_config_path()),
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

/// Loads the E2E test configuration wrapped in Arc for use in tests.
pub async fn get_config() -> Arc<Config> {
    Arc::new(tensorzero_core::test_helpers::get_e2e_config().await)
}
