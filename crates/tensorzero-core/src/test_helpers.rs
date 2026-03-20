#![expect(clippy::missing_panics_doc, clippy::unwrap_used)]

use std::path::PathBuf;

use crate::config::{Config, ConfigFileGlob};
use crate::db::delegating_connection::PrimaryDatastore;

/// Returns the glob path for the E2E test configuration files.
///
/// For ClickHouse primary: matches `tensorzero.*.toml` only.
/// For Postgres primary: also includes `postgres.*.toml` which sets `observability.backend = "postgres"`.
///
/// If `TENSORZERO_INTERNAL_TEST_CACHE_BACKEND` is set to `"clickhouse"` or `"valkey"`,
/// the corresponding `cache-{backend}.*.toml` config override is included in the glob.
pub fn get_e2e_config_path_for_datastore(primary: PrimaryDatastore) -> PathBuf {
    let mut config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let cache_prefix = std::env::var("TENSORZERO_INTERNAL_TEST_CACHE_BACKEND")
        .ok()
        .and_then(|v| match v.as_str() {
            "clickhouse" => Some("cache-clickhouse"),
            "valkey" => Some("cache-valkey"),
            _ => None,
        });

    let glob = match (primary, cache_prefix) {
        (PrimaryDatastore::ClickHouse | PrimaryDatastore::Disabled, None) => {
            "tests/e2e/config/tensorzero.*.toml".to_string()
        }
        (PrimaryDatastore::ClickHouse | PrimaryDatastore::Disabled, Some(cache)) => {
            format!("tests/e2e/config/{{tensorzero,{cache}}}.*.toml")
        }
        (PrimaryDatastore::Postgres, None) => {
            "tests/e2e/config/{tensorzero,postgres}.*.toml".to_string()
        }
        (PrimaryDatastore::Postgres, Some(cache)) => {
            format!("tests/e2e/config/{{tensorzero,postgres,{cache}}}.*.toml")
        }
    };

    config_path.push(glob);
    config_path
}

/// Returns the glob path for the E2E test configuration files,
/// automatically selecting the primary datastore from `TENSORZERO_INTERNAL_TEST_OBSERVABILITY_BACKEND` env var.
pub fn get_e2e_config_path() -> PathBuf {
    get_e2e_config_path_for_datastore(PrimaryDatastore::from_test_env())
}

/// Loads the E2E test configuration.
/// This function loads the configuration without verifying credentials,
/// which is useful for tests that don't make actual API calls.
pub async fn get_e2e_config() -> Config {
    let config_path = get_e2e_config_path();
    let config_glob = ConfigFileGlob::new_from_path(&config_path).unwrap();
    Config::load_from_path_optional_verify_credentials(&config_glob, false)
        .await
        .unwrap()
        .into_config_without_writing_for_tests()
}
