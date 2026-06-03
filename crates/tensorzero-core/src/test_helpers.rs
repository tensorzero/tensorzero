#![expect(clippy::missing_panics_doc, clippy::unwrap_used)]

use std::path::PathBuf;

use crate::config::{Config, ConfigFileGlob};
use crate::db::delegating_connection::PrimaryDatastore;

/// Selects which `[object_storage]` override file the e2e config glob picks up.
///
/// The default `Disabled` value keeps the gateway's object store off, matching
/// historical behavior for tests that never resolve a `StoragePath`. Tests that
/// resolve through the hardened `/internal/object_storage` endpoint or
/// `Client::resolve` need `S3` so the configured store matches the fixture
/// bucket — the hardened path only reads from the gateway's own store.
#[derive(Copy, Clone, Debug)]
pub enum ObjectStorageOverride {
    Disabled,
    S3,
}

impl ObjectStorageOverride {
    fn file_prefix(self) -> &'static str {
        match self {
            ObjectStorageOverride::Disabled => "object-storage-disabled",
            ObjectStorageOverride::S3 => "object-storage-s3",
        }
    }
}

/// Returns the glob path for the E2E test configuration files.
///
/// Always matches `tensorzero.*.toml` plus a `[object_storage]` override file
/// (`object-storage-disabled.*.toml` by default, `object-storage-s3.*.toml`
/// when `object_storage` is `S3`). For Postgres primary, also includes
/// `postgres.*.toml` which sets `observability.backend = "postgres"`.
///
/// If `TENSORZERO_INTERNAL_TEST_CACHE_BACKEND` is set to `"clickhouse"` or `"valkey"`,
/// the corresponding `cache-{backend}.*.toml` config override is included in the glob.
pub fn get_e2e_config_path_for_datastore_with_object_storage(
    primary: PrimaryDatastore,
    object_storage: ObjectStorageOverride,
) -> PathBuf {
    let mut config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let cache_prefix = std::env::var("TENSORZERO_INTERNAL_TEST_CACHE_BACKEND")
        .ok()
        .and_then(|v| match v.as_str() {
            "clickhouse" => Some("cache-clickhouse"),
            "valkey" => Some("cache-valkey"),
            _ => None,
        });

    let mut prefixes = vec!["tensorzero"];
    if matches!(primary, PrimaryDatastore::Postgres) {
        prefixes.push("postgres");
    }
    if let Some(cache) = cache_prefix {
        prefixes.push(cache);
    }
    prefixes.push(object_storage.file_prefix());

    let glob = format!("tests/e2e/config/{{{}}}.*.toml", prefixes.join(","));
    config_path.push(glob);
    config_path
}

/// Returns the glob path for the E2E test configuration files with the default
/// (`disabled`) `[object_storage]` override.
pub fn get_e2e_config_path_for_datastore(primary: PrimaryDatastore) -> PathBuf {
    get_e2e_config_path_for_datastore_with_object_storage(primary, ObjectStorageOverride::Disabled)
}

/// Returns the glob path for the E2E test configuration files,
/// automatically selecting the primary datastore from `TENSORZERO_INTERNAL_TEST_OBSERVABILITY_BACKEND` env var.
pub fn get_e2e_config_path() -> PathBuf {
    get_e2e_config_path_for_datastore(PrimaryDatastore::from_test_env())
}

/// Same as `get_e2e_config_path` but configures the gateway with an
/// `s3_compatible` `[object_storage]` pointing at the shared
/// `tensorzero-e2e-test-images` bucket. Use this for tests that resolve a
/// `StoragePath` through the hardened endpoint — the gateway's configured
/// store must match the fixture bucket because the hardened path no longer
/// honors caller-supplied `StorageKind`s.
pub fn get_e2e_config_path_with_object_storage() -> PathBuf {
    get_e2e_config_path_for_datastore_with_object_storage(
        PrimaryDatastore::from_test_env(),
        ObjectStorageOverride::S3,
    )
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
