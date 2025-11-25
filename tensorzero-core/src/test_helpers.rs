#![expect(clippy::missing_panics_doc, clippy::unwrap_used)]

use std::path::PathBuf;

use crate::config::{Config, ConfigFileGlob};

/// Returns the path to the E2E test configuration file.
/// The path is relative to the tensorzero-core crate root.
pub fn get_e2e_config_path() -> PathBuf {
    let mut config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("tests/e2e/config/tensorzero.*.toml");
    config_path
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
        .config
}
