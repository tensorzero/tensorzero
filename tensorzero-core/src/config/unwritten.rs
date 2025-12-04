use super::*;

/// A wrapper around `Config` that indicates the config has been loaded and validated,
/// but has **not yet been written to the database**.
///
/// This type exists to enforce correct sequencing in the config loading process:
/// 1. Config files are loaded and parsed
/// 2. The config is validated and initialized (producing `UnwrittenConfig`)
/// 3. Later, the config snapshot is written to the database (consuming `UnwrittenConfig`)
///
/// This wrapper is necessary because config loading happens **before** database connections
/// are established. The gateway needs to read database connection settings from the config
/// itself before it can connect to ClickHouse.
///
/// # Deref Behavior
/// This type implements `Deref<Target = Config>`, so you can access all `Config` methods
/// through an `UnwrittenConfig` reference.
///
/// # Consuming the Config
/// To get the inner `Config`, you must either:
/// - Call `ConfigLoadInfo::into_config()` to write the snapshot to the database
/// - Call `ConfigLoadInfo::dangerous_into_config_without_writing()` (test/special cases only)
#[derive(Debug)]
pub struct UnwrittenConfig {
    config: Config,
    snapshot: ConfigSnapshot,
}

impl UnwrittenConfig {
    pub fn new(config: Config, snapshot: ConfigSnapshot) -> Self {
        Self { config, snapshot }
    }

    /// Writes the config snapshot to ClickHouse and returns the config with its hash.
    ///
    /// This consumes the `ConfigLoadInfo` and:
    /// 1. Writes the `ConfigSnapshot` to the `ConfigSnapshot` table in ClickHouse
    /// 2. Returns a `ConfigWithHash` containing the config and its hash
    ///
    /// The hash is used to track which config version was used for each inference request.
    pub async fn into_config(self, clickhouse: &ClickHouseConnectionInfo) -> Result<Config, Error> {
        let UnwrittenConfig { config, snapshot } = self;
        write_config_snapshot(clickhouse, snapshot).await?;
        Ok(config)
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn into_config_without_writing_for_tests(self) -> Config {
        self.config
    }

    pub fn dangerous_into_config_without_writing(self) -> Config {
        self.config
    }
}

impl std::ops::Deref for UnwrittenConfig {
    type Target = Config;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}
