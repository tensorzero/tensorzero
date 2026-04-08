use super::*;
use crate::db::ConfigQueries;
use crate::error::DelayedError;

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
    uninitialized_config: UninitializedConfig,
    snapshot: ConfigSnapshot,
    runtime_overlay: RuntimeOverlay,
}

impl UnwrittenConfig {
    pub fn new(
        config: Config,
        uninitialized_config: UninitializedConfig,
        snapshot: ConfigSnapshot,
        runtime_overlay: RuntimeOverlay,
    ) -> Self {
        Self {
            config,
            uninitialized_config,
            snapshot,
            runtime_overlay,
        }
    }

    pub fn runtime_overlay(&self) -> &RuntimeOverlay {
        &self.runtime_overlay
    }

    pub fn uninitialized_config(&self) -> &UninitializedConfig {
        &self.uninitialized_config
    }

    /// Writes the config snapshot to the database and returns the config with its hash.
    ///
    /// This consumes the `UnwrittenConfig` and:
    /// 1. Writes the `ConfigSnapshot` to the database via the provided `ConfigQueries` impl
    /// 2. Returns the `Config`
    ///
    /// The hash is used to track which config version was used for each inference request.
    pub async fn into_config(
        self,
        db: &impl ConfigQueries,
    ) -> Result<(Config, RuntimeOverlay), DelayedError> {
        let UnwrittenConfig {
            config,
            uninitialized_config: _,
            snapshot,
            runtime_overlay,
        } = self;
        #[expect(clippy::disallowed_methods)]
        db.write_config_snapshot(&snapshot).await?;
        Ok((config, runtime_overlay))
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn into_config_without_writing_for_tests(self) -> Config {
        self.config
    }

    pub fn dangerous_into_config_without_writing(self) -> Config {
        self.config
    }

    /// Returns the extra templates discovered from the filesystem during config loading.
    /// These are templates resolved via MiniJinja `{% include %}` from the
    /// `template_filesystem_access` base path.
    pub fn extra_templates(&self) -> &HashMap<String, String> {
        &self.snapshot.extra_templates
    }
}

impl std::ops::Deref for UnwrittenConfig {
    type Target = Config;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}
