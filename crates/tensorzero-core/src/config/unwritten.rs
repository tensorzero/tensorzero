use std::sync::Arc;

use sqlx::PgPool;

use super::*;
use crate::db::ConfigQueries;
use crate::db::postgres::variant_version_queries;

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

    /// Writes the config snapshot to the database and returns the config with its hash.
    ///
    /// This consumes the `UnwrittenConfig` and:
    /// 1. Writes the `ConfigSnapshot` to the database via the provided `ConfigQueries` impl
    /// 2. Returns the `Config`
    ///
    /// The hash is used to track which config version was used for each inference request.
    pub async fn into_config(self, db: &impl ConfigQueries) -> Result<Config, Error> {
        let UnwrittenConfig { config, snapshot } = self;
        #[expect(clippy::disallowed_methods)]
        db.write_config_snapshot(&snapshot).await?;
        Ok(config)
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn into_config_without_writing_for_tests(self) -> Config {
        self.config
    }

    pub fn dangerous_into_config_without_writing(self) -> Config {
        self.config
    }

    /// Load all DB-authoritative variants from Postgres and merge them into this config.
    ///
    /// For each (function_name, variant_name) in the database:
    /// - If the function exists in the config, the DB variant overrides or supplements
    ///   the file-loaded variant with the same name.
    /// - If the function doesn't exist, the variant is skipped with a warning.
    ///
    /// This should be called after the Postgres pool is available but before
    /// `into_config()` finalizes the config.
    pub async fn merge_db_variants(&mut self, pool: &PgPool) -> Result<(), Error> {
        let db_variants = variant_version_queries::load_all_active_variants(pool).await?;

        if db_variants.is_empty() {
            return Ok(());
        }

        let variant_count = db_variants.len();
        let mut merged_count = 0u32;

        for ((function_name, variant_name), uninitialized_variant) in db_variants {
            let Some(function_arc) = self.config.functions.get_mut(&function_name) else {
                tracing::warn!(
                    function_name,
                    variant_name,
                    "DB variant references unknown function, skipping"
                );
                continue;
            };

            let function = Arc::get_mut(function_arc).ok_or_else(|| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "Cannot merge DB variant `{variant_name}` into function `{function_name}`: \
                         function config has multiple references"
                    ),
                })
            })?;
            let schemas = function.schemas();
            let error_context = ErrorContext {
                function_name: function_name.clone(),
                variant_name: variant_name.clone(),
            };

            let variant_info = uninitialized_variant.load(schemas, &error_context)?;
            function
                .variants_mut()
                .insert(variant_name, Arc::new(variant_info));
            merged_count += 1;
        }

        tracing::info!(
            total = variant_count,
            merged = merged_count,
            "Merged DB variants into config"
        );

        Ok(())
    }
}

impl std::ops::Deref for UnwrittenConfig {
    type Target = Config;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}
