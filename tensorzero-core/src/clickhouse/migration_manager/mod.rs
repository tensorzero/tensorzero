pub mod migration_trait;
pub mod migrations;

use std::collections::HashMap;
use std::env;
use std::time::{Duration, Instant};

use crate::clickhouse::{ClickHouseConnectionInfo, Rows, TableName};
use crate::config_parser::BatchWritesConfig;
use crate::endpoints::status::TENSORZERO_VERSION;
use crate::error::{Error, ErrorDetails};
use crate::serde_util::deserialize_u64;
use async_trait::async_trait;
use migration_trait::Migration;
use migrations::migration_0000::Migration0000;
use migrations::migration_0002::Migration0002;
use migrations::migration_0003::Migration0003;
use migrations::migration_0004::Migration0004;
use migrations::migration_0005::Migration0005;
use migrations::migration_0006::Migration0006;
use migrations::migration_0008::Migration0008;
use migrations::migration_0009::Migration0009;
use migrations::migration_0011::Migration0011;
use migrations::migration_0015::Migration0015;
use migrations::migration_0016::Migration0016;
use migrations::migration_0017::Migration0017;
use migrations::migration_0018::Migration0018;
use migrations::migration_0019::Migration0019;
use migrations::migration_0020::Migration0020;
use migrations::migration_0021::Migration0021;
use migrations::migration_0022::Migration0022;
use migrations::migration_0024::Migration0024;
use migrations::migration_0025::Migration0025;
use migrations::migration_0026::Migration0026;
use migrations::migration_0027::Migration0027;
use migrations::migration_0028::Migration0028;
use migrations::migration_0029::Migration0029;
use migrations::migration_0030::Migration0030;
use migrations::migration_0031::Migration0031;
use migrations::migration_0032::Migration0032;
use migrations::migration_0033::Migration0033;
use migrations::migration_0034::Migration0034;
use serde::{Deserialize, Serialize};

/// This must match the number of migrations returned by `make_all_migrations` - the tests
/// will panic if they don't match.
pub const NUM_MIGRATIONS: usize = 28;
fn get_run_migrations_command() -> String {
    let version = env!("CARGO_PKG_VERSION");
    format!("docker run --rm -e TENSORZERO_CLICKHOUSE_URL=$TENSORZERO_CLICKHOUSE_URL tensorzero/gateway:{version} --run-migrations-only")
}

/// Constructs (but does not run) a vector of all our database migrations.
/// This is the single source of truth for all migration - it's used during startup to migrate
/// the database, and in our ClickHouse tests to verify that the migrations apply correctly
/// to a fresh database.
pub fn make_all_migrations<'a>(
    clickhouse: &'a ClickHouseConnectionInfo,
) -> Vec<Box<dyn Migration + Send + Sync + 'a>> {
    let migrations: Vec<Box<dyn Migration + Send + Sync + 'a>> = vec![
        Box::new(Migration0000 { clickhouse }),
        // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0010
        // Box::new(Migration0001 { clickhouse }),
        Box::new(Migration0002 { clickhouse }),
        Box::new(Migration0003 { clickhouse }),
        Box::new(Migration0004 { clickhouse }),
        Box::new(Migration0005 { clickhouse }),
        Box::new(Migration0006 { clickhouse }),
        // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0013
        // Box::new(Migration0007 { clickhouse }),
        Box::new(Migration0008 { clickhouse }),
        Box::new(Migration0009 { clickhouse }),
        // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0013
        // Box::new(Migration0010 { clickhouse }),
        Box::new(Migration0011 { clickhouse }),
        // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0014
        // Box::new(Migration0012 { clickhouse }),
        // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0021
        // Box::new(Migration0013 { clickhouse }),
        // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0016
        // Box::new(Migration0014 { clickhouse }),
        Box::new(Migration0015 { clickhouse }),
        Box::new(Migration0016 { clickhouse }),
        Box::new(Migration0017 { clickhouse }),
        Box::new(Migration0018 { clickhouse }),
        Box::new(Migration0019 { clickhouse }),
        Box::new(Migration0020 { clickhouse }),
        Box::new(Migration0021 { clickhouse }),
        Box::new(Migration0022 { clickhouse }),
        Box::new(Migration0024 { clickhouse }),
        Box::new(Migration0025 { clickhouse }),
        Box::new(Migration0026 { clickhouse }),
        Box::new(Migration0027 { clickhouse }),
        Box::new(Migration0028 { clickhouse }),
        Box::new(Migration0029 { clickhouse }),
        Box::new(Migration0030 { clickhouse }),
        Box::new(Migration0031 { clickhouse }),
        Box::new(Migration0032 { clickhouse }),
        Box::new(Migration0033 { clickhouse }),
        Box::new(Migration0034 { clickhouse }),
    ];
    assert_eq!(
        migrations.len(),
        NUM_MIGRATIONS,
        "Please update the NUM_MIGRATIONS constant to match the number of migrations"
    );
    migrations
}

/// Returns `true` if our `TensorZeroMigration` table contains exactly `all_migrations`.
/// If the database or `TensorZeroMigration` table does not exist, or it contains either more
/// or fewer distinct migration ids than we expect, than we return `false` to be conservative.
/// Note that we allow multiple rows to exist per migration id (since the migrations
/// might have been run concurrently).
pub async fn should_skip_migrations(
    clickhouse: &ClickHouseConnectionInfo,
    all_migrations: &[Box<dyn Migration + Send + Sync + '_>],
) -> bool {
    let migration_records = match get_all_migration_records(clickhouse).await {
        Ok(records) => records,
        Err(e) => {
            if let ErrorDetails::ClickHouseMigration { message, .. } = e.get_details() {
                if message.contains("UNKNOWN_DATABASE") {
                    tracing::info!("Database not found, assuming clean start");
                    return false;
                }
                if message.contains("UNKNOWN_TABLE") {
                    tracing::info!("TensorZeroMigration table not found, assuming clean start");
                    return false;
                }
            }
            // Fall back to running all migrations as normal, and hopefully produce a better error message
            tracing::warn!("Failed to lookup migrations records: {e}");
            return false;
        }
    };
    let mut migration_ids = migration_records
        .iter()
        .map(|r| r.migration_id)
        .collect::<Vec<_>>();
    migration_ids.sort();

    let expected_migration_ids = all_migrations
        .iter()
        .map(|m| m.migration_num())
        .collect::<Result<Vec<_>, Error>>();
    let mut expected_migration_ids = match expected_migration_ids {
        Ok(ids) => ids,
        Err(e) => {
            // If we encounter any parse errors, just run the migrations as normal,
            // and hopefully produce a better error message
            tracing::warn!("Failed to get migration ids: {e}");
            return false;
        }
    };
    expected_migration_ids.sort();

    // We only want to skip running migrations if the database is in a known state
    // (we've run exactly the migrations that we expect to have run)
    tracing::debug!("Actual   migration ids: {migration_ids:?}");
    tracing::debug!("Expected migration ids: {expected_migration_ids:?}");
    migration_ids == expected_migration_ids
}

pub struct RunMigrationManagerArgs<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub skip_completed_migrations: bool,
    pub manual_run: bool,
}

pub async fn run(args: RunMigrationManagerArgs<'_>) -> Result<(), Error> {
    let RunMigrationManagerArgs {
        clickhouse,
        skip_completed_migrations,
        manual_run,
    } = args;
    clickhouse.health().await?;

    let migrations: Vec<Box<dyn Migration + Send + Sync>> = make_all_migrations(clickhouse);
    if skip_completed_migrations && should_skip_migrations(clickhouse, &migrations).await {
        tracing::debug!("All migrations have already been applied");
        return Ok(());
    }
    tracing::debug!("All migrations have not yet been applied, running migrations");
    let database_exists = clickhouse.check_database_exists().await?;
    if !database_exists {
        if clickhouse.is_cluster_configured() && !manual_run {
            let database = clickhouse.database();
            let run_migrations_command = get_run_migrations_command();
            return Err(ErrorDetails::ClickHouseConfiguration {
                message: format!("Database {database} does not exist. We do not automatically run migrations to create and set it up when replication is configured. Please run `{run_migrations_command}`."),
            }.into());
        } else {
            // This is a no-op if the database already exists
            clickhouse.create_database().await?;
        }
    }

    // Check if the ClickHouse instance is configured correctly for replication.
    check_replication_settings(clickhouse).await?;

    let is_replicated = clickhouse.is_cluster_configured();

    // If the first migration needs to run, we are starting from scratch and don't need to wait for data to migrate
    // The value we pass in for 'clean_start' is ignored for the first migration
    let clean_start = run_migration(RunMigrationArgs {
        clickhouse,
        migration: &*migrations[0],
        clean_start: false,
        manual_run,
        is_replicated,
    })
    .await?;
    for migration in &migrations[1..] {
        run_migration(RunMigrationArgs {
            clickhouse,
            migration: &**migration,
            clean_start,
            manual_run,
            is_replicated,
        })
        .await?;
    }
    Ok(())
}

/// Make sure that the ClickHouse instance is configured correctly for replication.
/// If the instance is configured to replicate, there must be a replicated non-cloud ClickHouse.
/// If the instance is not configured to replicate, there should be either a cloud ClickHouse,
/// a non-replicated OSS ClickHouse, or a replicated ClickHouse (this latter requires the
/// `TENSORZERO_OVERRIDE_NON_REPLICATED_CLICKHOUSE` environment variable to be set to "1").
async fn check_replication_settings(clickhouse: &ClickHouseConnectionInfo) -> Result<(), Error> {
    // First, let's check if we are using ClickHouse Cloud
    let cloud_mode_response = clickhouse
        .run_query_synchronous_no_params("SELECT getSetting('cloud_mode')".to_string())
        .await?;
    let cloud_mode: bool = cloud_mode_response.response.trim().parse().map_err(|e| {
        Error::new(ErrorDetails::ClickHouseDeserialization {
            message: format!("Failed to deserialize cloud mode response: {e}"),
        })
    })?;
    tracing::debug!("ClickHouse Cloud mode: {}", cloud_mode);
    // If we are using ClickHouse Cloud, we do not allow a cluster to be configured
    if cloud_mode && clickhouse.is_cluster_configured() {
        return Err(ErrorDetails::ClickHouseConfiguration {
            message: "Clusters cannot be configured when using ClickHouse Cloud.".to_string(),
        }
        .into());
    }

    // Next, let's check if there are replicated tables in our deployment
    let max_cluster_count_query = r"
        SELECT MAX(node_count) AS max_nodes_per_cluster
        FROM (
            SELECT
                cluster,
                COUNT() as node_count
            FROM system.clusters
            GROUP BY cluster
        );"
    .to_string();
    let max_cluster_count_response = clickhouse
        .run_query_synchronous_no_params(max_cluster_count_query)
        .await?;
    let max_cluster_count: u32 =
        max_cluster_count_response
            .response
            .trim()
            .parse()
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize max cluster count response: {e}"),
                })
            })?;
    tracing::debug!("Max cluster count: {}", max_cluster_count);
    // Let's check if the user has set the override to allow for non-replicated ClickHouse setup
    // on a ClickHouse deployment with a replicated cluster.
    let non_replicated_tensorzero_on_replicated_clickhouse_override =
        std::env::var("TENSORZERO_OVERRIDE_NON_REPLICATED_CLICKHOUSE").unwrap_or_default() == "1";
    // If the user has not set the override and the ClickHouse deployment is replicated
    // we fail if the ClickHouse deployment has not been configured to be replicated.
    if max_cluster_count > 1
        && !clickhouse.is_cluster_configured()
        && !cloud_mode
        && !non_replicated_tensorzero_on_replicated_clickhouse_override
    {
        return Err(Error::new(ErrorDetails::ClickHouseConfiguration {
            message: "TensorZero is not configured for replication but ClickHouse contains a replicated cluster. Please set the environment variable TENSORZERO_OVERRIDE_NON_REPLICATED_CLICKHOUSE=1 to override if you're sure you'd like a non-replicated ClickHouse setup.".to_string(),
        }));
    }

    // If the user has configured a replicated ClickHouse deployment but we don't have a replicated ClickHouse instance, we fail.
    if max_cluster_count <= 1 && clickhouse.is_cluster_configured() {
        return Err(Error::new(ErrorDetails::ClickHouseConfiguration {
            message: "TensorZero is configured for replication but ClickHouse is not configured for replication. Please ensure that ClickHouse is configured for replication.".to_string(),
        }));
    };

    Ok(())
}

#[derive(Deserialize, Debug, Serialize, PartialEq)]
pub struct MigrationRecordDatabaseInsert {
    pub migration_id: u32,
    pub migration_name: String,
    pub gateway_version: String,
    pub gateway_git_sha: String,
    #[serde(deserialize_with = "deserialize_u64")]
    pub execution_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_at: Option<String>,
}

/// Attempts to get all migration records from the `TensorZeroMigration` table.
/// We do not log any `Error`s that we return, as the caller may want to ignore them.
/// Returns at most one record per migration ID.
pub async fn get_all_migration_records(
    clickhouse: &ClickHouseConnectionInfo,
) -> Result<Vec<MigrationRecordDatabaseInsert>, Error> {
    let mut rows = Vec::new();
    for row in clickhouse
        .run_query_synchronous_with_err_logging(
            "SELECT DISTINCT ON (migration_id) * FROM TensorZeroMigration ORDER BY migration_id ASC, applied_at DESC FORMAT JSONEachRow"
                .to_string(),
            &HashMap::new(),
            false,
        )
        .await
        .map_err(|e| {
            Error::new_without_logging(ErrorDetails::ClickHouseMigration {
                id: "0000".to_string(),
                message: format!("Failed to get migration records: {e}"),
            })
        })?
        .response
        .lines()
    {
        rows.push(
            serde_json::from_str::<MigrationRecordDatabaseInsert>(row).map_err(|e| {
                Error::new_without_logging(ErrorDetails::ClickHouseMigration {
                    id: "0000".to_string(),
                    message: format!("Failed to parse migration record: {e}"),
                })
            })?,
        );
    }
    Ok(rows)
}

pub async fn insert_migration_record(
    clickhouse: &ClickHouseConnectionInfo,
    migration: &(impl Migration + ?Sized),
    execution_time: Duration,
) -> Result<(), Error> {
    let migration_id = migration.migration_num()?;
    let migration_name = migration.name();
    clickhouse
        .write_non_batched(
            Rows::Unserialized(&[MigrationRecordDatabaseInsert {
                migration_id,
                migration_name,
                gateway_version: TENSORZERO_VERSION.to_string(),
                gateway_git_sha: crate::built_info::GIT_COMMIT_HASH
                    .unwrap_or("unknown")
                    .to_string(),
                execution_time_ms: execution_time.as_millis() as u64,
                applied_at: None,
            }]),
            TableName::TensorZeroMigration,
        )
        .await?;
    Ok(())
}

pub struct RunMigrationArgs<'a, T: Migration + ?Sized> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub migration: &'a T,
    pub clean_start: bool,
    pub manual_run: bool,
    pub is_replicated: bool,
}

pub async fn manual_run_migrations() -> Result<(), Error> {
    let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL")
        .ok()
        .or_else(|| {
            std::env::var("CLICKHOUSE_URL").ok().inspect(|_| {
                tracing::warn!("Deprecation Warning: The environment variable \"CLICKHOUSE_URL\" has been renamed to \"TENSORZERO_CLICKHOUSE_URL\" and will be removed in a future version. Please update your environment to use \"TENSORZERO_CLICKHOUSE_URL\" instead.");
            })
        }).ok_or_else(|| Error::new(ErrorDetails::ClickHouseConfiguration { message: "TENSORZERO_CLICKHOUSE_URL not found".to_string() }))?;
    let clickhouse =
        ClickHouseConnectionInfo::new(&clickhouse_url, BatchWritesConfig::default()).await?;
    run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        // If we manually run the migrations, we should not skip any.
        skip_completed_migrations: false,
        manual_run: true,
    })
    .await
}

/// Returns Err(e) if the migration fails to apply.
/// Returns Ok(false) if the migration should not apply.
/// Returns Ok(true) if the migration succeeds.
pub async fn run_migration(
    args: RunMigrationArgs<'_, impl Migration + ?Sized>,
) -> Result<bool, Error> {
    let RunMigrationArgs {
        clickhouse,
        migration,
        clean_start,
        manual_run,
        is_replicated,
    } = args;

    migration.can_apply().await?;

    if migration.should_apply().await? {
        // Get the migration name (e.g. `Migration0000`)
        let migration_name = migration.name();

        if is_replicated && !manual_run {
            let run_migrations_command = get_run_migrations_command();
            return Err(ErrorDetails::ClickHouseMigration { id: migration_name, message: format!("Migrations must be run manually if using a replicated ClickHouse cluster. Please run `{run_migrations_command}`.") }.into());
        }

        tracing::info!("Applying migration: {migration_name} with clean_start: {clean_start}");

        let start_time = Instant::now();
        if let Err(e) = migration.apply(clean_start).await {
            tracing::error!(
                "Failed to apply migration: {migration_name}\n\n===== Rollback Instructions =====\n\n{}",
                migration.rollback_instructions()
            );
            return Err(e);
        }
        let execution_time = start_time.elapsed();

        match migration.has_succeeded().await {
            Ok(true) => {
                tracing::info!("Migration succeeded: {migration_name}");
                insert_migration_record(clickhouse, migration, execution_time).await?;
                return Ok(true);
            }
            Ok(false) => {
                tracing::error!(
                    "Failed migration success check: {migration_name}\n\n===== Rollback Instructions =====\n\n{}",
                    migration.rollback_instructions()
                );
                return Err(ErrorDetails::ClickHouseMigration {
                    id: migration_name.to_string(),
                    message: "Migration success check failed".to_string(),
                }
                .into());
            }
            Err(e) => {
                tracing::error!(
                    "Failed to verify migration: {migration_name}\n\n===== Rollback Instructions =====\n\n{}",
                    migration.rollback_instructions()
                );
                return Err(e);
            }
        }
    }

    Ok(false)
}

mod tests {
    use super::*;

    #[allow(clippy::allow_attributes, dead_code)] // False positive
    struct MockMigration {
        can_apply_result: bool,
        should_apply_result: bool,
        apply_result: bool,
        has_succeeded_result: bool,
        called_can_apply: std::sync::atomic::AtomicBool,
        called_should_apply: std::sync::atomic::AtomicBool,
        called_apply: std::sync::atomic::AtomicBool,
        called_has_succeeded: std::sync::atomic::AtomicBool,
    }

    impl Default for MockMigration {
        fn default() -> Self {
            Self {
                can_apply_result: true,
                should_apply_result: true,
                apply_result: true,
                has_succeeded_result: true,
                called_can_apply: std::sync::atomic::AtomicBool::new(false),
                called_should_apply: std::sync::atomic::AtomicBool::new(false),
                called_apply: std::sync::atomic::AtomicBool::new(false),
                called_has_succeeded: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    #[async_trait]
    impl Migration for MockMigration {
        fn name(&self) -> String {
            "Migration1".to_string()
        }

        async fn can_apply(&self) -> Result<(), Error> {
            self.called_can_apply
                .store(true, std::sync::atomic::Ordering::Relaxed);
            if self.can_apply_result {
                Ok(())
            } else {
                Err(ErrorDetails::ClickHouseMigration {
                    id: "0000".to_string(),
                    message: "MockMigration can_apply failed".to_string(),
                }
                .into())
            }
        }

        async fn should_apply(&self) -> Result<bool, Error> {
            self.called_should_apply
                .store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(self.should_apply_result)
        }

        async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
            self.called_apply
                .store(true, std::sync::atomic::Ordering::Relaxed);
            if self.apply_result {
                Ok(())
            } else {
                Err(ErrorDetails::ClickHouseMigration {
                    id: "0000".to_string(),
                    message: "MockMigration apply failed".to_string(),
                }
                .into())
            }
        }

        async fn has_succeeded(&self) -> Result<bool, Error> {
            self.called_has_succeeded
                .store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(self.has_succeeded_result)
        }

        fn rollback_instructions(&self) -> String {
            String::new()
        }
    }

    #[tokio::test]
    async fn test_run_migration_happy_path() {
        let mock_migration = MockMigration::default();

        // First check that method succeeds
        assert!(run_migration(RunMigrationArgs {
            clickhouse: &ClickHouseConnectionInfo::Disabled,
            migration: &mock_migration,
            clean_start: false,
            is_replicated: false,
            manual_run: false,
        })
        .await
        .is_ok());

        // Check that we called every method
        assert!(mock_migration
            .called_can_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_should_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_has_succeeded
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_run_migration_replicated_automatic_fails() {
        let mock_migration = MockMigration::default();

        // First check that method succeeds
        assert!(run_migration(RunMigrationArgs {
            clickhouse: &ClickHouseConnectionInfo::Disabled,
            migration: &mock_migration,
            clean_start: false,
            is_replicated: true,
            manual_run: false,
        })
        .await
        .is_err());

        // Check that we called can / should but not apply or has_succeeded
        assert!(mock_migration
            .called_can_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_should_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_has_succeeded
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_run_migration_replicated_manual() {
        let mock_migration = MockMigration::default();

        // First check that method succeeds
        assert!(run_migration(RunMigrationArgs {
            clickhouse: &ClickHouseConnectionInfo::Disabled,
            migration: &mock_migration,
            clean_start: false,
            is_replicated: true,
            manual_run: true,
        })
        .await
        .is_ok());

        // Check that we called every method
        assert!(mock_migration
            .called_can_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_should_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_has_succeeded
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_run_migration_can_apply_fails() {
        let mock_migration = MockMigration {
            can_apply_result: false,
            ..Default::default()
        };

        // First check that the method fails
        assert!(run_migration(RunMigrationArgs {
            clickhouse: &ClickHouseConnectionInfo::Disabled,
            migration: &mock_migration,
            clean_start: false,
            is_replicated: false,
            manual_run: false,
        })
        .await
        .is_err());

        // Check that we called every method
        assert!(mock_migration
            .called_can_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_should_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_has_succeeded
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_run_migration_should_apply_false() {
        let mock_migration = MockMigration {
            should_apply_result: false,
            ..Default::default()
        };

        // First check that the method succeeds
        assert!(run_migration(RunMigrationArgs {
            clickhouse: &ClickHouseConnectionInfo::Disabled,
            migration: &mock_migration,
            clean_start: false,
            is_replicated: false,
            manual_run: false,
        })
        .await
        .is_ok());

        // Check that we called every method
        assert!(mock_migration
            .called_can_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_should_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_has_succeeded
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_run_migration_should_apply_false_replicated() {
        let mock_migration = MockMigration {
            should_apply_result: false,
            ..Default::default()
        };

        // First check that the method succeeds
        assert!(run_migration(RunMigrationArgs {
            clickhouse: &ClickHouseConnectionInfo::Disabled,
            migration: &mock_migration,
            clean_start: false,
            is_replicated: true,
            manual_run: false,
        })
        .await
        .is_ok());

        // Check that we called can / should but not apply or has_succeeded
        assert!(mock_migration
            .called_can_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_should_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_has_succeeded
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_run_migration_apply_fails() {
        let mock_migration = MockMigration {
            apply_result: false,
            ..Default::default()
        };

        // First check that the method fails
        assert!(run_migration(RunMigrationArgs {
            clickhouse: &ClickHouseConnectionInfo::Disabled,
            migration: &mock_migration,
            clean_start: false,
            is_replicated: false,
            manual_run: false,
        })
        .await
        .is_err());

        // Check that we called every method
        assert!(mock_migration
            .called_can_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_should_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(!mock_migration
            .called_has_succeeded
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_run_migration_has_succeeded_false() {
        let mock_migration = MockMigration {
            has_succeeded_result: false,
            ..Default::default()
        };

        // First check that the method fails
        assert!(run_migration(RunMigrationArgs {
            clickhouse: &ClickHouseConnectionInfo::Disabled,
            migration: &mock_migration,
            clean_start: false,
            is_replicated: false,
            manual_run: false,
        })
        .await
        .is_err());

        // Check that we called every method
        assert!(mock_migration
            .called_can_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_should_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_apply
            .load(std::sync::atomic::Ordering::Relaxed));
        assert!(mock_migration
            .called_has_succeeded
            .load(std::sync::atomic::Ordering::Relaxed));
    }
}
