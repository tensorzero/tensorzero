pub mod migration_trait;
pub mod migrations;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
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

/// This must match the number of migrations returned by `make_all_migrations` - the tests
/// will panic if they don't match.
pub const NUM_MIGRATIONS: usize = 23;

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
    ];
    assert_eq!(
        migrations.len(),
        NUM_MIGRATIONS,
        "Please update the NUM_MIGRATIONS constant to match the number of migrations"
    );
    migrations
}

pub async fn run(clickhouse: &ClickHouseConnectionInfo) -> Result<(), Error> {
    clickhouse.health().await?;
    // This is a no-op if the database already exists
    clickhouse.create_database().await?;

    let migrations = make_all_migrations(clickhouse);

    // If the first migration needs to run, we are starting from scratch and don't need to wait for data to migrate
    // The value we pass in for 'clean_start' is ignored for the first migration
    let clean_start = run_migration(&*migrations[0], false).await?;
    for migration in &migrations[1..] {
        run_migration(&**migration, clean_start).await?;
    }
    Ok(())
}

/// Returns Err(e) if the migration fails to apply.
/// Returns Ok(false) if the migration should not apply.
/// Returns Ok(true) if the migration succeeds.
pub async fn run_migration(
    migration: &(impl Migration + ?Sized),
    clean_start: bool,
) -> Result<bool, Error> {
    migration.can_apply().await?;

    if migration.should_apply().await? {
        // Get the migration name (e.g. `Migration0000`)
        let migration_name = migration.name();

        tracing::info!("Applying migration: {migration_name} with clean_start: {clean_start}");

        if let Err(e) = migration.apply(clean_start).await {
            tracing::error!(
                "Failed to apply migration: {migration_name}\n\n===== Rollback Instructions =====\n\n{}",
                migration.rollback_instructions()
            );
            return Err(e);
        }

        match migration.has_succeeded().await {
            Ok(true) => {
                tracing::info!("Migration succeeded: {migration_name}");
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
            "".to_string()
        }
    }

    #[tokio::test]
    async fn test_run_migration_happy_path() {
        let mock_migration = MockMigration::default();

        // First check that method succeeds
        assert!(run_migration(&mock_migration, false).await.is_ok());

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
        assert!(run_migration(&mock_migration, false).await.is_err());

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
        assert!(run_migration(&mock_migration, false).await.is_ok());

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
    async fn test_run_migration_apply_fails() {
        let mock_migration = MockMigration {
            apply_result: false,
            ..Default::default()
        };

        // First check that the method fails
        assert!(run_migration(&mock_migration, false).await.is_err());

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
        assert!(run_migration(&mock_migration, false).await.is_err());

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
