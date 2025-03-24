pub mod migration_trait;
pub mod migrations;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
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

use async_trait::async_trait;

pub async fn run(clickhouse: &ClickHouseConnectionInfo) -> Result<(), Error> {
    clickhouse.health().await?;
    // This is a no-op if the database already exists
    clickhouse.create_database().await?;
    // If the first migration needs to run, we are starting from scratch and don't need to wait for data to migrate
    let clean_start = run_migration(&Migration0000 { clickhouse }).await?;
    // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0010
    // run_migration(&Migration0001 {
    //     clickhouse,
    //     clean_start,
    // })
    // .await?;
    run_migration(&Migration0002 { clickhouse }).await?;
    run_migration(&Migration0003 { clickhouse }).await?;
    run_migration(&Migration0004 { clickhouse }).await?;
    run_migration(&Migration0005 { clickhouse }).await?;
    run_migration(&Migration0006 { clickhouse }).await?;
    // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0013
    // run_migration(&Migration0007 {
    //     clickhouse,
    //     clean_start,
    // })
    // .await?;
    run_migration(&Migration0008 { clickhouse }).await?;
    run_migration(&Migration0009 {
        clickhouse,
        clean_start,
    })
    .await?;
    // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0013
    // run_migration(&Migration0010 {
    //     clickhouse,
    //     clean_start,
    // })
    // .await?;
    run_migration(&Migration0011 { clickhouse }).await?;
    // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0014
    // run_migration(&Migration0012 { clickhouse }).await?;
    // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0021
    // run_migration(&Migration0013 {
    //     clickhouse,
    //     clean_start,
    // })
    // .await?;
    // BANNED: This migration is no longer needed because it is deleted and replaced by migration 0016
    // run_migration(&Migration0014 { clickhouse }).await?;
    run_migration(&Migration0015 { clickhouse }).await?;
    run_migration(&Migration0016 { clickhouse }).await?;
    run_migration(&Migration0017 { clickhouse }).await?;
    run_migration(&Migration0018 { clickhouse }).await?;
    run_migration(&Migration0019 { clickhouse }).await?;
    run_migration(&Migration0020 {
        clickhouse,
        clean_start,
    })
    .await?;
    run_migration(&Migration0021 {
        clickhouse,
        clean_start,
    })
    .await?;
    // NOTE:
    // When we add more migrations, we need to add a test that applies them in a cumulative (N^2) way.
    //
    // In sequence:
    // - Migration0000
    // - Migration0000 (noop), Migration0001
    // - Migration0000 (noop), Migration0001 (noop), Migration0002
    //
    // We need to check that previous migrations return false for should_apply() (i.e. are noops).
    //
    // You should expand gateway::tests::e2e::clickhouse_migration_manager::clickhouse_migration_manager
    // to test this.

    Ok(())
}

/// Returns Err(e) if the migration fails to apply.
/// Returns Ok(false) if the migration should not apply.
/// Returns Ok(true) if the migration succeeds.
pub async fn run_migration(migration: &(impl Migration + ?Sized)) -> Result<bool, Error> {
    migration.can_apply().await?;

    if migration.should_apply().await? {
        // Get the migration name (e.g. `Migration0000`)
        let migration_name = migration.name();

        tracing::info!("Applying migration: {migration_name}");

        if let Err(e) = migration.apply().await {
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

        async fn apply(&self) -> Result<(), Error> {
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
        assert!(run_migration(&mock_migration).await.is_ok());

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
        assert!(run_migration(&mock_migration).await.is_err());

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
        assert!(run_migration(&mock_migration).await.is_ok());

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
        assert!(run_migration(&mock_migration).await.is_err());

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
        assert!(run_migration(&mock_migration).await.is_err());

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
