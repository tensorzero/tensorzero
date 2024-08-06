mod migration_trait;
mod migrations;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use migration_trait::Migration;
use migrations::migration_0000::Migration0000;

pub async fn run(clickhouse: &ClickHouseConnectionInfo) -> Result<(), Error> {
    run_migration(Migration0000 { clickhouse }).await?;

    Ok(())
}

async fn run_migration(migration: impl Migration) -> Result<(), Error> {
    migration.can_apply().await?;

    if migration.should_apply().await? {
        // Get the migration name (e.g. `Migration0000`)
        let migration_name = std::any::type_name_of_val(&migration)
            .split("::")
            .last()
            .unwrap_or("Unknown migration");

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
            }
            Ok(false) => {
                tracing::error!(
                    "Failed migration success check: {migration_name}\n\n===== Rollback Instructions =====\n\n{}",
                    migration.rollback_instructions()
                );
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

    Ok(())
}
