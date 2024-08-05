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
            .unwrap_or_else(|| "Unknown migration");

        tracing::info!("Applying migration: {migration_name}");

        migration.apply().await?;

        if !migration.has_succeeded().await? {
            migration.rollback().await?;
        }
    }

    Ok(())
}
