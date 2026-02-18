use std::collections::HashMap;

use async_trait::async_trait;
use sqlx::Row;

use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::db::ConfigQueries;
use crate::db::postgres::PostgresConnectionInfo;
use crate::error::{Error, ErrorDetails};

#[async_trait]
impl ConfigQueries for PostgresConnectionInfo {
    async fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> Result<ConfigSnapshot, Error> {
        let pool = self.get_pool_result()?;

        let row = sqlx::query(
            r"SELECT config, extra_templates, tags
               FROM tensorzero.config_snapshots
               WHERE hash = $1
               LIMIT 1",
        )
        .bind(&snapshot_hash)
        .fetch_optional(pool)
        .await?;

        let row = row.ok_or_else(|| {
            Error::new(ErrorDetails::ConfigSnapshotNotFound {
                snapshot_hash: snapshot_hash.to_string(),
            })
        })?;

        let config: String = row.try_get("config")?;
        let extra_templates_json: serde_json::Value = row.try_get("extra_templates")?;
        let tags_json: serde_json::Value = row.try_get("tags")?;

        let extra_templates: HashMap<String, String> =
            serde_json::from_value(extra_templates_json)?;
        let tags: HashMap<String, String> = serde_json::from_value(tags_json)?;

        ConfigSnapshot::from_stored(&config, extra_templates, tags, &snapshot_hash)
    }

    async fn write_config_snapshot(&self, snapshot: &ConfigSnapshot) -> Result<(), Error> {
        let pool = self.get_pool_result()?;

        let config_string = toml::to_string(&snapshot.config).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize config snapshot: {e}"),
            })
        })?;

        let extra_templates_json = serde_json::to_value(&snapshot.extra_templates)?;
        let tags_json = serde_json::to_value(&snapshot.tags)?;

        sqlx::query(
            r"INSERT INTO tensorzero.config_snapshots (hash, config, extra_templates, tensorzero_version, tags)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (hash) DO UPDATE SET
                 tags = tensorzero.config_snapshots.tags || EXCLUDED.tags,
                 last_used = NOW()",
        )
        .bind(snapshot.hash.as_bytes())
        .bind(&config_string)
        .bind(&extra_templates_json)
        .bind(crate::endpoints::status::TENSORZERO_VERSION)
        .bind(&tags_json)
        .execute(pool)
        .await?;

        Ok(())
    }
}
