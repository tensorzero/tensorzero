use super::check_column_exists;
use super::check_table_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::{ErrorDetails, delayed_error::DelayedError};
use async_trait::async_trait;

/// Adds `canonical_hash` and `config_json` columns to `ConfigSnapshot`
/// for ClickHouse parity with the Postgres canonical-hash work (see
/// migration `20260428000000_config_snapshots_jsonb.sql`).
///
/// The legacy `hash` and `config` (TOML) columns stay untouched — they
/// carry the values every existing `inferences.snapshot_hash` reference
/// expects. The two new columns carry the canonical-form pair from
/// `ConfigSnapshot::to_canonical_form`:
///
/// - `canonical_hash UInt256`: the structural Blake3 over the canonical
///   JSON tree. New inference rows reference this value going forward.
/// - `config_json String`: the canonical JSON bytes themselves. Read
///   directly via `from_stored_with_hash` so reads no longer reparse
///   the legacy TOML — important once `StoredConfig`'s shape evolves
///   past what older TOMLs can re-deserialize as.
///
/// Existing rows default both new columns to ClickHouse's column
/// defaults (`canonical_hash = 0`, `config_json = ''`). Those are the
/// "needs backfill" sentinels — the boot-time backfill in
/// `db::clickhouse::config_queries::backfill_config_snapshot_canonical_hash`
/// re-parses each row's `config` (TOML) text via the same
/// `to_canonical_form` helper and writes both new columns together.
/// Lookups by canonical hash against rows the backfill hasn't touched
/// (or that fail to backfill) return not-found, which is semantically
/// correct: the row was written before canonical hashing existed and
/// was never identified by it.
pub struct Migration0054<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0054";

#[async_trait]
impl Migration for Migration0054<'_> {
    async fn can_apply(&self) -> Result<(), DelayedError> {
        if !check_table_exists(self.clickhouse, "ConfigSnapshot", MIGRATION_ID).await? {
            return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "`ConfigSnapshot` table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, DelayedError> {
        let has_canonical_hash = check_column_exists(
            self.clickhouse,
            "ConfigSnapshot",
            "canonical_hash",
            MIGRATION_ID,
        )
        .await?;
        let has_config_json = check_column_exists(
            self.clickhouse,
            "ConfigSnapshot",
            "config_json",
            MIGRATION_ID,
        )
        .await?;
        Ok(!has_canonical_hash || !has_config_json)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), DelayedError> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        // `DEFAULT 0` so existing rows have a well-defined value (UInt256
        // has no NULL). `0` doubles as the "needs backfill" sentinel for
        // the boot-time backfill — the canonical hash of any real config
        // is overwhelmingly unlikely to collide with `0` in practice
        // (Blake3-256 collision space).
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                "ALTER TABLE ConfigSnapshot{on_cluster_name} \
                 ADD COLUMN IF NOT EXISTS canonical_hash UInt256 DEFAULT 0"
            ))
            .await?;

        // `DEFAULT ''` so existing rows have a well-defined value (String
        // has no NULL). Empty string doubles as the "needs backfill"
        // sentinel — a real canonical JSON document for any non-empty
        // `StoredConfig` is at minimum `"{}"` (`empty.toml` produces a
        // multi-key default object), so the empty string can't collide
        // with real content.
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(format!(
                "ALTER TABLE ConfigSnapshot{on_cluster_name} \
                 ADD COLUMN IF NOT EXISTS config_json String DEFAULT ''"
            ))
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        format!(
            "/* drop the canonical_hash and config_json columns added by migration {MIGRATION_ID} */\n\
             ALTER TABLE ConfigSnapshot DROP COLUMN IF EXISTS canonical_hash;\n\
             ALTER TABLE ConfigSnapshot DROP COLUMN IF EXISTS config_json;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, DelayedError> {
        // Migration is "succeeded" iff the column now exists. `should_apply`
        // already inverts the same check, so reuse it.
        Ok(!self.should_apply().await?)
    }
}
