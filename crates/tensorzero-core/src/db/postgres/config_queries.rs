use std::collections::HashMap;

use async_trait::async_trait;
use sqlx::{PgPool, Row};
use tensorzero_types::SnapshotHashScheme;

use crate::config::snapshot::{ConfigSnapshot, SnapshotHash, StoredConfig};
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::{ConfigQueries, ConfigSnapshotSearch};
use crate::error::{DelayedError, Error, ErrorDetails};

#[async_trait]
impl ConfigQueries for PostgresConnectionInfo {
    /// Read a snapshot by hash. JSON is the source of truth: we read
    /// `config_jsonb` and deserialize directly into `StoredConfig`. The
    /// `config TEXT` (TOML) column is the deprecated migration column —
    /// only consulted as a fallback for legacy rows whose `config_jsonb`
    /// is still NULL pre-backfill.
    ///
    /// The hash carries a `SnapshotHashScheme` that determines which
    /// column to query:
    ///
    /// - `LegacyToml` → `WHERE hash = $1` (the original primary key,
    ///                                     bound to canonical-TOML bytes).
    /// - `Canonical`  → `WHERE canonical_hash = $1` (the structural hash
    ///                                                introduced by this PR).
    ///
    /// New writes populate both columns so either lookup resolves the
    /// same row. Old `inferences.snapshot_hash` references parse as
    /// `LegacyToml` (unprefixed decimal) and continue to work.
    async fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> Result<ConfigSnapshot, Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;

        // Dispatch based on hash scheme. The two queries are otherwise
        // identical — only the column on the LHS of the WHERE clause
        // changes.
        let row = match snapshot_hash.scheme() {
            SnapshotHashScheme::LegacyToml => {
                sqlx::query(
                    r"SELECT config, config_jsonb, extra_templates, tags
                       FROM tensorzero.config_snapshots
                       WHERE hash = $1
                       LIMIT 1",
                )
                .bind(&snapshot_hash)
                .fetch_optional(pool)
                .await?
            }
            SnapshotHashScheme::Canonical => {
                sqlx::query(
                    r"SELECT config, config_jsonb, extra_templates, tags
                       FROM tensorzero.config_snapshots
                       WHERE canonical_hash = $1
                       LIMIT 1",
                )
                .bind(&snapshot_hash)
                .fetch_optional(pool)
                .await?
            }
        };

        let row = row.ok_or_else(|| {
            Error::new(ErrorDetails::ConfigSnapshotNotFound {
                snapshot_hash: snapshot_hash.to_string(),
            })
        })?;

        let config_jsonb_value: Option<serde_json::Value> = row.try_get("config_jsonb")?;
        let extra_templates_json: serde_json::Value = row.try_get("extra_templates")?;
        let tags_json: serde_json::Value = row.try_get("tags")?;

        let extra_templates: HashMap<String, String> =
            serde_json::from_value(extra_templates_json)?;
        let tags: HashMap<String, String> = serde_json::from_value(tags_json)?;

        if let Some(jsonb) = config_jsonb_value {
            // Canonical path: JSONB is the source of truth.
            let stored: StoredConfig = serde_json::from_value(jsonb).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Failed to deserialize config_jsonb for snapshot {snapshot_hash}: {e}",
                    ),
                })
            })?;
            Ok(ConfigSnapshot::from_stored_with_hash(
                stored,
                extra_templates,
                tags,
                snapshot_hash,
            ))
        } else {
            // Legacy fallback: row predates `config_jsonb` and the startup
            // backfill hasn't touched it yet (or skipped it on parse error).
            // Deserialize from the deprecated TOML column. After `hash_v2`
            // and the eventual drop of `config TEXT`, this branch goes
            // away.
            let config_toml: String = row.try_get("config")?;
            ConfigSnapshot::from_stored(&config_toml, extra_templates, tags, &snapshot_hash)
        }
    }

    async fn write_config_snapshot(&self, snapshot: &ConfigSnapshot) -> Result<(), DelayedError> {
        let pool = self.get_pool_result()?;

        let config_string = toml::to_string(&snapshot.config).map_err(|e| {
            DelayedError::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize config snapshot: {e}"),
            })
        })?;

        let extra_templates_json =
            serde_json::to_value(&snapshot.extra_templates).map_err(|e| {
                DelayedError::new(ErrorDetails::Serialization {
                    message: e.to_string(),
                })
            })?;
        let tags_json = serde_json::to_value(&snapshot.tags).map_err(|e| {
            DelayedError::new(ErrorDetails::Serialization {
                message: e.to_string(),
            })
        })?;
        // Canonical form of the snapshot — the storage value for the
        // `config_jsonb` column plus the structural identity hash for
        // `canonical_hash`. Derived in one pass via
        // `ConfigSnapshot::to_canonical_form()`. The JSONB column is
        // queryable by per-resource version using the GIN index on
        // `config_jsonb` (e.g. `@> '{"functions":{"foo":{"version":3}}}'`);
        // the hash is the content-addressed identity that lookups
        // dispatch to via `SnapshotHashScheme::Canonical`. The TOML
        // column is kept as a migration artifact only; the canonical
        // form is the source of truth for reads. Hash is stable across
        // serialize/deserialize round-trips, unlike the legacy
        // canonical-TOML-bytes hash in `snapshot.hash`.
        let canonical = snapshot.to_canonical_form().map_err(|e| {
            DelayedError::new(ErrorDetails::Serialization {
                message: format!("Failed to compute canonical form for snapshot write: {e}"),
            })
        })?;

        sqlx::query(
            r"INSERT INTO tensorzero.config_snapshots (hash, config, config_jsonb, canonical_hash, extra_templates, tensorzero_version, tags)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               ON CONFLICT (hash) DO UPDATE SET
                 tags = tensorzero.config_snapshots.tags || EXCLUDED.tags,
                 last_used = NOW(),
                 -- Lazily backfill `config_jsonb` and `canonical_hash` on
                 -- conflict if a prior write left them NULL. We never
                 -- overwrite an already-populated value.
                 config_jsonb = COALESCE(tensorzero.config_snapshots.config_jsonb, EXCLUDED.config_jsonb),
                 canonical_hash = COALESCE(tensorzero.config_snapshots.canonical_hash, EXCLUDED.canonical_hash)",
        )
        .bind(snapshot.hash.as_bytes())
        .bind(&config_string)
        .bind(canonical.as_jsonb())
        .bind(canonical.hash.as_bytes())
        .bind(&extra_templates_json)
        .bind(crate::endpoints::status::TENSORZERO_VERSION)
        .bind(&tags_json)
        .execute(pool)
        .await
        .map_err(|e| {
            DelayedError::new(ErrorDetails::PostgresQuery {
                message: e.to_string(),
            })
        })?;

        Ok(())
    }
}

/// Postgres-side `ConfigSnapshotSearch` implementation. Backed by the GIN
/// index `config_snapshots_config_jsonb_gin` (operator class
/// `jsonb_path_ops`), which makes `@>` containment queries cheap.
///
/// The path-value method is implemented in terms of containment rather than
/// JSONPath: `snapshots_with_path_value("functions/foo/version", json!(3))`
/// builds the nested fragment `{"functions": {"foo": {"version": 3}}}` and
/// delegates to `snapshots_containing`. This trades JSONPath flexibility
/// (no array indexing, no wildcards) for using the same index without a
/// second op-class — adequate for the simple property-path lookups every
/// caller in V0 needs.
#[async_trait]
impl ConfigSnapshotSearch for PostgresConnectionInfo {
    async fn snapshots_containing(
        &self,
        fragment: serde_json::Value,
    ) -> Result<Vec<SnapshotHash>, Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;
        let rows = sqlx::query(
            r"SELECT hash
               FROM tensorzero.config_snapshots
               WHERE config_jsonb @> $1
               ORDER BY created_at ASC, hash ASC",
        )
        .bind(&fragment)
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: e.to_string(),
            })
        })?;

        rows.into_iter()
            .map(|row| {
                let bytes: Vec<u8> = row.try_get("hash")?;
                Ok(SnapshotHash::from_bytes(&bytes))
            })
            .collect()
    }

    async fn snapshots_with_path_value(
        &self,
        path: &str,
        value: &serde_json::Value,
    ) -> Result<Vec<SnapshotHash>, Error> {
        let fragment = build_containment_fragment(path, value);
        self.snapshots_containing(fragment).await
    }
}

/// Build a nested JSON object that, used as the right-hand side of `@>`,
/// matches snapshots whose `config_jsonb` has `value` at `path`.
///
/// Example: `path = "functions/foo/version"`, `value = json!(3)` →
/// `{"functions": {"foo": {"version": 3}}}`.
///
/// `path` segments are split on `/` and `.`; both are accepted because
/// callers from external code paths (REST handlers, UI) will spell paths
/// either way.
fn build_containment_fragment(path: &str, value: &serde_json::Value) -> serde_json::Value {
    let segments: Vec<&str> = path.split(['/', '.']).filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return value.clone();
    }
    let mut current = value.clone();
    for segment in segments.iter().rev() {
        let mut obj = serde_json::Map::with_capacity(1);
        obj.insert((*segment).to_string(), current);
        current = serde_json::Value::Object(obj);
    }
    current
}

/// Populate `config_jsonb` AND `canonical_hash` on every
/// `tensorzero.config_snapshots` row that still has either NULL.
///
/// Runs once per gateway boot, immediately after migrations. Idempotent
/// (the `WHERE config_jsonb IS NULL OR canonical_hash IS NULL` guard
/// makes re-runs and concurrent boots no-ops on already-backfilled rows).
/// A row whose TOML cannot be parsed is logged and skipped rather than
/// failing startup — that lets older snapshots with formatting we no
/// longer accept stay readable for hash lookups even though they will
/// not appear in JSONB-filtered queries or in canonical-hash lookups.
///
/// Uses bytes-prefixed pagination over the `hash` primary key so we never
/// hold more than `BATCH` rows in memory at a time, and so the work is
/// resumable if interrupted.
pub async fn backfill_config_snapshot_jsonb(pool: &PgPool) -> Result<(), Error> {
    const BATCH: i64 = 256;

    let mut last_hash: Option<Vec<u8>> = None;
    let mut total_backfilled: u64 = 0;
    let mut total_skipped: u64 = 0;

    loop {
        let rows = sqlx::query(
            r"SELECT hash, config
               FROM tensorzero.config_snapshots
               WHERE (config_jsonb IS NULL OR canonical_hash IS NULL)
                 AND ($1::bytea IS NULL OR hash > $1)
               ORDER BY hash
               LIMIT $2",
        )
        .bind(last_hash.as_deref())
        .bind(BATCH)
        .fetch_all(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("config_snapshots backfill query failed: {e}"),
            })
        })?;

        if rows.is_empty() {
            break;
        }

        for row in &rows {
            let hash: Vec<u8> = row.try_get("hash").map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("config_snapshots backfill: hash column missing: {e}"),
                })
            })?;
            let config_toml: String = row.try_get("config").map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("config_snapshots backfill: config column missing: {e}"),
                })
            })?;
            last_hash = Some(hash.clone());

            // Parse TOML → StoredConfig. Any failure here means the
            // stored TOML is no longer parseable under the current type
            // definitions; we skip rather than fail startup.
            let stored: StoredConfig = match toml::from_str(&config_toml) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(
                        hash = %hex::encode(&hash),
                        "config_snapshots backfill: skipping unparseable TOML \
                         (file a bug if you don't expect this — backfill is best-effort, \
                         the row stays readable via legacy hash lookup): {e}"
                    );
                    total_skipped += 1;
                    continue;
                }
            };
            let config_jsonb_value = match serde_json::to_value(&stored) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(
                        hash = %hex::encode(&hash),
                        "config_snapshots backfill: skipping un-serializable config \
                         (StoredConfig should always serialize to JSON — please file a bug): {e}"
                    );
                    total_skipped += 1;
                    continue;
                }
            };
            let canonical_hash = match stored.canonical_hash() {
                Ok(h) => h,
                Err(e) => {
                    tracing::error!(
                        hash = %hex::encode(&hash),
                        "config_snapshots backfill: skipping (canonical_hash computation failed \
                         — the algorithm should be infallible for any valid StoredConfig; \
                         please file a bug): {e}"
                    );
                    total_skipped += 1;
                    continue;
                }
            };

            // The `WHERE … IS NULL` guards make the UPDATE a no-op for
            // any column another process has already populated — both
            // columns are populated independently using `COALESCE` so
            // races between two backfills, or between a backfill and a
            // new write, never overwrite a populated value.
            sqlx::query(
                r"UPDATE tensorzero.config_snapshots
                   SET config_jsonb = COALESCE(config_jsonb, $1),
                       canonical_hash = COALESCE(canonical_hash, $2)
                   WHERE hash = $3
                     AND (config_jsonb IS NULL OR canonical_hash IS NULL)",
            )
            .bind(&config_jsonb_value)
            .bind(canonical_hash.as_bytes())
            .bind(&hash)
            .execute(pool)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("config_snapshots backfill UPDATE failed: {e}"),
                })
            })?;
            total_backfilled += 1;
        }

        if rows.len() < BATCH as usize {
            break;
        }
    }

    if total_backfilled > 0 || total_skipped > 0 {
        tracing::info!(
            backfilled = total_backfilled,
            skipped = total_skipped,
            "config_snapshots backfill complete",
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::build_containment_fragment;

    #[test]
    fn build_containment_fragment_simple_path() {
        let frag = build_containment_fragment("functions/foo/version", &serde_json::json!(3));
        assert_eq!(
            frag,
            serde_json::json!({"functions": {"foo": {"version": 3}}}),
        );
    }

    #[test]
    fn build_containment_fragment_dot_path() {
        // Dot syntax is accepted as an alias for `/` because external callers
        // (REST clients, UI) commonly spell paths either way.
        let frag = build_containment_fragment("models.openai.routing", &serde_json::json!(["a"]));
        assert_eq!(
            frag,
            serde_json::json!({"models": {"openai": {"routing": ["a"]}}}),
        );
    }

    #[test]
    fn build_containment_fragment_empty_path_returns_value() {
        let frag = build_containment_fragment("", &serde_json::json!({"any": 1}));
        assert_eq!(frag, serde_json::json!({"any": 1}));
    }

    #[test]
    fn build_containment_fragment_collapses_redundant_separators() {
        let frag = build_containment_fragment("//a//b//", &serde_json::json!(true));
        assert_eq!(frag, serde_json::json!({"a": {"b": true}}));
    }
}
