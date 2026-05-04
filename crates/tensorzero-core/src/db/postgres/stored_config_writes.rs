use std::collections::{BTreeMap, HashMap, HashSet};

use sqlx::{Postgres, QueryBuilder, Transaction};
use tensorzero_stored_config::{
    ConfigObjectMetadata, PerObjectMetadata, STORED_AUTOPILOT_CONFIG_SCHEMA_REVISION,
    STORED_CLICKHOUSE_CONFIG_SCHEMA_REVISION, STORED_EMBEDDING_MODEL_CONFIG_SCHEMA_REVISION,
    STORED_EVALUATION_CONFIG_SCHEMA_REVISION, STORED_GATEWAY_CONFIG_SCHEMA_REVISION,
    STORED_METRIC_CONFIG_SCHEMA_REVISION, STORED_MODEL_CONFIG_SCHEMA_REVISION,
    STORED_OPTIMIZER_CONFIG_SCHEMA_REVISION, STORED_POSTGRES_CONFIG_SCHEMA_REVISION,
    STORED_PROVIDER_TYPES_CONFIG_SCHEMA_REVISION, STORED_RATE_LIMITING_CONFIG_SCHEMA_REVISION,
    STORED_STORAGE_KIND_SCHEMA_REVISION, STORED_TOOL_CONFIG_SCHEMA_REVISION, StoredAutopilotConfig,
    StoredClickHouseConfig, StoredEmbeddingModelConfig, StoredGatewayConfig, StoredMetricConfig,
    StoredModelConfig, StoredOptimizerConfig, StoredPostgresConfig, StoredProviderTypesConfig,
    StoredRateLimitingConfig, StoredStorageKind,
};
use uuid::Uuid;

use crate::config::Config;
use crate::config::UninitializedConfig;
use crate::config::path::ResolvedTomlPathData;
use crate::config::unwritten::UnwrittenConfig;
use crate::error::{Error, ErrorDetails};

use super::PostgresConnectionInfo;
use super::file_writes::{CollectedFile, add_file, write_collected_files};
use super::function_config_writes::write_function_config_in_tx_skipping_cas;

#[derive(Debug)]
pub struct WriteStoredConfigParams<'a> {
    pub config: &'a UninitializedConfig,
    pub creation_source: &'a str,
    pub source_autopilot_session_id: Option<Uuid>,
    /// Extra templates discovered from the filesystem (e.g. via `template_filesystem_access`).
    /// All prompts — whether explicitly specified in the config or dynamically included via
    /// MiniJinja `{% include %}` — must be stored in the database for the config to be
    /// self-contained.
    pub extra_templates: &'a HashMap<String, String>,
    /// Per-object descriptive metadata (notes, tags, source-file provenance,
    /// etc.) keyed by canonical object path (`"functions/foo"`,
    /// `"models/gpt-4o"`, `"gateway"`, …). Excluded from snapshot hashes.
    /// Empty by default; populated by the file-mode TOML-comment harvester
    /// and by REST endpoints that surface a Notes / tags UI.
    pub metadata: &'a PerObjectMetadata,
}

impl PostgresConnectionInfo {
    pub async fn write_stored_config(
        &self,
        params: WriteStoredConfigParams<'_>,
    ) -> Result<(), Error> {
        let pool = self.get_pool_result().map_err(|e| e.log())?;
        let mut tx = pool
            .begin()
            .await
            .map_err(|e| postgres_query_error("Failed to start stored config transaction", e))?;

        // `write_stored_config_in_tx` returns the validated `UnwrittenConfig`
        // for callers that want to hot-swap it into a live gateway; the
        // fire-and-forget wrapper has no use for it. `Box::pin` keeps the
        // outer future small (clippy::large_futures).
        let _ = Box::pin(write_stored_config_in_tx(&mut tx, params)).await?;

        tx.commit()
            .await
            .map_err(|e| postgres_query_error("Failed to commit stored config transaction", e))?;
        Ok(())
    }
}

pub(crate) async fn write_stored_config_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    params: WriteStoredConfigParams<'_>,
) -> Result<UnwrittenConfig, Error> {
    // Validate the config before touching any tables so invalid configs
    // (broken references, missing templates, schema errors, etc.) fail fast
    // without partial writes. We also return the resulting `UnwrittenConfig`
    // so callers can use its validated `Config` + hash directly (e.g. for the
    // post-write hot-swap on `--migrate-config`) instead of re-reading +
    // revalidating from the database on the happy path. Credential validation
    // is skipped: the caller is responsible for deciding when to exercise
    // provider credentials, since the write path runs under the global
    // stored-config advisory lock and shouldn't be making outbound network
    // calls.
    let unwritten = Config::load_from_uninitialized(params.config.clone(), false).await?;

    // Acquire a single global advisory lock to serialize concurrent
    // whole-config writes. This mirrors the per-function advisory lock in
    // `write_function_config_in_tx`, and prevents two whole-config writes
    // from interleaving across the many tables this function touches. The
    // per-function lock is still acquired separately inside
    // `write_function_config_in_tx`, so a whole-config write will also
    // conflict with concurrent single-function writes for any function it
    // touches.
    acquire_stored_config_advisory_lock(tx).await?;

    // Exhaustive destructure so the compiler forces us to handle every field
    // when new ones are added to `WriteStoredConfigParams`.
    let WriteStoredConfigParams {
        config,
        creation_source,
        source_autopilot_session_id,
        extra_templates: caller_extra_templates,
        metadata,
    } = params;

    // Merge the caller-provided templates with the ones discovered during
    // validation (e.g. via MiniJinja `{% include %}` from
    // `template_filesystem_access`). Using the validated snapshot's
    // discovered templates ensures the persisted config is self-contained
    // even when callers don't manually enumerate transitive includes — a
    // reload on another gateway (or after a restart without filesystem
    // access) will still find every referenced template.
    let mut merged_extra_templates: HashMap<String, String> = unwritten.extra_templates().clone();
    for (key, body) in caller_extra_templates {
        merged_extra_templates
            .entry(key.clone())
            .or_insert_with(|| body.clone());
    }
    let extra_templates = &merged_extra_templates;

    // Exhaustive destructure of `UninitializedConfig` so the compiler forces
    // us to handle every section when new ones are added.
    let UninitializedConfig {
        gateway,
        clickhouse,
        postgres,
        rate_limiting,
        object_storage,
        models,
        embedding_models,
        functions,
        metrics,
        tools,
        evaluations,
        provider_types,
        optimizers,
        autopilot,
    } = config;

    // 1. Singleton tables (append-only). Each singleton's metadata is
    //    looked up by its top-level path (`"gateway"`, `"clickhouse"`, …).
    if let Some(gateway) = gateway {
        let stored = StoredGatewayConfig::from(gateway.clone());
        insert_singleton_config_row(
            tx,
            "gateway_configs",
            STORED_GATEWAY_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
            &metadata_for(metadata, "gateway"),
        )
        .await?;
    }

    if let Some(clickhouse) = clickhouse {
        let stored = StoredClickHouseConfig::from(clickhouse);
        insert_singleton_config_row(
            tx,
            "clickhouse_configs",
            STORED_CLICKHOUSE_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
            &metadata_for(metadata, "clickhouse"),
        )
        .await?;
    }

    if let Some(postgres) = postgres {
        let stored = StoredPostgresConfig::from(postgres);
        insert_singleton_config_row(
            tx,
            "postgres_configs",
            STORED_POSTGRES_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
            &metadata_for(metadata, "postgres"),
        )
        .await?;
    }

    if let Some(object_storage) = object_storage {
        let stored = StoredStorageKind::from(object_storage);
        insert_singleton_config_row(
            tx,
            "object_storage_configs",
            STORED_STORAGE_KIND_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
            &metadata_for(metadata, "object_storage"),
        )
        .await?;
    }

    if let Some(rate_limiting) = rate_limiting {
        let stored = StoredRateLimitingConfig::from(rate_limiting);
        insert_singleton_config_row(
            tx,
            "rate_limiting_configs",
            STORED_RATE_LIMITING_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
            &metadata_for(metadata, "rate_limiting"),
        )
        .await?;
    }

    if let Some(autopilot) = autopilot {
        let stored = StoredAutopilotConfig::from(autopilot);
        insert_singleton_config_row(
            tx,
            "autopilot_configs",
            STORED_AUTOPILOT_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
            &metadata_for(metadata, "autopilot"),
        )
        .await?;
    }

    if let Some(provider_types) = provider_types {
        let stored = StoredProviderTypesConfig::from(provider_types);
        insert_singleton_config_row(
            tx,
            "provider_types_configs",
            STORED_PROVIDER_TYPES_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
            &metadata_for(metadata, "provider_types"),
        )
        .await?;
    }

    // 2. Named collection tables (upsert-by-name, tombstone removals).
    //
    // For each named collection we (a) upsert every row present in the new
    // config — which also revives any previously-tombstoned rows with the
    // same name via the `deleted_at = NULL` clause in `upsert_named_config_rows`
    // — and then (b) tombstone anything that was in the DB but is not in the
    // new config. This is what makes "remove a tool/model/etc. from the TOML
    // and apply" actually delete it on reload. The whole sequence is safe to
    // run inside the apply transaction because the caller holds the global
    // `config_editor` advisory lock and has already verified the full-TOML
    // signature against the user's base snapshot.
    let models_new_names = write_named_section(
        tx,
        "models_configs",
        "models",
        STORED_MODEL_CONFIG_SCHEMA_REVISION,
        models.as_ref().into_iter().flat_map(|m| m.iter()),
        |model_config| serialize_stored(&StoredModelConfig::try_from(model_config)?),
        metadata,
    )
    .await?;
    tombstone_removed_names(tx, "models_configs", &models_new_names).await?;

    let embedding_models_new_names = write_named_section(
        tx,
        "embedding_models_configs",
        "embedding_models",
        STORED_EMBEDDING_MODEL_CONFIG_SCHEMA_REVISION,
        embedding_models.as_ref().into_iter().flat_map(|m| m.iter()),
        |embedding_model_config| {
            serialize_stored(&StoredEmbeddingModelConfig::try_from(
                embedding_model_config,
            )?)
        },
        metadata,
    )
    .await?;
    tombstone_removed_names(tx, "embedding_models_configs", &embedding_models_new_names).await?;

    let metrics_new_names = write_named_section(
        tx,
        "metrics_configs",
        "metrics",
        STORED_METRIC_CONFIG_SCHEMA_REVISION,
        metrics.as_ref().into_iter().flat_map(|m| m.iter()),
        |metric_config| serialize_stored(&StoredMetricConfig::from(metric_config)),
        metadata,
    )
    .await?;
    tombstone_removed_names(tx, "metrics_configs", &metrics_new_names).await?;

    let optimizers_new_names = write_named_section(
        tx,
        "optimizers_configs",
        "optimizers",
        STORED_OPTIMIZER_CONFIG_SCHEMA_REVISION,
        optimizers.as_ref().into_iter().flat_map(|m| m.iter()),
        |optimizer_info| serialize_stored(&StoredOptimizerConfig::from(optimizer_info.clone())),
        metadata,
    )
    .await?;
    tombstone_removed_names(tx, "optimizers_configs", &optimizers_new_names).await?;

    // 3. Tools (with stored files)
    let mut tools_new_names: HashSet<String> = HashSet::new();
    let mut tool_rows: Vec<(String, serde_json::Value, serde_json::Value)> = Vec::new();
    for (name, tool_config) in tools.as_ref().into_iter().flat_map(|m| m.iter()) {
        let file_version_ids = write_files_in_tx(
            tx,
            std::iter::once(&tool_config.parameters),
            creation_source,
            source_autopilot_session_id,
        )
        .await?;
        let stored_tool = tool_config.convert_for_db(&file_version_ids)?;
        let config_json = serde_json::to_value(&stored_tool).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize tool `{name}` for DB: {e}"),
            })
        })?;
        let row_metadata = metadata_for(metadata, &format!("tools/{name}"));
        tool_rows.push((name.clone(), config_json, row_metadata));
        tools_new_names.insert(name.clone());
    }
    upsert_named_config_rows(
        tx,
        "tools_configs",
        STORED_TOOL_CONFIG_SCHEMA_REVISION,
        &tool_rows,
    )
    .await?;
    tombstone_removed_names(tx, "tools_configs", &tools_new_names).await?;

    // 4. Evaluations (with stored files)
    let mut evaluations_new_names: HashSet<String> = HashSet::new();
    let mut evaluation_rows: Vec<(String, serde_json::Value, serde_json::Value)> = Vec::new();
    for (name, eval_config) in evaluations.as_ref().into_iter().flat_map(|m| m.iter()) {
        let file_version_ids = write_files_in_tx(
            tx,
            eval_config.files_for_db().into_iter(),
            creation_source,
            source_autopilot_session_id,
        )
        .await?;
        let stored_eval = eval_config.to_stored_for_db(&file_version_ids)?;
        let config_json = serde_json::to_value(&stored_eval).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize evaluation `{name}` for DB: {e}"),
            })
        })?;
        let row_metadata = metadata_for(metadata, &format!("evaluations/{name}"));
        evaluation_rows.push((name.clone(), config_json, row_metadata));
        evaluations_new_names.insert(name.clone());
    }
    upsert_named_config_rows(
        tx,
        "evaluations_configs",
        STORED_EVALUATION_CONFIG_SCHEMA_REVISION,
        &evaluation_rows,
    )
    .await?;
    tombstone_removed_names(tx, "evaluations_configs", &evaluations_new_names).await?;

    // 5. Functions.
    //
    // We use the skipping-CAS variant here because this bulk path always runs
    // under the global `config_editor` advisory lock and only after the
    // apply-TOML handler has verified the full-TOML signature matches the
    // caller's base snapshot. See `write_function_config_in_tx_skipping_cas`
    // for the full invariant. A per-function CAS would require knowing the
    // current version ID of every existing function, which this bulk path
    // has no way to provide since it only sees the `UninitializedConfig`.
    let mut functions_new_names: HashSet<String> = HashSet::new();
    for (function_name, function_config) in functions.as_ref().into_iter().flat_map(|m| m.iter()) {
        // Lift any per-object metadata for this function and its variants
        // out of the `PerObjectMetadata` map. Variant metadata keys are
        // `functions/{name}/variants/{variant}` per the canonical path scheme.
        let function_metadata = metadata
            .get(&format!("functions/{function_name}"))
            .cloned()
            .unwrap_or_default();
        let mut variant_metadata: BTreeMap<String, ConfigObjectMetadata> = BTreeMap::new();
        let variant_prefix = format!("functions/{function_name}/variants/");
        for (path, m) in metadata {
            if let Some(variant_name) = path.strip_prefix(&variant_prefix) {
                variant_metadata.insert(variant_name.to_string(), m.clone());
            }
        }
        // This bulk path is the single approved caller of the skipping-CAS
        // variant (enforced via `disallowed-methods` in `clippy.toml`).
        #[expect(clippy::disallowed_methods)]
        write_function_config_in_tx_skipping_cas(
            tx,
            function_name,
            function_config,
            creation_source,
            source_autopilot_session_id,
            extra_templates,
            &function_metadata,
            &variant_metadata,
        )
        .await?;
        functions_new_names.insert(function_name.clone());
    }
    let active_function_names = load_active_function_names(tx).await?;
    let removed_functions: Vec<String> = active_function_names
        .difference(&functions_new_names)
        .cloned()
        .collect();
    tombstone_function_rows(tx, &removed_functions).await?;

    Ok(unwritten)
}

// ── Advisory lock ─────────────────────────────────────────────────────────────

/// Fixed advisory-lock key for whole-config writes. Derived once from the
/// BLAKE3 hash of `"tensorzero::stored_config::global"` (first 8 bytes
/// interpreted as a little-endian `i64`). Using a single fixed key means
/// every whole-config writer agrees on the same lock.
static STORED_CONFIG_ADVISORY_LOCK_KEY: std::sync::LazyLock<i64> = std::sync::LazyLock::new(|| {
    // BLAKE3 output is always 32 bytes; read the first 8 as a little-endian i64.
    let hash = blake3::hash(b"tensorzero::stored_config::global");
    let bytes: [u8; 8] = hash.as_bytes()[..8].try_into().unwrap_or([0u8; 8]);
    i64::from_le_bytes(bytes)
});

/// Acquires a transaction-level exclusive advisory lock that serializes all
/// concurrent whole-config writes. The lock is released automatically when
/// the transaction ends.
async fn acquire_stored_config_advisory_lock(
    tx: &mut Transaction<'_, Postgres>,
) -> Result<(), Error> {
    let acquired: bool = sqlx::query_scalar("SELECT pg_try_advisory_xact_lock($1)")
        .bind(*STORED_CONFIG_ADVISORY_LOCK_KEY)
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| {
            postgres_query_error("Failed to acquire advisory lock for stored config", e)
        })?;
    if !acquired {
        return Err(Error::new(ErrorDetails::PostgresQuery {
            message:
                "Failed to lock stored config for update; another client is writing the config. Please retry."
                    .to_string(),
        }));
    }
    Ok(())
}

/// Upserts every `(name, config)` pair in `items` into `table_name` in a
/// single batched query and returns the set of names that were written —
/// the caller uses this set as the "new" side of the diff when tombstoning
/// removed rows.
///
/// `metadata_path_prefix` (e.g. `"models"`, `"tools"`) is the canonical
/// section path; per-row metadata is looked up at `<prefix>/<name>` in
/// `per_object_metadata`.
async fn write_named_section<'a, K, T, I, F>(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &'static str,
    metadata_path_prefix: &str,
    schema_revision: i32,
    items: I,
    mut serialize: F,
    per_object_metadata: &PerObjectMetadata,
) -> Result<HashSet<String>, Error>
where
    K: AsRef<str> + 'a,
    T: 'a,
    I: Iterator<Item = (&'a K, &'a T)>,
    F: FnMut(&T) -> Result<serde_json::Value, Error>,
{
    let mut new_names: HashSet<String> = HashSet::new();
    let mut rows: Vec<(String, serde_json::Value, serde_json::Value)> = Vec::new();
    for (name, item) in items {
        let config_json = serialize(item)?;
        let name_str = name.as_ref().to_string();
        let row_metadata = metadata_for(
            per_object_metadata,
            &format!("{metadata_path_prefix}/{name_str}"),
        );
        new_names.insert(name_str.clone());
        rows.push((name_str, config_json, row_metadata));
    }
    upsert_named_config_rows(tx, table_name, schema_revision, &rows).await?;
    Ok(new_names)
}

/// Looks up metadata for a canonical object path (e.g. `"functions/foo"`,
/// `"gateway"`) in the provided map and serializes it as a `serde_json::Value`
/// suitable for binding into the `metadata JSONB` column.
///
/// Returns `'{}'` (an empty JSON object) when the path is absent from the
/// map — matching the column default in Postgres so a row that has no
/// associated metadata stays trivially equivalent to a default-row insert.
fn metadata_for(map: &PerObjectMetadata, path: &str) -> serde_json::Value {
    match map.get(path) {
        Some(m) => serde_json::to_value(m).unwrap_or_else(|_| serde_json::json!({})),
        None => serde_json::json!({}),
    }
}

/// Tombstones rows in `table_name` whose `name` is in the DB's active set
/// but not in `new_names` (i.e. names the user removed from the TOML on
/// this apply).
async fn tombstone_removed_names(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &'static str,
    new_names: &HashSet<String>,
) -> Result<(), Error> {
    let current_names = load_active_named_config_names(tx, table_name).await?;
    let to_delete: Vec<String> = current_names.difference(new_names).cloned().collect();
    tombstone_named_config_rows(tx, table_name, &to_delete).await
}

// ── SQL helpers ───────────────────────────────────────────────────────────────

fn serialize_stored(value: &impl serde::Serialize) -> Result<serde_json::Value, Error> {
    serde_json::to_value(value).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize stored config for DB: {e}"),
        })
    })
}

async fn insert_singleton_config_row(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &str,
    schema_revision: i32,
    config_json: &serde_json::Value,
    metadata: &serde_json::Value,
) -> Result<(), Error> {
    let id = Uuid::now_v7();
    let mut qb = QueryBuilder::new("INSERT INTO tensorzero.");
    qb.push(table_name);
    qb.push(" (id, schema_revision, config, metadata) VALUES (");
    qb.push_bind(id);
    qb.push(", ");
    qb.push_bind(schema_revision);
    qb.push(", ");
    qb.push_bind(config_json.clone());
    qb.push(", ");
    qb.push_bind(metadata.clone());
    qb.push(")");
    qb.build().execute(&mut **tx).await.map_err(|e| {
        postgres_query_error(
            &format!("Failed to insert singleton config row into `{table_name}`"),
            e,
        )
    })?;
    Ok(())
}

/// Upserts a batch of `(name, config, metadata)` rows into a named
/// collection table in a single query. Clears `deleted_at` on conflict so
/// that re-adding a previously tombstoned name revives its row.
///
/// On conflict, `metadata` is overwritten with the EXCLUDED value (the
/// caller's freshly-provided metadata). This is the right behavior for
/// file-mode applies (TOML-comment changes propagate) and for explicit
/// REST PATCHes (the caller is the source of truth). If you need
/// "preserve existing metadata on update" behavior, do the lookup +
/// merge above this layer.
async fn upsert_named_config_rows(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &str,
    schema_revision: i32,
    rows: &[(String, serde_json::Value, serde_json::Value)],
) -> Result<(), Error> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut qb = QueryBuilder::new("INSERT INTO tensorzero.");
    qb.push(table_name);
    qb.push(" (id, name, schema_revision, config, metadata) ");
    qb.push_values(rows, |mut b, (name, config_json, metadata)| {
        b.push_bind(Uuid::now_v7())
            .push_bind(name.clone())
            .push_bind(schema_revision)
            .push_bind(config_json.clone())
            .push_bind(metadata.clone());
    });
    qb.push(" ON CONFLICT (name) DO UPDATE SET schema_revision = EXCLUDED.schema_revision, config = EXCLUDED.config, metadata = EXCLUDED.metadata, updated_at = NOW(), deleted_at = NULL");
    qb.build().execute(&mut **tx).await.map_err(|e| {
        postgres_query_error(
            &format!("Failed to upsert named config rows into `{table_name}`"),
            e,
        )
    })?;
    Ok(())
}

/// Returns the set of currently-active (`deleted_at IS NULL`) names in a
/// named collection table.
async fn load_active_named_config_names(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &str,
) -> Result<HashSet<String>, Error> {
    let mut qb = QueryBuilder::<Postgres>::new("SELECT name FROM tensorzero.");
    qb.push(table_name);
    qb.push(" WHERE deleted_at IS NULL");
    let names: Vec<String> = qb
        .build_query_scalar()
        .fetch_all(&mut **tx)
        .await
        .map_err(|e| {
            postgres_query_error(
                &format!("Failed to load active names from `{table_name}`"),
                e,
            )
        })?;
    Ok(names.into_iter().collect())
}

/// Tombstones the rows in a named collection table whose `name` is in
/// `names_to_delete`. Named collection tables use `UNIQUE(name)`, so each
/// name has exactly one row, and tombstoning is a simple `UPDATE`.
async fn tombstone_named_config_rows(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &str,
    names_to_delete: &[String],
) -> Result<(), Error> {
    if names_to_delete.is_empty() {
        return Ok(());
    }
    let mut qb = QueryBuilder::<Postgres>::new("UPDATE tensorzero.");
    qb.push(table_name);
    qb.push(" SET deleted_at = NOW() WHERE deleted_at IS NULL AND name = ANY(");
    qb.push_bind(names_to_delete);
    qb.push(")");
    qb.build().execute(&mut **tx).await.map_err(|e| {
        postgres_query_error(
            &format!("Failed to tombstone removed rows in `{table_name}`"),
            e,
        )
    })?;
    Ok(())
}

/// Returns the set of function names whose most recent version is active
/// (`deleted_at IS NULL`). Functions are append-only with multiple versions
/// per name, so "active" means the latest version per name is not tombstoned.
async fn load_active_function_names(
    tx: &mut Transaction<'_, Postgres>,
) -> Result<HashSet<String>, Error> {
    let names: Vec<String> = sqlx::query_scalar(
        r"
        SELECT name
        FROM (
            SELECT DISTINCT ON (name) name, deleted_at
            FROM tensorzero.function_configs
            ORDER BY name ASC, created_at DESC, id DESC
        ) latest
        WHERE deleted_at IS NULL
        ",
    )
    .fetch_all(&mut **tx)
    .await
    .map_err(|e| postgres_query_error("Failed to load active function names", e))?;
    Ok(names.into_iter().collect())
}

/// Tombstones the latest version row for each function name in
/// `names_to_delete`. Uses `UPDATE` on the most-recent row per name rather
/// than `INSERT`-of-tombstone so we don't have to synthesize a fake
/// `function_type` / `config` for the tombstone row.
async fn tombstone_function_rows(
    tx: &mut Transaction<'_, Postgres>,
    names_to_delete: &[String],
) -> Result<(), Error> {
    if names_to_delete.is_empty() {
        return Ok(());
    }
    sqlx::query(
        r"
        UPDATE tensorzero.function_configs
        SET deleted_at = NOW()
        WHERE id IN (
            SELECT DISTINCT ON (name) id
            FROM tensorzero.function_configs
            WHERE name = ANY($1)
            ORDER BY name ASC, created_at DESC, id DESC
        )
        ",
    )
    .bind(names_to_delete)
    .execute(&mut **tx)
    .await
    .map_err(|e| postgres_query_error("Failed to tombstone removed function rows", e))?;
    Ok(())
}

// ── Stored file handling for standalone tools/evaluations ─────────────────────

/// Collect stored files for a standalone tool/evaluation config and
/// persist them via the shared writer, reusing existing rows that already
/// match `(file_path, content_hash)`.
async fn write_files_in_tx<'a>(
    tx: &mut Transaction<'_, Postgres>,
    templates: impl Iterator<Item = &'a ResolvedTomlPathData>,
    creation_source: &str,
    source_autopilot_session_id: Option<Uuid>,
) -> Result<HashMap<String, Uuid>, Error> {
    let mut collected: BTreeMap<String, CollectedFile> = BTreeMap::new();
    for template in templates {
        add_file(&mut collected, template)?;
    }
    write_collected_files(tx, &collected, creation_source, source_autopilot_session_id).await
}

fn postgres_query_error(context: &str, e: sqlx::Error) -> Error {
    Error::new(ErrorDetails::PostgresQuery {
        message: format!("{context}: {e}"),
    })
}
