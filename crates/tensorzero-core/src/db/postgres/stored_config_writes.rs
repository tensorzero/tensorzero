use std::collections::{BTreeMap, HashMap};

use sqlx::{Postgres, QueryBuilder, Transaction};
use tensorzero_stored_config::{
    STORED_AUTOPILOT_CONFIG_SCHEMA_REVISION, STORED_CLICKHOUSE_CONFIG_SCHEMA_REVISION,
    STORED_EMBEDDING_MODEL_CONFIG_SCHEMA_REVISION, STORED_EVALUATION_CONFIG_SCHEMA_REVISION,
    STORED_GATEWAY_CONFIG_SCHEMA_REVISION, STORED_METRIC_CONFIG_SCHEMA_REVISION,
    STORED_MODEL_CONFIG_SCHEMA_REVISION, STORED_OPTIMIZER_CONFIG_SCHEMA_REVISION,
    STORED_POSTGRES_CONFIG_SCHEMA_REVISION, STORED_PROVIDER_TYPES_CONFIG_SCHEMA_REVISION,
    STORED_RATE_LIMITING_CONFIG_SCHEMA_REVISION, STORED_STORAGE_KIND_SCHEMA_REVISION,
    STORED_TOOL_CONFIG_SCHEMA_REVISION, StoredAutopilotConfig, StoredClickHouseConfig,
    StoredEmbeddingModelConfig, StoredGatewayConfig, StoredMetricConfig, StoredModelConfig,
    StoredOptimizerConfig, StoredPostgresConfig, StoredProviderTypesConfig,
    StoredRateLimitingConfig, StoredStorageKind,
};
use uuid::Uuid;

use crate::config::UninitializedConfig;
use crate::config::path::ResolvedTomlPathData;
use crate::error::{Error, ErrorDetails};

use super::PostgresConnectionInfo;
use super::file_writes::{CollectedFile, add_file, write_collected_files};
use super::function_config_writes::{WriteFunctionConfigParams, write_function_config_in_tx};

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

        write_stored_config_in_tx(&mut tx, params).await?;

        tx.commit()
            .await
            .map_err(|e| postgres_query_error("Failed to commit stored config transaction", e))?;
        Ok(())
    }
}

async fn write_stored_config_in_tx(
    tx: &mut Transaction<'_, Postgres>,
    params: WriteStoredConfigParams<'_>,
) -> Result<(), Error> {
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
        extra_templates,
    } = params;

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

    // 1. Singleton tables (append-only)
    if let Some(gateway) = gateway {
        let stored = StoredGatewayConfig::from(gateway.clone());
        insert_singleton_config_row(
            tx,
            "gateway_configs",
            STORED_GATEWAY_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
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
        )
        .await?;
    }

    // 2. Named collection tables (upsert-by-name)
    for (name, model_config) in models.as_ref().into_iter().flat_map(|m| m.iter()) {
        let stored = StoredModelConfig::try_from(model_config)?;
        upsert_named_config_row(
            tx,
            "models_configs",
            STORED_MODEL_CONFIG_SCHEMA_REVISION,
            name,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    for (name, embedding_model_config) in
        embedding_models.as_ref().into_iter().flat_map(|m| m.iter())
    {
        let stored = StoredEmbeddingModelConfig::try_from(embedding_model_config)?;
        upsert_named_config_row(
            tx,
            "embedding_models_configs",
            STORED_EMBEDDING_MODEL_CONFIG_SCHEMA_REVISION,
            name,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    for (name, metric_config) in metrics.as_ref().into_iter().flat_map(|m| m.iter()) {
        let stored = StoredMetricConfig::from(metric_config);
        upsert_named_config_row(
            tx,
            "metrics_configs",
            STORED_METRIC_CONFIG_SCHEMA_REVISION,
            name,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    for (name, optimizer_info) in optimizers.as_ref().into_iter().flat_map(|m| m.iter()) {
        let stored = StoredOptimizerConfig::from(optimizer_info.clone());
        upsert_named_config_row(
            tx,
            "optimizers_configs",
            STORED_OPTIMIZER_CONFIG_SCHEMA_REVISION,
            name,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    // 3. Tools (with stored files)
    for (name, tool_config) in tools.as_ref().into_iter().flat_map(|m| m.iter()) {
        let file_version_ids = write_files_in_tx(
            tx,
            tool_config.files_for_db().into_iter(),
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
        upsert_named_config_row(
            tx,
            "tools_configs",
            STORED_TOOL_CONFIG_SCHEMA_REVISION,
            name,
            &config_json,
        )
        .await?;
    }

    // 4. Evaluations (with stored files)
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
        upsert_named_config_row(
            tx,
            "evaluations_configs",
            STORED_EVALUATION_CONFIG_SCHEMA_REVISION,
            name,
            &config_json,
        )
        .await?;
    }

    // 5. Functions (via existing write_function_config_in_tx)
    for (function_name, function_config) in functions.as_ref().into_iter().flat_map(|m| m.iter()) {
        write_function_config_in_tx(
            tx,
            WriteFunctionConfigParams {
                function_name,
                config: function_config,
                expected_current_version_id: None,
                creation_source,
                source_autopilot_session_id,
                extra_templates,
            },
        )
        .await?;
    }

    Ok(())
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
) -> Result<(), Error> {
    let id = Uuid::now_v7();
    let mut qb = QueryBuilder::new("INSERT INTO tensorzero.");
    qb.push(table_name);
    qb.push(" (id, schema_revision, config) VALUES (");
    qb.push_bind(id);
    qb.push(", ");
    qb.push_bind(schema_revision);
    qb.push(", ");
    qb.push_bind(config_json.clone());
    qb.push(")");
    qb.build().execute(&mut **tx).await.map_err(|e| {
        postgres_query_error(
            &format!("Failed to insert singleton config row into `{table_name}`"),
            e,
        )
    })?;
    Ok(())
}

async fn upsert_named_config_row(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &str,
    schema_revision: i32,
    name: &str,
    config_json: &serde_json::Value,
) -> Result<(), Error> {
    let id = Uuid::now_v7();
    let mut qb = QueryBuilder::new("INSERT INTO tensorzero.");
    qb.push(table_name);
    qb.push(" (id, name, schema_revision, config) VALUES (");
    qb.push_bind(id);
    qb.push(", ");
    qb.push_bind(name.to_string());
    qb.push(", ");
    qb.push_bind(schema_revision);
    qb.push(", ");
    qb.push_bind(config_json.clone());
    qb.push(") ON CONFLICT (name) DO UPDATE SET schema_revision = EXCLUDED.schema_revision, config = EXCLUDED.config, updated_at = NOW()");
    qb.build().execute(&mut **tx).await.map_err(|e| {
        postgres_query_error(
            &format!("Failed to upsert named config row into `{table_name}` for `{name}`"),
            e,
        )
    })?;
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
