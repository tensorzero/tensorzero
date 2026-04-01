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
    StoredOptimizerConfig, StoredPostgresConfig, StoredPromptTemplate, StoredProviderTypesConfig,
    StoredRateLimitingConfig, StoredStorageKind,
};
use uuid::Uuid;

use crate::config::UninitializedConfig;
use crate::config::path::ResolvedTomlPathData;
use crate::error::{Error, ErrorDetails};

use super::PostgresConnectionInfo;
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
        let pool = self.get_pool_result()?;
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
    let config = params.config;

    // 1. Singleton tables (append-only)
    if let Some(gateway) = &config.gateway {
        let stored = StoredGatewayConfig::from(gateway.clone());
        insert_singleton_config_row(
            tx,
            "gateway_configs",
            STORED_GATEWAY_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    if let Some(clickhouse) = &config.clickhouse {
        let stored = StoredClickHouseConfig::from(clickhouse);
        insert_singleton_config_row(
            tx,
            "clickhouse_configs",
            STORED_CLICKHOUSE_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    if let Some(postgres) = &config.postgres {
        let stored = StoredPostgresConfig::from(postgres);
        insert_singleton_config_row(
            tx,
            "postgres_configs",
            STORED_POSTGRES_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    if let Some(object_storage) = &config.object_storage {
        let stored = StoredStorageKind::from(object_storage);
        insert_singleton_config_row(
            tx,
            "object_storage_configs",
            STORED_STORAGE_KIND_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    if let Some(rate_limiting) = &config.rate_limiting {
        let stored = StoredRateLimitingConfig::from(rate_limiting);
        insert_singleton_config_row(
            tx,
            "rate_limiting_configs",
            STORED_RATE_LIMITING_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    if let Some(autopilot) = &config.autopilot {
        let stored = StoredAutopilotConfig::from(autopilot);
        insert_singleton_config_row(
            tx,
            "autopilot_configs",
            STORED_AUTOPILOT_CONFIG_SCHEMA_REVISION,
            &serialize_stored(&stored)?,
        )
        .await?;
    }

    if let Some(provider_types) = &config.provider_types {
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
    for (name, model_config) in config.models.as_ref().into_iter().flat_map(|m| m.iter()) {
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

    for (name, embedding_model_config) in config
        .embedding_models
        .as_ref()
        .into_iter()
        .flat_map(|m| m.iter())
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

    for (name, metric_config) in config.metrics.as_ref().into_iter().flat_map(|m| m.iter()) {
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

    for (name, optimizer_info) in config
        .optimizers
        .as_ref()
        .into_iter()
        .flat_map(|m| m.iter())
    {
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

    // 3. Tools (with prompt templates)
    for (name, tool_config) in config.tools.as_ref().into_iter().flat_map(|m| m.iter()) {
        let prompt_template_version_ids = write_prompt_templates_in_tx(
            tx,
            tool_config.prompt_templates_for_db().into_iter(),
            params.creation_source,
            params.source_autopilot_session_id,
        )
        .await?;
        let stored_tool = tool_config.convert_for_db(&prompt_template_version_ids)?;
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

    // 4. Evaluations (with prompt templates)
    for (name, eval_config) in config
        .evaluations
        .as_ref()
        .into_iter()
        .flat_map(|m| m.iter())
    {
        let prompt_template_version_ids = write_prompt_templates_in_tx(
            tx,
            eval_config.prompt_templates_for_db().into_iter(),
            params.creation_source,
            params.source_autopilot_session_id,
        )
        .await?;
        let stored_eval = eval_config.to_stored_for_db(&prompt_template_version_ids)?;
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
    for (function_name, function_config) in
        config.functions.as_ref().into_iter().flat_map(|m| m.iter())
    {
        write_function_config_in_tx(
            tx,
            WriteFunctionConfigParams {
                function_name,
                config: function_config,
                expected_current_version_id: None,
                creation_source: params.creation_source,
                source_autopilot_session_id: params.source_autopilot_session_id,
                extra_templates: params.extra_templates,
            },
        )
        .await?;
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

// ── Prompt-template handling for standalone tools/evaluations ─────────────────

async fn write_prompt_templates_in_tx<'a>(
    tx: &mut Transaction<'_, Postgres>,
    templates: impl Iterator<Item = &'a ResolvedTomlPathData>,
    creation_source: &str,
    source_autopilot_session_id: Option<Uuid>,
) -> Result<HashMap<String, Uuid>, Error> {
    let mut collected = BTreeMap::new();
    for template in templates {
        let template_key = template.get_template_key();
        let source_body = template.data().to_string();
        match collected.get(&template_key) {
            Some((existing_body, _)) if existing_body != &source_body => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Template key `{template_key}` was provided with conflicting source bodies."
                    ),
                }));
            }
            Some(_) => {}
            None => {
                collected.insert(template_key, (source_body, Uuid::now_v7()));
            }
        }
    }

    // Insert template rows
    for (template_key, (source_body, id)) in &collected {
        let content_hash = blake3::hash(source_body.as_bytes()).as_bytes().to_vec();
        let stored_template = StoredPromptTemplate {
            id: *id,
            template_key: template_key.clone(),
            source_body: source_body.clone(),
            content_hash,
            creation_source: creation_source.to_string(),
            source_autopilot_session_id,
        };
        sqlx::query(
            "INSERT INTO tensorzero.prompt_template_configs \
             (id, template_key, source_body, content_hash, creation_source, source_autopilot_session_id) \
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(stored_template.id)
        .bind(&stored_template.template_key)
        .bind(&stored_template.source_body)
        .bind(&stored_template.content_hash)
        .bind(&stored_template.creation_source)
        .bind(stored_template.source_autopilot_session_id)
        .execute(&mut **tx)
        .await
        .map_err(|e| postgres_query_error("Failed to insert prompt template version", e))?;
    }

    // Build result map: template_key -> version_id
    let result = collected
        .into_iter()
        .map(|(key, (_, id))| (key, id))
        .collect();
    Ok(result)
}

fn postgres_query_error(context: &str, e: sqlx::Error) -> Error {
    Error::new(ErrorDetails::PostgresQuery {
        message: format!("{context}: {e}"),
    })
}
