use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use sqlx::{Executor, FromRow, PgPool, Postgres, QueryBuilder, Transaction};
use tensorzero_stored_config::schema_dispatch::{
    deserialize_autopilot_config, deserialize_clickhouse_config,
    deserialize_embedding_model_config, deserialize_evaluation_config, deserialize_function_config,
    deserialize_gateway_config, deserialize_metric_config, deserialize_model_config,
    deserialize_optimizer_config, deserialize_postgres_config, deserialize_provider_types_config,
    deserialize_rate_limiting_config, deserialize_storage_kind, deserialize_tool_config,
    deserialize_variant_config,
};
use tensorzero_stored_config::{
    StoredEvaluationConfig, StoredEvaluatorConfig, StoredFile, StoredFileRef, StoredFunctionConfig,
    StoredLLMJudgeConfig, StoredLLMJudgeVariantConfig, StoredToolConfig, StoredVariantConfig,
    StoredVariantVersionConfig,
};
use uuid::Uuid;

use crate::config::rehydrate::{FileMap, rehydrate_evaluation, rehydrate_function, rehydrate_tool};
use crate::config::{UninitializedConfig, validate_user_config_names};
use crate::error::{Error, ErrorDetails};

#[derive(Clone, Debug, FromRow)]
struct VersionedConfigRow {
    schema_revision: i32,
    config: serde_json::Value,
}

#[derive(Clone, Debug, FromRow)]
struct NamedVersionedConfigRow {
    name: String,
    schema_revision: i32,
    config: serde_json::Value,
}

#[derive(Clone, Debug, FromRow)]
struct FunctionConfigRow {
    id: Uuid,
    name: String,
    deleted_at: Option<chrono::DateTime<chrono::Utc>>,
    #[expect(dead_code)]
    function_type: String,
    schema_revision: i32,
    config: serde_json::Value,
}

#[derive(Clone, Debug, FromRow)]
struct VariantConfigRow {
    id: Uuid,
    name: String,
    schema_revision: i32,
    config: serde_json::Value,
}

#[derive(Clone, Debug, FromRow)]
struct StoredFileRow {
    id: Uuid,
    file_path: String,
    source_body: String,
    content_hash: Vec<u8>,
    creation_source: String,
    source_autopilot_session_id: Option<Uuid>,
}

fn log_collection_skip(kind: &str, name: &str, error: &impl std::fmt::Display) {
    tracing::error!("Skipping {kind} `{name}` from DB config: {error}");
}

/// Open a new REPEATABLE READ READ ONLY transaction on its own connection and
/// import the given exported snapshot id, so the resulting transaction shares
/// a snapshot with the leader transaction that called `pg_export_snapshot()`.
///
/// `pg_export_snapshot()` returns an opaque identifier composed only of digits
/// and hyphens (e.g. `00000004-00000C53-1`), so interpolating it into the SQL
/// is safe — `SET TRANSACTION SNAPSHOT` does not accept bind parameters. We
/// still defensively reject any other characters before formatting.
async fn begin_snapshot_read_tx<'a>(
    pool: &'a PgPool,
    snapshot_id: &str,
) -> Result<Transaction<'a, Postgres>, Error> {
    if !snapshot_id
        .chars()
        .all(|c| c.is_ascii_hexdigit() || c == '-')
    {
        return Err(Error::new(ErrorDetails::Config {
            message: format!("Refusing to use unexpected snapshot id `{snapshot_id}`"),
        }));
    }
    let mut tx = pool.begin().await.map_err(|error| {
        Error::new(ErrorDetails::Config {
            message: format!("Failed to start config snapshot read transaction: {error}"),
        })
    })?;
    sqlx::query("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        .execute(&mut *tx)
        .await
        .map_err(|error| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to set REPEATABLE READ isolation level: {error}"),
            })
        })?;
    let mut set_snapshot = QueryBuilder::<Postgres>::new("SET TRANSACTION SNAPSHOT '");
    set_snapshot.push(snapshot_id).push("'");
    set_snapshot
        .build()
        .execute(&mut *tx)
        .await
        .map_err(|error| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to import snapshot `{snapshot_id}`: {error}"),
            })
        })?;
    Ok(tx)
}

fn schema_dispatch_error(error: impl std::fmt::Display) -> Error {
    Error::new(ErrorDetails::Config {
        message: error.to_string(),
    })
}

async fn load_latest_singleton<'e, E>(
    executor: E,
    table_name: &'static str,
) -> Result<Option<VersionedConfigRow>, Error>
where
    E: Executor<'e, Database = Postgres>,
{
    let mut query =
        QueryBuilder::<Postgres>::new("SELECT schema_revision, config FROM tensorzero.");
    query
        .push(table_name)
        .push(" ORDER BY created_at DESC, id DESC LIMIT 1");
    Ok(query.build_query_as().fetch_optional(executor).await?)
}

async fn load_named_collection<'e, E>(
    executor: E,
    table_name: &'static str,
) -> Result<Vec<NamedVersionedConfigRow>, Error>
where
    E: Executor<'e, Database = Postgres>,
{
    // Each row is uniquely keyed by `name` (UNIQUE constraint + upsert
    // semantics on the write side), so a simple filtered SELECT suffices.
    // Rows tombstoned via `deleted_at` are skipped — they represent names
    // the user removed from the TOML on the last full-config apply.
    let mut query =
        QueryBuilder::<Postgres>::new("SELECT name, schema_revision, config FROM tensorzero.");
    query
        .push(table_name)
        .push(" WHERE deleted_at IS NULL ORDER BY name ASC");
    Ok(query.build_query_as().fetch_all(executor).await?)
}

async fn load_latest_functions<'e, E>(executor: E) -> Result<Vec<FunctionConfigRow>, Error>
where
    E: Executor<'e, Database = Postgres>,
{
    Ok(sqlx::query_as::<_, FunctionConfigRow>(
        r"
        SELECT DISTINCT ON (name) id, name, deleted_at, function_type, schema_revision, config
        FROM tensorzero.function_configs
        ORDER BY name ASC
               , created_at DESC
               , id DESC
        ",
    )
    .fetch_all(executor)
    .await?)
}

async fn load_variant_versions<'e, E>(
    executor: E,
    ids: &[Uuid],
) -> Result<Vec<VariantConfigRow>, Error>
where
    E: Executor<'e, Database = Postgres>,
{
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    Ok(sqlx::query_as::<_, VariantConfigRow>(
        r"
        SELECT id, name, schema_revision, config
        FROM tensorzero.variant_configs
        WHERE id = ANY($1)
        ",
    )
    .bind(ids)
    .fetch_all(executor)
    .await?)
}

async fn load_files<'e, E>(executor: E, ids: &[Uuid]) -> Result<Vec<StoredFileRow>, Error>
where
    E: Executor<'e, Database = Postgres>,
{
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    Ok(sqlx::query_as::<_, StoredFileRow>(
        r"
        SELECT id, file_path, source_body, content_hash, creation_source, source_autopilot_session_id
        FROM tensorzero.stored_files
        WHERE id = ANY($1)
        ",
    )
    .bind(ids)
    .fetch_all(executor)
    .await?)
}

fn push_file_ref_ids(file_ids: &mut HashSet<Uuid>, file_ref: &StoredFileRef) {
    file_ids.insert(file_ref.file_version_id);
}

fn collect_evaluator_file_ids(
    evaluators: Option<&BTreeMap<String, StoredEvaluatorConfig>>,
    file_ids: &mut HashSet<Uuid>,
) {
    let Some(evaluators) = evaluators else {
        return;
    };

    for evaluator in evaluators.values() {
        match evaluator {
            StoredEvaluatorConfig::LLMJudge(StoredLLMJudgeConfig {
                variants: Some(variants),
                ..
            }) => {
                for variant in variants.values() {
                    collect_llm_judge_file_ids(&variant.variant, file_ids);
                }
            }
            StoredEvaluatorConfig::LLMJudge(_)
            | StoredEvaluatorConfig::ExactMatch(_)
            | StoredEvaluatorConfig::ToolUse(_)
            | StoredEvaluatorConfig::Regex(_) => {}
        }
    }
}

fn collect_llm_judge_file_ids(variant: &StoredLLMJudgeVariantConfig, file_ids: &mut HashSet<Uuid>) {
    match variant {
        StoredLLMJudgeVariantConfig::ChatCompletion(chat) => {
            push_file_ref_ids(file_ids, &chat.system_instructions);
        }
        StoredLLMJudgeVariantConfig::BestOfNSampling(best_of_n) => {
            push_file_ref_ids(file_ids, &best_of_n.evaluator.system_instructions);
        }
        StoredLLMJudgeVariantConfig::MixtureOfNSampling(mixture_of_n) => {
            push_file_ref_ids(file_ids, &mixture_of_n.fuser.system_instructions);
        }
        StoredLLMJudgeVariantConfig::Dicl(dicl) => {
            if let Some(system_instructions) = dicl.system_instructions.as_ref() {
                push_file_ref_ids(file_ids, system_instructions);
            }
        }
        StoredLLMJudgeVariantConfig::ChainOfThought(chain_of_thought) => {
            push_file_ref_ids(file_ids, &chain_of_thought.inner.system_instructions);
        }
    }
}

fn collect_function_file_ids(stored: &StoredFunctionConfig, file_ids: &mut HashSet<Uuid>) {
    match stored {
        StoredFunctionConfig::Chat(chat) => {
            if let Some(file_ref) = chat.system_schema.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = chat.user_schema.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = chat.assistant_schema.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(schemas) = chat.schemas.as_ref() {
                for file_ref in schemas.values() {
                    push_file_ref_ids(file_ids, file_ref);
                }
            }
            collect_evaluator_file_ids(chat.evaluators.as_ref(), file_ids);
        }
        StoredFunctionConfig::Json(json) => {
            if let Some(file_ref) = json.system_schema.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = json.user_schema.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = json.assistant_schema.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(schemas) = json.schemas.as_ref() {
                for file_ref in schemas.values() {
                    push_file_ref_ids(file_ids, file_ref);
                }
            }
            if let Some(file_ref) = json.output_schema.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            collect_evaluator_file_ids(json.evaluators.as_ref(), file_ids);
        }
    }
}

fn collect_function_variant_ids(stored: &StoredFunctionConfig, variant_ids: &mut HashSet<Uuid>) {
    let variants = match stored {
        StoredFunctionConfig::Chat(chat) => chat.variants.as_ref(),
        StoredFunctionConfig::Json(json) => json.variants.as_ref(),
    };
    let Some(variants) = variants else {
        return;
    };
    for variant_ref in variants.values() {
        variant_ids.insert(variant_ref.variant_version_id);
    }
}

fn collect_variant_file_ids(stored: &StoredVariantVersionConfig, file_ids: &mut HashSet<Uuid>) {
    match &stored.variant {
        StoredVariantConfig::ChatCompletion(chat) | StoredVariantConfig::ChainOfThought(chat) => {
            if let Some(file_ref) = chat.system_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = chat.user_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = chat.assistant_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(input_wrappers) = chat.input_wrappers.as_ref() {
                if let Some(file_ref) = input_wrappers.user.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
                if let Some(file_ref) = input_wrappers.assistant.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
                if let Some(file_ref) = input_wrappers.system.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
            }
            if let Some(templates) = chat.templates.as_ref() {
                for file_ref in templates.values() {
                    push_file_ref_ids(file_ids, file_ref);
                }
            }
        }
        StoredVariantConfig::BestOfNSampling(best_of_n) => {
            let evaluator = &best_of_n.evaluator;
            if let Some(file_ref) = evaluator.system_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = evaluator.user_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = evaluator.assistant_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(input_wrappers) = evaluator.input_wrappers.as_ref() {
                if let Some(file_ref) = input_wrappers.user.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
                if let Some(file_ref) = input_wrappers.assistant.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
                if let Some(file_ref) = input_wrappers.system.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
            }
            if let Some(templates) = evaluator.templates.as_ref() {
                for file_ref in templates.values() {
                    push_file_ref_ids(file_ids, file_ref);
                }
            }
        }
        StoredVariantConfig::MixtureOfN(mixture_of_n) => {
            let fuser = &mixture_of_n.fuser;
            if let Some(file_ref) = fuser.system_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = fuser.user_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(file_ref) = fuser.assistant_template.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
            if let Some(input_wrappers) = fuser.input_wrappers.as_ref() {
                if let Some(file_ref) = input_wrappers.user.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
                if let Some(file_ref) = input_wrappers.assistant.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
                if let Some(file_ref) = input_wrappers.system.as_ref() {
                    push_file_ref_ids(file_ids, file_ref);
                }
            }
            if let Some(templates) = fuser.templates.as_ref() {
                for file_ref in templates.values() {
                    push_file_ref_ids(file_ids, file_ref);
                }
            }
        }
        StoredVariantConfig::Dicl(dicl) => {
            if let Some(file_ref) = dicl.system_instructions.as_ref() {
                push_file_ref_ids(file_ids, file_ref);
            }
        }
    }
}

fn collect_tool_file_ids(stored: &StoredToolConfig, file_ids: &mut HashSet<Uuid>) {
    push_file_ref_ids(file_ids, &stored.parameters);
}

fn collect_evaluation_file_ids(stored: &StoredEvaluationConfig, file_ids: &mut HashSet<Uuid>) {
    match stored {
        StoredEvaluationConfig::Inference(inference) => {
            collect_evaluator_file_ids(inference.evaluators.as_ref(), file_ids);
        }
    }
}

fn rehydrate_named_collection<Stored, Item, DeserializeFn, RehydrateFn, DeserializeError>(
    rows: Vec<NamedVersionedConfigRow>,
    kind: &str,
    deserialize: DeserializeFn,
    rehydrate: RehydrateFn,
) -> HashMap<String, Item>
where
    DeserializeFn: Fn(i32, serde_json::Value) -> Result<Stored, DeserializeError>,
    RehydrateFn: Fn(Stored) -> Result<Item, Error>,
    DeserializeError: std::fmt::Display,
{
    let mut result = HashMap::new();

    for row in rows {
        let name = row.name;
        let stored = match deserialize(row.schema_revision, row.config) {
            Ok(stored) => stored,
            Err(error) => {
                log_collection_skip(kind, &name, &error);
                continue;
            }
        };
        let item = match rehydrate(stored) {
            Ok(item) => item,
            Err(error) => {
                log_collection_skip(kind, &name, &error);
                continue;
            }
        };
        result.insert(name, item);
    }

    result
}

struct LoadedStoredConfigRows {
    gateway_row: Option<VersionedConfigRow>,
    clickhouse_row: Option<VersionedConfigRow>,
    postgres_row: Option<VersionedConfigRow>,
    object_storage_row: Option<VersionedConfigRow>,
    model_rows: Vec<NamedVersionedConfigRow>,
    embedding_model_rows: Vec<NamedVersionedConfigRow>,
    metric_rows: Vec<NamedVersionedConfigRow>,
    tool_rows: Vec<NamedVersionedConfigRow>,
    evaluation_rows: Vec<NamedVersionedConfigRow>,
    optimizer_rows: Vec<NamedVersionedConfigRow>,
    rate_limiting_row: Option<VersionedConfigRow>,
    autopilot_row: Option<VersionedConfigRow>,
    provider_types_row: Option<VersionedConfigRow>,
    latest_function_rows: Vec<FunctionConfigRow>,
}

/// Rehydrate the given config rows into an `UninitializedConfig`. The provided
/// connection is used for the secondary sequential loads of variant versions
/// and prompt templates that depend on which functions are present.
async fn rehydrate_loaded_config_rows(
    rows: LoadedStoredConfigRows,
    conn: &mut sqlx::PgConnection,
) -> Result<UninitializedConfig, Vec<Error>> {
    let LoadedStoredConfigRows {
        gateway_row,
        clickhouse_row,
        postgres_row,
        object_storage_row,
        model_rows,
        embedding_model_rows,
        metric_rows,
        tool_rows,
        evaluation_rows,
        optimizer_rows,
        rate_limiting_row,
        autopilot_row,
        provider_types_row,
        latest_function_rows,
    } = rows;

    let gateway = match gateway_row {
        Some(row) => {
            let stored = deserialize_gateway_config(row.schema_revision, row.config)
                .map_err(|error| vec![schema_dispatch_error(error)])?;
            stored.try_into().map_err(|error| vec![error])?
        }
        None => Default::default(),
    };
    let clickhouse = match clickhouse_row {
        Some(row) => deserialize_clickhouse_config(row.schema_revision, row.config)
            .map_err(|error| vec![schema_dispatch_error(error)])?
            .into(),
        None => Default::default(),
    };
    let postgres = match postgres_row {
        Some(row) => deserialize_postgres_config(row.schema_revision, row.config)
            .map_err(|error| vec![schema_dispatch_error(error)])?
            .into(),
        None => Default::default(),
    };
    let object_storage = match object_storage_row {
        Some(row) => Some(
            deserialize_storage_kind(row.schema_revision, row.config)
                .map_err(|error| vec![schema_dispatch_error(error)])?
                .into(),
        ),
        None => None,
    };
    let rate_limiting = match rate_limiting_row {
        Some(row) => {
            let stored = deserialize_rate_limiting_config(row.schema_revision, row.config)
                .map_err(|error| vec![schema_dispatch_error(error)])?;
            stored.try_into().map_err(|error| vec![error])?
        }
        None => Default::default(),
    };
    let autopilot = match autopilot_row {
        Some(row) => deserialize_autopilot_config(row.schema_revision, row.config)
            .map_err(|error| vec![schema_dispatch_error(error)])?
            .into(),
        None => Default::default(),
    };
    let provider_types = match provider_types_row {
        Some(row) => deserialize_provider_types_config(row.schema_revision, row.config)
            .map_err(|error| vec![schema_dispatch_error(error)])?
            .into(),
        None => Default::default(),
    };

    let models = rehydrate_named_collection(
        model_rows,
        "model",
        deserialize_model_config,
        TryInto::try_into,
    )
    .into_iter()
    .map(|(name, config)| (Arc::<str>::from(name), config))
    .collect::<HashMap<_, _>>();
    let models = Some(models);
    let embedding_models: HashMap<_, _> = rehydrate_named_collection(
        embedding_model_rows,
        "embedding model",
        deserialize_embedding_model_config,
        TryInto::try_into,
    )
    .into_iter()
    .map(|(name, config)| (Arc::<str>::from(name), config))
    .collect();
    let embedding_models = Some(embedding_models);
    let metrics =
        rehydrate_named_collection(metric_rows, "metric", deserialize_metric_config, |stored| {
            Ok::<_, Error>(stored.into())
        });
    let optimizers = rehydrate_named_collection(
        optimizer_rows,
        "optimizer",
        deserialize_optimizer_config,
        TryInto::try_into,
    );

    let stored_tools =
        rehydrate_named_collection(tool_rows, "tool", deserialize_tool_config, |tool| {
            Ok::<_, Error>(tool)
        });
    let stored_evaluations = rehydrate_named_collection(
        evaluation_rows,
        "evaluation",
        deserialize_evaluation_config,
        Ok::<_, Error>,
    );

    let mut stored_functions = HashMap::new();
    let mut variant_ids = HashSet::new();
    for function_row in &latest_function_rows {
        if function_row.deleted_at.is_some() {
            continue;
        }
        let stored = match deserialize_function_config(
            function_row.schema_revision,
            function_row.config.clone(),
        ) {
            Ok(stored) => stored,
            Err(error) => {
                tracing::error!(
                    "Skipping function `{}` from DB config: failed to deserialize function version `{}`: {}",
                    function_row.name,
                    function_row.id,
                    error
                );
                continue;
            }
        };
        collect_function_variant_ids(&stored, &mut variant_ids);
        stored_functions.insert(function_row.name.clone(), stored);
    }

    let variant_ids = variant_ids.into_iter().collect::<Vec<_>>();
    let variant_version_rows = load_variant_versions(&mut *conn, &variant_ids)
        .await
        .map_err(|error| vec![error])?;
    let mut stored_variants = HashMap::new();
    for row in variant_version_rows {
        let stored = match deserialize_variant_config(row.schema_revision, row.config.clone()) {
            Ok(stored) => stored,
            Err(error) => {
                tracing::error!(
                    "Skipping variant version `{}` from DB config: {}",
                    row.id,
                    error
                );
                continue;
            }
        };
        stored_variants.insert(
            row.id,
            (
                row.name,
                StoredVariantVersionConfig {
                    variant: stored.variant,
                    timeouts: stored.timeouts,
                    namespace: stored.namespace,
                },
            ),
        );
    }

    let mut file_ids = HashSet::new();
    for stored_function in stored_functions.values() {
        collect_function_file_ids(stored_function, &mut file_ids);
    }
    for (_, stored_variant) in stored_variants.values() {
        collect_variant_file_ids(stored_variant, &mut file_ids);
    }
    for stored_tool in stored_tools.values() {
        collect_tool_file_ids(stored_tool, &mut file_ids);
    }
    for stored_evaluation in stored_evaluations.values() {
        collect_evaluation_file_ids(stored_evaluation, &mut file_ids);
    }

    let file_ids = file_ids.into_iter().collect::<Vec<_>>();
    let file_rows = load_files(&mut *conn, &file_ids)
        .await
        .map_err(|error| vec![error])?;
    let files = file_rows
        .into_iter()
        .map(|row| {
            (
                row.id,
                StoredFile {
                    id: row.id,
                    file_path: row.file_path,
                    source_body: row.source_body,
                    content_hash: row.content_hash,
                    creation_source: row.creation_source,
                    source_autopilot_session_id: row.source_autopilot_session_id,
                },
            )
        })
        .collect::<FileMap>();

    let mut tools = HashMap::new();
    for (name, stored_tool) in stored_tools {
        match rehydrate_tool(stored_tool, &files) {
            Ok(tool) => {
                tools.insert(name, tool);
            }
            Err(error) => {
                log_collection_skip("tool", &name, &error);
            }
        }
    }

    let mut evaluations = HashMap::new();
    for (name, stored_evaluation) in stored_evaluations {
        match rehydrate_evaluation(stored_evaluation, &files) {
            Ok(evaluation) => {
                evaluations.insert(name, evaluation);
            }
            Err(error) => {
                log_collection_skip("evaluation", &name, &error);
            }
        }
    }

    let mut functions = HashMap::new();
    for function_row in latest_function_rows {
        if function_row.deleted_at.is_some() {
            continue;
        }
        let Some(stored_function) = stored_functions.get(&function_row.name).cloned() else {
            continue;
        };
        match rehydrate_function(stored_function, &stored_variants, &files) {
            Ok(function) => {
                functions.insert(function_row.name, function);
            }
            Err(error) => {
                tracing::error!(
                    "Skipping function `{}` from DB config during rehydration: {}",
                    function_row.name,
                    error
                );
            }
        }
    }

    let config = UninitializedConfig {
        gateway: Some(gateway),
        clickhouse: Some(clickhouse),
        postgres: Some(postgres),
        rate_limiting: Some(rate_limiting),
        object_storage,
        models,
        embedding_models,
        functions: Some(functions),
        metrics: Some(metrics),
        tools: Some(tools),
        evaluations: Some(evaluations),
        provider_types: Some(provider_types),
        optimizers: Some(optimizers),
        autopilot: Some(autopilot),
    };

    validate_user_config_names(&config).map_err(|error| vec![error])?;

    Ok(config)
}

/// Collapses the `Vec<Error>` returned by `load_config_from_db` into a single
/// `Error` suitable for returning from a handler. Keeps the first error as-is
/// when it is the only one, and otherwise joins their display messages into a
/// single `Config` error.
pub fn merge_load_config_errors(errors: Vec<Error>) -> Error {
    let mut iter = errors.into_iter();
    let Some(first) = iter.next() else {
        return Error::new(ErrorDetails::Config {
            message: "Failed to load config from database".to_string(),
        });
    };
    if iter.len() == 0 {
        return first;
    }
    let message = std::iter::once(first)
        .chain(iter)
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("; ");
    Error::new(ErrorDetails::Config { message })
}

async fn load_singleton_in_snapshot(
    pool: &PgPool,
    snapshot_id: &str,
    table: &'static str,
) -> Result<Option<VersionedConfigRow>, Error> {
    let mut tx = begin_snapshot_read_tx(pool, snapshot_id).await?;
    load_latest_singleton(&mut *tx, table).await
}

async fn load_collection_in_snapshot(
    pool: &PgPool,
    snapshot_id: &str,
    table: &'static str,
) -> Result<Vec<NamedVersionedConfigRow>, Error> {
    let mut tx = begin_snapshot_read_tx(pool, snapshot_id).await?;
    load_named_collection(&mut *tx, table).await
}

async fn load_functions_in_snapshot(
    pool: &PgPool,
    snapshot_id: &str,
) -> Result<Vec<FunctionConfigRow>, Error> {
    let mut tx = begin_snapshot_read_tx(pool, snapshot_id).await?;
    load_latest_functions(&mut *tx).await
}

pub async fn load_config_from_db(pool: &PgPool) -> Result<UninitializedConfig, Vec<Error>> {
    // We want every read in this function to observe the same database state.
    // Without that, a concurrent writer (e.g. `write_stored_config` or
    // `write_function_config`) could move between our individual reads and
    // produce a torn snapshot — a new function row pointing at variant or
    // prompt rows we already missed, or new model rows that don't match the
    // provider-types row we already loaded.
    //
    // A Postgres transaction is bound to a single connection, so we cannot
    // simply wrap a `tokio::try_join!` of parallel queries in one tx. Instead
    // we use `pg_export_snapshot()`: open a leader REPEATABLE READ transaction
    // on one connection, export its snapshot id, and have every parallel
    // reader open its own REPEATABLE READ transaction that imports that
    // snapshot via `SET TRANSACTION SNAPSHOT`. This gives us both a single
    // consistent snapshot and full read parallelism. The leader transaction
    // must stay open until every reader has imported the snapshot, so we
    // commit it only after all reads complete.
    let mut leader_tx = pool.begin().await.map_err(|error| {
        vec![Error::new(ErrorDetails::Config {
            message: format!("Failed to start config snapshot leader transaction: {error}"),
        })]
    })?;
    sqlx::query("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        .execute(&mut *leader_tx)
        .await
        .map_err(|error| {
            vec![Error::new(ErrorDetails::Config {
                message: format!("Failed to set REPEATABLE READ on leader transaction: {error}"),
            })]
        })?;
    let snapshot_id: String = sqlx::query_scalar("SELECT pg_export_snapshot()")
        .fetch_one(&mut *leader_tx)
        .await
        .map_err(|error| {
            vec![Error::new(ErrorDetails::Config {
                message: format!("Failed to export config snapshot: {error}"),
            })]
        })?;

    // Each helper runs in its own snapshot transaction so the queries can
    // execute on independent connections in parallel under `tokio::try_join!`.
    // Futures are boxed at the call site so the outer `load_config_from_db`
    // future stays small (clippy::large_futures) — without this, fanning out
    // 14 sub-futures into a single state machine pushes it well past the
    // warning threshold.
    let snapshot_id = snapshot_id.as_str();
    let (
        gateway_row,
        clickhouse_row,
        postgres_row,
        object_storage_row,
        model_rows,
        embedding_model_rows,
        metric_rows,
        tool_rows,
        evaluation_rows,
        optimizer_rows,
        rate_limiting_row,
        autopilot_row,
        provider_types_row,
        latest_function_rows,
    ) = tokio::try_join!(
        Box::pin(load_singleton_in_snapshot(
            pool,
            snapshot_id,
            "gateway_configs"
        )),
        Box::pin(load_singleton_in_snapshot(
            pool,
            snapshot_id,
            "clickhouse_configs"
        )),
        Box::pin(load_singleton_in_snapshot(
            pool,
            snapshot_id,
            "postgres_configs"
        )),
        Box::pin(load_singleton_in_snapshot(
            pool,
            snapshot_id,
            "object_storage_configs"
        )),
        Box::pin(load_collection_in_snapshot(
            pool,
            snapshot_id,
            "models_configs"
        )),
        Box::pin(load_collection_in_snapshot(
            pool,
            snapshot_id,
            "embedding_models_configs"
        )),
        Box::pin(load_collection_in_snapshot(
            pool,
            snapshot_id,
            "metrics_configs"
        )),
        Box::pin(load_collection_in_snapshot(
            pool,
            snapshot_id,
            "tools_configs"
        )),
        Box::pin(load_collection_in_snapshot(
            pool,
            snapshot_id,
            "evaluations_configs"
        )),
        Box::pin(load_collection_in_snapshot(
            pool,
            snapshot_id,
            "optimizers_configs"
        )),
        Box::pin(load_singleton_in_snapshot(
            pool,
            snapshot_id,
            "rate_limiting_configs"
        )),
        Box::pin(load_singleton_in_snapshot(
            pool,
            snapshot_id,
            "autopilot_configs"
        )),
        Box::pin(load_singleton_in_snapshot(
            pool,
            snapshot_id,
            "provider_types_configs"
        )),
        Box::pin(load_functions_in_snapshot(pool, snapshot_id)),
    )
    .map_err(|error| vec![error])?;

    let rows = LoadedStoredConfigRows {
        gateway_row,
        clickhouse_row,
        postgres_row,
        object_storage_row,
        model_rows,
        embedding_model_rows,
        metric_rows,
        tool_rows,
        evaluation_rows,
        optimizer_rows,
        rate_limiting_row,
        autopilot_row,
        provider_types_row,
        latest_function_rows,
    };

    // Load variant versions and prompt templates inside the same snapshot
    // transaction so they observe the same consistent snapshot as the
    // parallel readers above.
    let mut secondary_tx = begin_snapshot_read_tx(pool, snapshot_id)
        .await
        .map_err(|error| vec![error])?;
    let config = rehydrate_loaded_config_rows(rows, &mut secondary_tx).await?;
    drop(secondary_tx);

    // All snapshot readers have finished, so the leader transaction is no
    // longer needed. Committing it (vs. dropping it) is just clearer about
    // intent — there is nothing to roll back.
    leader_tx.commit().await.map_err(|error| {
        vec![Error::new(ErrorDetails::Config {
            message: format!("Failed to commit config snapshot leader transaction: {error}"),
        })]
    })?;

    Ok(config)
}

/// Loads every non-tombstoned file in `stored_files`, keyed by `file_path`.
///
/// Used by the config editor to build the full `path_contents` map — both
/// files referenced in the TOML and "free" files the user added but has not
/// yet referenced. The write path maintains the invariant that at most one
/// active row exists per `file_path`, so we treat a duplicate as a data
/// integrity violation and surface an error rather than silently picking a
/// row.
pub async fn load_editor_path_contents(pool: &PgPool) -> Result<HashMap<String, String>, Error> {
    #[derive(FromRow)]
    struct EditorFileRow {
        file_path: String,
        source_body: String,
    }
    let rows = sqlx::query_as::<_, EditorFileRow>(
        r"
        SELECT file_path, source_body
        FROM tensorzero.stored_files
        WHERE deleted_at IS NULL
        ",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to load editor path contents: {e}"),
        })
    })?;
    let mut path_contents: HashMap<String, String> = HashMap::with_capacity(rows.len());
    for row in rows {
        if path_contents
            .insert(row.file_path.clone(), row.source_body)
            .is_some()
        {
            return Err(Error::new(ErrorDetails::PostgresQuery {
                message: format!(
                    "`stored_files` has multiple active rows for `file_path` `{}`; the write path should maintain at most one active row per path.",
                    row.file_path
                ),
            }));
        }
    }
    Ok(path_contents)
}
