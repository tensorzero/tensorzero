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
use crate::config::{ConfigLoadingError, UninitializedConfig, validate_user_config_names};
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

/// Best-effort conversion from a raw JSONB config value into a TOML string fragment.
/// Used to populate `ConfigLoadingError::raw_toml` for copy/paste debugging.
/// Returns `None` if the value cannot be rendered as TOML (e.g. top-level null).
fn json_to_toml_fragment(value: serde_json::Value) -> Option<String> {
    let toml_value = json_value_to_toml(value)?;
    toml::to_string_pretty(&toml_value).ok()
}

fn json_value_to_toml(value: serde_json::Value) -> Option<toml::Value> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(b) => Some(toml::Value::Boolean(b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(toml::Value::Integer(i))
            } else {
                n.as_f64().map(toml::Value::Float)
            }
        }
        serde_json::Value::String(s) => Some(toml::Value::String(s)),
        serde_json::Value::Array(arr) => {
            let items: Vec<toml::Value> = arr.into_iter().filter_map(json_value_to_toml).collect();
            Some(toml::Value::Array(items))
        }
        serde_json::Value::Object(obj) => {
            let map: toml::map::Map<String, toml::Value> = obj
                .into_iter()
                .filter_map(|(k, v)| json_value_to_toml(v).map(|tv| (k, tv)))
                .collect();
            Some(toml::Value::Table(map))
        }
    }
}

/// The result of loading config from the database. Contains the successfully
/// parsed config alongside any per-item errors encountered during loading.
#[derive(Debug)]
pub struct LoadedConfig {
    pub config: UninitializedConfig,
    pub loading_errors: Vec<ConfigLoadingError>,
}

/// Loads an optional singleton config row and converts it via `convert`.
///
/// - If the row is absent, returns `T::default()`.
/// - If the row is present and converts successfully, returns the converted value.
/// - If conversion fails, pushes a `ConfigLoadingError` and returns `T::default()`
///   so the gateway can still start with the built-in defaults.
///
/// `kind` is used as both the error `kind` and `name` (singleton rows have no
/// user-facing name). It must stay in sync with the TS binding consumers.
fn load_singleton_or_default<T, F>(
    row: Option<VersionedConfigRow>,
    kind: &'static str,
    convert: F,
    errors: &mut Vec<ConfigLoadingError>,
) -> T
where
    T: Default,
    F: FnOnce(i32, serde_json::Value) -> Result<T, Error>,
{
    let Some(row) = row else {
        return T::default();
    };
    let raw_toml = json_to_toml_fragment(row.config.clone());
    match convert(row.schema_revision, row.config) {
        Ok(value) => value,
        Err(error) => {
            errors.push(ConfigLoadingError {
                kind,
                name: kind.to_string(),
                parent: None,
                error: error.to_string(),
                raw_toml,
            });
            T::default()
        }
    }
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
            | StoredEvaluatorConfig::Regex(_)
            | StoredEvaluatorConfig::Typescript(_) => {}
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
    kind: &'static str,
    deserialize: DeserializeFn,
    rehydrate: RehydrateFn,
) -> (HashMap<String, Item>, Vec<ConfigLoadingError>)
where
    DeserializeFn: Fn(i32, serde_json::Value) -> Result<Stored, DeserializeError>,
    RehydrateFn: Fn(Stored) -> Result<Item, Error>,
    DeserializeError: std::fmt::Display,
{
    let mut result = HashMap::new();
    let mut errors = Vec::new();

    for row in rows {
        let name = row.name;
        let raw_toml = json_to_toml_fragment(row.config.clone());
        let stored = match deserialize(row.schema_revision, row.config) {
            Ok(stored) => stored,
            Err(error) => {
                errors.push(ConfigLoadingError {
                    kind,
                    name,
                    parent: None,
                    error: error.to_string(),
                    raw_toml,
                });
                continue;
            }
        };
        let item = match rehydrate(stored) {
            Ok(item) => item,
            Err(error) => {
                errors.push(ConfigLoadingError {
                    kind,
                    name,
                    parent: None,
                    error: error.to_string(),
                    raw_toml,
                });
                continue;
            }
        };
        result.insert(name, item);
    }

    (result, errors)
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

/// Rehydrate the given config rows into a `LoadedConfig`. The provided
/// connection is used for the secondary sequential loads of variant versions
/// and prompt templates that depend on which functions are present.
///
/// Per-item failures (broken JSONB, missing file refs, broken variant refs)
/// are collected into `LoadedConfig::loading_errors` rather than aborting the
/// load. Only unrecoverable structural failures (DB connectivity, snapshot
/// transaction errors) propagate as `Err`.
async fn rehydrate_loaded_config_rows(
    rows: LoadedStoredConfigRows,
    conn: &mut sqlx::PgConnection,
) -> Result<(UninitializedConfig, Vec<ConfigLoadingError>), Vec<Error>> {
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

    let mut loading_errors: Vec<ConfigLoadingError> = Vec::new();

    let gateway = load_singleton_or_default(
        gateway_row,
        "gateway_config",
        |sr, c| {
            deserialize_gateway_config(sr, c)
                .map_err(schema_dispatch_error)
                .and_then(TryInto::try_into)
        },
        &mut loading_errors,
    );
    let clickhouse = load_singleton_or_default(
        clickhouse_row,
        "clickhouse_config",
        |sr, c| {
            deserialize_clickhouse_config(sr, c)
                .map(Into::into)
                .map_err(schema_dispatch_error)
        },
        &mut loading_errors,
    );
    let postgres = load_singleton_or_default(
        postgres_row,
        "postgres_config",
        |sr, c| {
            deserialize_postgres_config(sr, c)
                .map(Into::into)
                .map_err(schema_dispatch_error)
        },
        &mut loading_errors,
    );
    let object_storage = load_singleton_or_default(
        object_storage_row,
        "object_storage_config",
        |sr, c| {
            deserialize_storage_kind(sr, c)
                .map(|stored| Some(stored.into()))
                .map_err(schema_dispatch_error)
        },
        &mut loading_errors,
    );
    let rate_limiting = load_singleton_or_default(
        rate_limiting_row,
        "rate_limiting_config",
        |sr, c| {
            deserialize_rate_limiting_config(sr, c)
                .map_err(schema_dispatch_error)
                .and_then(TryInto::try_into)
        },
        &mut loading_errors,
    );
    let autopilot = load_singleton_or_default(
        autopilot_row,
        "autopilot_config",
        |sr, c| {
            deserialize_autopilot_config(sr, c)
                .map(Into::into)
                .map_err(schema_dispatch_error)
        },
        &mut loading_errors,
    );
    let provider_types = load_singleton_or_default(
        provider_types_row,
        "provider_types_config",
        |sr, c| {
            deserialize_provider_types_config(sr, c)
                .map(Into::into)
                .map_err(schema_dispatch_error)
        },
        &mut loading_errors,
    );

    let (model_map, model_errors) = rehydrate_named_collection(
        model_rows,
        "model",
        deserialize_model_config,
        TryInto::try_into,
    );
    let models: HashMap<_, _> = model_map
        .into_iter()
        .map(|(name, config)| (Arc::<str>::from(name), config))
        .collect();
    loading_errors.extend(model_errors);

    let (embedding_model_map, embedding_model_errors) = rehydrate_named_collection(
        embedding_model_rows,
        "embedding_model",
        deserialize_embedding_model_config,
        TryInto::try_into,
    );
    let embedding_models: HashMap<_, _> = embedding_model_map
        .into_iter()
        .map(|(name, config)| (Arc::<str>::from(name), config))
        .collect();
    loading_errors.extend(embedding_model_errors);

    let (metrics, metric_errors) =
        rehydrate_named_collection(metric_rows, "metric", deserialize_metric_config, |stored| {
            Ok::<_, Error>(stored.into())
        });
    loading_errors.extend(metric_errors);

    let (optimizers, optimizer_errors) = rehydrate_named_collection(
        optimizer_rows,
        "optimizer",
        deserialize_optimizer_config,
        TryInto::try_into,
    );
    loading_errors.extend(optimizer_errors);

    let (stored_tools, tool_errors) =
        rehydrate_named_collection(tool_rows, "tool", deserialize_tool_config, |tool| {
            Ok::<_, Error>(tool)
        });
    loading_errors.extend(tool_errors);

    let (stored_evaluations, evaluation_errors) = rehydrate_named_collection(
        evaluation_rows,
        "evaluation",
        deserialize_evaluation_config,
        Ok::<_, Error>,
    );
    loading_errors.extend(evaluation_errors);

    let mut stored_functions = HashMap::new();
    let mut variant_ids = HashSet::new();
    for function_row in &latest_function_rows {
        if function_row.deleted_at.is_some() {
            continue;
        }
        // Only compute raw_toml on the error path — avoids the clone on the
        // happy path (which is the vast majority of rows).
        let stored = match deserialize_function_config(
            function_row.schema_revision,
            function_row.config.clone(),
        ) {
            Ok(stored) => stored,
            Err(error) => {
                loading_errors.push(ConfigLoadingError {
                    kind: "function",
                    name: function_row.name.clone(),
                    parent: None,
                    error: format!(
                        "Failed to deserialize function version `{}`: {error}",
                        function_row.id
                    ),
                    raw_toml: json_to_toml_fragment(function_row.config.clone()),
                });
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
        let raw_toml = json_to_toml_fragment(row.config.clone());
        let stored = match deserialize_variant_config(row.schema_revision, row.config) {
            Ok(stored) => stored,
            Err(error) => {
                // We don't know the parent function name here — that association
                // lives in the function config's variant refs. `name` is the
                // variant's own name and the error message carries the version
                // UUID so the operator can identify the specific row. The
                // function rehydration step will later surface a correlated
                // "Missing or broken variant version" error that includes the
                // parent function name.
                loading_errors.push(ConfigLoadingError {
                    kind: "variant",
                    name: row.name.clone(),
                    parent: None,
                    error: format!(
                        "Failed to deserialize variant version `{}`: {error}",
                        row.id
                    ),
                    raw_toml,
                });
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
                loading_errors.push(ConfigLoadingError {
                    kind: "tool",
                    name,
                    parent: None,
                    error: error.to_string(),
                    raw_toml: None,
                });
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
                loading_errors.push(ConfigLoadingError {
                    kind: "evaluation",
                    name,
                    parent: None,
                    error: error.to_string(),
                    raw_toml: None,
                });
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
            Ok((function, variant_errors)) => {
                for (variant_name, error) in variant_errors {
                    loading_errors.push(ConfigLoadingError {
                        kind: "variant",
                        name: variant_name,
                        parent: Some(function_row.name.clone()),
                        error: error.to_string(),
                        raw_toml: None,
                    });
                }
                functions.insert(function_row.name, function);
            }
            Err(error) => {
                loading_errors.push(ConfigLoadingError {
                    kind: "function",
                    name: function_row.name.clone(),
                    parent: None,
                    error: format!("Failed to rehydrate function: {error}"),
                    raw_toml: json_to_toml_fragment(function_row.config),
                });
            }
        }
    }

    let config = UninitializedConfig {
        gateway: Some(gateway),
        clickhouse: Some(clickhouse),
        postgres: Some(postgres),
        rate_limiting: Some(rate_limiting),
        object_storage,
        models: Some(models),
        embedding_models: Some(embedding_models),
        functions: Some(functions),
        metrics: Some(metrics),
        tools: Some(tools),
        evaluations: Some(evaluations),
        provider_types: Some(provider_types),
        optimizers: Some(optimizers),
        autopilot: Some(autopilot),
    };

    validate_user_config_names(&config).map_err(|error| vec![error])?;

    Ok((config, loading_errors))
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

pub async fn load_config_from_db(pool: &PgPool) -> Result<LoadedConfig, Vec<Error>> {
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
    let (config, loading_errors) = rehydrate_loaded_config_rows(rows, &mut secondary_tx).await?;
    drop(secondary_tx);

    // All snapshot readers have finished, so the leader transaction is no
    // longer needed. Committing it (vs. dropping it) is just clearer about
    // intent — there is nothing to roll back.
    leader_tx.commit().await.map_err(|error| {
        vec![Error::new(ErrorDetails::Config {
            message: format!("Failed to commit config snapshot leader transaction: {error}"),
        })]
    })?;

    if !loading_errors.is_empty() {
        let summary = loading_errors
            .iter()
            .map(|e| {
                if let Some(parent) = &e.parent {
                    format!("  - {} `{}` (in `{}`): {}", e.kind, e.name, parent, e.error)
                } else {
                    format!("  - {} `{}`: {}", e.kind, e.name, e.error)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        tracing::error!(
            "{} config item(s) failed to load from the database and will be unavailable:\n{summary}",
            loading_errors.len()
        );
    }

    Ok(LoadedConfig {
        config,
        loading_errors,
    })
}

/// Every Postgres table that holds stored-config rows, in declaration order.
/// `find_nonempty_stored_config_table` checks these in sequence; callers that
/// want to enumerate stored config state (for diagnostics, health checks, or
/// pre-bootstrap validation) should share this list rather than maintain
/// their own.
const STORED_CONFIG_TABLES: &[&str] = &[
    "tensorzero.function_configs",
    "tensorzero.variant_configs",
    "tensorzero.stored_files",
    "tensorzero.tools_configs",
    "tensorzero.evaluations_configs",
    "tensorzero.gateway_configs",
    "tensorzero.clickhouse_configs",
    "tensorzero.postgres_configs",
    "tensorzero.object_storage_configs",
    "tensorzero.models_configs",
    "tensorzero.embedding_models_configs",
    "tensorzero.metrics_configs",
    "tensorzero.rate_limiting_configs",
    "tensorzero.autopilot_configs",
    "tensorzero.provider_types_configs",
    "tensorzero.optimizers_configs",
];

/// Returns the name of the first stored-config table that contains at least
/// one row, or `None` if every stored-config table is empty. Short-circuits
/// on the first hit. Used by `gateway --bootstrap-config` to refuse to write
/// on top of existing state.
///
/// Soft-deleted rows still count as "present" — we want bootstrap to require
/// a fully-fresh schema, not a formerly-populated one that was later cleared.
pub async fn find_nonempty_stored_config_table(
    pool: &PgPool,
) -> Result<Option<&'static str>, Error> {
    for table in STORED_CONFIG_TABLES {
        let mut qb = QueryBuilder::<Postgres>::new("SELECT EXISTS (SELECT 1 FROM ");
        // Table names are static identifiers from `STORED_CONFIG_TABLES`, not
        // user input — `.push()` splices them as trusted SQL. `sqlx::query!`
        // can't bind identifiers, so a builder is the canonical pattern per
        // the crates AGENTS.md.
        qb.push(*table);
        qb.push(")");
        let has_any: bool = qb
            .build_query_scalar()
            .fetch_one(pool)
            .await
            .map_err(|error| {
                Error::new(ErrorDetails::PostgresQuery {
                    message: format!("Failed to check if `{table}` is empty: {error}"),
                })
            })?;
        if has_any {
            return Ok(Some(table));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use serde_json::json;

    // ─── json_value_to_toml ────────────────────────────────────────────────

    #[gtest]
    fn json_value_to_toml_null_returns_none() {
        expect_that!(json_value_to_toml(serde_json::Value::Null), none());
    }

    #[gtest]
    fn json_value_to_toml_bool_preserves_value() {
        expect_that!(
            json_value_to_toml(json!(true)),
            some(eq(&toml::Value::Boolean(true)))
        );
        expect_that!(
            json_value_to_toml(json!(false)),
            some(eq(&toml::Value::Boolean(false)))
        );
    }

    #[gtest]
    fn json_value_to_toml_integer_preserves_value() {
        expect_that!(
            json_value_to_toml(json!(42)),
            some(eq(&toml::Value::Integer(42)))
        );
        expect_that!(
            json_value_to_toml(json!(-17)),
            some(eq(&toml::Value::Integer(-17)))
        );
    }

    #[gtest]
    fn json_value_to_toml_float_preserves_value() {
        let Some(toml::Value::Float(f)) = json_value_to_toml(json!(3.5)) else {
            panic!("expected float toml value");
        };
        expect_that!(f, eq(3.5));
    }

    #[gtest]
    fn json_value_to_toml_string_preserves_value() {
        expect_that!(
            json_value_to_toml(json!("hello")),
            some(eq(&toml::Value::String("hello".to_string())))
        );
    }

    #[gtest]
    fn json_value_to_toml_array_filters_nulls() {
        // Nulls inside arrays are silently dropped — TOML has no null.
        let Some(toml::Value::Array(items)) = json_value_to_toml(json!([1, null, 2])) else {
            panic!("expected array toml value");
        };
        expect_that!(items.len(), eq(2));
        expect_that!(&items[0], eq(&toml::Value::Integer(1)));
        expect_that!(&items[1], eq(&toml::Value::Integer(2)));
    }

    #[gtest]
    fn json_value_to_toml_nested_object_preserves_shape() {
        let input = json!({
            "inner": {"key": "value", "n": 7},
            "flag": true,
        });
        let Some(toml::Value::Table(table)) = json_value_to_toml(input) else {
            panic!("expected table toml value");
        };
        expect_that!(table.get("flag"), some(eq(&toml::Value::Boolean(true))));
        let Some(toml::Value::Table(inner)) = table.get("inner") else {
            panic!("expected nested table");
        };
        expect_that!(
            inner.get("key"),
            some(eq(&toml::Value::String("value".to_string())))
        );
        expect_that!(inner.get("n"), some(eq(&toml::Value::Integer(7))));
    }

    #[gtest]
    fn json_value_to_toml_object_drops_null_entries() {
        // Null-valued keys are dropped, but the surrounding table survives.
        let input = json!({"present": 1, "missing": null});
        let Some(toml::Value::Table(table)) = json_value_to_toml(input) else {
            panic!("expected table toml value");
        };
        expect_that!(table.len(), eq(1));
        expect_that!(table.contains_key("missing"), eq(false));
        expect_that!(table.get("present"), some(eq(&toml::Value::Integer(1))));
    }

    // ─── json_to_toml_fragment ─────────────────────────────────────────────

    #[gtest]
    fn json_to_toml_fragment_top_level_null_returns_none() {
        expect_that!(json_to_toml_fragment(serde_json::Value::Null), none());
    }

    #[gtest]
    fn json_to_toml_fragment_renders_object_as_toml() {
        let fragment = json_to_toml_fragment(json!({
            "type": "chat_completion",
            "model": "gpt-4",
        }))
        .expect("object should render as toml");
        expect_that!(&fragment, contains_substring(r#"model = "gpt-4""#));
        expect_that!(&fragment, contains_substring(r#"type = "chat_completion""#));
    }

    #[gtest]
    fn json_to_toml_fragment_drops_top_level_null_entries() {
        // JSONB can contain nulls that TOML can't; they get silently dropped.
        let fragment = json_to_toml_fragment(json!({
            "kept": "value",
            "dropped": null,
        }))
        .expect("object with nulls should still render");
        expect_that!(&fragment, contains_substring(r#"kept = "value""#));
        expect_that!(&fragment, not(contains_substring("dropped")));
    }

    // ─── load_singleton_or_default ────────────────────────────────────────

    #[gtest]
    fn load_singleton_or_default_returns_default_when_row_absent() {
        let mut errors: Vec<ConfigLoadingError> = Vec::new();
        let out: i32 = load_singleton_or_default(
            None,
            "test_kind",
            |_sr, _c| panic!("convert should not be called when row is absent"),
            &mut errors,
        );
        expect_that!(out, eq(0));
        expect_that!(errors.is_empty(), eq(true));
    }

    #[gtest]
    fn load_singleton_or_default_returns_converted_value_on_success() {
        let mut errors: Vec<ConfigLoadingError> = Vec::new();
        let row = VersionedConfigRow {
            schema_revision: 1,
            config: json!({"value": 99}),
        };
        let out: i32 = load_singleton_or_default(
            Some(row),
            "test_kind",
            |sr, c| {
                expect_that!(sr, eq(1));
                expect_that!(c.get("value").and_then(|v| v.as_i64()), some(eq(99)));
                Ok(42)
            },
            &mut errors,
        );
        expect_that!(out, eq(42));
        expect_that!(errors.is_empty(), eq(true));
    }

    #[gtest]
    fn load_singleton_or_default_records_error_and_returns_default_on_failure() {
        let mut errors: Vec<ConfigLoadingError> = Vec::new();
        let row = VersionedConfigRow {
            schema_revision: 5,
            config: json!({"bad": "data"}),
        };
        let out: i32 = load_singleton_or_default(
            Some(row),
            "broken_kind",
            |_sr, _c| {
                Err(Error::new(ErrorDetails::Config {
                    message: "boom".to_string(),
                }))
            },
            &mut errors,
        );
        expect_that!(out, eq(0));
        assert_that!(errors.len(), eq(1));
        expect_that!(errors[0].kind, eq("broken_kind"));
        expect_that!(&errors[0].name, eq("broken_kind"));
        expect_that!(errors[0].parent, none());
        expect_that!(&errors[0].error, contains_substring("boom"));
        // raw_toml should be populated from the row's JSONB.
        expect_that!(
            errors[0].raw_toml.as_deref(),
            some(contains_substring(r#"bad = "data""#))
        );
    }

    #[gtest]
    fn load_singleton_or_default_with_option_type_returns_none_on_failure() {
        // Mirrors the object_storage case: convert returns Option<T> and the
        // fallback is None (via Option::default()).
        let mut errors: Vec<ConfigLoadingError> = Vec::new();
        let row = VersionedConfigRow {
            schema_revision: 1,
            config: json!({"whatever": true}),
        };
        let out: Option<i32> = load_singleton_or_default(
            Some(row),
            "optional_kind",
            |_sr, _c| {
                Err(Error::new(ErrorDetails::Config {
                    message: "no".to_string(),
                }))
            },
            &mut errors,
        );
        expect_that!(out, none());
        assert_that!(errors.len(), eq(1));
    }
}
