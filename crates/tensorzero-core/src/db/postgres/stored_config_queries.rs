use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Transaction};
use tensorzero_stored_config::schema_dispatch::{
    deserialize_autopilot_config, deserialize_clickhouse_config,
    deserialize_embedding_model_config, deserialize_evaluation_config, deserialize_function_config,
    deserialize_gateway_config, deserialize_metric_config, deserialize_model_config,
    deserialize_optimizer_config, deserialize_postgres_config, deserialize_provider_types_config,
    deserialize_rate_limiting_config, deserialize_storage_kind, deserialize_tool_config,
    deserialize_variant_config,
};
use tensorzero_stored_config::{
    StoredEvaluationConfig, StoredEvaluatorConfig, StoredFunctionConfig, StoredLLMJudgeConfig,
    StoredLLMJudgeVariantConfig, StoredPromptRef, StoredPromptTemplate, StoredToolConfig,
    StoredVariantConfig, StoredVariantVersionConfig,
};
use uuid::Uuid;

use crate::config::rehydrate::{
    PromptTemplateMap, rehydrate_evaluation, rehydrate_function, rehydrate_tool,
};
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
struct ActiveFunctionRow {
    name: String,
    #[expect(dead_code)]
    function_type: String,
    schema_revision: i32,
    config: serde_json::Value,
}

#[derive(Clone, Debug, FromRow)]
struct VariantVersionRow {
    id: Uuid,
    #[expect(dead_code)]
    function_name: String,
    name: String,
    #[expect(dead_code)]
    variant_type: String,
    schema_revision: i32,
    config: serde_json::Value,
}

#[derive(Clone, Debug, FromRow)]
struct PromptTemplateRow {
    id: Uuid,
    template_key: String,
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

async fn load_latest_singleton(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &'static str,
) -> Result<Option<VersionedConfigRow>, Error> {
    let mut query =
        QueryBuilder::<Postgres>::new("SELECT schema_revision, config FROM tensorzero.");
    query
        .push(table_name)
        .push(" ORDER BY created_at DESC, id DESC LIMIT 1");
    Ok(query.build_query_as().fetch_optional(&mut **tx).await?)
}

async fn load_named_collection(
    tx: &mut Transaction<'_, Postgres>,
    table_name: &'static str,
) -> Result<Vec<NamedVersionedConfigRow>, Error> {
    // Pick the most recent row per `name`, breaking ties by `id` so the result
    // is deterministic if two writes share an `updated_at` timestamp.
    let mut query = QueryBuilder::<Postgres>::new(
        "SELECT DISTINCT ON (name) name, schema_revision, config FROM tensorzero.",
    );
    query
        .push(table_name)
        .push(" ORDER BY name ASC, updated_at DESC, id DESC");
    Ok(query.build_query_as().fetch_all(&mut **tx).await?)
}

async fn load_active_functions(
    tx: &mut Transaction<'_, Postgres>,
) -> Result<Vec<ActiveFunctionRow>, Error> {
    // `function_configs` stores one row per function version. The active
    // version of a function is the most recently inserted row for that name
    // whose `deleted_at` is NULL.
    Ok(sqlx::query_as::<_, ActiveFunctionRow>(
        r"
        SELECT DISTINCT ON (name) name, function_type, schema_revision, config
        FROM tensorzero.function_configs
        WHERE deleted_at IS NULL
        ORDER BY name ASC, created_at DESC, id DESC
        ",
    )
    .fetch_all(&mut **tx)
    .await?)
}

async fn load_variant_versions(
    tx: &mut Transaction<'_, Postgres>,
    ids: &[Uuid],
) -> Result<Vec<VariantVersionRow>, Error> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    Ok(sqlx::query_as::<_, VariantVersionRow>(
        r"
        SELECT id, function_name, name, variant_type, schema_revision, config
        FROM tensorzero.variant_configs
        WHERE id = ANY($1)
        ",
    )
    .bind(ids)
    .fetch_all(&mut **tx)
    .await?)
}

async fn load_prompt_templates(
    tx: &mut Transaction<'_, Postgres>,
    ids: &[Uuid],
) -> Result<Vec<PromptTemplateRow>, Error> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    Ok(sqlx::query_as::<_, PromptTemplateRow>(
        r"
        SELECT id, template_key, source_body, content_hash, creation_source, source_autopilot_session_id
        FROM tensorzero.prompt_template_configs
        WHERE id = ANY($1)
        ",
    )
    .bind(ids)
    .fetch_all(&mut **tx)
    .await?)
}

fn push_prompt_ref_ids(prompt_ids: &mut HashSet<Uuid>, prompt_ref: &StoredPromptRef) {
    prompt_ids.insert(prompt_ref.prompt_template_version_id);
}

fn collect_evaluator_prompt_ids(
    evaluators: Option<&BTreeMap<String, StoredEvaluatorConfig>>,
    prompt_ids: &mut HashSet<Uuid>,
) {
    let Some(evaluators) = evaluators else {
        return;
    };

    for evaluator in evaluators.values() {
        if let StoredEvaluatorConfig::LLMJudge(StoredLLMJudgeConfig {
            variants: Some(variants),
            ..
        }) = evaluator
        {
            for variant in variants.values() {
                collect_llm_judge_prompt_ids(&variant.variant, prompt_ids);
            }
        }
    }
}

fn collect_llm_judge_prompt_ids(
    variant: &StoredLLMJudgeVariantConfig,
    prompt_ids: &mut HashSet<Uuid>,
) {
    match variant {
        StoredLLMJudgeVariantConfig::ChatCompletion(chat) => {
            push_prompt_ref_ids(prompt_ids, &chat.system_instructions);
        }
        StoredLLMJudgeVariantConfig::BestOfNSampling(best_of_n) => {
            push_prompt_ref_ids(prompt_ids, &best_of_n.evaluator.system_instructions);
        }
        StoredLLMJudgeVariantConfig::MixtureOfNSampling(mixture_of_n) => {
            push_prompt_ref_ids(prompt_ids, &mixture_of_n.fuser.system_instructions);
        }
        StoredLLMJudgeVariantConfig::Dicl(dicl) => {
            if let Some(system_instructions) = dicl.system_instructions.as_ref() {
                push_prompt_ref_ids(prompt_ids, system_instructions);
            }
        }
        StoredLLMJudgeVariantConfig::ChainOfThought(chain_of_thought) => {
            push_prompt_ref_ids(prompt_ids, &chain_of_thought.inner.system_instructions);
        }
    }
}

fn collect_function_prompt_ids(stored: &StoredFunctionConfig, prompt_ids: &mut HashSet<Uuid>) {
    match stored {
        StoredFunctionConfig::Chat(chat) => {
            if let Some(prompt_ref) = chat.system_schema.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = chat.user_schema.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = chat.assistant_schema.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(schemas) = chat.schemas.as_ref() {
                for prompt_ref in schemas.values() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
            }
            collect_evaluator_prompt_ids(chat.evaluators.as_ref(), prompt_ids);
        }
        StoredFunctionConfig::Json(json) => {
            if let Some(prompt_ref) = json.system_schema.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = json.user_schema.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = json.assistant_schema.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(schemas) = json.schemas.as_ref() {
                for prompt_ref in schemas.values() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
            }
            if let Some(prompt_ref) = json.output_schema.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            collect_evaluator_prompt_ids(json.evaluators.as_ref(), prompt_ids);
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

fn collect_variant_prompt_ids(stored: &StoredVariantVersionConfig, prompt_ids: &mut HashSet<Uuid>) {
    match &stored.variant {
        StoredVariantConfig::ChatCompletion(chat) | StoredVariantConfig::ChainOfThought(chat) => {
            if let Some(prompt_ref) = chat.system_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = chat.user_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = chat.assistant_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(input_wrappers) = chat.input_wrappers.as_ref() {
                if let Some(prompt_ref) = input_wrappers.user.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
                if let Some(prompt_ref) = input_wrappers.assistant.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
                if let Some(prompt_ref) = input_wrappers.system.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
            }
            if let Some(templates) = chat.templates.as_ref() {
                for prompt_ref in templates.values() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
            }
        }
        StoredVariantConfig::BestOfNSampling(best_of_n) => {
            let evaluator = &best_of_n.evaluator;
            if let Some(prompt_ref) = evaluator.system_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = evaluator.user_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = evaluator.assistant_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(input_wrappers) = evaluator.input_wrappers.as_ref() {
                if let Some(prompt_ref) = input_wrappers.user.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
                if let Some(prompt_ref) = input_wrappers.assistant.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
                if let Some(prompt_ref) = input_wrappers.system.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
            }
            if let Some(templates) = evaluator.templates.as_ref() {
                for prompt_ref in templates.values() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
            }
        }
        StoredVariantConfig::MixtureOfN(mixture_of_n) => {
            let fuser = &mixture_of_n.fuser;
            if let Some(prompt_ref) = fuser.system_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = fuser.user_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(prompt_ref) = fuser.assistant_template.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
            if let Some(input_wrappers) = fuser.input_wrappers.as_ref() {
                if let Some(prompt_ref) = input_wrappers.user.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
                if let Some(prompt_ref) = input_wrappers.assistant.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
                if let Some(prompt_ref) = input_wrappers.system.as_ref() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
            }
            if let Some(templates) = fuser.templates.as_ref() {
                for prompt_ref in templates.values() {
                    push_prompt_ref_ids(prompt_ids, prompt_ref);
                }
            }
        }
        StoredVariantConfig::Dicl(dicl) => {
            if let Some(prompt_ref) = dicl.system_instructions.as_ref() {
                push_prompt_ref_ids(prompt_ids, prompt_ref);
            }
        }
    }
}

fn collect_tool_prompt_ids(stored: &StoredToolConfig, prompt_ids: &mut HashSet<Uuid>) {
    push_prompt_ref_ids(prompt_ids, &stored.parameters);
}

fn collect_evaluation_prompt_ids(stored: &StoredEvaluationConfig, prompt_ids: &mut HashSet<Uuid>) {
    match stored {
        StoredEvaluationConfig::Inference(inference) => {
            collect_evaluator_prompt_ids(inference.evaluators.as_ref(), prompt_ids);
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

    // Each closure runs in its own snapshot transaction so the queries can
    // execute on independent connections in parallel under `tokio::try_join!`.
    // Futures are boxed so the outer `load_config_from_db` future stays small
    // (clippy::large_futures) — without this, fanning out 14 sub-futures into
    // a single state machine pushes it well past the warning threshold.
    let load_singleton = |table: &'static str| {
        let snapshot_id = snapshot_id.as_str();
        Box::pin(async move {
            let mut tx = begin_snapshot_read_tx(pool, snapshot_id).await?;
            load_latest_singleton(&mut tx, table).await
        })
            as std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Option<VersionedConfigRow>, Error>>>,
            >
    };
    let load_collection = |table: &'static str| {
        let snapshot_id = snapshot_id.as_str();
        Box::pin(async move {
            let mut tx = begin_snapshot_read_tx(pool, snapshot_id).await?;
            load_named_collection(&mut tx, table).await
        })
            as std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<NamedVersionedConfigRow>, Error>>>,
            >
    };
    let load_functions = || {
        let snapshot_id = snapshot_id.as_str();
        Box::pin(async move {
            let mut tx = begin_snapshot_read_tx(pool, snapshot_id).await?;
            load_active_functions(&mut tx).await
        })
            as std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<ActiveFunctionRow>, Error>>>,
            >
    };

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
        active_function_rows,
    ) = tokio::try_join!(
        load_singleton("gateway_configs"),
        load_singleton("clickhouse_configs"),
        load_singleton("postgres_configs"),
        load_singleton("object_storage_configs"),
        load_collection("models_configs"),
        load_collection("embedding_models_configs"),
        load_collection("metrics_configs"),
        load_collection("tools_configs"),
        load_collection("evaluations_configs"),
        load_collection("optimizers_configs"),
        load_singleton("rate_limiting_configs"),
        load_singleton("autopilot_configs"),
        load_singleton("provider_types_configs"),
        load_functions(),
    )
    .map_err(|error| vec![error])?;

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
    for active_function in &active_function_rows {
        let stored = match deserialize_function_config(
            active_function.schema_revision,
            active_function.config.clone(),
        ) {
            Ok(stored) => stored,
            Err(error) => {
                tracing::error!(
                    "Skipping function `{}` from DB config: failed to deserialize active function version: {}",
                    active_function.name,
                    error
                );
                continue;
            }
        };
        collect_function_variant_ids(&stored, &mut variant_ids);
        stored_functions.insert(active_function.name.clone(), stored);
    }

    let variant_ids = variant_ids.into_iter().collect::<Vec<_>>();
    let mut variant_tx = begin_snapshot_read_tx(pool, &snapshot_id)
        .await
        .map_err(|error| vec![error])?;
    let variant_version_rows = load_variant_versions(&mut variant_tx, &variant_ids)
        .await
        .map_err(|error| vec![error])?;
    drop(variant_tx);
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

    let mut prompt_ids = HashSet::new();
    for stored_function in stored_functions.values() {
        collect_function_prompt_ids(stored_function, &mut prompt_ids);
    }
    for (_, stored_variant) in stored_variants.values() {
        collect_variant_prompt_ids(stored_variant, &mut prompt_ids);
    }
    for stored_tool in stored_tools.values() {
        collect_tool_prompt_ids(stored_tool, &mut prompt_ids);
    }
    for stored_evaluation in stored_evaluations.values() {
        collect_evaluation_prompt_ids(stored_evaluation, &mut prompt_ids);
    }

    let prompt_ids = prompt_ids.into_iter().collect::<Vec<_>>();
    let mut prompt_tx = begin_snapshot_read_tx(pool, &snapshot_id)
        .await
        .map_err(|error| vec![error])?;
    let prompt_rows = load_prompt_templates(&mut prompt_tx, &prompt_ids)
        .await
        .map_err(|error| vec![error])?;
    drop(prompt_tx);

    // All snapshot readers have finished, so the leader transaction is no
    // longer needed. Committing it (vs. dropping it) is just clearer about
    // intent — there is nothing to roll back.
    leader_tx.commit().await.map_err(|error| {
        vec![Error::new(ErrorDetails::Config {
            message: format!("Failed to commit config snapshot leader transaction: {error}"),
        })]
    })?;
    let prompts = prompt_rows
        .into_iter()
        .map(|row| {
            (
                row.id,
                StoredPromptTemplate {
                    id: row.id,
                    template_key: row.template_key,
                    source_body: row.source_body,
                    content_hash: row.content_hash,
                    creation_source: row.creation_source,
                    source_autopilot_session_id: row.source_autopilot_session_id,
                },
            )
        })
        .collect::<PromptTemplateMap>();

    let mut tools = HashMap::new();
    for (name, stored_tool) in stored_tools {
        match rehydrate_tool(stored_tool, &prompts) {
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
        match rehydrate_evaluation(stored_evaluation, &prompts) {
            Ok(evaluation) => {
                evaluations.insert(name, evaluation);
            }
            Err(error) => {
                log_collection_skip("evaluation", &name, &error);
            }
        }
    }

    let mut functions = HashMap::new();
    for active_function in active_function_rows {
        let Some(stored_function) = stored_functions.get(&active_function.name).cloned() else {
            continue;
        };
        match rehydrate_function(stored_function, &stored_variants, &prompts) {
            Ok(function) => {
                functions.insert(active_function.name, function);
            }
            Err(error) => {
                tracing::error!(
                    "Skipping function `{}` from DB config during rehydration: {}",
                    active_function.name,
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
