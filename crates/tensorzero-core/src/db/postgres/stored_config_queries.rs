use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use sqlx::{FromRow, PgPool, Postgres, QueryBuilder};
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

fn schema_dispatch_error(error: impl std::fmt::Display) -> Error {
    Error::new(ErrorDetails::Config {
        message: error.to_string(),
    })
}

async fn load_latest_singleton(
    pool: &PgPool,
    table_name: &'static str,
) -> Result<Option<VersionedConfigRow>, Error> {
    let mut query =
        QueryBuilder::<Postgres>::new("SELECT schema_revision, config FROM tensorzero.");
    query
        .push(table_name)
        .push(" ORDER BY created_at DESC, id DESC LIMIT 1");
    Ok(query.build_query_as().fetch_optional(pool).await?)
}

async fn load_named_collection(
    pool: &PgPool,
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
    Ok(query.build_query_as().fetch_all(pool).await?)
}

async fn load_active_functions(pool: &PgPool) -> Result<Vec<ActiveFunctionRow>, Error> {
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
    .fetch_all(pool)
    .await?)
}

async fn load_variant_versions(
    pool: &PgPool,
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
    .fetch_all(pool)
    .await?)
}

async fn load_prompt_templates(
    pool: &PgPool,
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
    .fetch_all(pool)
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
        load_latest_singleton(pool, "gateway_configs"),
        load_latest_singleton(pool, "clickhouse_configs"),
        load_latest_singleton(pool, "postgres_configs"),
        load_latest_singleton(pool, "object_storage_configs"),
        load_named_collection(pool, "models_configs"),
        load_named_collection(pool, "embedding_models_configs"),
        load_named_collection(pool, "metrics_configs"),
        load_named_collection(pool, "tools_configs"),
        load_named_collection(pool, "evaluations_configs"),
        load_named_collection(pool, "optimizers_configs"),
        load_latest_singleton(pool, "rate_limiting_configs"),
        load_latest_singleton(pool, "autopilot_configs"),
        load_latest_singleton(pool, "provider_types_configs"),
        load_active_functions(pool),
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
    let variant_version_rows = load_variant_versions(pool, &variant_ids)
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
    let prompt_rows = load_prompt_templates(pool, &prompt_ids)
        .await
        .map_err(|error| vec![error])?;
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
