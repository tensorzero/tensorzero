use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use sqlx::PgPool;
use uuid::Uuid;

use crate::config::path::ResolvedTomlPathData;
use crate::config::{
    NonStreamingTimeouts, StreamingTimeouts, TimeoutsConfig, UninitializedFunctionConfig,
    UninitializedFunctionConfigChat, UninitializedFunctionConfigJson, UninitializedSchemas,
};
use crate::error::{Error, ErrorDetails};
use crate::evaluations::{
    ExactMatchConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat, LLMJudgeOptimize,
    LLMJudgeOutputType, RegexConfig, ToolUseConfig, UninitializedEvaluatorConfig,
    UninitializedLLMJudgeChatCompletionVariantConfig, UninitializedLLMJudgeConfig,
    UninitializedLLMJudgeVariantConfig, UninitializedLLMJudgeVariantInfo,
};
use crate::experimentation::adaptive_experimentation::{
    AdaptiveExperimentationAlgorithm, UninitializedAdaptiveExperimentationConfig,
};
use crate::experimentation::static_experimentation::{
    StaticExperimentationConfig, WeightedVariants,
};
use crate::experimentation::track_and_stop::UninitializedTrackAndStopExperimentationConfig;
use crate::experimentation::{
    UninitializedExperimentationConfig, UninitializedExperimentationConfigWithNamespaces,
};
use crate::utils::retries::RetryConfig;
use tensorzero_types::inference_params::{JsonMode, ServiceTier};
use tensorzero_types::tool::ToolChoice;

use super::variant_version_queries::{self, ExtraBodyRow, ExtraHeaderRow, PromptTemplateCollector};

// ---- Row types ----

#[derive(Debug, sqlx::FromRow)]
pub struct FunctionRow {
    pub id: Uuid,
    pub name: String,
    pub active_version_id: Option<Uuid>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct FunctionVersionRow {
    pub id: Uuid,
    pub function_id: Uuid,
    pub function_type: String,
    pub system_schema_prompt_id: Option<Uuid>,
    pub system_schema_key: Option<String>,
    pub user_schema_prompt_id: Option<Uuid>,
    pub user_schema_key: Option<String>,
    pub assistant_schema_prompt_id: Option<Uuid>,
    pub assistant_schema_key: Option<String>,
    pub output_schema_prompt_id: Option<Uuid>,
    pub output_schema_key: Option<String>,
    pub tools: Vec<String>,
    pub tool_choice: String,
    pub parallel_tool_calls: Option<bool>,
    pub description: Option<String>,
    pub creation_source: String,
    pub source_autopilot_session_id: Option<Uuid>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct FunctionVersionSchemaRow {
    pub function_version_id: Uuid,
    pub schema_name: String,
    pub prompt_template_version_id: Uuid,
    pub template_key: String,
}

#[derive(Debug, sqlx::FromRow)]
pub struct FunctionVersionVariantRow {
    pub function_version_id: Uuid,
    pub variant_name: String,
    pub variant_version_id: Uuid,
}

// ---- Evaluator row types ----

#[derive(Debug, sqlx::FromRow)]
pub struct EvaluatorRow {
    pub id: Uuid,
    pub function_version_id: Uuid,
    pub evaluator_name: String,
    pub evaluator_type: String,
}

#[derive(Debug, sqlx::FromRow)]
pub struct EvaluatorExactMatchRow {
    pub evaluator_id: Uuid,
    pub cutoff: Option<f32>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct EvaluatorRegexRow {
    pub evaluator_id: Uuid,
    pub must_match: Option<String>,
    pub must_not_match: Option<String>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct EvaluatorToolUseRow {
    pub evaluator_id: Uuid,
    pub tool_use_type: String,
}

#[derive(Debug, sqlx::FromRow)]
pub struct EvaluatorToolUseToolRow {
    pub evaluator_id: Uuid,
    pub tool_name: String,
}

#[derive(Debug, sqlx::FromRow)]
pub struct EvaluatorLLMJudgeConfigRow {
    pub evaluator_id: Uuid,
    pub input_format: String,
    pub output_type: String,
    pub optimize: String,
    pub include_reference_output: bool,
    pub cutoff: Option<f32>,
    pub description: Option<String>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct EvaluatorLLMJudgeVariantRow {
    pub id: Uuid,
    pub evaluator_id: Uuid,
    pub variant_name: String,
    pub variant_type: String,
    pub timeouts_non_streaming_total_ms: Option<i64>,
    pub timeouts_streaming_ttft_ms: Option<i64>,
    pub timeouts_streaming_total_ms: Option<i64>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct EvaluatorLLMJudgeCCRow {
    pub judge_variant_id: Uuid,
    pub active: Option<bool>,
    pub model: String,
    pub system_instructions_prompt_id: Uuid,
    pub system_instructions_key: String,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<i32>,
    pub json_mode: String,
    pub stop_sequences: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<String>,
    pub thinking_budget_tokens: Option<i32>,
    pub verbosity: Option<String>,
    pub num_retries: i32,
    pub max_retry_delay_s: f32,
}

// ---- Experimentation row types ----

#[derive(Debug, sqlx::FromRow)]
pub struct ExperimentationRow {
    pub id: Uuid,
    pub function_version_id: Uuid,
    pub namespace: Option<String>,
    pub experimentation_type: String,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ExperimentationStaticVariantRow {
    pub experimentation_id: Uuid,
    pub variant_name: String,
    pub weight: f64,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ExperimentationStaticFallbackRow {
    pub experimentation_id: Uuid,
    pub variant_name: String,
    pub position: i32,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ExperimentationAdaptiveConfigRow {
    pub experimentation_id: Uuid,
    pub algorithm: String,
    pub metric: String,
    pub min_samples_per_variant: i64,
    pub delta: f64,
    pub epsilon: f64,
    pub update_period_s: i64,
    pub min_prob: Option<f64>,
    pub max_samples_per_variant: Option<i64>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ExperimentationAdaptiveCandidateRow {
    pub experimentation_id: Uuid,
    pub variant_name: String,
    pub position: i32,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ExperimentationAdaptiveFallbackRow {
    pub experimentation_id: Uuid,
    pub variant_name: String,
    pub position: i32,
}

// ---- ToolChoice helpers ----

fn tool_choice_to_string(tc: &ToolChoice) -> String {
    match tc {
        ToolChoice::None => "none".to_string(),
        ToolChoice::Auto => "auto".to_string(),
        ToolChoice::Required => "required".to_string(),
        ToolChoice::Specific(name) => name.clone(),
    }
}

fn string_to_tool_choice(s: &str) -> ToolChoice {
    match s {
        "none" => ToolChoice::None,
        "auto" => ToolChoice::Auto,
        "required" => ToolChoice::Required,
        other => ToolChoice::Specific(other.to_string()),
    }
}

// ---- Timeouts helpers ----

fn timeouts_to_ms(timeouts: Option<&TimeoutsConfig>) -> (Option<i64>, Option<i64>, Option<i64>) {
    match timeouts {
        Some(t) => (
            t.non_streaming.total_ms.map(|v| v as i64),
            t.streaming.ttft_ms.map(|v| v as i64),
            t.streaming.total_ms.map(|v| v as i64),
        ),
        None => (None, None, None),
    }
}

fn ms_to_timeouts(
    non_streaming_total_ms: Option<i64>,
    streaming_ttft_ms: Option<i64>,
    streaming_total_ms: Option<i64>,
) -> Option<TimeoutsConfig> {
    if non_streaming_total_ms.is_none()
        && streaming_ttft_ms.is_none()
        && streaming_total_ms.is_none()
    {
        return None;
    }
    Some(TimeoutsConfig {
        non_streaming: NonStreamingTimeouts {
            total_ms: non_streaming_total_ms.map(|v| v as u64),
        },
        streaming: StreamingTimeouts {
            ttft_ms: streaming_ttft_ms.map(|v| v as u64),
            total_ms: streaming_total_ms.map(|v| v as u64),
        },
    })
}

// ---- Enum string helpers (reused for LLM judge config fields) ----

fn serialize_json_mode(jm: JsonMode) -> Result<String, Error> {
    serde_json::to_value(jm)
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to serialize json_mode: {e}"),
            })
        })?
        .as_str()
        .map(String::from)
        .ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: "json_mode did not serialize to a string".to_string(),
            })
        })
}

fn deserialize_json_mode(s: &str) -> Result<JsonMode, Error> {
    serde_json::from_value(serde_json::Value::String(s.to_string())).map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to parse json_mode `{s}`: {e}"),
        })
    })
}

fn serialize_service_tier(st: &ServiceTier) -> Result<String, Error> {
    serde_json::to_value(st)
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to serialize service_tier: {e}"),
            })
        })?
        .as_str()
        .map(String::from)
        .ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: "service_tier did not serialize to a string".to_string(),
            })
        })
}

fn deserialize_service_tier(s: &str) -> Result<ServiceTier, Error> {
    serde_json::from_value(serde_json::Value::String(s.to_string())).map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to parse service_tier `{s}`: {e}"),
        })
    })
}

// ===========================================================================
// Write path
// ===========================================================================

/// Ensures a `tensorzero.functions` row exists for the given name, returning its ID.
async fn ensure_function_row(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    function_name: &str,
) -> Result<Uuid, Error> {
    // Try to select existing
    let existing: Option<(Uuid,)> =
        sqlx::query_as("SELECT id FROM tensorzero.functions WHERE name = $1")
            .bind(function_name)
            .fetch_optional(&mut **tx)
            .await?;

    if let Some((id,)) = existing {
        return Ok(id);
    }

    let id = Uuid::now_v7();
    sqlx::query("INSERT INTO tensorzero.functions (id, name) VALUES ($1, $2)")
        .bind(id)
        .bind(function_name)
        .execute(&mut **tx)
        .await?;

    Ok(id)
}

/// Writes prompt templates referenced by function schemas, returning key->UUID map.
async fn write_schema_prompt_templates(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    config: &UninitializedFunctionConfig,
    creation_source: &str,
) -> Result<HashMap<String, Uuid>, Error> {
    let mut collector = PromptTemplateCollector::new();

    let (system_schema, user_schema, assistant_schema, schemas) = match config {
        UninitializedFunctionConfig::Chat(c) => (
            c.system_schema.as_ref(),
            c.user_schema.as_ref(),
            c.assistant_schema.as_ref(),
            &c.schemas,
        ),
        UninitializedFunctionConfig::Json(c) => (
            c.system_schema.as_ref(),
            c.user_schema.as_ref(),
            c.assistant_schema.as_ref(),
            &c.schemas,
        ),
    };

    collector.add_option(system_schema);
    collector.add_option(user_schema);
    collector.add_option(assistant_schema);

    if let UninitializedFunctionConfig::Json(c) = config {
        collector.add_option(c.output_schema.as_ref());
    }

    for (_name, path_data) in schemas.iter() {
        collector.add(path_data);
    }

    let mut key_to_id: HashMap<String, Uuid> = HashMap::new();
    for (key, (_template_key, source_body)) in &collector.templates {
        let id = Uuid::now_v7();
        variant_version_queries::insert_prompt_template_version(
            tx,
            id,
            key,
            source_body,
            creation_source,
        )
        .await?;
        key_to_id.insert(key.to_string(), id);
    }
    Ok(key_to_id)
}

/// Writes a function version and all its sub-tables (schemas, variants, evaluators).
/// Returns the function_version_id.
pub async fn write_function_version(
    pool: &PgPool,
    function_name: &str,
    config: &UninitializedFunctionConfig,
    creation_source: &str,
) -> Result<Uuid, Error> {
    let mut tx = pool.begin().await?;
    let function_version_id =
        write_function_version_in_tx(&mut tx, function_name, config, creation_source).await?;
    tx.commit().await?;
    Ok(function_version_id)
}

/// Writes a function version within an existing transaction.
pub async fn write_function_version_in_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    function_name: &str,
    config: &UninitializedFunctionConfig,
    creation_source: &str,
) -> Result<Uuid, Error> {
    let function_id = ensure_function_row(tx, function_name).await?;
    let function_version_id = Uuid::now_v7();

    // Write prompt templates for schemas
    let schema_key_to_id = write_schema_prompt_templates(tx, config, creation_source).await?;

    // Write the function_version row
    let (function_type, description, tools, tool_choice_str, parallel_tool_calls) = match config {
        UninitializedFunctionConfig::Chat(c) => (
            "chat",
            c.description.as_deref(),
            &c.tools,
            tool_choice_to_string(&c.tool_choice),
            c.parallel_tool_calls,
        ),
        UninitializedFunctionConfig::Json(c) => (
            "json",
            c.description.as_deref(),
            &Vec::new(),
            "auto".to_string(),
            None,
        ),
    };

    let (system_schema, user_schema, assistant_schema) = match config {
        UninitializedFunctionConfig::Chat(c) => (
            c.system_schema.as_ref(),
            c.user_schema.as_ref(),
            c.assistant_schema.as_ref(),
        ),
        UninitializedFunctionConfig::Json(c) => (
            c.system_schema.as_ref(),
            c.user_schema.as_ref(),
            c.assistant_schema.as_ref(),
        ),
    };

    let output_schema = match config {
        UninitializedFunctionConfig::Json(c) => c.output_schema.as_ref(),
        UninitializedFunctionConfig::Chat(_) => None,
    };

    sqlx::query(
        r"INSERT INTO tensorzero.function_versions (
            id, function_id, function_type,
            system_schema_prompt_id, system_schema_key,
            user_schema_prompt_id, user_schema_key,
            assistant_schema_prompt_id, assistant_schema_key,
            output_schema_prompt_id, output_schema_key,
            tools, tool_choice, parallel_tool_calls,
            description, creation_source
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
        )",
    )
    .bind(function_version_id)
    .bind(function_id)
    .bind(function_type)
    .bind(variant_version_queries::prompt_id_for(
        &schema_key_to_id,
        system_schema,
    ))
    .bind(variant_version_queries::prompt_key_for(system_schema))
    .bind(variant_version_queries::prompt_id_for(
        &schema_key_to_id,
        user_schema,
    ))
    .bind(variant_version_queries::prompt_key_for(user_schema))
    .bind(variant_version_queries::prompt_id_for(
        &schema_key_to_id,
        assistant_schema,
    ))
    .bind(variant_version_queries::prompt_key_for(assistant_schema))
    .bind(variant_version_queries::prompt_id_for(
        &schema_key_to_id,
        output_schema,
    ))
    .bind(variant_version_queries::prompt_key_for(output_schema))
    .bind(tools)
    .bind(&tool_choice_str)
    .bind(parallel_tool_calls)
    .bind(description)
    .bind(creation_source)
    .execute(&mut **tx)
    .await?;

    // Write named schemas
    let schemas = match config {
        UninitializedFunctionConfig::Chat(c) => &c.schemas,
        UninitializedFunctionConfig::Json(c) => &c.schemas,
    };
    for (schema_name, path_data) in schemas.iter() {
        let prompt_id = schema_key_to_id
            .get(&path_data.get_template_key())
            .ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Missing prompt template ID for schema `{schema_name}`"),
                })
            })?;
        sqlx::query(
            r"INSERT INTO tensorzero.function_version_schemas
               (function_version_id, schema_name, prompt_template_version_id, template_key)
               VALUES ($1, $2, $3, $4)",
        )
        .bind(function_version_id)
        .bind(schema_name)
        .bind(prompt_id)
        .bind(path_data.get_template_key())
        .execute(&mut **tx)
        .await?;
    }

    // Write variant versions and junction rows
    let variants = match config {
        UninitializedFunctionConfig::Chat(c) => &c.variants,
        UninitializedFunctionConfig::Json(c) => &c.variants,
    };
    for (variant_name, variant_info) in variants {
        let variant_version_id = variant_version_queries::write_variant_version_in_tx(
            tx,
            variant_info,
            Some(function_name),
            Some(variant_name),
            creation_source,
        )
        .await?;

        sqlx::query(
            r"INSERT INTO tensorzero.function_version_variants
               (function_version_id, variant_name, variant_version_id)
               VALUES ($1, $2, $3)",
        )
        .bind(function_version_id)
        .bind(variant_name)
        .bind(variant_version_id)
        .execute(&mut **tx)
        .await?;
    }

    // Write experimentation config
    let experimentation = match config {
        UninitializedFunctionConfig::Chat(c) => c.experimentation.as_ref(),
        UninitializedFunctionConfig::Json(c) => c.experimentation.as_ref(),
    };
    if let Some(exp) = experimentation {
        write_experimentation(tx, function_version_id, exp).await?;
    }

    // Write evaluators
    let evaluators = match config {
        UninitializedFunctionConfig::Chat(c) => &c.evaluators,
        UninitializedFunctionConfig::Json(c) => &c.evaluators,
    };
    for (evaluator_name, evaluator_config) in evaluators {
        write_evaluator(
            tx,
            function_version_id,
            evaluator_name,
            evaluator_config,
            creation_source,
        )
        .await?;
    }

    // Update active_version_id on the functions row
    sqlx::query(
        r"UPDATE tensorzero.functions
           SET active_version_id = $1, updated_at = NOW()
           WHERE id = $2",
    )
    .bind(function_version_id)
    .bind(function_id)
    .execute(&mut **tx)
    .await?;

    Ok(function_version_id)
}

// ---- Experimentation write path ----

async fn write_experimentation(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    function_version_id: Uuid,
    config: &UninitializedExperimentationConfigWithNamespaces,
) -> Result<(), Error> {
    // Write the base config (namespace = NULL)
    write_single_experimentation(tx, function_version_id, None, &config.base).await?;

    // Write namespace overrides
    for (namespace, ns_config) in &config.namespaces {
        write_single_experimentation(tx, function_version_id, Some(namespace), ns_config).await?;
    }

    Ok(())
}

async fn write_single_experimentation(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    function_version_id: Uuid,
    namespace: Option<&str>,
    config: &UninitializedExperimentationConfig,
) -> Result<(), Error> {
    let experimentation_id = Uuid::now_v7();

    // Normalize legacy types on write
    let (experimentation_type, normalized) = normalize_experimentation_config(config);

    sqlx::query(
        r"INSERT INTO tensorzero.function_version_experimentation
           (id, function_version_id, namespace, experimentation_type)
           VALUES ($1, $2, $3, $4)",
    )
    .bind(experimentation_id)
    .bind(function_version_id)
    .bind(namespace)
    .bind(experimentation_type)
    .execute(&mut **tx)
    .await?;

    match &normalized {
        NormalizedExperimentation::Static {
            candidate_variants,
            fallback_variants,
        } => {
            for (variant_name, weight) in candidate_variants {
                sqlx::query(
                    r"INSERT INTO tensorzero.experimentation_static_variants
                       (experimentation_id, variant_name, weight)
                       VALUES ($1, $2, $3)",
                )
                .bind(experimentation_id)
                .bind(variant_name)
                .bind(*weight)
                .execute(&mut **tx)
                .await?;
            }
            for (i, variant_name) in fallback_variants.iter().enumerate() {
                sqlx::query(
                    r"INSERT INTO tensorzero.experimentation_static_fallbacks
                       (experimentation_id, variant_name, position)
                       VALUES ($1, $2, $3)",
                )
                .bind(experimentation_id)
                .bind(variant_name)
                .bind(i as i32)
                .execute(&mut **tx)
                .await?;
            }
        }
        NormalizedExperimentation::Adaptive(adaptive) => {
            sqlx::query(
                r"INSERT INTO tensorzero.experimentation_adaptive_configs
                   (experimentation_id, algorithm, metric,
                    min_samples_per_variant, delta, epsilon, update_period_s,
                    min_prob, max_samples_per_variant)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            )
            .bind(experimentation_id)
            .bind("track_and_stop")
            .bind(adaptive.inner.metric())
            .bind(adaptive.inner.min_samples_per_variant() as i64)
            .bind(adaptive.inner.delta())
            .bind(adaptive.inner.epsilon())
            .bind(adaptive.inner.update_period_s() as i64)
            .bind(adaptive.inner.min_prob())
            .bind(adaptive.inner.max_samples_per_variant().map(|v| v as i64))
            .execute(&mut **tx)
            .await?;

            for (i, variant_name) in adaptive.inner.candidate_variants().iter().enumerate() {
                sqlx::query(
                    r"INSERT INTO tensorzero.experimentation_adaptive_candidates
                       (experimentation_id, variant_name, position)
                       VALUES ($1, $2, $3)",
                )
                .bind(experimentation_id)
                .bind(variant_name)
                .bind(i as i32)
                .execute(&mut **tx)
                .await?;
            }

            for (i, variant_name) in adaptive.inner.fallback_variants().iter().enumerate() {
                sqlx::query(
                    r"INSERT INTO tensorzero.experimentation_adaptive_fallbacks
                       (experimentation_id, variant_name, position)
                       VALUES ($1, $2, $3)",
                )
                .bind(experimentation_id)
                .bind(variant_name)
                .bind(i as i32)
                .execute(&mut **tx)
                .await?;
            }
        }
    }

    Ok(())
}

/// Intermediate type for normalized experimentation config (legacy types resolved).
enum NormalizedExperimentation {
    Static {
        candidate_variants: BTreeMap<String, f64>,
        fallback_variants: Vec<String>,
    },
    Adaptive(UninitializedAdaptiveExperimentationConfig),
}

/// Normalizes legacy experimentation types to Static/Adaptive.
fn normalize_experimentation_config(
    config: &UninitializedExperimentationConfig,
) -> (&'static str, NormalizedExperimentation) {
    match config {
        UninitializedExperimentationConfig::Static(c) => (
            "static",
            NormalizedExperimentation::Static {
                candidate_variants: c.candidate_variants.inner().clone(),
                fallback_variants: c.fallback_variants.clone(),
            },
        ),
        UninitializedExperimentationConfig::Adaptive(c) => {
            ("adaptive", NormalizedExperimentation::Adaptive(c.clone()))
        }
        // Legacy types: normalize on write by converting through the into_static_config path
        UninitializedExperimentationConfig::StaticWeights(c) => {
            let static_config = c.clone().into_static_config();
            (
                "static",
                NormalizedExperimentation::Static {
                    candidate_variants: static_config.candidate_variants.inner().clone(),
                    fallback_variants: static_config.fallback_variants.clone(),
                },
            )
        }
        UninitializedExperimentationConfig::Uniform(c) => {
            let static_config = c.clone().into_static_config();
            match static_config {
                Some(sc) => (
                    "static",
                    NormalizedExperimentation::Static {
                        candidate_variants: sc.candidate_variants.inner().clone(),
                        fallback_variants: sc.fallback_variants.clone(),
                    },
                ),
                // Uniform with no candidates/fallbacks = empty static
                None => (
                    "static",
                    NormalizedExperimentation::Static {
                        candidate_variants: BTreeMap::new(),
                        fallback_variants: Vec::new(),
                    },
                ),
            }
        }
        UninitializedExperimentationConfig::TrackAndStop(c) => (
            "adaptive",
            NormalizedExperimentation::Adaptive(UninitializedAdaptiveExperimentationConfig {
                algorithm: AdaptiveExperimentationAlgorithm::TrackAndStop,
                inner: c.clone(),
            }),
        ),
    }
}

// ---- Evaluator write path ----

async fn write_evaluator(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    function_version_id: Uuid,
    evaluator_name: &str,
    config: &UninitializedEvaluatorConfig,
    creation_source: &str,
) -> Result<Uuid, Error> {
    let evaluator_id = Uuid::now_v7();
    let evaluator_type = match config {
        UninitializedEvaluatorConfig::ExactMatch(_) => "exact_match",
        UninitializedEvaluatorConfig::LLMJudge(_) => "llm_judge",
        UninitializedEvaluatorConfig::ToolUse(_) => "tool_use",
        UninitializedEvaluatorConfig::Regex(_) => "regex",
    };

    sqlx::query(
        r"INSERT INTO tensorzero.function_version_evaluators
           (id, function_version_id, evaluator_name, evaluator_type)
           VALUES ($1, $2, $3, $4)",
    )
    .bind(evaluator_id)
    .bind(function_version_id)
    .bind(evaluator_name)
    .bind(evaluator_type)
    .execute(&mut **tx)
    .await?;

    match config {
        UninitializedEvaluatorConfig::ExactMatch(c) => {
            write_evaluator_exact_match(tx, evaluator_id, c).await?;
        }
        UninitializedEvaluatorConfig::Regex(c) => {
            write_evaluator_regex(tx, evaluator_id, c).await?;
        }
        UninitializedEvaluatorConfig::ToolUse(c) => {
            write_evaluator_tool_use(tx, evaluator_id, c).await?;
        }
        UninitializedEvaluatorConfig::LLMJudge(c) => {
            write_evaluator_llm_judge(tx, evaluator_id, c, creation_source).await?;
        }
    }

    Ok(evaluator_id)
}

#[expect(deprecated)]
async fn write_evaluator_exact_match(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    evaluator_id: Uuid,
    config: &ExactMatchConfig,
) -> Result<(), Error> {
    sqlx::query(
        "INSERT INTO tensorzero.evaluator_exact_match_configs (evaluator_id, cutoff) VALUES ($1, $2)",
    )
    .bind(evaluator_id)
    .bind(config.cutoff)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn write_evaluator_regex(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    evaluator_id: Uuid,
    config: &RegexConfig,
) -> Result<(), Error> {
    sqlx::query(
        r"INSERT INTO tensorzero.evaluator_regex_configs (evaluator_id, must_match, must_not_match)
           VALUES ($1, $2, $3)",
    )
    .bind(evaluator_id)
    .bind(&config.must_match)
    .bind(&config.must_not_match)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn write_evaluator_tool_use(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    evaluator_id: Uuid,
    config: &ToolUseConfig,
) -> Result<(), Error> {
    let (tool_use_type, tools) = match config {
        ToolUseConfig::None => ("none", vec![]),
        ToolUseConfig::NoneOf { tools } => ("none_of", tools.clone()),
        ToolUseConfig::Any => ("any", vec![]),
        ToolUseConfig::AnyOf { tools } => ("any_of", tools.clone()),
        ToolUseConfig::AllOf { tools } => ("all_of", tools.clone()),
    };

    sqlx::query(
        r"INSERT INTO tensorzero.evaluator_tool_use_configs (evaluator_id, tool_use_type)
           VALUES ($1, $2)",
    )
    .bind(evaluator_id)
    .bind(tool_use_type)
    .execute(&mut **tx)
    .await?;

    for tool_name in &tools {
        sqlx::query(
            r"INSERT INTO tensorzero.evaluator_tool_use_tools (evaluator_id, tool_name)
               VALUES ($1, $2)",
        )
        .bind(evaluator_id)
        .bind(tool_name)
        .execute(&mut **tx)
        .await?;
    }

    Ok(())
}

#[expect(deprecated)]
async fn write_evaluator_llm_judge(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    evaluator_id: Uuid,
    config: &UninitializedLLMJudgeConfig,
    creation_source: &str,
) -> Result<(), Error> {
    let input_format = match config.input_format {
        LLMJudgeInputFormat::Serialized => "serialized",
        LLMJudgeInputFormat::Messages => "messages",
    };
    let output_type = match config.output_type {
        LLMJudgeOutputType::Float => "float",
        LLMJudgeOutputType::Boolean => "boolean",
    };
    let optimize = match config.optimize {
        LLMJudgeOptimize::Min => "min",
        LLMJudgeOptimize::Max => "max",
    };

    sqlx::query(
        r"INSERT INTO tensorzero.evaluator_llm_judge_configs
           (evaluator_id, input_format, output_type, optimize, include_reference_output, cutoff, description)
           VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(evaluator_id)
    .bind(input_format)
    .bind(output_type)
    .bind(optimize)
    .bind(config.include.reference_output)
    .bind(config.cutoff)
    .bind(&config.description)
    .execute(&mut **tx)
    .await?;

    // Write each variant
    for (variant_name, variant_info) in &config.variants {
        write_evaluator_llm_judge_variant(
            tx,
            evaluator_id,
            variant_name,
            variant_info,
            creation_source,
        )
        .await?;
    }

    Ok(())
}

async fn write_evaluator_llm_judge_variant(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    evaluator_id: Uuid,
    variant_name: &str,
    variant_info: &UninitializedLLMJudgeVariantInfo,
    creation_source: &str,
) -> Result<(), Error> {
    let judge_variant_id = Uuid::now_v7();

    let variant_type = match &variant_info.inner {
        UninitializedLLMJudgeVariantConfig::ChatCompletion(_) => "chat_completion",
        UninitializedLLMJudgeVariantConfig::BestOfNSampling(_) => "experimental_best_of_n_sampling",
        UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(_) => "experimental_mixture_of_n",
        UninitializedLLMJudgeVariantConfig::Dicl(_) => "experimental_dynamic_in_context_learning",
        UninitializedLLMJudgeVariantConfig::ChainOfThought(_) => "experimental_chain_of_thought",
    };

    let (non_streaming_total_ms, streaming_ttft_ms, streaming_total_ms) =
        timeouts_to_ms(variant_info.timeouts.as_ref());

    sqlx::query(
        r"INSERT INTO tensorzero.evaluator_llm_judge_variants
           (id, evaluator_id, variant_name, variant_type,
            timeouts_non_streaming_total_ms, timeouts_streaming_ttft_ms, timeouts_streaming_total_ms)
           VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(judge_variant_id)
    .bind(evaluator_id)
    .bind(variant_name)
    .bind(variant_type)
    .bind(non_streaming_total_ms)
    .bind(streaming_ttft_ms)
    .bind(streaming_total_ms)
    .execute(&mut **tx)
    .await?;

    match &variant_info.inner {
        UninitializedLLMJudgeVariantConfig::ChatCompletion(cc) => {
            write_evaluator_llm_judge_cc(tx, judge_variant_id, cc, creation_source).await?;
        }
        other => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Unsupported LLM judge variant type for DB storage: `{}`",
                    variant_type_display(other)
                ),
            }));
        }
    }

    Ok(())
}

fn variant_type_display(config: &UninitializedLLMJudgeVariantConfig) -> &'static str {
    match config {
        UninitializedLLMJudgeVariantConfig::ChatCompletion(_) => "chat_completion",
        UninitializedLLMJudgeVariantConfig::BestOfNSampling(_) => "best_of_n_sampling",
        UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(_) => "mixture_of_n",
        UninitializedLLMJudgeVariantConfig::Dicl(_) => "dicl",
        UninitializedLLMJudgeVariantConfig::ChainOfThought(_) => "chain_of_thought",
    }
}

async fn write_evaluator_llm_judge_cc(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    judge_variant_id: Uuid,
    config: &UninitializedLLMJudgeChatCompletionVariantConfig,
    creation_source: &str,
) -> Result<(), Error> {
    // Write prompt template for system_instructions
    let prompt_id = Uuid::now_v7();
    variant_version_queries::insert_prompt_template_version(
        tx,
        prompt_id,
        &config.system_instructions.get_template_key(),
        config.system_instructions.data(),
        creation_source,
    )
    .await?;

    let json_mode_str = serialize_json_mode(config.json_mode)?;
    let service_tier_str = config
        .service_tier
        .as_ref()
        .map(serialize_service_tier)
        .transpose()?;

    sqlx::query(
        r"INSERT INTO tensorzero.evaluator_llm_judge_cc_configs (
            judge_variant_id, active, model,
            system_instructions_prompt_id, system_instructions_key,
            temperature, top_p, max_tokens, presence_penalty, frequency_penalty,
            seed, json_mode, stop_sequences, reasoning_effort, service_tier,
            thinking_budget_tokens, verbosity, num_retries, max_retry_delay_s
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
        )",
    )
    .bind(judge_variant_id)
    .bind(config.active)
    .bind(config.model.as_ref())
    .bind(prompt_id)
    .bind(config.system_instructions.get_template_key())
    .bind(config.temperature)
    .bind(config.top_p)
    .bind(config.max_tokens.map(|v| v as i32))
    .bind(config.presence_penalty)
    .bind(config.frequency_penalty)
    .bind(config.seed.map(|v| v as i32))
    .bind(&json_mode_str)
    .bind(&config.stop_sequences)
    .bind(&config.reasoning_effort)
    .bind(&service_tier_str)
    .bind(config.thinking_budget_tokens)
    .bind(&config.verbosity)
    .bind(config.retries.num_retries as i32)
    .bind(config.retries.max_delay_s)
    .execute(&mut **tx)
    .await?;

    // Write extra_body rows
    variant_version_queries::write_extra_body_rows(
        tx,
        judge_variant_id,
        config.extra_body.as_ref(),
        INSERT_LLM_JUDGE_CC_EXTRA_BODY,
    )
    .await?;

    // Write extra_headers rows
    variant_version_queries::write_extra_headers_rows(
        tx,
        judge_variant_id,
        config.extra_headers.as_ref(),
        INSERT_LLM_JUDGE_CC_EXTRA_HEADERS,
    )
    .await?;

    Ok(())
}

const INSERT_LLM_JUDGE_CC_EXTRA_BODY: &str = "INSERT INTO tensorzero.evaluator_llm_judge_cc_extra_body (judge_variant_id, position, pointer, kind, replacement_value) VALUES ($1, $2, $3, $4, $5)";
const INSERT_LLM_JUDGE_CC_EXTRA_HEADERS: &str = "INSERT INTO tensorzero.evaluator_llm_judge_cc_extra_headers (judge_variant_id, position, header_name, kind, header_value) VALUES ($1, $2, $3, $4, $5)";

// ===========================================================================
// Read path
// ===========================================================================

/// Reads a function version from the database and rehydrates it.
/// Returns (function_name, UninitializedFunctionConfig).
pub async fn read_function_version(
    pool: &PgPool,
    function_version_id: Uuid,
) -> Result<(String, UninitializedFunctionConfig), Error> {
    // Load the function_version row
    let fv_row: FunctionVersionRow = sqlx::query_as(
        r"SELECT id, function_id, function_type,
                  system_schema_prompt_id, system_schema_key,
                  user_schema_prompt_id, user_schema_key,
                  assistant_schema_prompt_id, assistant_schema_key,
                  output_schema_prompt_id, output_schema_key,
                  tools, tool_choice, parallel_tool_calls,
                  description, creation_source, source_autopilot_session_id
           FROM tensorzero.function_versions
           WHERE id = $1",
    )
    .bind(function_version_id)
    .fetch_one(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to load function version `{function_version_id}`: {e}"),
        })
    })?;

    // Load function name
    let function_row: FunctionRow = sqlx::query_as(
        "SELECT id, name, active_version_id FROM tensorzero.functions WHERE id = $1",
    )
    .bind(fv_row.function_id)
    .fetch_one(pool)
    .await?;

    // Load schemas
    let schema_rows: Vec<FunctionVersionSchemaRow> = sqlx::query_as(
        r"SELECT function_version_id, schema_name, prompt_template_version_id, template_key
           FROM tensorzero.function_version_schemas
           WHERE function_version_id = $1",
    )
    .bind(function_version_id)
    .fetch_all(pool)
    .await?;

    // Collect all prompt IDs for schemas
    let mut prompt_ids: Vec<Uuid> = Vec::new();
    if let Some(id) = fv_row.system_schema_prompt_id {
        prompt_ids.push(id);
    }
    if let Some(id) = fv_row.user_schema_prompt_id {
        prompt_ids.push(id);
    }
    if let Some(id) = fv_row.assistant_schema_prompt_id {
        prompt_ids.push(id);
    }
    if let Some(id) = fv_row.output_schema_prompt_id {
        prompt_ids.push(id);
    }
    for s in &schema_rows {
        prompt_ids.push(s.prompt_template_version_id);
    }

    let prompt_rows =
        variant_version_queries::load_prompt_template_versions(pool, &prompt_ids).await?;

    // Rehydrate schema refs
    let system_schema = variant_version_queries::rehydrate_prompt_ref(
        fv_row.system_schema_prompt_id,
        fv_row.system_schema_key.as_ref(),
        &prompt_rows,
    )?;
    let user_schema = variant_version_queries::rehydrate_prompt_ref(
        fv_row.user_schema_prompt_id,
        fv_row.user_schema_key.as_ref(),
        &prompt_rows,
    )?;
    let assistant_schema = variant_version_queries::rehydrate_prompt_ref(
        fv_row.assistant_schema_prompt_id,
        fv_row.assistant_schema_key.as_ref(),
        &prompt_rows,
    )?;
    let output_schema = variant_version_queries::rehydrate_prompt_ref(
        fv_row.output_schema_prompt_id,
        fv_row.output_schema_key.as_ref(),
        &prompt_rows,
    )?;

    // Rehydrate named schemas
    let mut schemas_map = HashMap::new();
    for s_row in &schema_rows {
        let prompt_row = prompt_rows
            .get(&s_row.prompt_template_version_id)
            .ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Missing prompt template version row for schema `{}`",
                        s_row.schema_name
                    ),
                })
            })?;
        schemas_map.insert(
            s_row.schema_name.clone(),
            ResolvedTomlPathData::new_fake_path(
                s_row.template_key.clone(),
                prompt_row.source_body.clone(),
            ),
        );
    }
    let schemas = UninitializedSchemas::from_paths(schemas_map);

    // Load variant junction rows
    let variant_junction_rows: Vec<FunctionVersionVariantRow> = sqlx::query_as(
        r"SELECT function_version_id, variant_name, variant_version_id
           FROM tensorzero.function_version_variants
           WHERE function_version_id = $1",
    )
    .bind(function_version_id)
    .fetch_all(pool)
    .await?;

    // Read each variant
    let mut variants = HashMap::new();
    for vj in &variant_junction_rows {
        let info =
            variant_version_queries::read_variant_version(pool, vj.variant_version_id).await?;
        variants.insert(vj.variant_name.clone(), info);
    }

    // Load evaluators
    let evaluators = read_evaluators(pool, function_version_id).await?;

    // Load experimentation config
    let experimentation = read_experimentation(pool, function_version_id).await?;

    // Build the config based on function_type
    let config = match fv_row.function_type.as_str() {
        "chat" => UninitializedFunctionConfig::Chat(UninitializedFunctionConfigChat {
            variants,
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            tools: fv_row.tools,
            tool_choice: string_to_tool_choice(&fv_row.tool_choice),
            parallel_tool_calls: fv_row.parallel_tool_calls,
            description: fv_row.description,
            experimentation,
            evaluators,
        }),
        "json" => UninitializedFunctionConfig::Json(UninitializedFunctionConfigJson {
            variants,
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            output_schema,
            description: fv_row.description,
            experimentation,
            evaluators,
        }),
        other => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Unknown function type `{other}`"),
            }));
        }
    };

    Ok((function_row.name, config))
}

// ---- Evaluator read path ----

async fn read_evaluators(
    pool: &PgPool,
    function_version_id: Uuid,
) -> Result<HashMap<String, UninitializedEvaluatorConfig>, Error> {
    let evaluator_rows: Vec<EvaluatorRow> = sqlx::query_as(
        r"SELECT id, function_version_id, evaluator_name, evaluator_type
           FROM tensorzero.function_version_evaluators
           WHERE function_version_id = $1",
    )
    .bind(function_version_id)
    .fetch_all(pool)
    .await?;

    let mut evaluators = HashMap::new();
    for row in &evaluator_rows {
        let config = match row.evaluator_type.as_str() {
            "exact_match" => read_evaluator_exact_match(pool, row.id).await?,
            "regex" => read_evaluator_regex(pool, row.id).await?,
            "tool_use" => read_evaluator_tool_use(pool, row.id).await?,
            "llm_judge" => read_evaluator_llm_judge(pool, row.id).await?,
            other => {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: format!("Unknown evaluator type `{other}`"),
                }));
            }
        };
        evaluators.insert(row.evaluator_name.clone(), config);
    }
    Ok(evaluators)
}

#[expect(deprecated)]
async fn read_evaluator_exact_match(
    pool: &PgPool,
    evaluator_id: Uuid,
) -> Result<UninitializedEvaluatorConfig, Error> {
    let row: EvaluatorExactMatchRow = sqlx::query_as(
        "SELECT evaluator_id, cutoff FROM tensorzero.evaluator_exact_match_configs WHERE evaluator_id = $1",
    )
    .bind(evaluator_id)
    .fetch_one(pool)
    .await?;
    Ok(UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig {
        cutoff: row.cutoff,
    }))
}

async fn read_evaluator_regex(
    pool: &PgPool,
    evaluator_id: Uuid,
) -> Result<UninitializedEvaluatorConfig, Error> {
    let row: EvaluatorRegexRow = sqlx::query_as(
        r"SELECT evaluator_id, must_match, must_not_match
           FROM tensorzero.evaluator_regex_configs
           WHERE evaluator_id = $1",
    )
    .bind(evaluator_id)
    .fetch_one(pool)
    .await?;
    Ok(UninitializedEvaluatorConfig::Regex(RegexConfig {
        must_match: row.must_match,
        must_not_match: row.must_not_match,
    }))
}

async fn read_evaluator_tool_use(
    pool: &PgPool,
    evaluator_id: Uuid,
) -> Result<UninitializedEvaluatorConfig, Error> {
    let row: EvaluatorToolUseRow = sqlx::query_as(
        r"SELECT evaluator_id, tool_use_type
           FROM tensorzero.evaluator_tool_use_configs
           WHERE evaluator_id = $1",
    )
    .bind(evaluator_id)
    .fetch_one(pool)
    .await?;

    let tools: Vec<EvaluatorToolUseToolRow> = sqlx::query_as(
        r"SELECT evaluator_id, tool_name
           FROM tensorzero.evaluator_tool_use_tools
           WHERE evaluator_id = $1",
    )
    .bind(evaluator_id)
    .fetch_all(pool)
    .await?;

    let tool_names: Vec<String> = tools.iter().map(|t| t.tool_name.clone()).collect();

    let config = match row.tool_use_type.as_str() {
        "none" => ToolUseConfig::None,
        "none_of" => ToolUseConfig::NoneOf { tools: tool_names },
        "any" => ToolUseConfig::Any,
        "any_of" => ToolUseConfig::AnyOf { tools: tool_names },
        "all_of" => ToolUseConfig::AllOf { tools: tool_names },
        other => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Unknown tool_use_type `{other}`"),
            }));
        }
    };

    Ok(UninitializedEvaluatorConfig::ToolUse(config))
}

#[expect(deprecated)]
async fn read_evaluator_llm_judge(
    pool: &PgPool,
    evaluator_id: Uuid,
) -> Result<UninitializedEvaluatorConfig, Error> {
    let lj_row: EvaluatorLLMJudgeConfigRow = sqlx::query_as(
        r"SELECT evaluator_id, input_format, output_type, optimize,
                  include_reference_output, cutoff, description
           FROM tensorzero.evaluator_llm_judge_configs
           WHERE evaluator_id = $1",
    )
    .bind(evaluator_id)
    .fetch_one(pool)
    .await?;

    let input_format = match lj_row.input_format.as_str() {
        "serialized" => LLMJudgeInputFormat::Serialized,
        "messages" => LLMJudgeInputFormat::Messages,
        other => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Unknown LLM judge input_format `{other}`"),
            }));
        }
    };

    let output_type = match lj_row.output_type.as_str() {
        "float" => LLMJudgeOutputType::Float,
        "boolean" => LLMJudgeOutputType::Boolean,
        other => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Unknown LLM judge output_type `{other}`"),
            }));
        }
    };

    let optimize = match lj_row.optimize.as_str() {
        "min" => LLMJudgeOptimize::Min,
        "max" => LLMJudgeOptimize::Max,
        other => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Unknown LLM judge optimize `{other}`"),
            }));
        }
    };

    // Load variants
    let variant_rows: Vec<EvaluatorLLMJudgeVariantRow> = sqlx::query_as(
        r"SELECT id, evaluator_id, variant_name, variant_type,
                  timeouts_non_streaming_total_ms, timeouts_streaming_ttft_ms,
                  timeouts_streaming_total_ms
           FROM tensorzero.evaluator_llm_judge_variants
           WHERE evaluator_id = $1",
    )
    .bind(evaluator_id)
    .fetch_all(pool)
    .await?;

    let mut variants = HashMap::new();
    for vrow in &variant_rows {
        let timeouts = ms_to_timeouts(
            vrow.timeouts_non_streaming_total_ms,
            vrow.timeouts_streaming_ttft_ms,
            vrow.timeouts_streaming_total_ms,
        );

        let inner = match vrow.variant_type.as_str() {
            "chat_completion" => {
                let cc = read_evaluator_llm_judge_cc(pool, vrow.id).await?;
                UninitializedLLMJudgeVariantConfig::ChatCompletion(cc)
            }
            other => {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: format!(
                        "Unsupported LLM judge variant type for DB loading: `{other}`"
                    ),
                }));
            }
        };

        variants.insert(
            vrow.variant_name.clone(),
            UninitializedLLMJudgeVariantInfo { inner, timeouts },
        );
    }

    Ok(UninitializedEvaluatorConfig::LLMJudge(
        UninitializedLLMJudgeConfig {
            input_format,
            variants,
            output_type,
            optimize,
            include: LLMJudgeIncludeConfig {
                reference_output: lj_row.include_reference_output,
            },
            cutoff: lj_row.cutoff,
            description: lj_row.description,
        },
    ))
}

async fn read_evaluator_llm_judge_cc(
    pool: &PgPool,
    judge_variant_id: Uuid,
) -> Result<UninitializedLLMJudgeChatCompletionVariantConfig, Error> {
    let cc_row: EvaluatorLLMJudgeCCRow = sqlx::query_as(
        r"SELECT judge_variant_id, active, model,
                  system_instructions_prompt_id, system_instructions_key,
                  temperature, top_p, max_tokens, presence_penalty, frequency_penalty,
                  seed, json_mode, stop_sequences, reasoning_effort, service_tier,
                  thinking_budget_tokens, verbosity, num_retries, max_retry_delay_s
           FROM tensorzero.evaluator_llm_judge_cc_configs
           WHERE judge_variant_id = $1",
    )
    .bind(judge_variant_id)
    .fetch_one(pool)
    .await?;

    // Load the prompt template for system_instructions
    let prompt_rows = variant_version_queries::load_prompt_template_versions(
        pool,
        &[cc_row.system_instructions_prompt_id],
    )
    .await?;

    let system_instructions = variant_version_queries::rehydrate_prompt_ref(
        Some(cc_row.system_instructions_prompt_id),
        Some(&cc_row.system_instructions_key),
        &prompt_rows,
    )?
    .ok_or_else(|| {
        Error::new(ErrorDetails::InternalError {
            message: "LLM judge system_instructions is required but missing".to_string(),
        })
    })?;

    let json_mode = deserialize_json_mode(&cc_row.json_mode)?;

    let service_tier: Option<ServiceTier> = cc_row
        .service_tier
        .as_deref()
        .map(deserialize_service_tier)
        .transpose()?;

    // Load extra_body and extra_headers
    let extra_body_rows: Vec<ExtraBodyRow> = sqlx::query_as(
        r"SELECT judge_variant_id AS variant_version_id, position, pointer, kind, replacement_value
           FROM tensorzero.evaluator_llm_judge_cc_extra_body
           WHERE judge_variant_id = $1
           ORDER BY position",
    )
    .bind(judge_variant_id)
    .fetch_all(pool)
    .await?;

    let extra_header_rows: Vec<ExtraHeaderRow> = sqlx::query_as(
        r"SELECT judge_variant_id AS variant_version_id, position, header_name, kind, header_value
           FROM tensorzero.evaluator_llm_judge_cc_extra_headers
           WHERE judge_variant_id = $1
           ORDER BY position",
    )
    .bind(judge_variant_id)
    .fetch_all(pool)
    .await?;

    let extra_body = variant_version_queries::rehydrate_extra_body(&extra_body_rows)?;
    let extra_headers = variant_version_queries::rehydrate_extra_headers(&extra_header_rows)?;

    Ok(UninitializedLLMJudgeChatCompletionVariantConfig {
        active: cc_row.active,
        model: Arc::from(cc_row.model.as_str()),
        system_instructions,
        temperature: cc_row.temperature,
        top_p: cc_row.top_p,
        max_tokens: cc_row.max_tokens.map(|v| v as u32),
        presence_penalty: cc_row.presence_penalty,
        frequency_penalty: cc_row.frequency_penalty,
        seed: cc_row.seed.map(|v| v as u32),
        json_mode,
        stop_sequences: cc_row.stop_sequences,
        reasoning_effort: cc_row.reasoning_effort,
        service_tier,
        thinking_budget_tokens: cc_row.thinking_budget_tokens,
        verbosity: cc_row.verbosity,
        retries: RetryConfig {
            num_retries: cc_row.num_retries as usize,
            max_delay_s: cc_row.max_retry_delay_s,
        },
        extra_body,
        extra_headers,
    })
}

// ---- Experimentation read path ----

async fn read_experimentation(
    pool: &PgPool,
    function_version_id: Uuid,
) -> Result<Option<UninitializedExperimentationConfigWithNamespaces>, Error> {
    let rows: Vec<ExperimentationRow> = sqlx::query_as(
        r"SELECT id, function_version_id, namespace, experimentation_type
           FROM tensorzero.function_version_experimentation
           WHERE function_version_id = $1",
    )
    .bind(function_version_id)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(None);
    }

    let mut base: Option<UninitializedExperimentationConfig> = None;
    let mut namespaces = HashMap::new();

    for row in &rows {
        let config = match row.experimentation_type.as_str() {
            "static" => read_static_experimentation(pool, row.id).await?,
            "adaptive" => read_adaptive_experimentation(pool, row.id).await?,
            other => {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: format!("Unknown experimentation type `{other}`"),
                }));
            }
        };

        match &row.namespace {
            None => base = Some(config),
            Some(ns) => {
                namespaces.insert(ns.clone(), config);
            }
        }
    }

    let base = base.ok_or_else(|| {
        Error::new(ErrorDetails::InternalError {
            message: format!(
                "Experimentation config for function version `{function_version_id}` \
                 has namespace overrides but no base config"
            ),
        })
    })?;

    Ok(Some(UninitializedExperimentationConfigWithNamespaces {
        base,
        namespaces,
    }))
}

async fn read_static_experimentation(
    pool: &PgPool,
    experimentation_id: Uuid,
) -> Result<UninitializedExperimentationConfig, Error> {
    let variant_rows: Vec<ExperimentationStaticVariantRow> = sqlx::query_as(
        r"SELECT experimentation_id, variant_name, weight
           FROM tensorzero.experimentation_static_variants
           WHERE experimentation_id = $1",
    )
    .bind(experimentation_id)
    .fetch_all(pool)
    .await?;

    let fallback_rows: Vec<ExperimentationStaticFallbackRow> = sqlx::query_as(
        r"SELECT experimentation_id, variant_name, position
           FROM tensorzero.experimentation_static_fallbacks
           WHERE experimentation_id = $1
           ORDER BY position",
    )
    .bind(experimentation_id)
    .fetch_all(pool)
    .await?;

    let candidate_variants = WeightedVariants::from_map(
        variant_rows
            .iter()
            .map(|v| (v.variant_name.clone(), v.weight))
            .collect(),
    );

    let fallback_variants: Vec<String> = fallback_rows
        .iter()
        .map(|f| f.variant_name.clone())
        .collect();

    Ok(UninitializedExperimentationConfig::Static(
        StaticExperimentationConfig {
            candidate_variants,
            fallback_variants,
        },
    ))
}

async fn read_adaptive_experimentation(
    pool: &PgPool,
    experimentation_id: Uuid,
) -> Result<UninitializedExperimentationConfig, Error> {
    let ac_row: ExperimentationAdaptiveConfigRow = sqlx::query_as(
        r"SELECT experimentation_id, algorithm, metric,
                  min_samples_per_variant, delta, epsilon, update_period_s,
                  min_prob, max_samples_per_variant
           FROM tensorzero.experimentation_adaptive_configs
           WHERE experimentation_id = $1",
    )
    .bind(experimentation_id)
    .fetch_one(pool)
    .await?;

    let candidate_rows: Vec<ExperimentationAdaptiveCandidateRow> = sqlx::query_as(
        r"SELECT experimentation_id, variant_name, position
           FROM tensorzero.experimentation_adaptive_candidates
           WHERE experimentation_id = $1
           ORDER BY position",
    )
    .bind(experimentation_id)
    .fetch_all(pool)
    .await?;

    let fallback_rows: Vec<ExperimentationAdaptiveFallbackRow> = sqlx::query_as(
        r"SELECT experimentation_id, variant_name, position
           FROM tensorzero.experimentation_adaptive_fallbacks
           WHERE experimentation_id = $1
           ORDER BY position",
    )
    .bind(experimentation_id)
    .fetch_all(pool)
    .await?;

    let candidate_variants: Vec<String> = candidate_rows
        .iter()
        .map(|c| c.variant_name.clone())
        .collect();
    let fallback_variants: Vec<String> = fallback_rows
        .iter()
        .map(|f| f.variant_name.clone())
        .collect();

    let inner = UninitializedTrackAndStopExperimentationConfig::from_fields(
        ac_row.metric,
        candidate_variants,
        fallback_variants,
        ac_row.min_samples_per_variant as u64,
        ac_row.delta,
        ac_row.epsilon,
        ac_row.update_period_s as u64,
        ac_row.min_prob,
        ac_row.max_samples_per_variant.map(|v| v as u64),
    );

    Ok(UninitializedExperimentationConfig::Adaptive(
        UninitializedAdaptiveExperimentationConfig {
            algorithm: AdaptiveExperimentationAlgorithm::TrackAndStop,
            inner,
        },
    ))
}

// ===========================================================================
// Load latest / merge helpers
// ===========================================================================

/// Row for the "active version per function" query.
#[derive(Debug, sqlx::FromRow)]
struct ActiveFunctionVersionRow {
    active_version_id: Uuid,
    name: String,
}

/// Loads the active function version for each function that has one.
/// Returns a map from function_name -> UninitializedFunctionConfig.
pub async fn load_all_active_function_versions(
    pool: &PgPool,
) -> Result<HashMap<String, UninitializedFunctionConfig>, Error> {
    let rows: Vec<ActiveFunctionVersionRow> = sqlx::query_as(
        r"SELECT name, active_version_id
           FROM tensorzero.functions
           WHERE active_version_id IS NOT NULL AND deleted_at IS NULL",
    )
    .fetch_all(pool)
    .await?;

    let mut result = HashMap::with_capacity(rows.len());
    for row in rows {
        let (function_name, config) = read_function_version(pool, row.active_version_id).await?;
        debug_assert_eq!(function_name, row.name);
        result.insert(function_name, config);
    }

    Ok(result)
}

/// Merges database-sourced function configs into an uninitialized config.
///
/// For each DB function:
/// - If the function exists in the config, its variants and evaluators from the DB
///   are merged (DB wins on conflicts).
/// - If the function doesn't exist in the config, it is added.
pub fn merge_db_functions(
    config: &mut crate::config::UninitializedConfig,
    db_functions: HashMap<String, UninitializedFunctionConfig>,
) {
    for (function_name, db_function_config) in db_functions {
        config.functions.insert(function_name, db_function_config);
    }
}
