//! Stored types for function versions persisted in the database as JSONB.
//!
//! These types are the durable schema for `function_versions.config`. The function type
//! discriminator (`"chat"` or `"json"`) lives on the row, not in the JSONB blob.
//!
//! Schema evolution is controlled by `schema_version` on the row.
//!
//! Key design decisions:
//! - **Schemas and LLM judge system_instructions** use `StoredPromptRef` (replacing `ResolvedTomlPathData`).
//! - **Experimentation** serializes directly as `UninitializedExperimentationConfigWithNamespaces`.
//! - **Evaluators** need stored types because LLM judge variants contain prompt references.
//! - **Variants** become `StoredVariantRef` (UUID pointers to `variant_versions` rows).

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::TimeoutsConfig;
use crate::config::path::ResolvedTomlPathData;
use crate::config::variant_versions::{
    PromptTemplateVersionRow, StoredPromptRef, rehydrate_prompt_ref, rehydrate_prompt_ref_opt,
    resolve_prompt_ref, resolve_prompt_ref_opt,
};
use crate::config::{
    UninitializedFunctionConfig, UninitializedFunctionConfigChat, UninitializedFunctionConfigJson,
    UninitializedSchemas, UninitializedVariantInfo,
};
use crate::error::{Error, ErrorDetails};
use crate::evaluations::{
    ExactMatchConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat, LLMJudgeOptimize,
    LLMJudgeOutputType, RegexConfig, ToolUseConfig, UninitializedEvaluatorConfig,
    UninitializedLLMJudgeBestOfNVariantConfig, UninitializedLLMJudgeChainOfThoughtVariantConfig,
    UninitializedLLMJudgeChatCompletionVariantConfig, UninitializedLLMJudgeConfig,
    UninitializedLLMJudgeDiclVariantConfig, UninitializedLLMJudgeMixtureOfNVariantConfig,
    UninitializedLLMJudgeVariantConfig, UninitializedLLMJudgeVariantInfo,
};
use crate::experimentation::UninitializedExperimentationConfigWithNamespaces;
use crate::inference::types::chat_completion_inference_params::ServiceTier;
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::utils::retries::RetryConfig;
use crate::variant::JsonMode;
use tensorzero_types::ToolChoice;

/// Current schema version for newly written function versions.
pub const CURRENT_SCHEMA_VERSION: i32 = 1;

// ─── StoredVariantRef ───────────────────────────────────────────────────────

/// A reference to a row in `variant_versions`, used in place of inline variant configs.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredVariantRef {
    pub variant_version_id: Uuid,
}

// ─── Stored function config (schema_version = 1) ───────────────────────────

/// The JSONB blob stored in `function_versions.config`.
/// Uses a single struct for both chat and json function types — the discriminator
/// is on the row (`function_type` column), not inside the JSONB.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StoredFunctionConfig {
    pub variants: HashMap<String, StoredVariantRef>,

    // ── Schemas (shared by chat and json) ──
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_schema: Option<StoredPromptRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_schema: Option<StoredPromptRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assistant_schema: Option<StoredPromptRef>,
    #[serde(default)]
    pub schemas: HashMap<String, StoredPromptRef>,

    // ── Chat-only fields ──
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    // ── Json-only fields ──
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<StoredPromptRef>,

    // ── Common fields ──
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    // Experimentation serializes directly — no prompt refs inside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experimentation: Option<UninitializedExperimentationConfigWithNamespaces>,

    // Evaluators need stored types because LLM judge variants contain prompt refs.
    #[serde(default)]
    pub evaluators: HashMap<String, StoredEvaluatorConfig>,
}

// ─── Stored evaluator types ─────────────────────────────────────────────────

/// Stored form of `UninitializedEvaluatorConfig`.
/// ExactMatch, Regex, and ToolUse have no prompt refs and reuse their original types.
/// LLMJudge needs stored types for its variant configs.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredEvaluatorConfig {
    ExactMatch(ExactMatchConfig),
    #[serde(rename = "llm_judge")]
    LLMJudge(StoredLLMJudgeConfig),
    ToolUse(ToolUseConfig),
    Regex(RegexConfig),
}

/// Stored form of `UninitializedLLMJudgeConfig`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StoredLLMJudgeConfig {
    #[serde(default)]
    pub input_format: LLMJudgeInputFormat,
    pub variants: HashMap<String, StoredLLMJudgeVariantInfo>,
    pub output_type: LLMJudgeOutputType,
    pub optimize: LLMJudgeOptimize,
    #[serde(default)]
    pub include: LLMJudgeIncludeConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cutoff: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Stored form of `UninitializedLLMJudgeVariantInfo`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StoredLLMJudgeVariantInfo {
    #[serde(flatten)]
    pub inner: StoredLLMJudgeVariantConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeouts: Option<TimeoutsConfig>,
}

/// Stored form of `UninitializedLLMJudgeVariantConfig`.
/// Uses adjacently-tagged enum to keep variant-specific fields separate from `timeouts`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "snake_case")]
pub enum StoredLLMJudgeVariantConfig {
    ChatCompletion(StoredLLMJudgeChatCompletionConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(StoredLLMJudgeBestOfNConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfNSampling(StoredLLMJudgeMixtureOfNConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(StoredLLMJudgeDiclConfig),
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(StoredLLMJudgeChainOfThoughtConfig),
}

/// Stored form of `UninitializedLLMJudgeChatCompletionVariantConfig`.
/// `system_instructions` is replaced with `StoredPromptRef`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredLLMJudgeChatCompletionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
    pub model: Arc<str>,
    pub system_instructions: StoredPromptRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    pub json_mode: JsonMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    #[serde(default)]
    pub retries: RetryConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<ExtraBodyConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_headers: Option<ExtraHeadersConfig>,
}

/// Stored form of `UninitializedLLMJudgeBestOfNVariantConfig`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredLLMJudgeBestOfNConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
    // Deprecated `timeout_s` is canonicalized away on write — not stored.
    #[serde(default)]
    pub candidates: Vec<String>,
    pub evaluator: StoredLLMJudgeChatCompletionConfig,
}

/// Stored form of `UninitializedLLMJudgeMixtureOfNVariantConfig`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredLLMJudgeMixtureOfNConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
    // Deprecated `timeout_s` is canonicalized away on write — not stored.
    #[serde(default)]
    pub candidates: Vec<String>,
    pub fuser: StoredLLMJudgeChatCompletionConfig,
}

/// Stored form of `UninitializedLLMJudgeDiclVariantConfig`.
/// `system_instructions` is replaced with `StoredPromptRef`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredLLMJudgeDiclConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
    pub embedding_model: String,
    pub k: u32,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instructions: Option<StoredPromptRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_mode: Option<JsonMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<ExtraBodyConfig>,
    #[serde(default)]
    pub retries: RetryConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_headers: Option<ExtraHeadersConfig>,
}

/// Stored form of `UninitializedLLMJudgeChainOfThoughtVariantConfig`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredLLMJudgeChainOfThoughtConfig {
    #[serde(flatten)]
    pub inner: StoredLLMJudgeChatCompletionConfig,
}

// ─── Schema version dispatch ────────────────────────────────────────────────

/// Deserialize a `StoredFunctionConfig` from a JSONB value, dispatching on `schema_version`.
pub fn deserialize_stored_function_config(
    schema_version: i32,
    config: serde_json::Value,
) -> Result<StoredFunctionConfig, Error> {
    match schema_version {
        1 => serde_json::from_value(config).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to deserialize function version (schema_version=1): {e}"),
            })
        }),
        _ => Err(Error::new(ErrorDetails::Config {
            message: format!("Unknown function version schema_version: {schema_version}"),
        })),
    }
}

// ─── Write path: UninitializedFunctionConfig → StoredFunctionConfig ─────────

/// Collects all `ResolvedTomlPathData` references from a function config
/// (schemas and evaluator LLM judge system_instructions).
/// Returns (template_key, template_data) pairs.
pub fn collect_function_prompt_templates(
    config: &UninitializedFunctionConfig,
) -> Vec<(String, String)> {
    let mut templates = Vec::new();
    match config {
        UninitializedFunctionConfig::Chat(c) => {
            collect_schema_templates(
                c.system_schema.as_ref(),
                c.user_schema.as_ref(),
                c.assistant_schema.as_ref(),
                &c.schemas,
                &mut templates,
            );
            collect_evaluator_templates(&c.evaluators, &mut templates);
        }
        UninitializedFunctionConfig::Json(c) => {
            collect_schema_templates(
                c.system_schema.as_ref(),
                c.user_schema.as_ref(),
                c.assistant_schema.as_ref(),
                &c.schemas,
                &mut templates,
            );
            if let Some(ref os) = c.output_schema {
                templates.push((os.get_template_key(), os.data().to_string()));
            }
            collect_evaluator_templates(&c.evaluators, &mut templates);
        }
    }
    templates
}

fn collect_schema_templates(
    system_schema: Option<&ResolvedTomlPathData>,
    user_schema: Option<&ResolvedTomlPathData>,
    assistant_schema: Option<&ResolvedTomlPathData>,
    schemas: &UninitializedSchemas,
    templates: &mut Vec<(String, String)>,
) {
    if let Some(s) = system_schema {
        templates.push((s.get_template_key(), s.data().to_string()));
    }
    if let Some(s) = user_schema {
        templates.push((s.get_template_key(), s.data().to_string()));
    }
    if let Some(s) = assistant_schema {
        templates.push((s.get_template_key(), s.data().to_string()));
    }
    for (_, schema) in schemas.iter() {
        templates.push((
            schema.path.get_template_key(),
            schema.path.data().to_string(),
        ));
    }
}

fn collect_evaluator_templates(
    evaluators: &HashMap<String, UninitializedEvaluatorConfig>,
    templates: &mut Vec<(String, String)>,
) {
    for eval in evaluators.values() {
        if let UninitializedEvaluatorConfig::LLMJudge(judge) = eval {
            for variant_info in judge.variants.values() {
                collect_llm_judge_variant_templates(&variant_info.inner, templates);
            }
        }
    }
}

fn collect_llm_judge_variant_templates(
    variant: &UninitializedLLMJudgeVariantConfig,
    templates: &mut Vec<(String, String)>,
) {
    match variant {
        UninitializedLLMJudgeVariantConfig::ChatCompletion(c) => {
            templates.push((
                c.system_instructions.get_template_key(),
                c.system_instructions.data().to_string(),
            ));
        }
        UninitializedLLMJudgeVariantConfig::BestOfNSampling(c) => {
            templates.push((
                c.evaluator.system_instructions.get_template_key(),
                c.evaluator.system_instructions.data().to_string(),
            ));
        }
        UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(c) => {
            templates.push((
                c.fuser.system_instructions.get_template_key(),
                c.fuser.system_instructions.data().to_string(),
            ));
        }
        UninitializedLLMJudgeVariantConfig::Dicl(c) => {
            if let Some(ref si) = c.system_instructions {
                templates.push((si.get_template_key(), si.data().to_string()));
            }
        }
        UninitializedLLMJudgeVariantConfig::ChainOfThought(c) => {
            templates.push((
                c.inner.system_instructions.get_template_key(),
                c.inner.system_instructions.data().to_string(),
            ));
        }
    }
}

/// Convert an `UninitializedFunctionConfig` to a `StoredFunctionConfig`.
///
/// `prompt_ids` maps template_key → prompt_template_version_id.
/// `variant_ids` maps variant_name → variant_version_id.
pub fn to_stored_function_config(
    config: &UninitializedFunctionConfig,
    prompt_ids: &HashMap<String, Uuid>,
    variant_ids: &HashMap<String, Uuid>,
) -> Result<StoredFunctionConfig, Error> {
    match config {
        UninitializedFunctionConfig::Chat(c) => to_stored_chat(c, prompt_ids, variant_ids),
        UninitializedFunctionConfig::Json(c) => to_stored_json(c, prompt_ids, variant_ids),
    }
}

fn to_stored_variants(
    variants: &HashMap<String, UninitializedVariantInfo>,
    variant_ids: &HashMap<String, Uuid>,
) -> Result<HashMap<String, StoredVariantRef>, Error> {
    let mut stored = HashMap::with_capacity(variants.len());
    for name in variants.keys() {
        let id = variant_ids.get(name).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("Missing variant_version_id for variant `{name}`"),
            })
        })?;
        stored.insert(
            name.clone(),
            StoredVariantRef {
                variant_version_id: *id,
            },
        );
    }
    Ok(stored)
}

fn to_stored_schemas(
    schemas: &UninitializedSchemas,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<HashMap<String, StoredPromptRef>, Error> {
    let mut stored = HashMap::new();
    for (name, schema) in schemas.iter() {
        stored.insert(name.clone(), resolve_prompt_ref(&schema.path, prompt_ids)?);
    }
    Ok(stored)
}

fn to_stored_chat(
    c: &UninitializedFunctionConfigChat,
    prompt_ids: &HashMap<String, Uuid>,
    variant_ids: &HashMap<String, Uuid>,
) -> Result<StoredFunctionConfig, Error> {
    Ok(StoredFunctionConfig {
        variants: to_stored_variants(&c.variants, variant_ids)?,
        system_schema: resolve_prompt_ref_opt(c.system_schema.as_ref(), prompt_ids)?,
        user_schema: resolve_prompt_ref_opt(c.user_schema.as_ref(), prompt_ids)?,
        assistant_schema: resolve_prompt_ref_opt(c.assistant_schema.as_ref(), prompt_ids)?,
        schemas: to_stored_schemas(&c.schemas, prompt_ids)?,
        tools: c.tools.clone(),
        tool_choice: c.tool_choice.clone(),
        parallel_tool_calls: c.parallel_tool_calls,
        output_schema: None,
        description: c.description.clone(),
        experimentation: c.experimentation.clone(),
        evaluators: to_stored_evaluators(&c.evaluators, prompt_ids)?,
    })
}

fn to_stored_json(
    c: &UninitializedFunctionConfigJson,
    prompt_ids: &HashMap<String, Uuid>,
    variant_ids: &HashMap<String, Uuid>,
) -> Result<StoredFunctionConfig, Error> {
    Ok(StoredFunctionConfig {
        variants: to_stored_variants(&c.variants, variant_ids)?,
        system_schema: resolve_prompt_ref_opt(c.system_schema.as_ref(), prompt_ids)?,
        user_schema: resolve_prompt_ref_opt(c.user_schema.as_ref(), prompt_ids)?,
        assistant_schema: resolve_prompt_ref_opt(c.assistant_schema.as_ref(), prompt_ids)?,
        schemas: to_stored_schemas(&c.schemas, prompt_ids)?,
        tools: Vec::new(),
        tool_choice: ToolChoice::default(),
        parallel_tool_calls: None,
        output_schema: resolve_prompt_ref_opt(c.output_schema.as_ref(), prompt_ids)?,
        description: c.description.clone(),
        experimentation: c.experimentation.clone(),
        evaluators: to_stored_evaluators(&c.evaluators, prompt_ids)?,
    })
}

fn to_stored_evaluators(
    evaluators: &HashMap<String, UninitializedEvaluatorConfig>,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<HashMap<String, StoredEvaluatorConfig>, Error> {
    let mut stored = HashMap::with_capacity(evaluators.len());
    for (name, eval) in evaluators {
        stored.insert(name.clone(), to_stored_evaluator(eval, prompt_ids)?);
    }
    Ok(stored)
}

fn to_stored_evaluator(
    eval: &UninitializedEvaluatorConfig,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<StoredEvaluatorConfig, Error> {
    #[expect(deprecated)] // cutoff is deprecated but we must read it
    match eval {
        UninitializedEvaluatorConfig::ExactMatch(c) => {
            Ok(StoredEvaluatorConfig::ExactMatch(c.clone()))
        }
        UninitializedEvaluatorConfig::Regex(c) => Ok(StoredEvaluatorConfig::Regex(c.clone())),
        UninitializedEvaluatorConfig::ToolUse(c) => Ok(StoredEvaluatorConfig::ToolUse(c.clone())),
        UninitializedEvaluatorConfig::LLMJudge(c) => {
            let mut stored_variants = HashMap::with_capacity(c.variants.len());
            for (name, vi) in &c.variants {
                stored_variants.insert(
                    name.clone(),
                    to_stored_llm_judge_variant_info(vi, prompt_ids)?,
                );
            }
            Ok(StoredEvaluatorConfig::LLMJudge(StoredLLMJudgeConfig {
                input_format: c.input_format.clone(),
                variants: stored_variants,
                output_type: c.output_type,
                optimize: c.optimize,
                include: c.include.clone(),
                cutoff: c.cutoff,
                description: c.description.clone(),
            }))
        }
    }
}

fn to_stored_llm_judge_variant_info(
    vi: &UninitializedLLMJudgeVariantInfo,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeVariantInfo, Error> {
    Ok(StoredLLMJudgeVariantInfo {
        inner: to_stored_llm_judge_variant(&vi.inner, prompt_ids)?,
        timeouts: vi.timeouts.clone(),
    })
}

fn to_stored_llm_judge_variant(
    variant: &UninitializedLLMJudgeVariantConfig,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeVariantConfig, Error> {
    match variant {
        UninitializedLLMJudgeVariantConfig::ChatCompletion(c) => Ok(
            StoredLLMJudgeVariantConfig::ChatCompletion(to_stored_llm_judge_cc(c, prompt_ids)?),
        ),
        UninitializedLLMJudgeVariantConfig::BestOfNSampling(c) => Ok(
            StoredLLMJudgeVariantConfig::BestOfNSampling(StoredLLMJudgeBestOfNConfig {
                active: c.active,
                candidates: c.candidates.clone(),
                evaluator: to_stored_llm_judge_cc(&c.evaluator, prompt_ids)?,
            }),
        ),
        UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(c) => Ok(
            StoredLLMJudgeVariantConfig::MixtureOfNSampling(StoredLLMJudgeMixtureOfNConfig {
                active: c.active,
                candidates: c.candidates.clone(),
                fuser: to_stored_llm_judge_cc(&c.fuser, prompt_ids)?,
            }),
        ),
        UninitializedLLMJudgeVariantConfig::Dicl(c) => Ok(StoredLLMJudgeVariantConfig::Dicl(
            StoredLLMJudgeDiclConfig {
                active: c.active,
                embedding_model: c.embedding_model.clone(),
                k: c.k,
                model: c.model.clone(),
                system_instructions: resolve_prompt_ref_opt(
                    c.system_instructions.as_ref(),
                    prompt_ids,
                )?,
                temperature: c.temperature,
                top_p: c.top_p,
                presence_penalty: c.presence_penalty,
                frequency_penalty: c.frequency_penalty,
                max_tokens: c.max_tokens,
                seed: c.seed,
                json_mode: c.json_mode,
                stop_sequences: c.stop_sequences.clone(),
                extra_body: c.extra_body.clone(),
                retries: c.retries,
                extra_headers: c.extra_headers.clone(),
            },
        )),
        UninitializedLLMJudgeVariantConfig::ChainOfThought(c) => Ok(
            StoredLLMJudgeVariantConfig::ChainOfThought(StoredLLMJudgeChainOfThoughtConfig {
                inner: to_stored_llm_judge_cc(&c.inner, prompt_ids)?,
            }),
        ),
    }
}

fn to_stored_llm_judge_cc(
    c: &UninitializedLLMJudgeChatCompletionVariantConfig,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<StoredLLMJudgeChatCompletionConfig, Error> {
    Ok(StoredLLMJudgeChatCompletionConfig {
        active: c.active,
        model: c.model.clone(),
        system_instructions: resolve_prompt_ref(&c.system_instructions, prompt_ids)?,
        temperature: c.temperature,
        top_p: c.top_p,
        max_tokens: c.max_tokens,
        presence_penalty: c.presence_penalty,
        frequency_penalty: c.frequency_penalty,
        seed: c.seed,
        json_mode: c.json_mode,
        stop_sequences: c.stop_sequences.clone(),
        reasoning_effort: c.reasoning_effort.clone(),
        service_tier: c.service_tier.clone(),
        thinking_budget_tokens: c.thinking_budget_tokens,
        verbosity: c.verbosity.clone(),
        retries: c.retries,
        extra_body: c.extra_body.clone(),
        extra_headers: c.extra_headers.clone(),
    })
}

// ─── Read path: StoredFunctionConfig → UninitializedFunctionConfig ──────────

/// Rehydrate a `StoredFunctionConfig` back to `UninitializedFunctionConfig`.
///
/// `function_type` is `"chat"` or `"json"` from the row.
/// `variant_versions` maps variant_version_id → UninitializedVariantInfo (already rehydrated).
/// `prompt_rows` maps prompt_template_version_id → PromptTemplateVersionRow.
pub fn rehydrate_function(
    function_type: &str,
    stored: &StoredFunctionConfig,
    variant_versions: &HashMap<Uuid, (String, UninitializedVariantInfo)>,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedFunctionConfig, Error> {
    // Rehydrate variants: resolve StoredVariantRef → UninitializedVariantInfo
    let mut variants = HashMap::with_capacity(stored.variants.len());
    for (name, vref) in &stored.variants {
        let (_, variant_info) =
            variant_versions
                .get(&vref.variant_version_id)
                .ok_or_else(|| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Missing variant version `{}` for variant `{name}`",
                            vref.variant_version_id
                        ),
                    })
                })?;
        variants.insert(name.clone(), variant_info.clone());
    }

    // Rehydrate schemas
    let system_schema = rehydrate_prompt_ref_opt(stored.system_schema.as_ref(), prompt_rows)?;
    let user_schema = rehydrate_prompt_ref_opt(stored.user_schema.as_ref(), prompt_rows)?;
    let assistant_schema = rehydrate_prompt_ref_opt(stored.assistant_schema.as_ref(), prompt_rows)?;
    let schemas = rehydrate_schemas(&stored.schemas, prompt_rows)?;

    // Rehydrate evaluators
    let evaluators = rehydrate_evaluators(&stored.evaluators, prompt_rows)?;

    match function_type {
        "chat" => Ok(UninitializedFunctionConfig::Chat(
            UninitializedFunctionConfigChat {
                variants,
                system_schema,
                user_schema,
                assistant_schema,
                schemas,
                tools: stored.tools.clone(),
                tool_choice: stored.tool_choice.clone(),
                parallel_tool_calls: stored.parallel_tool_calls,
                description: stored.description.clone(),
                experimentation: stored.experimentation.clone(),
                evaluators,
            },
        )),
        "json" => {
            let output_schema =
                rehydrate_prompt_ref_opt(stored.output_schema.as_ref(), prompt_rows)?;
            Ok(UninitializedFunctionConfig::Json(
                UninitializedFunctionConfigJson {
                    variants,
                    system_schema,
                    user_schema,
                    assistant_schema,
                    schemas,
                    output_schema,
                    description: stored.description.clone(),
                    experimentation: stored.experimentation.clone(),
                    evaluators,
                },
            ))
        }
        _ => Err(Error::new(ErrorDetails::Config {
            message: format!("Unknown function type: `{function_type}`"),
        })),
    }
}

fn rehydrate_schemas(
    stored: &HashMap<String, StoredPromptRef>,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedSchemas, Error> {
    let mut paths = HashMap::with_capacity(stored.len());
    for (name, pref) in stored {
        paths.insert(name.clone(), rehydrate_prompt_ref(pref, prompt_rows)?);
    }
    Ok(UninitializedSchemas::from_paths(paths))
}

fn rehydrate_evaluators(
    stored: &HashMap<String, StoredEvaluatorConfig>,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<HashMap<String, UninitializedEvaluatorConfig>, Error> {
    let mut evaluators = HashMap::with_capacity(stored.len());
    for (name, eval) in stored {
        evaluators.insert(name.clone(), rehydrate_evaluator(eval, prompt_rows)?);
    }
    Ok(evaluators)
}

#[expect(deprecated)] // cutoff is deprecated but we must construct the structs
fn rehydrate_evaluator(
    stored: &StoredEvaluatorConfig,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedEvaluatorConfig, Error> {
    match stored {
        StoredEvaluatorConfig::ExactMatch(c) => {
            Ok(UninitializedEvaluatorConfig::ExactMatch(c.clone()))
        }
        StoredEvaluatorConfig::Regex(c) => Ok(UninitializedEvaluatorConfig::Regex(c.clone())),
        StoredEvaluatorConfig::ToolUse(c) => Ok(UninitializedEvaluatorConfig::ToolUse(c.clone())),
        StoredEvaluatorConfig::LLMJudge(c) => {
            let mut variants = HashMap::with_capacity(c.variants.len());
            for (name, vi) in &c.variants {
                variants.insert(
                    name.clone(),
                    rehydrate_llm_judge_variant_info(vi, prompt_rows)?,
                );
            }
            Ok(UninitializedEvaluatorConfig::LLMJudge(
                UninitializedLLMJudgeConfig {
                    input_format: c.input_format.clone(),
                    variants,
                    output_type: c.output_type,
                    optimize: c.optimize,
                    include: c.include.clone(),
                    cutoff: c.cutoff,
                    description: c.description.clone(),
                },
            ))
        }
    }
}

fn rehydrate_llm_judge_variant_info(
    stored: &StoredLLMJudgeVariantInfo,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedLLMJudgeVariantInfo, Error> {
    Ok(UninitializedLLMJudgeVariantInfo {
        inner: rehydrate_llm_judge_variant(&stored.inner, prompt_rows)?,
        timeouts: stored.timeouts.clone(),
    })
}

#[expect(deprecated)] // timeout_s is deprecated but we must construct the structs
fn rehydrate_llm_judge_variant(
    stored: &StoredLLMJudgeVariantConfig,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedLLMJudgeVariantConfig, Error> {
    match stored {
        StoredLLMJudgeVariantConfig::ChatCompletion(c) => {
            Ok(UninitializedLLMJudgeVariantConfig::ChatCompletion(
                rehydrate_llm_judge_cc(c, prompt_rows)?,
            ))
        }
        StoredLLMJudgeVariantConfig::BestOfNSampling(c) => {
            Ok(UninitializedLLMJudgeVariantConfig::BestOfNSampling(
                UninitializedLLMJudgeBestOfNVariantConfig {
                    active: c.active,
                    timeout_s: None,
                    candidates: c.candidates.clone(),
                    evaluator: rehydrate_llm_judge_cc(&c.evaluator, prompt_rows)?,
                },
            ))
        }
        StoredLLMJudgeVariantConfig::MixtureOfNSampling(c) => {
            Ok(UninitializedLLMJudgeVariantConfig::MixtureOfNSampling(
                UninitializedLLMJudgeMixtureOfNVariantConfig {
                    active: c.active,
                    timeout_s: None,
                    candidates: c.candidates.clone(),
                    fuser: rehydrate_llm_judge_cc(&c.fuser, prompt_rows)?,
                },
            ))
        }
        StoredLLMJudgeVariantConfig::Dicl(c) => Ok(UninitializedLLMJudgeVariantConfig::Dicl(
            UninitializedLLMJudgeDiclVariantConfig {
                active: c.active,
                embedding_model: c.embedding_model.clone(),
                k: c.k,
                model: c.model.clone(),
                system_instructions: rehydrate_prompt_ref_opt(
                    c.system_instructions.as_ref(),
                    prompt_rows,
                )?,
                temperature: c.temperature,
                top_p: c.top_p,
                presence_penalty: c.presence_penalty,
                frequency_penalty: c.frequency_penalty,
                max_tokens: c.max_tokens,
                seed: c.seed,
                json_mode: c.json_mode,
                stop_sequences: c.stop_sequences.clone(),
                extra_body: c.extra_body.clone(),
                retries: c.retries,
                extra_headers: c.extra_headers.clone(),
            },
        )),
        StoredLLMJudgeVariantConfig::ChainOfThought(c) => {
            Ok(UninitializedLLMJudgeVariantConfig::ChainOfThought(
                UninitializedLLMJudgeChainOfThoughtVariantConfig {
                    inner: rehydrate_llm_judge_cc(&c.inner, prompt_rows)?,
                },
            ))
        }
    }
}

fn rehydrate_llm_judge_cc(
    c: &StoredLLMJudgeChatCompletionConfig,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedLLMJudgeChatCompletionVariantConfig, Error> {
    Ok(UninitializedLLMJudgeChatCompletionVariantConfig {
        active: c.active,
        model: c.model.clone(),
        system_instructions: rehydrate_prompt_ref(&c.system_instructions, prompt_rows)?,
        temperature: c.temperature,
        top_p: c.top_p,
        max_tokens: c.max_tokens,
        presence_penalty: c.presence_penalty,
        frequency_penalty: c.frequency_penalty,
        seed: c.seed,
        json_mode: c.json_mode,
        stop_sequences: c.stop_sequences.clone(),
        reasoning_effort: c.reasoning_effort.clone(),
        service_tier: c.service_tier.clone(),
        thinking_budget_tokens: c.thinking_budget_tokens,
        verbosity: c.verbosity.clone(),
        retries: c.retries,
        extra_body: c.extra_body.clone(),
        extra_headers: c.extra_headers.clone(),
    })
}
