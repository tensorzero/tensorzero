//! Stored types for variant versions persisted in the database as JSONB.
//!
//! These types are the durable schema for `variant_versions.config`. Each variant type
//! has a corresponding `Stored*VariantConfig` that replaces `ResolvedTomlPathData` fields
//! with `StoredPromptRef` (a pointer to a `prompt_template_versions` row).
//!
//! The variant type discriminator lives inside the JSONB itself via serde's adjacently-tagged
//! enum (`#[serde(tag = "type", content = "config")]`), so there is no separate `variant_type`
//! column. Schema evolution is controlled by `schema_version` on the row.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::TimeoutsConfig;
use crate::config::namespace::Namespace;
use crate::config::path::ResolvedTomlPathData;
use crate::inference::types::chat_completion_inference_params::ServiceTier;
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::utils::retries::RetryConfig;
use crate::variant::JsonMode;
use crate::variant::chat_completion::{
    UninitializedChatCompletionConfig, UninitializedChatTemplate, UninitializedChatTemplates,
    UninitializedInputWrappers,
};

use crate::config::UninitializedVariantConfig;
use crate::config::UninitializedVariantInfo;
use crate::variant::best_of_n_sampling::{
    UninitializedBestOfNEvaluatorConfig, UninitializedBestOfNSamplingConfig,
};
use crate::variant::dicl::UninitializedDiclConfig;
use crate::variant::mixture_of_n::{UninitializedFuserConfig, UninitializedMixtureOfNConfig};

use crate::error::{Error, ErrorDetails};

/// Current schema version for newly written variant versions.
/// Bumped from 1 → 2 to remove `chain_of_thought` as a valid variant type.
pub const CURRENT_SCHEMA_VERSION: i32 = 2;

// ─── StoredPromptRef ─────────────────────────────────────────────────────────

/// A reference to a row in `prompt_template_versions`, used in place of
/// `ResolvedTomlPathData` in stored variant configs.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredPromptRef {
    pub prompt_template_version_id: Uuid,
    pub template_key: String,
}

// ─── Stored variant config types (schema_version = 1) ────────────────────────

/// Stored form of `UninitializedChatCompletionConfig`.
/// `ResolvedTomlPathData` fields are replaced with `StoredPromptRef`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredChatCompletionVariantConfig {
    pub weight: Option<f64>,
    pub model: Arc<str>,
    pub system_template: Option<StoredPromptRef>,
    pub user_template: Option<StoredPromptRef>,
    pub assistant_template: Option<StoredPromptRef>,
    pub input_wrappers: Option<StoredInputWrappers>,
    #[serde(default)]
    pub templates: StoredChatTemplates,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    pub json_mode: Option<JsonMode>,
    #[serde(default)]
    pub retries: RetryConfig,
    pub extra_body: Option<ExtraBodyConfig>,
    pub extra_headers: Option<ExtraHeadersConfig>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredInputWrappers {
    pub user: Option<StoredPromptRef>,
    pub assistant: Option<StoredPromptRef>,
    pub system: Option<StoredPromptRef>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct StoredChatTemplates {
    #[serde(flatten)]
    pub inner: HashMap<String, StoredChatTemplate>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredChatTemplate {
    pub path: StoredPromptRef,
}

/// Stored form of `UninitializedBestOfNSamplingConfig`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredBestOfNVariantConfig {
    pub weight: Option<f64>,
    // Deprecated `timeout_s` is canonicalized away on write — not stored.
    pub candidates: Vec<String>,
    pub evaluator: StoredBestOfNEvaluatorConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredBestOfNEvaluatorConfig {
    #[serde(flatten)]
    pub inner: StoredChatCompletionVariantConfig,
}

/// Stored form of `UninitializedMixtureOfNConfig`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredMixtureOfNVariantConfig {
    pub weight: Option<f64>,
    // Deprecated `timeout_s` is canonicalized away on write — not stored.
    pub candidates: Vec<String>,
    pub fuser: StoredFuserConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredFuserConfig {
    #[serde(flatten)]
    pub inner: StoredChatCompletionVariantConfig,
}

/// Stored form of `UninitializedDiclConfig`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StoredDiclVariantConfig {
    pub weight: Option<f64>,
    pub embedding_model: String,
    pub k: u32,
    pub model: String,
    pub system_instructions: Option<StoredPromptRef>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    pub json_mode: Option<JsonMode>,
    pub extra_body: Option<ExtraBodyConfig>,
    #[serde(default)]
    pub retries: RetryConfig,
    pub extra_headers: Option<ExtraHeadersConfig>,
    pub max_distance: Option<f32>,
}

// ─── Schema version 2 (current) ──────────────────────────────────────────────

/// Variant config enum (schema_version=2). No `ChainOfThought` — it was
/// canonicalized to `ChatCompletion` in the expand release and removed here.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", content = "config")]
pub enum StoredVariantConfig {
    #[serde(rename = "chat_completion")]
    ChatCompletion(StoredChatCompletionVariantConfig),
    #[serde(rename = "best_of_n_sampling")]
    BestOfNSampling(StoredBestOfNVariantConfig),
    #[serde(rename = "mixture_of_n")]
    MixtureOfN(StoredMixtureOfNVariantConfig),
    #[serde(rename = "dicl")]
    Dicl(StoredDiclVariantConfig),
}

/// The top-level JSONB blob stored in `variant_versions.config` (schema_version=2).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StoredVariantVersion {
    #[serde(flatten)]
    pub config: StoredVariantConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeouts: Option<TimeoutsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<Namespace>,
}

impl StoredVariantVersion {
    /// Collect all prompt_template_version IDs referenced by this variant.
    /// Used to batch-load prompt templates from the database.
    pub fn referenced_prompt_template_ids(&self) -> Vec<Uuid> {
        let mut ids = Vec::new();
        match &self.config {
            StoredVariantConfig::ChatCompletion(c) => {
                collect_cc_prompt_ids(c, &mut ids);
            }
            StoredVariantConfig::BestOfNSampling(c) => {
                collect_cc_prompt_ids(&c.evaluator.inner, &mut ids);
            }
            StoredVariantConfig::MixtureOfN(c) => {
                collect_cc_prompt_ids(&c.fuser.inner, &mut ids);
            }
            StoredVariantConfig::Dicl(c) => {
                if let Some(ref si) = c.system_instructions {
                    ids.push(si.prompt_template_version_id);
                }
            }
        }
        ids
    }
}

fn collect_cc_prompt_ids(c: &StoredChatCompletionVariantConfig, ids: &mut Vec<Uuid>) {
    if let Some(ref t) = c.system_template {
        ids.push(t.prompt_template_version_id);
    }
    if let Some(ref t) = c.user_template {
        ids.push(t.prompt_template_version_id);
    }
    if let Some(ref t) = c.assistant_template {
        ids.push(t.prompt_template_version_id);
    }
    if let Some(ref w) = c.input_wrappers {
        if let Some(ref t) = w.user {
            ids.push(t.prompt_template_version_id);
        }
        if let Some(ref t) = w.assistant {
            ids.push(t.prompt_template_version_id);
        }
        if let Some(ref t) = w.system {
            ids.push(t.prompt_template_version_id);
        }
    }
    for ct in c.templates.inner.values() {
        ids.push(ct.path.prompt_template_version_id);
    }
}

// ─── Schema version dispatch ─────────────────────────────────────────────────

/// Deserialize a `StoredVariantVersion` from a JSONB value, dispatching on `schema_version`.
pub fn deserialize_stored_variant_version(
    schema_version: i32,
    config: serde_json::Value,
) -> Result<StoredVariantVersion, Error> {
    match schema_version {
        // V1 support was removed after the `deprecate_chain_of_thought_v1` background
        // migration rewrote all v1 rows to v2. If v1 rows somehow remain, this will
        // return an error — check that the migration completed successfully.
        2 => serde_json::from_value(config).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to deserialize variant version (schema_version=2): {e}"),
            })
        }),
        _ => Err(Error::new(ErrorDetails::Config {
            message: format!("Unknown variant version schema_version: {schema_version}"),
        })),
    }
}

// ─── Write path: UninitializedVariantInfo → StoredVariantVersion ─────────────

/// Collects all `ResolvedTomlPathData` references from a variant config.
/// Returns (template_key, template_data) pairs.
pub fn collect_prompt_templates(variant_info: &UninitializedVariantInfo) -> Vec<(String, String)> {
    let mut templates = Vec::new();
    match &variant_info.inner {
        UninitializedVariantConfig::ChatCompletion(c) => {
            collect_chat_completion_templates(c, &mut templates);
        }
        UninitializedVariantConfig::BestOfNSampling(c) => {
            collect_chat_completion_templates(&c.evaluator.inner, &mut templates);
        }
        UninitializedVariantConfig::MixtureOfN(c) => {
            collect_chat_completion_templates(&c.fuser.inner, &mut templates);
        }
        UninitializedVariantConfig::Dicl(c) => {
            if let Some(ref si) = c.system_instructions {
                templates.push((si.get_template_key(), si.data().to_string()));
            }
        }
        UninitializedVariantConfig::ChainOfThought(c) => {
            collect_chat_completion_templates(&c.inner, &mut templates);
        }
    }
    templates
}

fn collect_chat_completion_templates(
    config: &UninitializedChatCompletionConfig,
    templates: &mut Vec<(String, String)>,
) {
    if let Some(ref t) = config.system_template {
        templates.push((t.get_template_key(), t.data().to_string()));
    }
    if let Some(ref t) = config.user_template {
        templates.push((t.get_template_key(), t.data().to_string()));
    }
    if let Some(ref t) = config.assistant_template {
        templates.push((t.get_template_key(), t.data().to_string()));
    }
    if let Some(ref wrappers) = config.input_wrappers {
        if let Some(ref t) = wrappers.user {
            templates.push((t.get_template_key(), t.data().to_string()));
        }
        if let Some(ref t) = wrappers.assistant {
            templates.push((t.get_template_key(), t.data().to_string()));
        }
        if let Some(ref t) = wrappers.system {
            templates.push((t.get_template_key(), t.data().to_string()));
        }
    }
    for ct in config.templates.inner.values() {
        templates.push((ct.path.get_template_key(), ct.path.data().to_string()));
    }
}

/// Convert an `UninitializedVariantInfo` to a `StoredVariantVersion`, given a mapping
/// from template_key → prompt_template_version_id (produced by inserting templates first).
pub fn to_stored_variant_version(
    variant_info: &UninitializedVariantInfo,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<StoredVariantVersion, Error> {
    let config = match &variant_info.inner {
        UninitializedVariantConfig::ChatCompletion(c) => {
            StoredVariantConfig::ChatCompletion(to_stored_chat_completion(c, prompt_ids)?)
        }
        UninitializedVariantConfig::BestOfNSampling(c) => {
            StoredVariantConfig::BestOfNSampling(StoredBestOfNVariantConfig {
                weight: c.weight,
                candidates: c.candidates.clone(),
                evaluator: StoredBestOfNEvaluatorConfig {
                    inner: to_stored_chat_completion(&c.evaluator.inner, prompt_ids)?,
                },
            })
        }
        UninitializedVariantConfig::MixtureOfN(c) => {
            StoredVariantConfig::MixtureOfN(StoredMixtureOfNVariantConfig {
                weight: c.weight,
                candidates: c.candidates.clone(),
                fuser: StoredFuserConfig {
                    inner: to_stored_chat_completion(&c.fuser.inner, prompt_ids)?,
                },
            })
        }
        UninitializedVariantConfig::Dicl(c) => {
            StoredVariantConfig::Dicl(to_stored_dicl(c, prompt_ids)?)
        }
        // ChainOfThought is deprecated — canonicalize to ChatCompletion on write.
        // Default reasoning_effort to "medium" to preserve reasoning behavioral intent.
        // We don't set thinking_budget_tokens because it requires a provider-specific
        // value and can conflict with reasoning_effort on some providers.
        UninitializedVariantConfig::ChainOfThought(c) => {
            let mut stored = to_stored_chat_completion(&c.inner, prompt_ids)?;
            if stored.reasoning_effort.is_none() {
                stored.reasoning_effort = Some("medium".to_string());
            }
            StoredVariantConfig::ChatCompletion(stored)
        }
    };
    Ok(StoredVariantVersion {
        config,
        timeouts: variant_info.timeouts.clone(),
        namespace: variant_info.namespace.clone(),
    })
}

fn resolve_prompt_ref(
    path_data: &ResolvedTomlPathData,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<StoredPromptRef, Error> {
    let key = path_data.get_template_key();
    let id = prompt_ids.get(&key).ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("Missing prompt_template_version_id for template_key `{key}`"),
        })
    })?;
    Ok(StoredPromptRef {
        prompt_template_version_id: *id,
        template_key: key,
    })
}

fn resolve_prompt_ref_opt(
    path_data: Option<&ResolvedTomlPathData>,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<Option<StoredPromptRef>, Error> {
    match path_data {
        Some(pd) => Ok(Some(resolve_prompt_ref(pd, prompt_ids)?)),
        None => Ok(None),
    }
}

fn to_stored_chat_completion(
    c: &UninitializedChatCompletionConfig,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<StoredChatCompletionVariantConfig, Error> {
    let input_wrappers = match &c.input_wrappers {
        Some(w) => Some(StoredInputWrappers {
            user: resolve_prompt_ref_opt(w.user.as_ref(), prompt_ids)?,
            assistant: resolve_prompt_ref_opt(w.assistant.as_ref(), prompt_ids)?,
            system: resolve_prompt_ref_opt(w.system.as_ref(), prompt_ids)?,
        }),
        None => None,
    };

    let mut stored_templates = HashMap::new();
    for (name, ct) in &c.templates.inner {
        stored_templates.insert(
            name.clone(),
            StoredChatTemplate {
                path: resolve_prompt_ref(&ct.path, prompt_ids)?,
            },
        );
    }

    Ok(StoredChatCompletionVariantConfig {
        weight: c.weight,
        model: c.model.clone(),
        system_template: resolve_prompt_ref_opt(c.system_template.as_ref(), prompt_ids)?,
        user_template: resolve_prompt_ref_opt(c.user_template.as_ref(), prompt_ids)?,
        assistant_template: resolve_prompt_ref_opt(c.assistant_template.as_ref(), prompt_ids)?,
        input_wrappers,
        templates: StoredChatTemplates {
            inner: stored_templates,
        },
        temperature: c.temperature,
        top_p: c.top_p,
        max_tokens: c.max_tokens,
        presence_penalty: c.presence_penalty,
        frequency_penalty: c.frequency_penalty,
        seed: c.seed,
        stop_sequences: c.stop_sequences.clone(),
        reasoning_effort: c.reasoning_effort.clone(),
        service_tier: c.service_tier.clone(),
        thinking_budget_tokens: c.thinking_budget_tokens,
        verbosity: c.verbosity.clone(),
        json_mode: c.json_mode,
        retries: c.retries,
        extra_body: c.extra_body.clone(),
        extra_headers: c.extra_headers.clone(),
    })
}

fn to_stored_dicl(
    c: &UninitializedDiclConfig,
    prompt_ids: &HashMap<String, Uuid>,
) -> Result<StoredDiclVariantConfig, Error> {
    Ok(StoredDiclVariantConfig {
        weight: c.weight,
        embedding_model: c.embedding_model.clone(),
        k: c.k,
        model: c.model.clone(),
        system_instructions: resolve_prompt_ref_opt(c.system_instructions.as_ref(), prompt_ids)?,
        temperature: c.temperature,
        top_p: c.top_p,
        stop_sequences: c.stop_sequences.clone(),
        presence_penalty: c.presence_penalty,
        frequency_penalty: c.frequency_penalty,
        max_tokens: c.max_tokens,
        seed: c.seed,
        reasoning_effort: c.reasoning_effort.clone(),
        thinking_budget_tokens: c.thinking_budget_tokens,
        verbosity: c.verbosity.clone(),
        json_mode: c.json_mode,
        extra_body: c.extra_body.clone(),
        retries: c.retries,
        extra_headers: c.extra_headers.clone(),
        max_distance: c.max_distance,
    })
}

// ─── Read path: StoredVariantVersion → UninitializedVariantInfo ──────────────

/// A loaded prompt template version row from the database.
#[derive(Clone, Debug)]
pub struct PromptTemplateVersionRow {
    pub id: Uuid,
    pub template_key: String,
    pub source_body: String,
}

/// Rehydrate a `StoredVariantVersion` back to `UninitializedVariantInfo` using
/// prompt template data loaded from the database.
#[expect(deprecated)] // timeout_s is deprecated but we must construct the structs
pub fn rehydrate_variant(
    stored: &StoredVariantVersion,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedVariantInfo, Error> {
    let inner = match &stored.config {
        StoredVariantConfig::ChatCompletion(c) => {
            UninitializedVariantConfig::ChatCompletion(rehydrate_chat_completion(c, prompt_rows)?)
        }
        StoredVariantConfig::BestOfNSampling(c) => {
            UninitializedVariantConfig::BestOfNSampling(UninitializedBestOfNSamplingConfig {
                weight: c.weight,
                timeout_s: None,
                candidates: c.candidates.clone(),
                evaluator: UninitializedBestOfNEvaluatorConfig {
                    inner: rehydrate_chat_completion(&c.evaluator.inner, prompt_rows)?,
                },
            })
        }
        StoredVariantConfig::MixtureOfN(c) => {
            UninitializedVariantConfig::MixtureOfN(UninitializedMixtureOfNConfig {
                weight: c.weight,
                timeout_s: None,
                candidates: c.candidates.clone(),
                fuser: UninitializedFuserConfig {
                    inner: rehydrate_chat_completion(&c.fuser.inner, prompt_rows)?,
                },
            })
        }
        StoredVariantConfig::Dicl(c) => {
            UninitializedVariantConfig::Dicl(rehydrate_dicl(c, prompt_rows)?)
        }
    };
    Ok(UninitializedVariantInfo {
        inner,
        timeouts: stored.timeouts.clone(),
        namespace: stored.namespace.clone(),
    })
}

fn rehydrate_prompt_ref(
    pref: &StoredPromptRef,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<ResolvedTomlPathData, Error> {
    let row = prompt_rows
        .get(&pref.prompt_template_version_id)
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Missing prompt template version `{}`",
                    pref.prompt_template_version_id
                ),
            })
        })?;
    Ok(ResolvedTomlPathData::new_fake_path(
        pref.template_key.clone(),
        row.source_body.clone(),
    ))
}

fn rehydrate_prompt_ref_opt(
    pref: Option<&StoredPromptRef>,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<Option<ResolvedTomlPathData>, Error> {
    match pref {
        Some(p) => Ok(Some(rehydrate_prompt_ref(p, prompt_rows)?)),
        None => Ok(None),
    }
}

fn rehydrate_chat_completion(
    c: &StoredChatCompletionVariantConfig,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedChatCompletionConfig, Error> {
    let input_wrappers = match &c.input_wrappers {
        Some(w) => Some(UninitializedInputWrappers {
            user: rehydrate_prompt_ref_opt(w.user.as_ref(), prompt_rows)?,
            assistant: rehydrate_prompt_ref_opt(w.assistant.as_ref(), prompt_rows)?,
            system: rehydrate_prompt_ref_opt(w.system.as_ref(), prompt_rows)?,
        }),
        None => None,
    };

    let mut templates = HashMap::new();
    for (name, ct) in &c.templates.inner {
        templates.insert(
            name.clone(),
            UninitializedChatTemplate {
                path: rehydrate_prompt_ref(&ct.path, prompt_rows)?,
            },
        );
    }

    Ok(UninitializedChatCompletionConfig {
        weight: c.weight,
        model: c.model.clone(),
        system_template: rehydrate_prompt_ref_opt(c.system_template.as_ref(), prompt_rows)?,
        user_template: rehydrate_prompt_ref_opt(c.user_template.as_ref(), prompt_rows)?,
        assistant_template: rehydrate_prompt_ref_opt(c.assistant_template.as_ref(), prompt_rows)?,
        input_wrappers,
        templates: UninitializedChatTemplates { inner: templates },
        temperature: c.temperature,
        top_p: c.top_p,
        max_tokens: c.max_tokens,
        presence_penalty: c.presence_penalty,
        frequency_penalty: c.frequency_penalty,
        seed: c.seed,
        stop_sequences: c.stop_sequences.clone(),
        reasoning_effort: c.reasoning_effort.clone(),
        service_tier: c.service_tier.clone(),
        thinking_budget_tokens: c.thinking_budget_tokens,
        verbosity: c.verbosity.clone(),
        json_mode: c.json_mode,
        retries: c.retries,
        extra_body: c.extra_body.clone(),
        extra_headers: c.extra_headers.clone(),
    })
}

fn rehydrate_dicl(
    c: &StoredDiclVariantConfig,
    prompt_rows: &HashMap<Uuid, PromptTemplateVersionRow>,
) -> Result<UninitializedDiclConfig, Error> {
    Ok(UninitializedDiclConfig {
        weight: c.weight,
        embedding_model: c.embedding_model.clone(),
        k: c.k,
        model: c.model.clone(),
        system_instructions: rehydrate_prompt_ref_opt(c.system_instructions.as_ref(), prompt_rows)?,
        temperature: c.temperature,
        top_p: c.top_p,
        stop_sequences: c.stop_sequences.clone(),
        presence_penalty: c.presence_penalty,
        frequency_penalty: c.frequency_penalty,
        max_tokens: c.max_tokens,
        seed: c.seed,
        reasoning_effort: c.reasoning_effort.clone(),
        thinking_budget_tokens: c.thinking_budget_tokens,
        verbosity: c.verbosity.clone(),
        json_mode: c.json_mode,
        extra_body: c.extra_body.clone(),
        retries: c.retries,
        extra_headers: c.extra_headers.clone(),
        max_distance: c.max_distance,
    })
}
