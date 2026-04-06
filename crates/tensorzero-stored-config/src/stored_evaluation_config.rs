use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tensorzero_types::inference_params::{JsonMode, ServiceTier};

use crate::StoredPromptRef;

pub const STORED_EVALUATION_CONFIG_SCHEMA_REVISION: i32 = 1;

// --- Top-level evaluation config ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredEvaluationConfig {
    Inference(StoredInferenceEvaluationConfig),
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredInferenceEvaluationConfig {
    pub evaluators: Option<BTreeMap<String, StoredEvaluatorConfig>>,
    pub function_name: String,
    pub description: Option<String>,
}

// --- Evaluator config (tagged enum) ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredEvaluatorConfig {
    ExactMatch(StoredExactMatchConfig),
    #[serde(rename = "llm_judge")]
    LLMJudge(StoredLLMJudgeConfig),
    ToolUse(StoredToolUseConfig),
    Regex(StoredRegexConfig),
}

// --- Simple evaluator stored types ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredExactMatchConfig {
    pub cutoff: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "behavior")]
#[serde(rename_all = "snake_case")]
pub enum StoredToolUseConfig {
    None,
    NoneOf { tools: Vec<String> },
    Any,
    AnyOf { tools: Vec<String> },
    AllOf { tools: Vec<String> },
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRegexConfig {
    pub must_match: Option<String>,
    pub must_not_match: Option<String>,
}

// --- LLM judge config ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredLLMJudgeConfig {
    pub input_format: Option<StoredLLMJudgeInputFormat>,
    pub variants: Option<BTreeMap<String, StoredLLMJudgeVariantInfo>>,
    pub output_type: StoredLLMJudgeOutputType,
    pub optimize: StoredLLMJudgeOptimize,
    pub cutoff: Option<f32>,
    pub include: Option<StoredLLMJudgeIncludeConfig>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredLLMJudgeInputFormat {
    Serialized,
    Messages,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredLLMJudgeOutputType {
    Float,
    Boolean,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredLLMJudgeOptimize {
    Min,
    Max,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredLLMJudgeIncludeConfig {
    pub reference_output: bool,
}

// --- LLM judge variant types ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredLLMJudgeVariantInfo {
    pub variant: StoredLLMJudgeVariantConfig,
    pub timeouts: Option<StoredTimeoutsConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredLLMJudgeVariantConfig {
    ChatCompletion(StoredLLMJudgeChatCompletionVariantConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(StoredLLMJudgeBestOfNVariantConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfNSampling(StoredLLMJudgeMixtureOfNVariantConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(StoredLLMJudgeDiclVariantConfig),
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(StoredLLMJudgeChainOfThoughtVariantConfig),
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredLLMJudgeChatCompletionVariantConfig {
    pub active: Option<bool>,
    pub model: Arc<str>,
    pub system_instructions: StoredPromptRef,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub json_mode: JsonMode,
    pub stop_sequences: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<ServiceTier>,
    pub thinking_budget_tokens: Option<i32>,
    pub verbosity: Option<String>,
    pub retries: Option<StoredRetryConfig>,
    pub extra_body: Option<crate::StoredExtraBodyConfig>,
    pub extra_headers: Option<crate::StoredExtraHeadersConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredLLMJudgeBestOfNVariantConfig {
    pub active: Option<bool>,
    pub timeout_s: Option<f64>,
    pub candidates: Option<Vec<String>>,
    pub evaluator: StoredLLMJudgeChatCompletionVariantConfig,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredLLMJudgeMixtureOfNVariantConfig {
    pub active: Option<bool>,
    pub timeout_s: Option<f64>,
    pub candidates: Option<Vec<String>>,
    pub fuser: StoredLLMJudgeChatCompletionVariantConfig,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredLLMJudgeDiclVariantConfig {
    pub active: Option<bool>,
    pub embedding_model: String,
    pub k: u32,
    pub model: String,
    pub system_instructions: Option<StoredPromptRef>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u32>,
    pub json_mode: Option<JsonMode>,
    pub stop_sequences: Option<Vec<String>>,
    pub extra_body: Option<crate::StoredExtraBodyConfig>,
    pub retries: Option<StoredRetryConfig>,
    pub extra_headers: Option<crate::StoredExtraHeadersConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredLLMJudgeChainOfThoughtVariantConfig {
    pub inner: StoredLLMJudgeChatCompletionVariantConfig,
}

// --- Shared stored types ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRetryConfig {
    pub num_retries: u32,
    pub max_delay_s: f32,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredTimeoutsConfig {
    pub non_streaming: Option<StoredNonStreamingTimeouts>,
    pub streaming: Option<StoredStreamingTimeouts>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredNonStreamingTimeouts {
    pub total_ms: Option<u64>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredStreamingTimeouts {
    pub ttft_ms: Option<u64>,
    pub total_ms: Option<u64>,
}
