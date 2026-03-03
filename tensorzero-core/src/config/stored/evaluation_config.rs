use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::TimeoutsConfig;
use crate::evaluations::{
    ExactMatchConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat, LLMJudgeOptimize,
    LLMJudgeOutputType, ToolUseConfig, UninitializedEvaluationConfig, UninitializedEvaluatorConfig,
    UninitializedInferenceEvaluationConfig, UninitializedLLMJudgeBestOfNVariantConfig,
    UninitializedLLMJudgeChainOfThoughtVariantConfig,
    UninitializedLLMJudgeChatCompletionVariantConfig, UninitializedLLMJudgeConfig,
    UninitializedLLMJudgeDiclVariantConfig, UninitializedLLMJudgeMixtureOfNVariantConfig,
    UninitializedLLMJudgeVariantInfo,
};

/// Stored version of `UninitializedEvaluationConfig`.
///
/// Does NOT use `deny_unknown_fields` for forward-compatibility.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredEvaluationConfig {
    #[serde(alias = "static")]
    Inference(StoredInferenceEvaluationConfig),
}

/// Stored version of `UninitializedInferenceEvaluationConfig`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredInferenceEvaluationConfig {
    #[serde(default)]
    pub evaluators: HashMap<String, StoredEvaluatorConfig>,
    pub function_name: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// Stored version of `UninitializedEvaluatorConfig`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredEvaluatorConfig {
    ExactMatch(ExactMatchConfig),
    #[serde(rename = "llm_judge")]
    LLMJudge(StoredLLMJudgeConfig),
    ToolUse(ToolUseConfig),
}

/// Stored version of `UninitializedLLMJudgeConfig`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredLLMJudgeConfig {
    #[serde(default)]
    pub input_format: LLMJudgeInputFormat,
    pub variants: HashMap<String, StoredLLMJudgeVariantInfo>,
    pub output_type: LLMJudgeOutputType,
    pub optimize: LLMJudgeOptimize,
    #[serde(default)]
    pub include: LLMJudgeIncludeConfig,
    #[serde(default)]
    pub cutoff: Option<f32>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Stored version of `UninitializedLLMJudgeVariantInfo`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredLLMJudgeVariantInfo {
    #[serde(flatten)]
    pub inner: StoredLLMJudgeVariantConfig,
    #[serde(default)]
    pub timeouts: Option<TimeoutsConfig>,
}

/// Stored version of `UninitializedLLMJudgeVariantConfig`.
///
/// Uses stored types for BestOfN/MixtureOfN variants that had `timeout_s`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredLLMJudgeVariantConfig {
    ChatCompletion(UninitializedLLMJudgeChatCompletionVariantConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(StoredLLMJudgeBestOfNVariantConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfNSampling(StoredLLMJudgeMixtureOfNVariantConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(UninitializedLLMJudgeDiclVariantConfig),
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(UninitializedLLMJudgeChainOfThoughtVariantConfig),
}

/// Stored version of `UninitializedLLMJudgeBestOfNVariantConfig`.
///
/// Retains the deprecated `timeout_s` field.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredLLMJudgeBestOfNVariantConfig {
    #[serde(default)]
    pub active: Option<bool>,
    /// DEPRECATED: was a no-op in evaluations, silently dropped during migration.
    #[serde(default)]
    pub timeout_s: Option<f64>,
    #[serde(default)]
    pub candidates: Vec<String>,
    pub evaluator: UninitializedLLMJudgeChatCompletionVariantConfig,
}

/// Stored version of `UninitializedLLMJudgeMixtureOfNVariantConfig`.
///
/// Retains the deprecated `timeout_s` field.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredLLMJudgeMixtureOfNVariantConfig {
    #[serde(default)]
    pub active: Option<bool>,
    /// DEPRECATED: was a no-op in evaluations, silently dropped during migration.
    #[serde(default)]
    pub timeout_s: Option<f64>,
    #[serde(default)]
    pub candidates: Vec<String>,
    pub fuser: UninitializedLLMJudgeChatCompletionVariantConfig,
}

// --- From<Uninitialized*> for Stored* conversions ---

impl From<UninitializedEvaluationConfig> for StoredEvaluationConfig {
    fn from(config: UninitializedEvaluationConfig) -> Self {
        match config {
            UninitializedEvaluationConfig::Inference(c) => Self::Inference(c.into()),
        }
    }
}

impl From<UninitializedInferenceEvaluationConfig> for StoredInferenceEvaluationConfig {
    fn from(config: UninitializedInferenceEvaluationConfig) -> Self {
        let UninitializedInferenceEvaluationConfig {
            evaluators,
            function_name,
            description,
        } = config;

        Self {
            evaluators: evaluators.into_iter().map(|(k, v)| (k, v.into())).collect(),
            function_name,
            description,
        }
    }
}

impl From<UninitializedEvaluatorConfig> for StoredEvaluatorConfig {
    fn from(config: UninitializedEvaluatorConfig) -> Self {
        match config {
            UninitializedEvaluatorConfig::ExactMatch(c) => Self::ExactMatch(c),
            UninitializedEvaluatorConfig::LLMJudge(c) => Self::LLMJudge(c.into()),
            UninitializedEvaluatorConfig::ToolUse(c) => Self::ToolUse(c),
        }
    }
}

impl From<UninitializedLLMJudgeConfig> for StoredLLMJudgeConfig {
    fn from(config: UninitializedLLMJudgeConfig) -> Self {
        let UninitializedLLMJudgeConfig {
            input_format,
            variants,
            output_type,
            optimize,
            include,
            cutoff,
            description,
        } = config;

        Self {
            input_format,
            variants: variants.into_iter().map(|(k, v)| (k, v.into())).collect(),
            output_type,
            optimize,
            include,
            cutoff,
            description,
        }
    }
}

impl From<UninitializedLLMJudgeVariantInfo> for StoredLLMJudgeVariantInfo {
    fn from(info: UninitializedLLMJudgeVariantInfo) -> Self {
        let UninitializedLLMJudgeVariantInfo { inner, timeouts } = info;

        Self {
            inner: inner.into(),
            timeouts,
        }
    }
}

impl From<crate::evaluations::UninitializedLLMJudgeVariantConfig> for StoredLLMJudgeVariantConfig {
    fn from(config: crate::evaluations::UninitializedLLMJudgeVariantConfig) -> Self {
        use crate::evaluations::UninitializedLLMJudgeVariantConfig as Src;
        match config {
            Src::ChatCompletion(c) => Self::ChatCompletion(c),
            Src::BestOfNSampling(c) => Self::BestOfNSampling(c.into()),
            Src::MixtureOfNSampling(c) => Self::MixtureOfNSampling(c.into()),
            Src::Dicl(c) => Self::Dicl(c),
            Src::ChainOfThought(c) => Self::ChainOfThought(c),
        }
    }
}

impl From<UninitializedLLMJudgeBestOfNVariantConfig> for StoredLLMJudgeBestOfNVariantConfig {
    fn from(config: UninitializedLLMJudgeBestOfNVariantConfig) -> Self {
        let UninitializedLLMJudgeBestOfNVariantConfig {
            active,
            candidates,
            evaluator,
        } = config;

        Self {
            active,
            timeout_s: None,
            candidates,
            evaluator,
        }
    }
}

impl From<UninitializedLLMJudgeMixtureOfNVariantConfig> for StoredLLMJudgeMixtureOfNVariantConfig {
    fn from(config: UninitializedLLMJudgeMixtureOfNVariantConfig) -> Self {
        let UninitializedLLMJudgeMixtureOfNVariantConfig {
            active,
            candidates,
            fuser,
        } = config;

        Self {
            active,
            timeout_s: None,
            candidates,
            fuser,
        }
    }
}

// --- From<Stored*> for Uninitialized* conversions ---

impl From<StoredEvaluationConfig> for UninitializedEvaluationConfig {
    fn from(stored: StoredEvaluationConfig) -> Self {
        match stored {
            StoredEvaluationConfig::Inference(c) => Self::Inference(c.into()),
        }
    }
}

impl From<StoredInferenceEvaluationConfig> for UninitializedInferenceEvaluationConfig {
    fn from(stored: StoredInferenceEvaluationConfig) -> Self {
        let StoredInferenceEvaluationConfig {
            evaluators,
            function_name,
            description,
        } = stored;

        Self {
            evaluators: evaluators.into_iter().map(|(k, v)| (k, v.into())).collect(),
            function_name,
            description,
        }
    }
}

impl From<StoredEvaluatorConfig> for UninitializedEvaluatorConfig {
    fn from(stored: StoredEvaluatorConfig) -> Self {
        match stored {
            StoredEvaluatorConfig::ExactMatch(c) => Self::ExactMatch(c),
            StoredEvaluatorConfig::LLMJudge(c) => Self::LLMJudge(c.into()),
            StoredEvaluatorConfig::ToolUse(c) => Self::ToolUse(c),
        }
    }
}

impl From<StoredLLMJudgeConfig> for UninitializedLLMJudgeConfig {
    fn from(stored: StoredLLMJudgeConfig) -> Self {
        let StoredLLMJudgeConfig {
            input_format,
            variants,
            output_type,
            optimize,
            include,
            cutoff,
            description,
        } = stored;

        Self {
            input_format,
            variants: variants.into_iter().map(|(k, v)| (k, v.into())).collect(),
            output_type,
            optimize,
            include,
            cutoff,
            description,
        }
    }
}

impl From<StoredLLMJudgeVariantInfo> for UninitializedLLMJudgeVariantInfo {
    fn from(stored: StoredLLMJudgeVariantInfo) -> Self {
        let StoredLLMJudgeVariantInfo { inner, timeouts } = stored;

        Self {
            inner: inner.into(),
            timeouts,
        }
    }
}

impl From<StoredLLMJudgeVariantConfig> for crate::evaluations::UninitializedLLMJudgeVariantConfig {
    fn from(stored: StoredLLMJudgeVariantConfig) -> Self {
        use crate::evaluations::UninitializedLLMJudgeVariantConfig as Dst;
        match stored {
            StoredLLMJudgeVariantConfig::ChatCompletion(c) => Dst::ChatCompletion(c),
            StoredLLMJudgeVariantConfig::BestOfNSampling(c) => Dst::BestOfNSampling(c.into()),
            StoredLLMJudgeVariantConfig::MixtureOfNSampling(c) => Dst::MixtureOfNSampling(c.into()),
            StoredLLMJudgeVariantConfig::Dicl(c) => Dst::Dicl(c),
            StoredLLMJudgeVariantConfig::ChainOfThought(c) => Dst::ChainOfThought(c),
        }
    }
}

impl From<StoredLLMJudgeBestOfNVariantConfig> for UninitializedLLMJudgeBestOfNVariantConfig {
    fn from(stored: StoredLLMJudgeBestOfNVariantConfig) -> Self {
        let StoredLLMJudgeBestOfNVariantConfig {
            active,
            timeout_s: _, // silently dropped — was a no-op in evaluations
            candidates,
            evaluator,
        } = stored;

        Self {
            active,
            candidates,
            evaluator,
        }
    }
}

impl From<StoredLLMJudgeMixtureOfNVariantConfig> for UninitializedLLMJudgeMixtureOfNVariantConfig {
    fn from(stored: StoredLLMJudgeMixtureOfNVariantConfig) -> Self {
        let StoredLLMJudgeMixtureOfNVariantConfig {
            active,
            timeout_s: _, // silently dropped — was a no-op in evaluations
            candidates,
            fuser,
        } = stored;

        Self {
            active,
            candidates,
            fuser,
        }
    }
}
