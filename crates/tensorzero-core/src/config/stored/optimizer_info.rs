use serde::{Deserialize, Serialize};

use crate::optimization::dicl::UninitializedDiclOptimizationConfig;
use crate::optimization::fireworks_sft::UninitializedFireworksSFTConfig;
use crate::optimization::gcp_vertex_gemini_sft::UninitializedGCPVertexGeminiSFTConfig;
use crate::optimization::gepa::UninitializedGEPAConfig;
use crate::optimization::openai_rft::UninitializedOpenAIRFTConfig;
use crate::optimization::openai_sft::UninitializedOpenAISFTConfig;
use crate::optimization::together_sft::UninitializedTogetherSFTConfig;
use crate::optimization::{UninitializedOptimizerConfig, UninitializedOptimizerInfo};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredOptimizerInfo {
    #[serde(flatten)]
    pub inner: StoredOptimizerConfig,
}

impl From<UninitializedOptimizerInfo> for StoredOptimizerInfo {
    fn from(value: UninitializedOptimizerInfo) -> Self {
        Self {
            inner: value.inner.into(),
        }
    }
}

impl From<StoredOptimizerInfo> for UninitializedOptimizerInfo {
    fn from(value: StoredOptimizerInfo) -> Self {
        Self {
            inner: value.inner.into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredOptimizerConfig {
    #[serde(rename = "dicl")]
    Dicl(UninitializedDiclOptimizationConfig),
    #[serde(rename = "openai_sft")]
    OpenAISFT(UninitializedOpenAISFTConfig),
    #[serde(rename = "openai_rft")]
    OpenAIRFT(Box<UninitializedOpenAIRFTConfig>),
    #[serde(rename = "fireworks_sft")]
    FireworksSFT(UninitializedFireworksSFTConfig),
    #[serde(rename = "gcp_vertex_gemini_sft")]
    GCPVertexGeminiSFT(UninitializedGCPVertexGeminiSFTConfig),
    #[serde(rename = "gepa")]
    GEPA(StoredGEPAConfig),
    #[serde(rename = "together_sft")]
    TogetherSFT(Box<UninitializedTogetherSFTConfig>),
}

impl From<UninitializedOptimizerConfig> for StoredOptimizerConfig {
    fn from(value: UninitializedOptimizerConfig) -> Self {
        match value {
            UninitializedOptimizerConfig::Dicl(config) => Self::Dicl(config),
            UninitializedOptimizerConfig::OpenAISFT(config) => Self::OpenAISFT(config),
            UninitializedOptimizerConfig::OpenAIRFT(config) => Self::OpenAIRFT(config),
            UninitializedOptimizerConfig::FireworksSFT(config) => Self::FireworksSFT(config),
            UninitializedOptimizerConfig::GCPVertexGeminiSFT(config) => {
                Self::GCPVertexGeminiSFT(config)
            }
            UninitializedOptimizerConfig::GEPA(config) => Self::GEPA(config.into()),
            UninitializedOptimizerConfig::TogetherSFT(config) => Self::TogetherSFT(config),
        }
    }
}

impl From<StoredOptimizerConfig> for UninitializedOptimizerConfig {
    fn from(value: StoredOptimizerConfig) -> Self {
        match value {
            StoredOptimizerConfig::Dicl(config) => Self::Dicl(config),
            StoredOptimizerConfig::OpenAISFT(config) => Self::OpenAISFT(config),
            StoredOptimizerConfig::OpenAIRFT(config) => Self::OpenAIRFT(config),
            StoredOptimizerConfig::FireworksSFT(config) => Self::FireworksSFT(config),
            StoredOptimizerConfig::GCPVertexGeminiSFT(config) => Self::GCPVertexGeminiSFT(config),
            StoredOptimizerConfig::GEPA(config) => Self::GEPA(config.into()),
            StoredOptimizerConfig::TogetherSFT(config) => Self::TogetherSFT(config),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredGEPAConfig {
    pub function_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_names: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_variants: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_prefix: Option<String>,
    #[serde(default = "crate::optimization::gepa::default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "crate::optimization::gepa::default_max_iterations")]
    pub max_iterations: u32,
    #[serde(default = "crate::optimization::gepa::default_max_concurrency")]
    pub max_concurrency: u32,
    pub analysis_model: String,
    pub mutation_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(default = "crate::optimization::gepa::default_timeout")]
    pub timeout: u64,
    #[serde(default = "crate::optimization::gepa::default_include_inference_for_mutation")]
    pub include_inference_for_mutation: bool,
    #[serde(default)]
    pub retries: crate::utils::retries::RetryConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

impl From<UninitializedGEPAConfig> for StoredGEPAConfig {
    fn from(value: UninitializedGEPAConfig) -> Self {
        let UninitializedGEPAConfig {
            function_name,
            evaluation_name,
            evaluator_names,
            initial_variants,
            variant_prefix,
            batch_size,
            max_iterations,
            max_concurrency,
            analysis_model,
            mutation_model,
            seed,
            timeout,
            include_inference_for_mutation,
            retries,
            max_tokens,
        } = value;

        Self {
            function_name,
            evaluation_name,
            evaluator_names,
            initial_variants,
            variant_prefix,
            batch_size,
            max_iterations,
            max_concurrency,
            analysis_model,
            mutation_model,
            seed,
            timeout,
            include_inference_for_mutation,
            retries,
            max_tokens,
        }
    }
}

impl From<StoredGEPAConfig> for UninitializedGEPAConfig {
    fn from(value: StoredGEPAConfig) -> Self {
        let StoredGEPAConfig {
            function_name,
            evaluation_name,
            evaluator_names,
            initial_variants,
            variant_prefix,
            batch_size,
            max_iterations,
            max_concurrency,
            analysis_model,
            mutation_model,
            seed,
            timeout,
            include_inference_for_mutation,
            retries,
            max_tokens,
        } = value;

        Self {
            function_name,
            evaluation_name,
            evaluator_names,
            initial_variants,
            variant_prefix,
            batch_size,
            max_iterations,
            max_concurrency,
            analysis_model,
            mutation_model,
            seed,
            timeout,
            include_inference_for_mutation,
            retries,
            max_tokens,
        }
    }
}
