use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::error::Error;
use crate::model_table::ProviderTypeDefaultCredentials;
use crate::utils::retries::RetryConfig;
use crate::variant::chat_completion::UninitializedChatCompletionConfig;

// Default functions
fn default_batch_size() -> usize {
    5
}

fn default_max_iterations() -> u32 {
    1
}

fn default_max_concurrency() -> u32 {
    10
}

fn default_timeout() -> u64 {
    300
}

fn default_analysis_model() -> String {
    "openai::gpt-5-mini".to_string()
}

fn default_mutation_model() -> String {
    "openai::gpt-5".to_string()
}

fn default_include_inference_input_for_mutation() -> bool {
    false
}

fn default_max_tokens() -> u32 {
    16_384
}

/// GEPA (Genetic Evolution with Pareto Analysis) optimization configuration
///
/// GEPA is a multi-objective optimization algorithm that maintains a Pareto frontier
/// of high-performing variants. It uses genetic programming techniques to evolve
/// prompt templates based on evaluation results.
#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GEPAConfig {
    /// Name of the function being optimized
    pub function_name: String,

    /// Name of the evaluation used to score candidate variants
    pub evaluation_name: String,

    /// Optional list of variant_names to initialize GEPA with.
    /// If None, will use all variants defined for the function.
    pub initial_variants: Option<Vec<String>>,

    /// Prefix for the name of the new optimized variants
    pub variant_prefix: Option<String>,

    /// Number of training samples to analyze per iteration
    pub batch_size: usize,

    /// Maximum number of training iterations
    pub max_iterations: u32,

    /// Maximum number of concurrent inference calls
    pub max_concurrency: u32,

    /// Model for analysis/prediction (e.g., "openai::gpt-5-mini")
    pub analysis_model: String,

    /// Model for mutation (e.g., "openai::gpt-5")
    pub mutation_model: String,

    /// Optional random seed for reproducibility
    pub seed: Option<u32>,

    /// Client timeout in seconds for TensorZero gateway operations
    pub timeout: u64,

    /// Whether to include inference input in InferenceWithAnalysis for mutation
    ///
    /// If true, the mutate function will see the inference input in addition to the output and analysis for each example in the batch.
    /// This provides additional context but increases token usage significantly.
    ///
    /// **Warning:** Use with caution, especially with:
    /// - Multi-turn conversations (many input messages)
    /// - Large batch sizes (many analyses per mutation)
    ///
    /// These can cause context length overflow for the mutation model.
    pub include_inference_input_for_mutation: bool,

    /// Retry configuration for inference calls during GEPA optimization
    /// Applies to analyze function calls, mutate function calls, and all mutated variants
    pub retries: RetryConfig,

    /// Maximum number of tokens to generate for analysis and mutation model calls
    /// Default: 16_384
    pub max_tokens: u32,
}

/// Uninitialized GEPA configuration (deserializable from TOML)
#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "GEPAConfig"))]
pub struct UninitializedGEPAConfig {
    pub function_name: String,
    pub evaluation_name: String,
    pub initial_variants: Option<Vec<String>>,
    pub variant_prefix: Option<String>,

    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: u32,

    #[serde(default = "default_analysis_model")]
    pub analysis_model: String,

    #[serde(default = "default_mutation_model")]
    pub mutation_model: String,

    pub seed: Option<u32>,

    #[serde(default = "default_timeout")]
    pub timeout: u64,

    #[serde(default = "default_include_inference_input_for_mutation")]
    pub include_inference_input_for_mutation: bool,

    #[serde(default)]
    pub retries: RetryConfig,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

impl Default for UninitializedGEPAConfig {
    fn default() -> Self {
        Self {
            function_name: String::new(),
            evaluation_name: String::new(),
            initial_variants: None,
            variant_prefix: None,
            batch_size: default_batch_size(),
            max_iterations: default_max_iterations(),
            max_concurrency: default_max_concurrency(),
            analysis_model: default_analysis_model(),
            mutation_model: default_mutation_model(),
            seed: None,
            timeout: default_timeout(),
            include_inference_input_for_mutation: default_include_inference_input_for_mutation(),
            retries: RetryConfig::default(),
            max_tokens: default_max_tokens(),
        }
    }
}

impl std::fmt::Display for UninitializedGEPAConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedGEPAConfig {
    #[new]
    #[pyo3(signature = (
        function_name,
        evaluation_name,
        initial_variants=None,
        variant_prefix=None,
        batch_size=None,
        max_iterations=None,
        max_concurrency=None,
        analysis_model=None,
        mutation_model=None,
        seed=None,
        timeout=None,
        include_inference_input_for_mutation=None,
        retries=None,
        max_tokens=None,
    ))]
    #[expect(clippy::too_many_arguments)]
    fn py_new(
        function_name: String,
        evaluation_name: String,
        initial_variants: Option<Vec<String>>,
        variant_prefix: Option<String>,
        batch_size: Option<usize>,
        max_iterations: Option<u32>,
        max_concurrency: Option<u32>,
        analysis_model: Option<String>,
        mutation_model: Option<String>,
        seed: Option<u32>,
        timeout: Option<u64>,
        include_inference_input_for_mutation: Option<bool>,
        retries: Option<RetryConfig>,
        max_tokens: Option<u32>,
    ) -> Self {
        Self {
            function_name,
            evaluation_name,
            initial_variants,
            variant_prefix,
            batch_size: batch_size.unwrap_or_else(default_batch_size),
            max_iterations: max_iterations.unwrap_or_else(default_max_iterations),
            max_concurrency: max_concurrency.unwrap_or_else(default_max_concurrency),
            analysis_model: analysis_model.unwrap_or_else(default_analysis_model),
            mutation_model: mutation_model.unwrap_or_else(default_mutation_model),
            seed,
            timeout: timeout.unwrap_or_else(default_timeout),
            include_inference_input_for_mutation: include_inference_input_for_mutation
                .unwrap_or_else(default_include_inference_input_for_mutation),
            retries: retries.unwrap_or_default(),
            max_tokens: max_tokens.unwrap_or_else(default_max_tokens),
        }
    }

    /// Initialize the GEPAConfig. All parameters are optional except for `function_name` and `evaluation_name`.
    ///
    /// :param function_name: Name of the function being optimized.
    /// :param evaluation_name: Name of the evaluation used to score candidate variants.
    /// :param initial_variants: Optional list of variant names to initialize GEPA with. If None, uses all variants defined for the function.
    /// :param variant_prefix: Prefix for the name of the new optimized variants.
    /// :param batch_size: Number of training samples to analyze per iteration. Default: 5.
    /// :param max_iterations: Maximum number of training iterations. Default: 1.
    /// :param max_concurrency: Maximum number of concurrent inference calls. Default: 10.
    /// :param analysis_model: Model for analysis/prediction. Default: "openai::gpt-5-mini".
    /// :param mutation_model: Model for mutation. Default: "openai::gpt-5".
    /// :param seed: Optional random seed for reproducibility.
    /// :param timeout: Client timeout in seconds for TensorZero gateway operations. Default: 300.
    /// :param include_inference_input_for_mutation: Whether to include inference input in InferenceWithAnalysis for mutation. If True, the mutate function will see the inference input in addition to the output and analysis for each example in the batch. Use with caution for multi-turn conversations or large batch sizes. Default: False.
    /// :param retries: Retry configuration for inference calls during GEPA optimization.
    /// :param max_tokens: Maximum number of tokens to generate for analysis and mutation model calls. Default: 16_384.
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, function_name, evaluation_name, initial_variants=None, variant_prefix=None, batch_size=None, max_iterations=None, max_concurrency=None, analysis_model=None, mutation_model=None, seed=None, timeout=None, include_inference_input_for_mutation=None, retries=None, max_tokens=None))]
    fn __init__(
        this: Py<Self>,
        function_name: String,
        evaluation_name: String,
        initial_variants: Option<Vec<String>>,
        variant_prefix: Option<String>,
        batch_size: Option<usize>,
        max_iterations: Option<u32>,
        max_concurrency: Option<u32>,
        analysis_model: Option<String>,
        mutation_model: Option<String>,
        seed: Option<u32>,
        timeout: Option<u64>,
        include_inference_input_for_mutation: Option<bool>,
        retries: Option<RetryConfig>,
        max_tokens: Option<u32>,
    ) -> Py<Self> {
        this
    }

    fn __repr__(&self) -> String {
        format!("{self}")
    }
}

impl UninitializedGEPAConfig {
    /// Load the configuration (GEPA doesn't need credential resolution)
    pub async fn load(
        self,
        _default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<GEPAConfig, Error> {
        Ok(GEPAConfig {
            function_name: self.function_name,
            evaluation_name: self.evaluation_name,
            initial_variants: self.initial_variants,
            variant_prefix: self.variant_prefix,
            batch_size: self.batch_size,
            max_iterations: self.max_iterations,
            max_concurrency: self.max_concurrency,
            analysis_model: self.analysis_model,
            mutation_model: self.mutation_model,
            seed: self.seed,
            timeout: self.timeout,
            include_inference_input_for_mutation: self.include_inference_input_for_mutation,
            retries: self.retries,
            max_tokens: self.max_tokens,
        })
    }
}

/// Job handle for GEPA optimization
///
/// Contains the final Pareto frontier of optimized variants or an error message.
/// GEPA optimization is synchronous, so polling immediately returns the completed
/// results or failure status.
#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct GEPAJobHandle {
    /// Result of the GEPA optimization - either a map of variant names to their
    /// configurations (the Pareto frontier) or an error message
    pub result: Result<HashMap<String, UninitializedChatCompletionConfig>, String>,
}

impl std::fmt::Display for GEPAJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.result {
            Ok(variants) => {
                let json = serde_json::to_string_pretty(variants).map_err(|_| std::fmt::Error)?;
                write!(f, "Success: {json}")
            }
            Err(msg) => write!(f, "Failed: {msg}"),
        }
    }
}

// Manual PartialEq implementation since UninitializedChatCompletionConfig doesn't derive PartialEq
// We compare based on success/failure status and variant names (not the full configs)
impl PartialEq for GEPAJobHandle {
    fn eq(&self, other: &Self) -> bool {
        match (&self.result, &other.result) {
            (Ok(self_variants), Ok(other_variants)) => {
                // Compare by checking if both have the same set of variant names
                if self_variants.len() != other_variants.len() {
                    return false;
                }
                self_variants.keys().all(|k| other_variants.contains_key(k))
            }
            (Err(self_msg), Err(other_msg)) => self_msg == other_msg,
            _ => false,
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl GEPAJobHandle {
    fn __repr__(&self) -> String {
        format!("{self}")
    }
}
