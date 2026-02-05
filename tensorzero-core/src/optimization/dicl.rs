use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

fn default_batch_size() -> usize {
    128
}

fn default_max_concurrency() -> usize {
    10
}

fn default_k() -> u32 {
    10
}

/// Deprecated default model for DICL optimization.
/// This will be removed in a future release when `model` becomes mandatory.
pub const DEPRECATED_DEFAULT_MODEL: &str = "openai::gpt-4o-mini-2024-07-18";

fn default_append_to_existing_variants() -> bool {
    false
}

/// Initialized DICL optimization configuration (per-job settings only).
/// Credentials come from `provider_types.openai.defaults` in the gateway configuration.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct DiclOptimizationConfig {
    pub embedding_model: Arc<str>,
    pub variant_name: String,
    pub function_name: String,
    pub dimensions: Option<u32>,
    pub batch_size: usize,
    pub max_concurrency: usize,
    pub k: u32,
    pub model: Arc<str>,
    pub append_to_existing_variants: bool,
}

/// Uninitialized DICL optimization configuration (per-job settings only).
/// Credentials come from `provider_types.openai.defaults` in the gateway configuration.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "DICLOptimizationConfig"))]
pub struct UninitializedDiclOptimizationConfig {
    pub embedding_model: String,
    pub variant_name: String,
    pub function_name: String,
    pub dimensions: Option<u32>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    #[serde(default = "default_k")]
    pub k: u32,
    /// The model to use for the DICL variant.
    /// This field will be required in a future release.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_append_to_existing_variants")]
    pub append_to_existing_variants: bool,
}

impl Default for UninitializedDiclOptimizationConfig {
    fn default() -> Self {
        Self {
            embedding_model: String::new(),
            variant_name: String::new(),
            function_name: String::new(),
            dimensions: None,
            batch_size: default_batch_size(),
            max_concurrency: default_max_concurrency(),
            k: default_k(),
            model: None,
            append_to_existing_variants: default_append_to_existing_variants(),
        }
    }
}

impl std::fmt::Display for UninitializedDiclOptimizationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedDiclOptimizationConfig {
    /// Initialize the DiclOptimizationConfig.
    ///
    /// Credentials come from `provider_types.openai.defaults` in the gateway configuration.
    ///
    /// :param embedding_model: The embedding model to use (required).
    /// :param variant_name: The name to be used for the DICL variant (required).
    /// :param function_name: The name of the function to optimize (required).
    /// :param dimensions: The dimensions of the embeddings. If None, uses the model's default.
    /// :param batch_size: The batch size to use for getting embeddings.
    /// :param max_concurrency: The maximum concurrency to use for getting embeddings.
    /// :param k: The number of nearest neighbors to use for the DICL variant.
    /// :param model: The model to use for the DICL variant. This field will be required in a future release.
    /// :param append_to_existing_variants: Whether to append to existing variants. If False (default), raises an error if the variant already exists.
    #[new]
    #[expect(clippy::too_many_arguments)]
    #[pyo3(signature = (*, embedding_model, variant_name, function_name, dimensions=None, batch_size=None, max_concurrency=None, k=None, model=None, append_to_existing_variants=None))]
    pub fn new(
        embedding_model: String,
        variant_name: String,
        function_name: String,
        dimensions: Option<u32>,
        batch_size: Option<usize>,
        max_concurrency: Option<usize>,
        k: Option<u32>,
        model: Option<String>,
        append_to_existing_variants: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self {
            embedding_model,
            variant_name,
            function_name,
            dimensions,
            batch_size: batch_size.unwrap_or_else(default_batch_size),
            max_concurrency: max_concurrency.unwrap_or_else(default_max_concurrency),
            k: k.unwrap_or_else(default_k),
            model,
            append_to_existing_variants: append_to_existing_variants
                .unwrap_or_else(default_append_to_existing_variants),
        })
    }

    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, embedding_model, variant_name, function_name, dimensions=None, batch_size=None, max_concurrency=None, k=None, model=None, append_to_existing_variants=None))]
    fn __init__(
        this: Py<Self>,
        embedding_model: String,
        variant_name: String,
        function_name: String,
        dimensions: Option<u32>,
        batch_size: Option<usize>,
        max_concurrency: Option<usize>,
        k: Option<u32>,
        model: Option<String>,
        append_to_existing_variants: Option<bool>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedDiclOptimizationConfig {
    pub fn load(self) -> DiclOptimizationConfig {
        let model = match self.model {
            Some(m) => Arc::from(m),
            None => {
                tracing::warn!(
                    "The `model` field in `DICLOptimizationConfig` was not specified. \
                     Using deprecated default `{}`. \
                     This field will be required in a future release. (#5616)",
                    DEPRECATED_DEFAULT_MODEL
                );
                Arc::from(DEPRECATED_DEFAULT_MODEL)
            }
        };

        DiclOptimizationConfig {
            embedding_model: Arc::from(self.embedding_model),
            variant_name: self.variant_name,
            function_name: self.function_name,
            dimensions: self.dimensions,
            batch_size: self.batch_size,
            max_concurrency: self.max_concurrency,
            k: self.k,
            model,
            append_to_existing_variants: self.append_to_existing_variants,
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct DiclOptimizationJobHandle {
    pub embedding_model: String,
    pub k: u32,
    pub model: String,
}

impl std::fmt::Display for DiclOptimizationJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
