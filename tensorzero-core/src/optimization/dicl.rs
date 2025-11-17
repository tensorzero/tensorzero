use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    error::Error,
    model::CredentialLocationWithFallback,
    model_table::{OpenAIKind, ProviderKind, ProviderTypeDefaultCredentials},
    providers::openai::OpenAICredentials,
};

#[cfg(feature = "pyo3")]
use crate::model::CredentialLocation;
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

fn default_model() -> String {
    "openai::gpt-4o-mini-2024-07-18".to_string()
}

fn default_append_to_existing_variants() -> bool {
    false
}

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
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
    #[serde(skip)]
    pub credentials: OpenAICredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize)]
#[ts(export)]
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
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_append_to_existing_variants")]
    pub append_to_existing_variants: bool,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocationWithFallback>,
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
            model: default_model(),
            append_to_existing_variants: default_append_to_existing_variants(),
            credentials: None,
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
    // We allow too many arguments since it is a Python constructor
    /// NOTE: This signature currently does not work:
    /// print(DiclOptimizationConfig.__init__.__text_signature__)
    /// prints out signature:
    /// ($self, /, *args, **kwargs)
    #[new]
    #[pyo3(signature = (*, embedding_model, variant_name, function_name, dimensions=None, batch_size=None, max_concurrency=None, k=None, model=None, append_to_existing_variants=None, credentials=None))]
    #[expect(clippy::too_many_arguments)]
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
        credentials: Option<String>,
    ) -> PyResult<Self> {
        // Use Deserialize to convert the string to a CredentialLocationWithFallback
        let credentials = credentials.map(|s| {
            serde_json::from_str(&s).unwrap_or(CredentialLocationWithFallback::Single(
                CredentialLocation::Env(s),
            ))
        });
        Ok(Self {
            embedding_model,
            variant_name,
            function_name,
            dimensions,
            batch_size: batch_size.unwrap_or_else(default_batch_size),
            max_concurrency: max_concurrency.unwrap_or_else(default_max_concurrency),
            k: k.unwrap_or_else(default_k),
            model: model.unwrap_or_else(default_model),
            append_to_existing_variants: append_to_existing_variants
                .unwrap_or_else(default_append_to_existing_variants),
            credentials,
        })
    }

    /// Initialize the DiclOptimizationConfig. All parameters are optional except for `embedding_model`.
    ///
    /// :param embedding_model: The embedding model to use.
    /// :param variant_name: The name to be used for the DICL variant.
    /// :param function_name: The name of the function to optimize.
    /// :param dimensions: The dimensions of the embeddings. If None, uses the model's default.
    /// :param batch_size: The batch size to use for getting embeddings.
    /// :param max_concurrency: The maximum concurrency to use for getting embeddings.
    /// :param k: The number of nearest neighbors to use for the DICL variant.
    /// :param model: The model to use for the DICL variant.
    /// :param append_to_existing_variants: Whether to append to existing variants. If False (default), raises an error if the variant already exists.
    /// :param credentials: The credentials to use for embedding. This should be a string like `env::OPENAI_API_KEY`. See docs for more details.
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, embedding_model, variant_name, function_name, dimensions=None, batch_size=None, max_concurrency=None, k=None, model=None, append_to_existing_variants=None, credentials=None))]
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
        credentials: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedDiclOptimizationConfig {
    pub async fn load(
        self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<DiclOptimizationConfig, Error> {
        Ok(DiclOptimizationConfig {
            embedding_model: Arc::from(self.embedding_model),
            variant_name: self.variant_name,
            function_name: self.function_name,
            dimensions: self.dimensions,
            batch_size: self.batch_size,
            max_concurrency: self.max_concurrency,
            k: self.k,
            model: Arc::from(self.model),
            append_to_existing_variants: self.append_to_existing_variants,
            credentials: OpenAIKind
                .get_defaulted_credential(self.credentials.as_ref(), default_credentials)
                .await?,
            credential_location: self.credentials,
        })
    }
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
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
