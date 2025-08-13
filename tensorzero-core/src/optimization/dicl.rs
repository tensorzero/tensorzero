#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use url::Url;

use crate::{
    clickhouse::ClickHouseConnectionInfo,
    config_parser::{BatchWritesConfig, ProviderTypesConfig},
    embeddings::{
        EmbeddingEncodingFormat, EmbeddingInput, EmbeddingProvider, EmbeddingRequest,
        UninitializedEmbeddingProviderConfig,
    },
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    model::{build_creds_caching_default, CredentialLocation},
    optimization::{JobHandle, OptimizationJobInfo, Optimizer, OptimizerOutput},
    providers::openai::{
        default_api_key_location, OpenAICredentials, DEFAULT_CREDENTIALS, PROVIDER_TYPE,
    },
    stored_inference::RenderedSample,
};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct DiclOptimizationConfig {
    pub embedding_model: String,
    pub variant_name: String,
    pub function_name: String,
    pub clickhouse_url: String,
    #[serde(skip)]
    pub credentials: OpenAICredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
    pub api_base: Option<Url>,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "DiclOptimizationConfig"))]
pub struct UninitializedDiclOptimizationConfig {
    pub embedding_model: String,
    pub variant_name: String,
    pub function_name: String,
    pub clickhouse_url: String,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocation>,
    pub api_base: Option<Url>,
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
    #[pyo3(signature = (*, embedding_model, variant_name, function_name, clickhouse_url, credentials=None, api_base=None))]
    pub fn new(
        embedding_model: String,
        variant_name: String,
        function_name: String,
        clickhouse_url: String,
        credentials: Option<String>,
        api_base: Option<String>,
    ) -> PyResult<Self> {
        // Use Deserialize to convert the string to a CredentialLocation
        let credentials =
            credentials.map(|s| serde_json::from_str(&s).unwrap_or(CredentialLocation::Env(s)));
        let api_base = api_base
            .map(|s| {
                Url::parse(&s)
                    .map_err(|e| PyErr::new::<PyValueError, std::string::String>(e.to_string()))
            })
            .transpose()?;
        Ok(Self {
            embedding_model,
            variant_name,
            function_name,
            clickhouse_url,
            credentials,
            api_base,
        })
    }

    /// Initialize the DiclOptimizationConfig. All parameters are optional except for `embedding_model`.
    ///
    /// :param embedding_model: The embedding model to use.
    /// :param variant_name: The name to be used for the DICL variant.
    /// :param function_name: The name of the function to optimize.
    /// :param clickhouse_url: The URL of the ClickHouse database to use for storing the DICL examples.
    /// :param credentials: The credentials to use for the fine-tuning job. This should be a string like "env::OPENAI_API_KEY". See docs for more details.
    /// :param api_base: The base URL to use for the fine-tuning job. This is primarily used for testing.
    #[expect(unused_variables)]
    #[pyo3(signature = (*, embedding_model, variant_name, function_name, clickhouse_url, credentials=None, api_base=None))]
    fn __init__(
        this: Py<Self>,
        embedding_model: String,
        variant_name: String,
        function_name: String,
        clickhouse_url: String,
        credentials: Option<String>,
        api_base: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedDiclOptimizationConfig {
    pub fn load(self) -> Result<DiclOptimizationConfig, Error> {
        Ok(DiclOptimizationConfig {
            embedding_model: self.embedding_model,
            variant_name: self.variant_name,
            function_name: self.function_name,
            clickhouse_url: self.clickhouse_url,
            api_base: self.api_base,
            credentials: build_creds_caching_default(
                self.credentials.clone(),
                default_api_key_location(),
                PROVIDER_TYPE,
                &DEFAULT_CREDENTIALS,
            )?,
            credential_location: self.credentials,
        })
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct DiclOptimizationJobHandle {
    pub job_id: String,
    /// A url to a human-readable page for the job.
    pub job_url: Url,
    pub job_api_url: Url,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
}

impl std::fmt::Display for DiclOptimizationJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl Optimizer for DiclOptimizationConfig {
    type Handle = DiclOptimizationJobHandle;

    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::Handle, Error> {
        // Warn if val_examples is provided (not used in DICL)
        if val_examples.is_some() {
            tracing::warn!("val_examples provided for DICL optimization but will be ignored");
        }

        // Check if we have examples to process
        if train_examples.is_empty() {
            return Err(Error::new(ErrorDetails::AppState {
                message: "No training examples provided for DICL optimization".to_string(),
            }));
        }

        // Convert RenderedSample inputs to strings for embedding
        let input_texts: Vec<String> = train_examples
            .iter()
            .map(|sample| {
                serde_json::to_string(&sample.input).unwrap_or_else(|_| sample.input.to_string())
            })
            .collect();

        // Initialize the embedding provider
        // Using OpenAI provider with the specified embedding model
        let provider_config_str = format!(
            r#"
            type = "openai"
            model_name = "{}"
            "#,
            self.embedding_model
        );

        if let Some(_api_base) = &self.api_base {
            // If api_base is provided, we need to use it
            // TODO: Support custom API base for embeddings
            tracing::warn!("Custom api_base for embeddings not yet supported in DICL optimization");
        }

        let provider_config =
            toml::from_str::<UninitializedEmbeddingProviderConfig>(&provider_config_str)
                .map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to create embedding provider config: {e}"),
                    })
                })?
                .load(
                    &ProviderTypesConfig::default(),
                    Arc::from(self.embedding_model.clone()),
                )
                .await?;

        // Get embeddings for all examples in batch
        let embedding_request = EmbeddingRequest {
            input: EmbeddingInput::Batch(input_texts.clone()),
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::Float,
        };

        let embeddings_response = provider_config
            .embed(&embedding_request, client, credentials)
            .await?;

        // Prepare data for insertion into ClickHouse
        let mut rows = Vec::new();
        for (i, sample) in train_examples.iter().enumerate() {
            let output = sample
                .output
                .as_ref()
                .and_then(|outputs| outputs.first())
                .map(|output| serde_json::to_string(output).unwrap_or_default())
                .unwrap_or_default();

            let row = json!({
                "id": Uuid::now_v7(),
                "function_name": self.function_name,
                "variant_name": self.variant_name,
                "namespace": "", // Empty namespace for now
                "input": input_texts[i],
                "output": output,
                "embedding": embeddings_response.embeddings[i],
            });
            rows.push(serde_json::to_string(&row).map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: format!("Failed to serialize DICL example: {e}"),
                })
            })?);
        }

        // Insert into ClickHouse
        // Note: In a real implementation, we'd need access to the ClickHouse connection
        // from the app state. For now, we'll return a success indicating the job is complete.
        let clickhouse =
            &ClickHouseConnectionInfo::new(&self.clickhouse_url, BatchWritesConfig::default())
                .await?;

        // Join all rows with newlines for JSONEachRow format
        let query = format!(
            "INSERT INTO DynamicInContextLearningExample\n\
            SETTINGS async_insert=1, wait_for_async_insert=1\n\
            FORMAT JSONEachRow\n\
            {}",
            rows.join("\n")
        );

        clickhouse.run_query_synchronous_no_params(query).await?;

        // Create a job handle indicating immediate success
        let job_handle = DiclOptimizationJobHandle {
            job_id: Uuid::now_v7().to_string(),
            job_url: Url::parse("https://tensorzero.com/dicl").map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse job URL: {e}"),
                })
            })?,
            job_api_url: Url::parse("https://api.tensorzero.com/dicl").map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse job API URL: {e}"),
                })
            })?,
            credential_location: self.credential_location.clone(),
        };

        // TODO: Actually insert the rows into ClickHouse when we have access to the connection
        tracing::info!(
            "DICL optimization prepared {} examples for function '{}' variant '{}'",
            rows.len(),
            self.function_name,
            self.variant_name
        );

        Ok(job_handle)
    }
}

impl JobHandle for DiclOptimizationJobHandle {
    async fn poll(
        &self,
        client: &reqwest::Client,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        // DICL optimization is synchronous, so it's always complete once launched
        let _ = (client, credentials);

        // DICL doesn't produce a new model or variant, it just updates the examples table
        // For now, return a successful completion without output
        // TODO: Consider adding a new OptimizerOutput variant for DICL
        // DICL optimization is synchronous, so we need to store the variant info in the handle
        // For now, return an empty model since we don't have access to the original config
        // TODO: Store variant config in the job handle or pass it through another way
        Ok(OptimizationJobInfo::Completed {
            output: OptimizerOutput::Model(crate::model::UninitializedModelConfig {
                routing: vec![],
                providers: std::collections::HashMap::new(),
                timeouts: crate::config_parser::TimeoutsConfig::default(),
            }),
        })
    }
}
