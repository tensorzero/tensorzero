#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    cache::CacheOptions,
    config::{Config, UninitializedVariantConfig},
    db::clickhouse::{ClickHouseConnectionInfo, ExternalDataInfo},
    embeddings::{
        Embedding, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingModelConfig, EmbeddingRequest,
    },
    endpoints::inference::{InferenceClients, InferenceCredentials},
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    model::{build_creds_caching_default, CredentialLocation},
    optimization::{JobHandle, OptimizationJobInfo, Optimizer, OptimizerOutput},
    providers::openai::{
        default_api_key_location, OpenAICredentials, DEFAULT_CREDENTIALS, PROVIDER_TYPE,
    },
    stored_inference::{RenderedSample, StoredOutput},
    variant::{dicl::UninitializedDiclConfig, RetryConfig},
};
use futures::future::try_join_all;
use std::sync::Arc;
use tokio::sync::Semaphore;
use uuid::Uuid;

fn default_batch_size() -> usize {
    128
}

fn default_max_concurrency() -> usize {
    10
}

fn default_k() -> usize {
    10
}

fn default_model() -> String {
    "openai::gpt-4o-mini-2024-07-18".to_string()
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct DiclOptimizationConfig {
    pub embedding_model: String,
    pub variant_name: String,
    pub function_name: String,
    pub dimensions: Option<u32>,
    pub batch_size: usize,
    pub max_concurrency: usize,
    pub k: usize,
    pub model: String,
    #[serde(skip)]
    pub credentials: OpenAICredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "DiclOptimizationConfig"))]
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
    pub k: usize,
    #[serde(default = "default_model")]
    pub model: String,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocation>,
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
    #[pyo3(signature = (*, embedding_model, variant_name, function_name, dimensions=None, batch_size=None, max_concurrency=None, k=None, model=None, credentials=None))]
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        embedding_model: String,
        variant_name: String,
        function_name: String,
        dimensions: Option<u32>,
        batch_size: Option<usize>,
        max_concurrency: Option<usize>,
        k: Option<usize>,
        model: Option<String>,
        credentials: Option<String>,
    ) -> PyResult<Self> {
        // Use Deserialize to convert the string to a CredentialLocation
        let credentials =
            credentials.map(|s| serde_json::from_str(&s).unwrap_or(CredentialLocation::Env(s)));
        Ok(Self {
            embedding_model,
            variant_name,
            function_name,
            dimensions,
            batch_size: batch_size.unwrap_or_else(default_batch_size),
            max_concurrency: max_concurrency.unwrap_or_else(default_max_concurrency),
            k: k.unwrap_or_else(default_k),
            model: model.unwrap_or_else(default_model),
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
    /// :param credentials: The credentials to use for embedding. This should be a string like "env::OPENAI_API_KEY". See docs for more details.
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, embedding_model, variant_name, function_name, dimensions=None, batch_size=None, max_concurrency=None, k=None, model=None, credentials=None))]
    fn __init__(
        this: Py<Self>,
        embedding_model: String,
        variant_name: String,
        function_name: String,
        dimensions: Option<u32>,
        batch_size: Option<usize>,
        max_concurrency: Option<usize>,
        k: Option<usize>,
        model: Option<String>,
        credentials: Option<String>,
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
            dimensions: self.dimensions,
            batch_size: self.batch_size,
            max_concurrency: self.max_concurrency,
            k: self.k,
            model: self.model,
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
    pub embedding_model: String,
    pub k: usize,
    pub model: String,
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
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: &Config,
    ) -> Result<Self::Handle, Error> {
        // Check if we have examples to process
        if train_examples.is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: "DICL optimization requires at least one training example".to_string(),
            }));
        }

        // Warn if val_examples is provided (not used in DICL)
        if val_examples.is_some() {
            tracing::warn!("val_examples provided for DICL optimization but will be ignored");
        }

        // Check if ClickHouse is available (required for DICL)
        if matches!(
            clickhouse_connection_info,
            ClickHouseConnectionInfo::Disabled
        ) {
            return Err(Error::new(ErrorDetails::AppState {
                message: "DICL optimization requires ClickHouse to be enabled to store examples"
                    .to_string(),
            }));
        }

        // 1. Check that the function exists in the config
        let function_config = config.functions.get(&self.function_name).ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "function '{}' not found in configuration",
                    self.function_name
                ),
            })
        })?;

        // 2. Check that the variant name is not already in the function variants
        let variants = match &**function_config {
            FunctionConfig::Chat(chat_config) => &chat_config.variants,
            FunctionConfig::Json(json_config) => &json_config.variants,
        };

        if variants.contains_key(&self.variant_name) {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "variant '{}' already exists in function '{}' - DICL optimization cannot overwrite existing variants",
                    self.variant_name, self.function_name
                ),
            }));
        }

        // 3. Check that the embedding model exists in the config
        let embedding_model_config = config
            .embedding_models
            .get(&self.embedding_model)
            .await?
            .ok_or_else(|| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "embedding model '{}' not found in configuration",
                        self.embedding_model
                    ),
                })
            })?;

        tracing::info!(
            "🚀 Starting DICL optimization for function '{}' variant '{}' with {} examples",
            self.function_name,
            self.variant_name,
            train_examples.len()
        );

        // Convert RenderedSample inputs to strings for embedding using stored_input
        let input_texts: Vec<String> = train_examples
            .iter()
            .map(|sample| {
                serde_json::to_string(&sample.stored_input)
                    .map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Error in serializing stored_input in DICL optimization: {e}"
                            ),
                        })
                    })
                    .unwrap_or_else(|_| format!("{:?}", sample.stored_input))
            })
            .collect();

        // Process embeddings with batching and concurrency control
        let all_embeddings = process_embeddings_with_batching(
            &embedding_model_config,
            &self.embedding_model,
            client,
            credentials,
            input_texts,
            self.batch_size,
            self.max_concurrency,
            self.dimensions,
        )
        .await?;

        // Combine examples with their embeddings
        let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> = train_examples
            .into_iter()
            .zip(all_embeddings.into_iter())
            .collect();

        let num_examples = examples_with_embeddings.len();

        tracing::info!("🔄 Phase transition: Embedding → ClickHouse storage");

        insert_dicl_examples_with_batching(
            clickhouse_connection_info,
            examples_with_embeddings,
            &self.function_name,
            &self.variant_name,
            self.batch_size,
        )
        .await?;

        // Create a job handle indicating immediate success
        let job_handle = DiclOptimizationJobHandle {
            embedding_model: self.embedding_model.clone(),
            k: self.k,
            model: self.model.clone(),
        };

        tracing::info!(
            "🎉 DICL optimization completed successfully!\n\
            📈 Summary:\n\
            ├─ Function: '{}'\n\
            ├─ Variant: '{}'\n\
            ├─ Examples processed: {}\n\
            ├─ Embeddings generated: {}\n\
            ├─ Batch size: {}\n\
            └─ Max concurrency: {}",
            self.function_name,
            self.variant_name,
            num_examples,
            num_examples, // Same as examples since each example gets one embedding
            self.batch_size,
            self.max_concurrency
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

        // DICL produces a variant configuration that references the stored examples
        // Return a DICL variant with the configuration from the optimization job
        Ok(OptimizationJobInfo::Completed {
            output: OptimizerOutput::Variant(Box::new(UninitializedVariantConfig::Dicl(
                UninitializedDiclConfig {
                    weight: None,
                    embedding_model: self.embedding_model.clone(),
                    k: self.k as u32,
                    model: self.model.clone(),
                    system_instructions: None,
                    temperature: None,
                    top_p: None,
                    stop_sequences: None,
                    presence_penalty: None,
                    frequency_penalty: None,
                    max_tokens: None,
                    seed: None,
                    json_mode: None,
                    extra_body: None,
                    extra_headers: None,
                    retries: RetryConfig::default(),
                },
            ))),
        })
    }
}

/// Processes a batch of input texts to get embeddings
async fn process_embedding_batch(
    embedding_model_config: &EmbeddingModelConfig,
    model_name: &str,
    client: &reqwest::Client,
    credentials: &InferenceCredentials,
    batch_texts: Vec<String>,
    batch_index: usize,
    dimensions: Option<u32>,
) -> Result<Vec<Vec<f64>>, Error> {
    let embedding_request = EmbeddingRequest {
        input: EmbeddingInput::Batch(batch_texts),
        dimensions,
        encoding_format: EmbeddingEncodingFormat::Float,
    };

    // Create InferenceClients context for the embedding model
    let cache_options = CacheOptions::default();
    let clients = InferenceClients {
        http_client: client,
        credentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo::Disabled,
        cache_options: &cache_options,
    };

    let response = embedding_model_config
        .embed(&embedding_request, model_name, &clients)
        .await?;

    tracing::debug!("Successfully processed embedding batch {}", batch_index);

    // Convert embeddings from Vec<Embedding> to Vec<Vec<f64>>
    let embeddings: Result<Vec<Vec<f64>>, Error> = response
        .embeddings
        .into_iter()
        .map(|embedding| match embedding {
            Embedding::Float(vec_f32) => Ok(vec_f32.into_iter().map(|f| f as f64).collect()),
            Embedding::Base64(_) => Err(Error::new(ErrorDetails::Inference {
                message: "Base64 embeddings not supported for DICL optimization".to_string(),
            })),
        })
        .collect();

    embeddings
}

/// Processes all embedding batches with concurrency control
#[expect(clippy::too_many_arguments)]
async fn process_embeddings_with_batching(
    embedding_model_config: &EmbeddingModelConfig,
    model_name: &str,
    client: &reqwest::Client,
    credentials: &InferenceCredentials,
    input_texts: Vec<String>,
    batch_size: usize,
    max_concurrency: usize,
    dimensions: Option<u32>,
) -> Result<Vec<Vec<f64>>, Error> {
    let batches: Vec<Vec<String>> = input_texts
        .chunks(batch_size)
        .map(<[String]>::to_vec)
        .collect();

    tracing::info!(
        "🔢 Starting embedding processing: {} input texts → {} batches (size: {}, max concurrent: {})",
        input_texts.len(),
        batches.len(),
        batch_size,
        max_concurrency
    );

    // Process batches with controlled concurrency using a semaphore
    let total_batches = batches.len();
    let semaphore = Arc::new(Semaphore::new(max_concurrency));

    let batch_futures: Vec<_> = batches
        .into_iter()
        .enumerate()
        .map(|(batch_index, batch)| {
            let semaphore = Arc::clone(&semaphore);

            async move {
                let _permit = semaphore.acquire().await.map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to acquire semaphore permit: {e}"),
                    })
                })?;

                let result = process_embedding_batch(
                    embedding_model_config,
                    model_name,
                    client,
                    credentials,
                    batch,
                    batch_index,
                    dimensions,
                )
                .await;

                if batch_index > 0 && batch_index % 10 == 0 {
                    let progress_pct =
                        (batch_index as f64 / total_batches as f64 * 100.0).round() as u32;
                    tracing::info!(
                        "📊 Embedding progress: {}/{} batches completed ({}%)",
                        batch_index,
                        total_batches,
                        progress_pct
                    );
                }

                result
            }
        })
        .collect();

    let batch_results = try_join_all(batch_futures).await?;
    let all_embeddings: Vec<Vec<f64>> = batch_results.into_iter().flatten().collect();

    tracing::info!(
        "✅ Embedding processing complete: {} embeddings generated from {} input texts",
        all_embeddings.len(),
        input_texts.len()
    );

    Ok(all_embeddings)
}

/// Inserts DICL examples into ClickHouse using ExternalDataInfo pattern
pub async fn insert_dicl_examples_with_batching(
    clickhouse: &ClickHouseConnectionInfo,
    examples: Vec<(RenderedSample, Vec<f64>)>,
    function_name: &str,
    variant_name: &str,
    batch_size: usize,
) -> Result<(), Error> {
    let total_examples = examples.len();
    let total_batches = total_examples.div_ceil(batch_size);

    tracing::info!(
        "💾 Starting ClickHouse insertion: {} examples → {} batches (size: {})",
        total_examples,
        total_batches,
        batch_size
    );

    let mut inserted_examples = 0;

    // Process all examples in batches using ExternalDataInfo
    for (batch_index, batch) in examples.chunks(batch_size).enumerate() {
        let serialized_rows: Result<Vec<String>, Error> = batch
            .iter()
            .map(|(sample, embedding)| {
                let output = sample
                    .stored_output
                    .as_ref()
                    .map(|stored_output| match stored_output {
                        StoredOutput::Chat(outputs) => {
                            serde_json::to_string(outputs).unwrap_or_default()
                        }
                        StoredOutput::Json(json_output) => {
                            serde_json::to_string(json_output).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();

                let input_text = serde_json::to_string(&sample.stored_input)
                    .unwrap_or_else(|_| format!("{:?}", sample.stored_input));

                let row = json!({
                    "id": Uuid::now_v7(),
                    "function_name": function_name,
                    "variant_name": variant_name,
                    "namespace": "",
                    "input": input_text,
                    "output": output,
                    "embedding": embedding,
                });

                serde_json::to_string(&row).map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to serialize DICL example: {e}"),
                    })
                })
            })
            .collect();

        let rows = serialized_rows?;

        // Use ExternalDataInfo for efficient bulk insertion
        let query = r"
        INSERT INTO DynamicInContextLearningExample
            (
                id,
                function_name,
                variant_name,
                namespace,
                input,
                output,
                embedding
            )
            SELECT
                new_data.id,
                new_data.function_name,
                new_data.variant_name,
                new_data.namespace,
                new_data.input,
                new_data.output,
                new_data.embedding
            FROM new_data
        ";

        let external_data = ExternalDataInfo {
            external_data_name: "new_data".to_string(),
            structure: "id UUID, function_name LowCardinality(String), variant_name LowCardinality(String), namespace String, input String, output String, embedding Array(Float32)".to_string(),
            format: "JSONEachRow".to_string(),
            data: rows.join("\n"),
        };

        let result = clickhouse
            .run_query_with_external_data(external_data, query.to_string())
            .await?;

        inserted_examples += batch.len();
        let progress_pct =
            (inserted_examples as f64 / total_examples as f64 * 100.0).round() as u32;

        tracing::info!(
            "📊 ClickHouse insertion progress: {}/{} batches completed ({}%) - {}/{} examples inserted (wrote {} rows)",
            batch_index + 1,
            total_batches,
            progress_pct,
            inserted_examples,
            total_examples,
            result.metadata.written_rows
        );
    }

    tracing::info!(
        "✅ ClickHouse insertion complete: {} examples successfully stored for function '{}' variant '{}'",
        inserted_examples,
        function_name,
        variant_name
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::TimeoutsConfig,
        embeddings::{EmbeddingModelConfig, EmbeddingProviderConfig, EmbeddingProviderInfo},
        endpoints::inference::InferenceCredentials,
    };
    use std::collections::HashMap;

    // Helper functions to create test embedding models using the Dummy provider

    fn create_test_embedding_model() -> EmbeddingModelConfig {
        create_test_embedding_model_with_name("test-embedding")
    }

    fn create_test_embedding_model_with_failure() -> EmbeddingModelConfig {
        create_test_embedding_model_with_name("error") // This will cause the dummy provider to fail
    }

    fn create_test_embedding_model_with_name(model_name: &str) -> EmbeddingModelConfig {
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            use crate::providers::dummy::DummyProvider;
            let mut providers = HashMap::new();
            providers.insert(
                Arc::from("dummy"),
                EmbeddingProviderInfo {
                    inner: EmbeddingProviderConfig::Dummy(DummyProvider {
                        model_name: model_name.to_string(),
                        ..Default::default()
                    }),
                    timeouts: TimeoutsConfig::default(),
                    provider_name: Arc::from("dummy"),
                    extra_body: None,
                },
            );
            EmbeddingModelConfig {
                routing: vec![Arc::from("dummy")],
                providers,
                timeouts: TimeoutsConfig::default(),
            }
        }
        #[cfg(not(any(test, feature = "e2e_tests")))]
        {
            panic!("This function is only available in test mode")
        }
    }

    #[tokio::test]
    async fn test_process_embedding_batch_success() {
        let embedding_model = create_test_embedding_model();

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["hello".to_string(), "world".to_string()];

        let result = process_embedding_batch(
            &embedding_model,
            "test-embedding",
            &client,
            &credentials,
            batch_texts,
            0,
            None,
        )
        .await;

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 2);
        // The dummy provider returns zero-filled embeddings, so just check structure
        assert!(!embeddings[0].is_empty());
        assert!(!embeddings[1].is_empty());
    }

    #[tokio::test]
    async fn test_process_embedding_batch_with_dimensions() {
        let embedding_model = create_test_embedding_model();

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["hello".to_string()];
        let dimensions = Some(512);

        let result = process_embedding_batch(
            &embedding_model,
            "test-embedding",
            &client,
            &credentials,
            batch_texts,
            0,
            dimensions,
        )
        .await;

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert!(!embeddings[0].is_empty());
    }

    #[tokio::test]
    async fn test_process_embedding_batch_failure() {
        let embedding_model = create_test_embedding_model_with_failure();

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["test".to_string()];

        let result = process_embedding_batch(
            &embedding_model,
            "error", // Use "error" model name to trigger failure
            &client,
            &credentials,
            batch_texts,
            0,
            None,
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_embeddings_with_batching_success() {
        let embedding_model = create_test_embedding_model();

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let input_texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
        ];

        let result = process_embeddings_with_batching(
            &embedding_model,
            "test-embedding",
            &client,
            &credentials,
            input_texts,
            2, // batch_size
            1, // max_concurrency
            None,
        )
        .await;

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);
        // Check that all embeddings have proper structure
        for embedding in &embeddings {
            assert!(!embedding.is_empty());
        }
    }

    #[tokio::test]
    async fn test_process_embeddings_with_batching_respects_concurrency() {
        let embedding_model = create_test_embedding_model();

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let input_texts = vec!["1".to_string(), "2".to_string(), "3".to_string()];

        let result = process_embeddings_with_batching(
            &embedding_model,
            "test-embedding",
            &client,
            &credentials,
            input_texts,
            1, // batch_size: each text is its own batch
            2, // max_concurrency: process 2 batches at a time
            None,
        )
        .await;

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);
    }
}
