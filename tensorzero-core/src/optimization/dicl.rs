#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use url::Url;

use crate::{
    clickhouse::{ClickHouseConnectionInfo, ExternalDataInfo},
    config_parser::ProviderTypesConfig,
    embeddings::{
        EmbeddingEncodingFormat, EmbeddingInput, EmbeddingProvider, EmbeddingProviderInfo,
        EmbeddingRequest, UninitializedEmbeddingProviderConfig,
    },
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    model::{build_creds_caching_default, CredentialLocation},
    optimization::{JobHandle, OptimizationJobInfo, Optimizer, OptimizerOutput},
    providers::openai::{
        default_api_key_location, OpenAICredentials, DEFAULT_CREDENTIALS, PROVIDER_TYPE,
    },
    stored_inference::RenderedSample,
    variant::{dicl::DiclConfig, RetryConfig, VariantConfig},
};
use futures::future::try_join_all;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
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
    pub provider: String,
    pub embedding_model: String,
    pub variant_name: String,
    pub function_name: String,
    pub dimensions: Option<u32>,
    pub batch_size: usize,
    pub max_concurrency: usize,
    pub retries: RetryConfig,
    pub k: usize,
    pub model: String,
    #[serde(skip)]
    pub credentials: OpenAICredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
    pub api_base: Option<Url>,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "DiclOptimizationConfig"))]
pub struct UninitializedDiclOptimizationConfig {
    pub provider: String,
    pub embedding_model: String,
    pub variant_name: String,
    pub function_name: String,
    pub dimensions: Option<u32>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    #[serde(default)]
    pub retries: RetryConfig,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default = "default_model")]
    pub model: String,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocation>,
    pub api_base: Option<Url>,
}

impl Default for UninitializedDiclOptimizationConfig {
    fn default() -> Self {
        Self {
            provider: String::new(),
            embedding_model: String::new(),
            variant_name: String::new(),
            function_name: String::new(),
            dimensions: None,
            batch_size: default_batch_size(),
            max_concurrency: default_max_concurrency(),
            retries: RetryConfig::default(),
            k: default_k(),
            model: default_model(),
            credentials: None,
            api_base: None,
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
    #[pyo3(signature = (*, provider, embedding_model, variant_name, function_name, dimensions=None, batch_size=None, max_concurrency=None, k=None, model=None, credentials=None, api_base=None))]
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        provider: String,
        embedding_model: String,
        variant_name: String,
        function_name: String,
        dimensions: Option<u32>,
        batch_size: Option<usize>,
        max_concurrency: Option<usize>,
        k: Option<usize>,
        model: Option<String>,
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
            provider,
            embedding_model,
            variant_name,
            function_name,
            dimensions,
            batch_size: batch_size.unwrap_or_else(default_batch_size),
            max_concurrency: max_concurrency.unwrap_or_else(default_max_concurrency),
            retries: RetryConfig::default(),
            k: k.unwrap_or_else(default_k),
            model: model.unwrap_or_else(default_model),
            credentials,
            api_base,
        })
    }

    /// Initialize the DiclOptimizationConfig. All parameters are optional except for `embedding_model`.
    ///
    /// :param provider: The provider of the embedding model.
    /// :param embedding_model: The embedding model to use.
    /// :param variant_name: The name to be used for the DICL variant.
    /// :param function_name: The name of the function to optimize.
    /// :param dimensions: The dimensions of the embeddings. If None, uses the model's default.
    /// :param batch_size: The batch size to use for getting embeddings.
    /// :param max_concurrency: The maximum concurrency to use for getting embeddings.
    /// :param k: The number of nearest neighbors to use for the DICL variant.
    /// :param model: The model to use for the DICL variant.
    /// :param credentials: The credentials to use for embedding. This should be a string like "env::OPENAI_API_KEY". See docs for more details.
    /// :param api_base: The base URL to use for embedding. This is primarily used for testing.
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, provider, embedding_model, variant_name, function_name, dimensions=None, batch_size=None, max_concurrency=None, k=None, model=None, credentials=None, api_base=None))]
    fn __init__(
        this: Py<Self>,
        provider: String,
        embedding_model: String,
        variant_name: String,
        function_name: String,
        dimensions: Option<u32>,
        batch_size: Option<usize>,
        max_concurrency: Option<usize>,
        k: Option<usize>,
        model: Option<String>,
        credentials: Option<String>,
        api_base: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedDiclOptimizationConfig {
    pub fn load(self) -> Result<DiclOptimizationConfig, Error> {
        Ok(DiclOptimizationConfig {
            provider: self.provider,
            embedding_model: self.embedding_model,
            variant_name: self.variant_name,
            function_name: self.function_name,
            dimensions: self.dimensions,
            batch_size: self.batch_size,
            max_concurrency: self.max_concurrency,
            retries: self.retries,
            k: self.k,
            model: self.model,
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
    // TODO: Remove job_url. Currently causes error in UI if not provided because DICL is being loaded as a fine tuning option.
    pub job_url: Url,
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
    ) -> Result<Self::Handle, Error> {
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

        // Check if we have examples to process
        if train_examples.is_empty() {
            tracing::warn!(
                "⚠️  No training examples provided for DICL optimization - creating empty variant"
            );

            // Create a job handle indicating immediate success with empty processing
            let job_handle = DiclOptimizationJobHandle {
                job_url: Url::parse("https://tensorzero.com/dicl").map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to parse job URL: {e}"),
                    })
                })?, // TODO: Remove job_url once UI error is addressed
                embedding_model: self.embedding_model.clone(),
                k: self.k,
                model: self.model.clone(),
            };

            tracing::info!(
                "🎉 DICL optimization completed (no examples)!\n\
                📈 Summary:\n\
                ├─ Function: '{}'\n\
                ├─ Variant: '{}'\n\
                ├─ Examples processed: 0\n\
                └─ Status: Empty variant created",
                self.function_name,
                self.variant_name
            );

            return Ok(job_handle);
        }

        tracing::info!(
            "🚀 Starting DICL optimization for function '{}' variant '{}' with {} examples",
            self.function_name,
            self.variant_name,
            train_examples.len()
        );

        // Convert RenderedSample inputs to strings for embedding (only messages, not system)
        let input_texts: Vec<String> = train_examples
            .iter()
            .map(|sample| {
                serde_json::to_string(&sample.input.messages)
                    .unwrap_or_else(|_| format!("{:?}", sample.input.messages))
            })
            .collect();

        // Initialize the embedding provider
        let provider_config_str = format!(
            r#"
            type = "{}"
            model_name = "{}"
            "#,
            self.provider, self.embedding_model
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

        // Process embeddings with batching and concurrency control
        let all_embeddings = process_embeddings_with_batching(
            &provider_config,
            client,
            credentials,
            input_texts,
            self.batch_size,
            self.max_concurrency,
            &self.retries,
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
            job_url: Url::parse("https://tensorzero.com/dicl").map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse job URL: {e}"),
                })
            })?, // TODO: Remove job_url once UI issue is resolved.
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
            output: OptimizerOutput::Variant {
                variant: Box::new(VariantConfig::Dicl(DiclConfig {
                    weight: None,
                    embedding_model: Arc::from(self.embedding_model.as_str()),
                    k: self.k as u32,
                    model: Arc::from(self.model.as_str()),
                    system_instructions: String::new(),
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
                })),
            },
        })
    }
}

/// Processes a batch of input texts to get embeddings with retry logic
async fn process_embedding_batch(
    provider_config: &EmbeddingProviderInfo,
    client: &reqwest::Client,
    credentials: &InferenceCredentials,
    batch_texts: Vec<String>,
    batch_index: usize,
    retry_config: &RetryConfig,
    dimensions: Option<u32>,
) -> Result<Vec<Vec<f64>>, Error> {
    let max_retries = retry_config.num_retries;
    let mut retries = 0;

    loop {
        let embedding_request = EmbeddingRequest {
            input: EmbeddingInput::Batch(batch_texts.clone()),
            dimensions,
            encoding_format: EmbeddingEncodingFormat::Float,
        };

        match provider_config
            .embed(&embedding_request, client, credentials)
            .await
        {
            Ok(response) => {
                tracing::debug!("Successfully processed embedding batch {}", batch_index);
                // Convert embeddings from Vec<Embedding> to Vec<Vec<f64>>
                let embeddings: Result<Vec<Vec<f64>>, Error> = response
                    .embeddings
                    .into_iter()
                    .map(|embedding| match embedding {
                        crate::embeddings::Embedding::Float(vec_f32) => {
                            Ok(vec_f32.into_iter().map(|f| f as f64).collect())
                        }
                        crate::embeddings::Embedding::Base64(_) => {
                            Err(Error::new(ErrorDetails::Inference {
                                message: "Base64 embeddings not supported for DICL optimization"
                                    .to_string(),
                            }))
                        }
                    })
                    .collect();
                return embeddings;
            }
            Err(e) if retries < max_retries => {
                retries += 1;
                let delay_secs = (retries as f32).min(retry_config.max_delay_s);
                tracing::warn!(
                    "Embedding batch {} failed (attempt {}/{}): {}. Retrying in {:.1}s...",
                    batch_index,
                    retries,
                    max_retries + 1,
                    e,
                    delay_secs
                );
                sleep(Duration::from_secs_f32(delay_secs)).await;
            }
            Err(e) => {
                tracing::error!(
                    "Embedding batch {} failed after {} attempts: {}",
                    batch_index,
                    max_retries + 1,
                    e
                );
                return Err(e);
            }
        }
    }
}

/// Processes all embedding batches with concurrency control
#[expect(clippy::too_many_arguments)]
async fn process_embeddings_with_batching(
    provider_config: &EmbeddingProviderInfo,
    client: &reqwest::Client,
    credentials: &InferenceCredentials,
    input_texts: Vec<String>,
    batch_size: usize,
    max_concurrency: usize,
    retry_config: &RetryConfig,
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

    // Process batches with controlled concurrency
    let mut all_embeddings = Vec::new();
    let total_batches = batches.len();
    let mut processed_batches = 0;

    for batch_chunk in batches.chunks(max_concurrency) {
        let batch_futures: Vec<_> = batch_chunk
            .iter()
            .enumerate()
            .map(|(i, batch)| {
                let batch_index = processed_batches + i;
                process_embedding_batch(
                    provider_config,
                    client,
                    credentials,
                    batch.clone(),
                    batch_index,
                    retry_config,
                    dimensions,
                )
            })
            .collect();

        let batch_results = try_join_all(batch_futures).await?;
        for batch_embeddings in batch_results {
            all_embeddings.extend(batch_embeddings);
        }

        processed_batches += batch_chunk.len();
        let progress_pct = (processed_batches as f64 / total_batches as f64 * 100.0).round() as u32;
        tracing::info!(
            "📊 Embedding progress: {}/{} batches completed ({}%) - {} embeddings generated",
            processed_batches,
            total_batches,
            progress_pct,
            all_embeddings.len()
        );
    }

    tracing::info!(
        "✅ Embedding processing complete: {} embeddings generated from {} input texts",
        all_embeddings.len(),
        input_texts.len()
    );

    Ok(all_embeddings)
}

/// Inserts DICL examples into ClickHouse using ExternalDataInfo pattern
async fn insert_dicl_examples_with_batching(
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
                    .output
                    .as_ref()
                    .and_then(|outputs| outputs.first())
                    .map(|output| serde_json::to_string(output).unwrap_or_default())
                    .unwrap_or_default();

                let input_text = serde_json::to_string(&sample.input.messages)
                    .unwrap_or_else(|_| format!("{:?}", sample.input.messages));

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
        clickhouse::test_helpers::get_clickhouse,
        config_parser::TimeoutsConfig,
        embeddings::{EmbeddingProviderConfig, EmbeddingProviderInfo},
        endpoints::inference::InferenceCredentials,
        inference::types::{
            ContentBlock, ContentBlockChatOutput, ModelInput, RequestMessage, Role, Text,
        },
        variant::RetryConfig,
    };
    use std::collections::HashMap;
    use uuid::Uuid;

    // Helper functions to create test providers using the Dummy provider

    fn create_test_embedding_provider(_embeddings: Vec<Vec<f32>>) -> EmbeddingProviderInfo {
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            use crate::providers::dummy::DummyProvider;
            EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::Dummy(DummyProvider {
                    model_name: "test-embedding".into(),
                    ..Default::default()
                }),
                timeouts: TimeoutsConfig::default(),
                provider_name: Arc::from("dummy"),
            }
        }
        #[cfg(not(any(test, feature = "e2e_tests")))]
        {
            panic!("This function is only available in test mode")
        }
    }

    fn create_test_embedding_provider_with_failure(
        _embeddings: Vec<Vec<f32>>,
        _fail_attempt: usize,
    ) -> EmbeddingProviderInfo {
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            use crate::providers::dummy::DummyProvider;
            EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::Dummy(DummyProvider {
                    model_name: "error".into(), // This will cause the dummy provider to fail
                    ..Default::default()
                }),
                timeouts: TimeoutsConfig::default(),
                provider_name: Arc::from("dummy"),
            }
        }
        #[cfg(not(any(test, feature = "e2e_tests")))]
        {
            panic!("This function is only available in test mode")
        }
    }

    fn create_test_rendered_sample(input: &str, output: &str) -> RenderedSample {
        RenderedSample {
            function_name: "test_function".to_string(),
            input: ModelInput {
                system: None,
                messages: vec![RequestMessage {
                    role: Role::User,
                    content: vec![ContentBlock::Text(Text {
                        text: input.to_string(),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: output.to_string(),
            })]),
            episode_id: Some(Uuid::now_v7()),
            inference_id: Some(Uuid::now_v7()),
            tool_params: None,
            output_schema: None,
            dispreferred_outputs: vec![],
            tags: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_process_embedding_batch_success() {
        let provider =
            create_test_embedding_provider(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["hello".to_string(), "world".to_string()];
        let retry_config = RetryConfig::default();

        let result = process_embedding_batch(
            &provider,
            &client,
            &credentials,
            batch_texts,
            0,
            &retry_config,
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
        let provider = create_test_embedding_provider(vec![vec![1.0, 2.0, 3.0]]);

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["hello".to_string()];
        let retry_config = RetryConfig::default();
        let dimensions = Some(512);

        let result = process_embedding_batch(
            &provider,
            &client,
            &credentials,
            batch_texts,
            0,
            &retry_config,
            dimensions,
        )
        .await;

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert!(!embeddings[0].is_empty());
    }

    #[tokio::test]
    async fn test_process_embedding_batch_retry_exhausted() {
        let provider = create_test_embedding_provider_with_failure(vec![], 0);

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["test".to_string()];
        let retry_config = RetryConfig {
            num_retries: 0, // No retries, should fail immediately
            max_delay_s: 1.0,
        };

        let result = process_embedding_batch(
            &provider,
            &client,
            &credentials,
            batch_texts,
            0,
            &retry_config,
            None,
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_embeddings_with_batching_success() {
        let provider = create_test_embedding_provider(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
            vec![9.0, 10.0],
        ]);

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let input_texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
        ];
        let retry_config = RetryConfig::default();

        let result = process_embeddings_with_batching(
            &provider,
            &client,
            &credentials,
            input_texts,
            2, // batch_size
            1, // max_concurrency
            &retry_config,
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
        let provider = create_test_embedding_provider(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ]);

        let client = reqwest::Client::new();
        let credentials = InferenceCredentials::default();
        let input_texts = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let retry_config = RetryConfig::default();

        let result = process_embeddings_with_batching(
            &provider,
            &client,
            &credentials,
            input_texts,
            1, // batch_size: each text is its own batch
            2, // max_concurrency: process 2 batches at a time
            &retry_config,
            None,
        )
        .await;

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);
    }

    #[tokio::test]
    async fn test_insert_dicl_examples_with_batching_success() {
        let clickhouse = get_clickhouse().await;

        let samples = vec![
            create_test_rendered_sample("test input 1", "test output 1"),
            create_test_rendered_sample("test input 2", "test output 2"),
        ];
        let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> =
            samples.into_iter().zip(embeddings.into_iter()).collect();

        let result = insert_dicl_examples_with_batching(
            &clickhouse,
            examples_with_embeddings,
            "test_function_unit",
            "test_variant_unit",
            10, // batch_size
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_insert_dicl_examples_batching_logic() {
        let clickhouse = get_clickhouse().await;

        // Create 5 examples with batch size of 2, should create 3 batches (2+2+1)
        let samples = (0..5)
            .map(|i| {
                create_test_rendered_sample(&format!("test input {i}"), &format!("test output {i}"))
            })
            .collect::<Vec<_>>();
        let embeddings = (0..5)
            .map(|i| vec![i as f64, (i + 1) as f64, (i + 2) as f64])
            .collect::<Vec<_>>();
        let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> =
            samples.into_iter().zip(embeddings.into_iter()).collect();

        let result = insert_dicl_examples_with_batching(
            &clickhouse,
            examples_with_embeddings,
            "test_function_batch",
            "test_variant_batch",
            2, // batch_size
        )
        .await;

        assert!(result.is_ok());

        // Verify that data was actually inserted by querying the table
        let count_query = "SELECT count() FROM DynamicInContextLearningExample WHERE function_name = 'test_function_batch' AND variant_name = 'test_variant_batch' FORMAT JSONEachRow";
        let result = clickhouse
            .run_query_synchronous_no_params(count_query.to_string())
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_insert_dicl_examples_with_empty_batch() {
        let clickhouse = get_clickhouse().await;

        let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> = vec![];

        let result = insert_dicl_examples_with_batching(
            &clickhouse,
            examples_with_embeddings,
            "test_function_empty",
            "test_variant_empty",
            10,
        )
        .await;

        // Should succeed with empty batch
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_insert_dicl_examples_json_serialization() {
        let clickhouse = get_clickhouse().await;

        // Test with special characters that need proper JSON encoding
        let sample = create_test_rendered_sample(
            "input with \"quotes\" and \n newlines",
            "output with special chars: àáâã",
        );
        let embedding = vec![1.0, 2.0, 3.0];
        let examples_with_embeddings = vec![(sample, embedding)];

        let result = insert_dicl_examples_with_batching(
            &clickhouse,
            examples_with_embeddings,
            "test_function_json",
            "test_variant_json",
            1,
        )
        .await;

        assert!(result.is_ok());
    }
}
