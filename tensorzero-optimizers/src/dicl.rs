use futures::future::try_join_all;
use serde_json::json;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Semaphore;
use uuid::Uuid;

use tensorzero_core::{
    cache::CacheOptions,
    config::{Config, UninitializedVariantConfig, provider_types::ProviderTypesConfig},
    db::{
        clickhouse::{
            ClickHouseConnectionInfo, ExternalDataInfo, clickhouse_client::ClickHouseClientType,
        },
        postgres::PostgresConnectionInfo,
    },
    embeddings::{Embedding, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest},
    endpoints::inference::{InferenceClients, InferenceCredentials},
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    function::FunctionConfig,
    http::TensorzeroHttpClient,
    inference::types::StoredInputMessageContent,
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        OptimizationJobInfo, OptimizerOutput,
        dicl::{DEPRECATED_DEFAULT_MODEL, DiclOptimizationConfig, DiclOptimizationJobHandle},
    },
    rate_limiting::{RateLimitingManager, ScopeInfo},
    stored_inference::RenderedSample,
    variant::dicl::UninitializedDiclConfig,
};

use crate::{JobHandle, Optimizer};

impl Optimizer for DiclOptimizationConfig {
    type Handle = DiclOptimizationJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        // Warn if using deprecated default model
        if self.model.as_ref() == DEPRECATED_DEFAULT_MODEL {
            tracing::warn!(
                "DICL optimization is using the deprecated default model `{}`. \
                 Please specify the `model` field explicitly. \
                 This field will be required in a future release. (#5616)",
                DEPRECATED_DEFAULT_MODEL
            );
        }

        // Validate training examples
        validate_train_examples(&train_examples)?;

        // Warn if val_examples is provided (not used in DICL)
        if val_examples.is_some() {
            tracing::warn!("val_examples provided for DICL optimization but will be ignored");
        }

        // Check if ClickHouse is available (required for DICL)
        if clickhouse_connection_info.client_type() == ClickHouseClientType::Disabled {
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

        // 2. Check that the function does not have tools configured (DICL doesn't support tools)
        validate_function_config(&self.function_name, function_config)?;

        // 3. Check if DICL examples already exist in the database for this variant (unless appending is enabled)
        if !self.append_to_existing_variants
            && dicl_examples_exist(
                clickhouse_connection_info,
                &self.function_name,
                &self.variant_name,
            )
            .await?
        {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "variant '{}' already has DICL examples in the database for function '{}' - set append_to_existing_variants=true to append to existing variants",
                    self.variant_name, self.function_name
                ),
            }));
        }

        tracing::info!(
            "Starting DICL optimization for function '{}' variant '{}' with {} examples",
            self.function_name,
            self.variant_name,
            train_examples.len()
        );

        // Convert RenderedSample inputs to strings for embedding using stored_input
        // IMPORTANT: we use `sample.stored_input` which has NOT templated the prompts
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
            &config,
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
        let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> =
            train_examples.into_iter().zip(all_embeddings).collect();

        let num_examples = examples_with_embeddings.len();

        tracing::info!("Phase transition: Embedding → ClickHouse storage");

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
            embedding_model: self.embedding_model.to_string(),
            k: self.k,
            model: self.model.to_string(),
        };

        tracing::info!(
            "DICL optimization completed successfully!\n\
            Summary:\n\
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
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        _default_credentials: &ProviderTypeDefaultCredentials,
        _provider_types: &ProviderTypesConfig,
    ) -> Result<OptimizationJobInfo, Error> {
        // DICL optimization is synchronous, so it's always complete once launched
        let _ = (client, credentials);

        // DICL produces a variant configuration that references the stored examples
        // Return a DICL variant with the configuration from the optimization job
        Ok(OptimizationJobInfo::Completed {
            output: OptimizerOutput::Variant(Box::new(UninitializedVariantConfig::Dicl(
                UninitializedDiclConfig {
                    embedding_model: self.embedding_model.to_string(),
                    k: self.k,
                    model: self.model.to_string(),
                    ..Default::default()
                },
            ))),
        })
    }
}

/// Validates function config for DICL optimization
/// Checks that the function does not have tools configured (DICL doesn't support tools)
fn validate_function_config(
    function_name: &str,
    function_config: &FunctionConfig,
) -> Result<(), Error> {
    match function_config {
        FunctionConfig::Chat(chat_config) => {
            if !chat_config.tools.is_empty() {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "DICL optimization does not support functions with tools. Function '{}' has {} tools configured. Please use a function without tools for DICL optimization.",
                        function_name,
                        chat_config.tools.len()
                    ),
                }));
            }
        }
        FunctionConfig::Json(json_config) => {
            // JSON functions should have exactly one implicit tool for schema validation
            let tools_count = json_config
                .json_mode_tool_call_config
                .tools_available()?
                .count();
            if tools_count != 1 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!(
                        "DICL optimization expected JSON function '{function_name}' to have exactly 1 implicit tool, but found {tools_count}. This indicates a configuration issue."
                    ),
                }));
            }
        }
    }
    Ok(())
}

/// Validates training examples for DICL optimization
/// Checks for emptiness and presence of tool calls which are not supported
fn validate_train_examples(train_examples: &[RenderedSample]) -> Result<(), Error> {
    // Check if we have examples to process
    if train_examples.is_empty() {
        return Err(Error::new(ErrorDetails::Config {
            message: "DICL optimization requires at least one training example".to_string(),
        }));
    }

    // Check for tool calls in training examples
    for (i, example) in train_examples.iter().enumerate() {
        // Check that each example has an output
        if example.stored_output.is_none() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "DICL optimization requires all training examples to have outputs. Training example {} is missing an output.",
                    i + 1
                ),
            }));
        }
        // Check if tools are available
        let has_additional_tools = example
            .tool_params
            .additional_tools
            .as_ref()
            .map(|tools| !tools.is_empty())
            .unwrap_or(false);
        let has_allowed_tools = example
            .tool_params
            .allowed_tools
            .as_ref()
            .map(|tools| !tools.is_empty())
            .unwrap_or(false);

        if has_additional_tools || has_allowed_tools {
            let num_tools = example
                .tool_params
                .additional_tools
                .as_ref()
                .map(std::vec::Vec::len)
                .unwrap_or(0);
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!(
                    "DICL optimization does not support tool calls. Training example {} contains {} available tools.",
                    i + 1,
                    num_tools
                ),
            }));
        }

        // Check stored_input messages for ToolCall or ToolResult content
        for message in &example.stored_input.messages {
            for content in &message.content {
                match content {
                    StoredInputMessageContent::ToolCall(_) => {
                        return Err(Error::new(ErrorDetails::InvalidRequest {
                            message: format!(
                                "DICL optimization does not support tool calls. Training example {} contains a tool call in message content.",
                                i + 1
                            ),
                        }));
                    }
                    StoredInputMessageContent::ToolResult(_) => {
                        return Err(Error::new(ErrorDetails::InvalidRequest {
                            message: format!(
                                "DICL optimization does not support tool calls. Training example {} contains a tool result in message content.",
                                i + 1
                            ),
                        }));
                    }
                    _ => {} // Other content types are fine
                }
            }
        }
    }

    Ok(())
}

/// Processes a batch of input texts to get embeddings
async fn process_embedding_batch(
    config: &Config,
    model_name: &str,
    client: &TensorzeroHttpClient,
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

    let embedding_model_config = config
        .embedding_models
        .get(model_name, config.gateway.relay.as_ref())
        .await?
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!("embedding model '{model_name}' not found in configuration",),
            })
        })?;

    let tags = Arc::new(HashMap::default());

    // Create InferenceClients context for the embedding model
    let deferred_tasks = tokio_util::task::TaskTracker::new();
    let rate_limiting_config = Arc::new(config.rate_limiting.clone());
    let postgres_connection_info = PostgresConnectionInfo::Disabled;
    let rate_limiting_manager = Arc::new(RateLimitingManager::new(
        rate_limiting_config.clone(),
        Arc::new(postgres_connection_info.clone()),
    ));
    let clients = InferenceClients {
        http_client: client.clone(),
        credentials: Arc::new(credentials.clone()),
        clickhouse_connection_info: ClickHouseConnectionInfo::new_disabled(),
        postgres_connection_info,
        cache_options: CacheOptions::default(),
        tags: tags.clone(),
        rate_limiting_manager,
        // We don't currently perform any OTLP export in optimization workflows
        otlp_config: Default::default(),
        deferred_tasks: deferred_tasks.clone(),
        // We don't currently use API keys for optimization workflows
        scope_info: ScopeInfo::new(tags.clone(), None),
        relay: None,
        include_raw_usage: false,
        include_raw_response: false,
    };

    let response = embedding_model_config
        .embed(&embedding_request, model_name, &clients)
        .await?;

    // We're running an optimization, so we don't really have a gateway to shutdown
    // Let's just wait for the deferred tasks to finish immediately
    deferred_tasks.close();
    deferred_tasks.wait().await;

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
    config: &Config,
    model_name: &str,
    client: &TensorzeroHttpClient,
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
        "Starting embedding processing: {} input texts → {} batches (size: {}, max concurrent: {})",
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
                    config,
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
                        "Embedding progress: {}/{} batches completed ({}%)",
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
        "Embedding processing complete: {} embeddings generated from {} input texts",
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
        "Starting ClickHouse insertion: {} examples → {} batches (size: {})",
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
                    .ok_or_else(|| Error::new(ErrorDetails::InternalError {
                        message: format!("Training example is missing output after validation. {IMPOSSIBLE_ERROR_MESSAGE}")
                    }))
                    .and_then(|stored_output| serde_json::to_string(stored_output).map_err(Into::into))?;

                let input_text = serde_json::to_string(&sample.stored_input)?;

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
            "ClickHouse insertion progress: {}/{} batches completed ({}%) - {}/{} examples inserted (wrote {} rows)",
            batch_index + 1,
            total_batches,
            progress_pct,
            inserted_examples,
            total_examples,
            result.metadata.written_rows
        );
    }

    tracing::info!(
        "ClickHouse insertion complete: {} examples successfully stored for function '{}' variant '{}'",
        inserted_examples,
        function_name,
        variant_name
    );

    Ok(())
}

/// Checks if DICL examples exist in ClickHouse for a given function and variant
pub async fn dicl_examples_exist(
    clickhouse: &ClickHouseConnectionInfo,
    function_name: &str,
    variant_name: &str,
) -> Result<bool, Error> {
    let query = r"
        SELECT 1
        FROM DynamicInContextLearningExample
        WHERE function_name = {function_name:String}
        AND variant_name = {variant_name:String}
        LIMIT 1
    ";

    let params = HashMap::from([
        ("function_name", function_name),
        ("variant_name", variant_name),
    ]);

    let result = clickhouse
        .run_query_synchronous(query.to_string(), &params)
        .await?;

    // If the query returns "1", examples exist; if empty, they don't
    Ok(result.response.trim() == "1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use tensorzero_core::tool::Tool;
    use uuid::Uuid;

    use tensorzero_core::{
        config::{Config, SchemaData, provider_types::ProviderTypesConfig},
        embeddings::{
            EmbeddingModelConfig, EmbeddingModelTable, EmbeddingProviderConfig,
            EmbeddingProviderInfo,
        },
        endpoints::inference::InferenceCredentials,
        experimentation::ExperimentationConfig,
        function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson},
        http::TensorzeroHttpClient,
        inference::types::{
            ContentBlockChatOutput, ModelInput, ResolvedContentBlock, ResolvedRequestMessage, Role,
            StoredInput, StoredInputMessage, StoredInputMessageContent, System, Text,
        },
        jsonschema_util::JSONSchema,
        model_table::ProviderTypeDefaultCredentials,
        providers::dummy::DummyProvider,
        stored_inference::{RenderedSample, StoredOutput},
        tool::{
            DynamicToolParams, FunctionTool, ToolCall, ToolCallConfig, ToolChoice, ToolResult,
            create_json_mode_tool_call_config,
        },
    };

    // Helper functions to create test embedding models using the Dummy provider

    fn create_test_embedding_model_config() -> Config {
        create_test_embedding_model_with_name("test-embedding")
    }

    fn create_test_embedding_model_with_failure_config() -> Config {
        create_test_embedding_model_with_name("error") // This will cause the dummy provider to fail
    }

    fn create_test_embedding_model_with_name(model_name: &str) -> Config {
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            let mut providers = HashMap::new();
            providers.insert(
                Arc::from("dummy"),
                EmbeddingProviderInfo {
                    inner: EmbeddingProviderConfig::Dummy(DummyProvider {
                        model_name: model_name.to_string(),
                        ..Default::default()
                    }),
                    timeout_ms: None,
                    provider_name: Arc::from("dummy"),
                    extra_body: None,
                    extra_headers: None,
                },
            );
            let embedding_model_config = EmbeddingModelConfig {
                routing: vec![Arc::from("dummy")],
                providers,
                timeout_ms: None,
            };
            let provider_types = ProviderTypesConfig::default();
            Config {
                embedding_models: Arc::new(
                    EmbeddingModelTable::new(
                        HashMap::from([(Arc::from(model_name), embedding_model_config)]),
                        ProviderTypeDefaultCredentials::new(&provider_types).into(),
                        chrono::Duration::seconds(120),
                    )
                    .unwrap(),
                ),
                ..Default::default()
            }
        }
        #[cfg(not(any(test, feature = "e2e_tests")))]
        {
            panic!("This function is only available in test mode")
        }
    }

    #[tokio::test]
    async fn test_process_embedding_batch_success() {
        let config = create_test_embedding_model_config();

        let client = TensorzeroHttpClient::new_testing().unwrap();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["hello".to_string(), "world".to_string()];

        let result = process_embedding_batch(
            &config,
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
        let config = create_test_embedding_model_config();

        let client = TensorzeroHttpClient::new_testing().unwrap();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["hello".to_string()];
        let dimensions = Some(512);

        let result = process_embedding_batch(
            &config,
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
        let config = create_test_embedding_model_with_failure_config();

        let client = TensorzeroHttpClient::new_testing().unwrap();
        let credentials = InferenceCredentials::default();
        let batch_texts = vec!["test".to_string()];

        let result = process_embedding_batch(
            &config,
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
        let config = create_test_embedding_model_config();

        let client = TensorzeroHttpClient::new_testing().unwrap();
        let credentials = InferenceCredentials::default();
        let input_texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
        ];

        let result = process_embeddings_with_batching(
            &config,
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
        let config = create_test_embedding_model_config();

        let client = TensorzeroHttpClient::new_testing().unwrap();
        let credentials = InferenceCredentials::default();
        let input_texts = vec!["1".to_string(), "2".to_string(), "3".to_string()];

        let result = process_embeddings_with_batching(
            &config,
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

    // Helper function to create a basic RenderedSample for testing
    fn create_test_rendered_sample() -> RenderedSample {
        RenderedSample {
            function_name: "test_function".to_string(),
            input: ModelInput {
                system: Some("Test system".to_string()),
                messages: vec![ResolvedRequestMessage {
                    role: Role::User,
                    content: vec![ResolvedContentBlock::Text(Text {
                        text: "Test message".to_string(),
                    })],
                }],
            },
            stored_input: StoredInput {
                system: Some(System::Text("Test system".to_string())),
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "Test message".to_string(),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Test output".to_string(),
            })]),
            stored_output: Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Text(
                Text {
                    text: "Test output".to_string(),
                },
            )])),
            episode_id: Some(Uuid::now_v7()),
            inference_id: Some(Uuid::now_v7()),
            tool_params: DynamicToolParams::default(),
            output_schema: None,
            dispreferred_outputs: vec![],
            tags: HashMap::new(),
        }
    }

    fn create_test_rendered_sample_with_tools(tools: Vec<FunctionTool>) -> RenderedSample {
        let mut sample = create_test_rendered_sample();
        sample.tool_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: Some(tools.into_iter().map(Tool::Function).collect()),
            tool_choice: Some(ToolChoice::Auto),
            parallel_tool_calls: Some(true),
            provider_tools: vec![],
        };
        sample
    }

    fn create_test_rendered_sample_with_tool_content() -> RenderedSample {
        let mut sample = create_test_rendered_sample();

        // Add a message with tool call content
        sample.stored_input.messages.push(StoredInputMessage {
            role: Role::Assistant,
            content: vec![StoredInputMessageContent::ToolCall(ToolCall {
                id: "test_call".to_string(),
                name: "test_tool".to_string(),
                arguments: serde_json::json!({"arg": "value"}).to_string(),
            })],
        });

        // Also add empty tool params to make it realistic
        sample.tool_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: None,
            tool_choice: Some(ToolChoice::None),
            parallel_tool_calls: Some(false),
            provider_tools: vec![],
        };

        sample
    }

    fn create_test_rendered_sample_with_tool_result() -> RenderedSample {
        let mut sample = create_test_rendered_sample();

        // Add a message with tool result content
        sample.stored_input.messages.push(StoredInputMessage {
            role: Role::User,
            content: vec![StoredInputMessageContent::ToolResult(ToolResult {
                id: "test_call".to_string(),
                name: "test_tool".to_string(),
                result: "Tool result".to_string(),
            })],
        });

        // Also add empty tool params to make it realistic
        sample.tool_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: None,
            tool_choice: Some(ToolChoice::None),
            parallel_tool_calls: Some(false),
            provider_tools: vec![],
        };

        sample
    }

    #[test]
    fn test_validate_train_examples_empty() {
        let result = validate_train_examples(&[]);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires at least one training example")
        );
    }

    #[test]
    fn test_validate_train_examples_valid_no_tools() {
        let sample = create_test_rendered_sample();
        let result = validate_train_examples(&[sample]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_train_examples_valid_empty_tools() {
        let sample = create_test_rendered_sample_with_tools(vec![]);
        let result = validate_train_examples(&[sample]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_train_examples_rejects_tools_available() {
        let tool = FunctionTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            strict: false,
        };

        let sample = create_test_rendered_sample_with_tools(vec![tool]);
        let result = validate_train_examples(&[sample]);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("does not support tool calls"));
        assert!(error.to_string().contains("contains 1 available tools"));
    }

    #[test]
    fn test_validate_train_examples_rejects_tool_call_content() {
        let sample = create_test_rendered_sample_with_tool_content();
        let result = validate_train_examples(&[sample]);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("does not support tool calls"));
        assert!(
            error
                .to_string()
                .contains("contains a tool call in message content")
        );
    }

    #[test]
    fn test_validate_train_examples_rejects_tool_result_content() {
        let sample = create_test_rendered_sample_with_tool_result();
        let result = validate_train_examples(&[sample]);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("does not support tool calls"));
        assert!(
            error
                .to_string()
                .contains("contains a tool result in message content")
        );
    }

    #[test]
    fn test_validate_train_examples_multiple_samples() {
        let valid_sample = create_test_rendered_sample();

        let tool = FunctionTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            strict: false,
        };
        let invalid_sample = create_test_rendered_sample_with_tools(vec![tool]);

        // Should fail on the second sample
        let result = validate_train_examples(&[valid_sample, invalid_sample]);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Training example 2"));
    }

    #[test]
    fn test_validate_train_examples_rejects_missing_output() {
        let mut sample = create_test_rendered_sample();
        sample.stored_output = None;

        let result = validate_train_examples(&[sample]);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("missing an output"));
        assert!(error.to_string().contains("Training example 1"));
    }

    #[test]
    fn test_validate_train_examples_multiple_samples_with_missing_output() {
        let valid_sample = create_test_rendered_sample();
        let mut invalid_sample = create_test_rendered_sample();
        invalid_sample.stored_output = None;

        // Should fail on the second sample
        let result = validate_train_examples(&[valid_sample, invalid_sample]);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Training example 2"));
        assert!(error.to_string().contains("missing an output"));
    }

    fn create_test_chat_function_config_no_tools() -> FunctionConfig {
        FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![], // No tools
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            description: None,

            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        })
    }

    fn create_test_chat_function_config_with_tools() -> FunctionConfig {
        FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec!["test_tool".to_string()],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        })
    }

    fn create_test_json_function_config() -> FunctionConfig {
        let output_schema = JSONSchema::from_value(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"]
        }))
        .unwrap();

        let json_mode_tool_call_config = create_json_mode_tool_call_config(output_schema.clone());

        FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema,
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        })
    }

    fn create_test_json_function_config_invalid_tools() -> FunctionConfig {
        let output_schema = JSONSchema::from_value(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"]
        }))
        .unwrap();

        // Create an invalid config with no tools (should have exactly 1)
        let invalid_tool_call_config = ToolCallConfig::default();

        FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema,
            json_mode_tool_call_config: invalid_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        })
    }

    #[test]
    fn test_validate_function_config_chat_no_tools() {
        let function_config = create_test_chat_function_config_no_tools();
        let result = validate_function_config("test_function", &function_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_function_config_chat_with_tools() {
        let function_config = create_test_chat_function_config_with_tools();
        let result = validate_function_config("test_function", &function_config);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not support functions with tools")
        );
        assert!(error.to_string().contains("test_function"));
        assert!(error.to_string().contains("1 tools configured"));
    }

    #[test]
    fn test_validate_function_config_json_valid() {
        let function_config = create_test_json_function_config();
        let result = validate_function_config("test_json_function", &function_config);
        assert!(result.is_ok(), "Validation error: {result:?}");
    }

    #[test]
    fn test_validate_function_config_json_invalid() {
        let function_config = create_test_json_function_config_invalid_tools();
        let result = validate_function_config("test_json_function", &function_config);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("expected JSON function"));
        assert!(error.to_string().contains("test_json_function"));
        assert!(error.to_string().contains("exactly 1 implicit tool"));
        assert!(error.to_string().contains("found 0"));
    }
}
