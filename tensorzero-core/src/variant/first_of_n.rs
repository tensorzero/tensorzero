//! # First-of-N Variant
//!
//! The first-of-n variant type executes multiple candidate variants concurrently and returns
//! the result from whichever completes successfully first. This optimizes for **latency**
//! rather than quality, making it ideal for scenarios where speed is critical.
//!
//! ## Use Cases
//!
//! - **Latency optimization**: When multiple models can solve the same task but with different
//!   response times, first-of-n ensures you get the fastest available result.
//! - **Provider redundancy**: Race the same model across multiple providers for improved
//!   reliability and faster response times.
//! - **Fast fallback**: Use a fast but potentially lower-quality model alongside a slower
//!   higher-quality model, letting the fast one provide immediate results.
//!
//! ## Configuration Example
//!
//! ```toml
//! [functions.my_function.variants.racing_variant]
//! type = "experimental_first_of_n"
//! timeout_s = 10.0  # Per-candidate timeout in seconds (default: 300s)
//! candidates = ["primary_variant", "fallback_variant", "backup_variant"]
//! weight = 0.5      # Optional: for A/B testing with other variants
//! ```
//!
//! ## Behavior
//!
//! - **Concurrent execution**: All candidates execute simultaneously using Rust's async runtime
//! - **First successful wins**: Returns immediately when any candidate completes successfully
//! - **Cancellation**: Other candidates are cancelled once a winner is found
//! - **Minimal observability**: Only the winning candidate is logged to the ModelInference table
//! - **Error handling**: If all candidates fail or timeout, returns an error
//! - **Independent timeouts**: Each candidate has its own timeout (not cumulative)
//!
//! ## Comparison to Other Variants
//!
//! - **First-of-N**: Returns immediately when any candidate succeeds (optimizes for latency)
//! - **Best-of-N**: Waits for all candidates, uses a judge to select the best (optimizes for quality)
//! - **Mixture-of-N**: Combines outputs from multiple candidates using a fuser model
//!
//! ## Streaming Behavior
//!
//! Since candidates run in parallel and we don't know which will finish first until runtime,
//! streaming is not supported during candidate execution. The first successful non-streaming
//! result is automatically converted to a stream for the client if streaming is requested.
//!
//! ## Implementation Details
//!
//! The implementation uses `FuturesUnordered` to poll all candidate inference futures concurrently.
//! As soon as any future completes successfully, that result is returned. Errors from failed
//! candidates are collected but don't block the return of a successful result.

use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use futures::stream::{FuturesUnordered, StreamExt};
use tokio::time::{timeout, Duration};

use crate::config::{LoadableConfig, PathWithContents};
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels};
use crate::error::ErrorDetails;
use crate::inference::types::batch::StartBatchModelInferenceWithMetadata;
use crate::inference::types::resolved_input::LazyResolvedInput;
use crate::model::ModelTable;
use crate::variant::mixture_of_n::stream_inference_from_non_stream;
use crate::{
    endpoints::inference::InferenceParams,
    error::Error,
    function::FunctionConfig,
    inference::types::{InferenceResult, InferenceResultStream},
    minijinja_util::TemplateConfig,
};

use super::{InferenceConfig, ModelUsedInfo, Variant};

/// Configuration for a first-of-n variant.
///
/// This variant races multiple candidate variants concurrently and returns the first successful result.
/// All candidates are executed in parallel, and whichever completes successfully first provides
/// the final inference result.
///
/// # Fields
///
/// * `weight` - Optional weight for A/B testing this variant against others
/// * `timeout_s` - Timeout in seconds for each individual candidate (default: 300s)
/// * `candidates` - List of variant names to race concurrently
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct FirstOfNConfig {
    weight: Option<f64>,
    timeout_s: f64,
    candidates: Vec<String>,
}

impl FirstOfNConfig {
    pub fn weight(&self) -> Option<f64> {
        self.weight
    }

    pub fn set_weight(&mut self, weight: Option<f64>) {
        self.weight = weight;
    }

    /// Returns the timeout in seconds for each candidate.
    ///
    /// Each candidate has an independent timeout. If a candidate doesn't complete within
    /// this duration, it will be cancelled and treated as a failure. This does not affect
    /// other candidates that are still running.
    pub fn timeout_s(&self) -> f64 {
        self.timeout_s
    }

    /// Returns the list of candidate variant names that will be raced.
    ///
    /// All candidates must be valid variant names defined in the same function.
    pub fn candidates(&self) -> &Vec<String> {
        &self.candidates
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedFirstOfNConfig {
    #[serde(default)]
    pub weight: Option<f64>,
    #[serde(default = "default_timeout")]
    pub timeout_s: f64,
    pub candidates: Vec<String>,
}

fn default_timeout() -> f64 {
    300.0
}

impl LoadableConfig<FirstOfNConfig> for UninitializedFirstOfNConfig {
    fn load(self) -> Result<FirstOfNConfig, Error> {
        Ok(FirstOfNConfig {
            weight: self.weight,
            timeout_s: self.timeout_s,
            candidates: self.candidates,
        })
    }
}

impl Variant for FirstOfNConfig {
    /// Executes all candidate variants concurrently and returns the first successful result.
    ///
    /// This method starts inference for all candidates in parallel using `FuturesUnordered`.
    /// As candidates complete, their results are checked in order of completion. The first
    /// successful result is returned immediately, and remaining candidates are cancelled.
    ///
    /// # Behavior
    ///
    /// - All candidates start execution simultaneously
    /// - Each candidate has an independent timeout specified by `timeout_s`
    /// - First successful completion wins and is returned immediately
    /// - Remaining candidates are cancelled when a winner is found
    /// - Failed candidates (errors or timeouts) don't block success
    /// - If all candidates fail, an error is returned
    /// - Only the winning candidate is logged to the ModelInference table
    ///
    /// # Error Handling
    ///
    /// - Individual candidate errors are collected but don't prevent success
    /// - Timeouts are converted to timeout errors
    /// - Only returns an error if all candidates fail or timeout
    async fn infer(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        _inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        // TODO: for bookkeeping we will have to keep these futures around and wait until they return
        // without blocking after the first resolves.
        //
        // There are two options:
        // - make all the inputs to `variant.infer` owned or Arced so there aren't lifetimes
        //   in them
        // - Figure out how to return a FuturesUnordered and manage the lifetimes all the way through a nonblocking write task
        // the latter seems quite difficult
        let mut candidate_results: FuturesUnordered<_> = self
            .candidates
            .iter()
            .map(|candidate| async {
                let variant = function.variants().get(candidate).ok_or_else(|| {
                    Error::new(ErrorDetails::UnknownCandidate {
                        name: candidate.to_string(),
                    })
                })?;
                timeout(
                    Duration::from_secs_f64(self.timeout_s),
                    variant.infer(
                        input,
                        models,
                        Arc::clone(&function),
                        Arc::clone(&inference_config),
                        clients.clone(),
                        InferenceParams::default(),
                    ),
                )
                .await
                .map_err(|_timeout_err| {
                    Error::new(ErrorDetails::InferenceTimeout {
                        variant_name: candidate.clone(),
                    })
                })
            })
            .collect();
        let mut errors = Vec::new();
        let mut first = Err(Error::new(ErrorDetails::Inference {
            message: "Inference result stream returned `None` immediately: Should not happen"
                .to_string(),
        }));
        while let Some(res) = candidate_results.next().await {
            match res {
                Ok(Ok(res)) => {
                    first = Ok(res);
                    break;
                }
                Ok(Err(err)) => {
                    errors.push(err);
                }
                Err(err) => {
                    errors.push(err);
                }
            }
        }
        if first.is_err() {
            return Err(Error::new(ErrorDetails::Inference {
                message: "Inference result stream returned `None`: {errors:?}".to_string(),
            }));
        }
        first
    }

    /// Executes candidates and returns a streaming response.
    ///
    /// Since candidates run in parallel, true streaming during candidate execution is not
    /// possible. Instead, this method waits for the first successful candidate (using the
    /// non-streaming `infer` method) and then converts the result to a stream for the client.
    ///
    /// # Note
    ///
    /// The streaming conversion happens after candidate selection, not during candidate
    /// execution. This means the latency benefit of streaming is limited compared to variants
    /// that support native streaming.
    async fn infer_stream(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        let inference_result = self
            .infer(
                input,
                models,
                function,
                inference_config,
                clients,
                InferenceParams::default(),
            )
            .await?;
        stream_inference_from_non_stream(inference_result, InferenceParams::default())
    }

    /// Validates all candidate variants.
    ///
    /// This method ensures that:
    /// - All candidate variant names reference valid variants in the same function
    /// - Each candidate variant passes its own validation checks
    ///
    /// # Errors
    ///
    /// Returns an error if any candidate is invalid or if any candidate's validation fails.
    async fn validate(
        &self,
        function: Arc<FunctionConfig>,
        models: &ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig<'_>,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        for candidate in &self.candidates {
            let variant = function.variants().get(candidate).ok_or_else(|| {
                Error::new(ErrorDetails::UnknownCandidate {
                    name: candidate.to_string(),
                })
            })?;
            Box::pin(variant.validate(
                Arc::clone(&function),
                models,
                embedding_models,
                templates,
                function_name,
                candidate,
            ))
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidCandidate {
                    variant_name: variant_name.to_string(),
                    message: e.to_string(),
                })
            })?;
        }
        Ok(())
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        // we don't use an inner ChatCompletionConfig so we return an empty Vec
        Vec::new()
    }

    fn get_all_explicit_template_names(&self) -> HashSet<String> {
        HashSet::new()
    }

    /// Batch inference is not supported for first-of-n variants.
    ///
    /// The concurrent, first-wins nature of first-of-n makes it incompatible with
    /// batch inference patterns. Always returns an error.
    async fn start_batch_inference<'a>(
        &'a self,
        _input: &[LazyResolvedInput],
        _models: InferenceModels,
        _function: &'a FunctionConfig,
        _inference_configs: &'a [InferenceConfig],
        _clients: InferenceClients,
        _inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        Err(ErrorDetails::UnsupportedVariantForBatchInference { variant_name: None }.into())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use uuid::Uuid;

    use crate::config::provider_types::ProviderTypesConfig;
    use crate::config::{ErrorContext, OtlpConfig, SchemaData};
    use crate::db::clickhouse::ClickHouseConnectionInfo;
    use crate::db::postgres::PostgresConnectionInfo;
    use crate::embeddings::EmbeddingModelTable;
    use crate::http::TensorzeroHttpClient;
    use crate::model_table::ProviderTypeDefaultCredentials;
    use crate::rate_limiting::RateLimitingConfig;
    use crate::variant::chat_completion::UninitializedChatCompletionConfig;
    use crate::{
        cache::{CacheEnabledMode, CacheOptions},
        endpoints::inference::{InferenceCredentials, InferenceIds},
        function::FunctionConfigChat,
        minijinja_util::tests::get_test_template_config,
        model::{ModelConfig, ModelProvider, ProviderConfig},
        providers::dummy::DummyProvider,
        variant::{VariantConfig, VariantInfo},
    };

    use super::*;

    struct TestSetup {
        models: ModelTable,
        embedding_models: EmbeddingModelTable,
        function: FunctionConfig,
        input: LazyResolvedInput,
        templates: TemplateConfig<'static>,
        client: TensorzeroHttpClient,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        postgres_connection_info: PostgresConnectionInfo,
        api_keys: InferenceCredentials,
        cache_options: CacheOptions,
        tags: HashMap<String, String>,
        rate_limiting_config: RateLimitingConfig,
        otlp_config: OtlpConfig,
    }

    impl TestSetup {
        fn create_inference_config<'a>(&'a self) -> InferenceConfig<'a> {
            InferenceConfig {
                ids: InferenceIds {
                    inference_id: Uuid::now_v7(),
                    episode_id: Uuid::now_v7(),
                },
                templates: &self.templates,
                tool_config: None,
                dynamic_output_schema: None,
                function_name: "test_function",
                variant_name: "first_of_n_variant",
                fetch_and_encode_input_files_before_inference: false,
                extra_body: Default::default(),
                extra_headers: Default::default(),
                extra_cache_key: None,
            }
        }
        fn create_inference_clients<'a>(&'a self) -> InferenceClients<'a> {
            InferenceClients {
                http_client: &self.client,
                clickhouse_connection_info: &self.clickhouse_connection_info,
                postgres_connection_info: &self.postgres_connection_info,
                credentials: &self.api_keys,
                cache_options: &self.cache_options,
                tags: &self.tags,
                rate_limiting_config: &self.rate_limiting_config,
                otlp_config: &self.otlp_config,
            }
        }
    }

    async fn create_test_setup() -> TestSetup {
        let templates = get_test_template_config();
        let provider_types = ProviderTypesConfig::default();

        let models = ModelTable::new(
            HashMap::from([
                (
                    "fast".into(),
                    ModelConfig {
                        routing: vec!["fast".into()],
                        providers: HashMap::from([(
                            "fast".into(),
                            ModelProvider {
                                name: "fast".into(),
                                config: ProviderConfig::Dummy(DummyProvider {
                                    model_name: "fast".into(),
                                    ..Default::default()
                                }),
                                extra_body: Default::default(),
                                extra_headers: Default::default(),
                                timeouts: Default::default(),
                                discard_unknown_chunks: false,
                            },
                        )]),
                        timeouts: Default::default(),
                    },
                ),
                (
                    "slow".into(),
                    ModelConfig {
                        routing: vec!["slow".into()],
                        providers: HashMap::from([(
                            "slow".into(),
                            ModelProvider {
                                name: "slow".into(),
                                config: ProviderConfig::Dummy(DummyProvider {
                                    model_name: "slow".into(),
                                    ..Default::default()
                                }),
                                extra_body: Default::default(),
                                extra_headers: Default::default(),
                                timeouts: Default::default(),
                                discard_unknown_chunks: false,
                            },
                        )]),
                        timeouts: Default::default(),
                    },
                ),
                (
                    "error".into(),
                    ModelConfig {
                        routing: vec!["error".into()],
                        providers: HashMap::from([(
                            "error_model".into(),
                            ModelProvider {
                                name: "error_model".into(),
                                config: ProviderConfig::Dummy(DummyProvider {
                                    model_name: "error".into(),
                                    ..Default::default()
                                }),
                                extra_body: Default::default(),
                                extra_headers: Default::default(),
                                timeouts: Default::default(),
                                discard_unknown_chunks: false,
                            },
                        )]),
                        timeouts: Default::default(),
                    },
                ),
            ]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
        )
        .expect("Failed to create model table");

        let mut variants = HashMap::new();
        variants.insert(
            "fast_candidate".to_string(),
            Arc::new(VariantInfo {
                inner: VariantConfig::ChatCompletion(
                    UninitializedChatCompletionConfig {
                        model: "fast".into(),
                        weight: None,
                        ..Default::default()
                    }
                    .load(&SchemaData::default(), &ErrorContext::new_test())
                    .unwrap(),
                ),
                timeouts: Default::default(),
            }),
        );
        variants.insert(
            "slow_candidate".to_string(),
            Arc::new(VariantInfo {
                inner: VariantConfig::ChatCompletion(
                    UninitializedChatCompletionConfig {
                        model: "slow".into(),
                        weight: None,
                        ..Default::default()
                    }
                    .load(&SchemaData::default(), &ErrorContext::new_test())
                    .unwrap(),
                ),
                timeouts: Default::default(),
            }),
        );
        variants.insert(
            "error_candidate".to_string(),
            Arc::new(VariantInfo {
                inner: VariantConfig::ChatCompletion(
                    UninitializedChatCompletionConfig {
                        model: "error".into(),
                        weight: None,
                        ..Default::default()
                    }
                    .load(&SchemaData::default(), &ErrorContext::new_test())
                    .unwrap(),
                ),
                timeouts: Default::default(),
            }),
        );

        let function = FunctionConfig::Chat(FunctionConfigChat {
            variants,
            ..Default::default()
        });

        let embedding_models = EmbeddingModelTable::default();

        let input = LazyResolvedInput {
            system: None,
            messages: vec![],
        };

        let client = TensorzeroHttpClient::new().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let postgres_connection_info = PostgresConnectionInfo::Disabled;
        let api_keys = InferenceCredentials::default();
        let cache_options = CacheOptions {
            max_age_s: None,
            enabled: CacheEnabledMode::WriteOnly,
        };

        TestSetup {
            models,
            embedding_models,
            function,
            input,
            templates,
            client,
            clickhouse_connection_info,
            postgres_connection_info,
            api_keys,
            cache_options,
            tags: Default::default(),
            rate_limiting_config: Default::default(),
            otlp_config: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_first_of_n_normal_operation() {
        let setup = create_test_setup().await;
        let inference_models = InferenceModels {
            models: &setup.models,
            embedding_models: &setup.embedding_models,
        };
        let inference_config = setup.create_inference_config();
        let inference_clients = setup.create_inference_clients();

        let first_of_n_config = FirstOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec!["fast_candidate".to_string(), "slow_candidate".to_string()],
        };

        let result = first_of_n_config
            .infer(
                &setup.input,
                &inference_models,
                &setup.function,
                &inference_config,
                &inference_clients,
                InferenceParams::default(),
            )
            .await;

        assert!(result.is_ok());
        let inference_result = result.unwrap();

        match inference_result {
            InferenceResult::Chat(chat_result) => {
                assert!(!chat_result.content.is_empty());
                assert_eq!(chat_result.model_inference_results.len(), 1);
                assert_eq!(
                    chat_result.model_inference_results[0].model_name,
                    "fast".into()
                );
                assert_eq!(
                    chat_result.model_inference_results[0].model_provider_name,
                    "fast".into()
                );
            }
            InferenceResult::Json(..) => panic!("Expected Chat inference result"),
        }
    }

    #[tokio::test]
    async fn test_first_of_n_mixed_success_with_error() {
        let setup = create_test_setup().await;
        let inference_models = InferenceModels {
            models: &setup.models,
            embedding_models: &setup.embedding_models,
        };
        let inference_config = setup.create_inference_config();
        let inference_clients = setup.create_inference_clients();

        let mixed_first_of_n_config = FirstOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec!["error_candidate".to_string(), "fast_candidate".to_string()],
        };

        let result = mixed_first_of_n_config
            .infer(
                &setup.input,
                &inference_models,
                &setup.function,
                &inference_config,
                &inference_clients,
                InferenceParams::default(),
            )
            .await;

        assert!(result.is_ok());
        let inference_result = result.unwrap();
        match inference_result {
            InferenceResult::Chat(chat_result) => {
                assert!(!chat_result.content.is_empty());
                assert_eq!(chat_result.model_inference_results.len(), 1);
                assert_eq!(
                    chat_result.model_inference_results[0].model_name,
                    "fast".into()
                );
            }
            InferenceResult::Json(..) => panic!("Expected Chat inference result"),
        }
    }

    #[tokio::test]
    async fn test_first_of_n_empty_candidates() {
        let setup = create_test_setup().await;
        let _inference_models = InferenceModels {
            models: &setup.models,
            embedding_models: &setup.embedding_models,
        };
        let inference_config = setup.create_inference_config();
        let inference_clients = setup.create_inference_clients();

        let empty_first_of_n_config = FirstOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
        };

        let result = empty_first_of_n_config
            .infer(
                &setup.input,
                &_inference_models,
                &setup.function,
                &inference_config,
                &inference_clients,
                InferenceParams::default(),
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_first_of_n_single_candidate() {
        let setup = create_test_setup().await;
        let inference_models = InferenceModels {
            models: &setup.models,
            embedding_models: &setup.embedding_models,
        };
        let inference_config = setup.create_inference_config();
        let inference_clients = setup.create_inference_clients();

        let single_first_of_n_config = FirstOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec!["fast_candidate".to_string()],
        };

        let result = single_first_of_n_config
            .infer(
                &setup.input,
                &inference_models,
                &setup.function,
                &inference_config,
                &inference_clients,
                InferenceParams::default(),
            )
            .await;

        assert!(result.is_ok());
        let inference_result = result.unwrap();
        match inference_result {
            InferenceResult::Chat(chat_result) => {
                assert!(!chat_result.content.is_empty());
                assert_eq!(chat_result.model_inference_results.len(), 1);
                assert_eq!(
                    chat_result.model_inference_results[0].model_name,
                    "fast".into()
                );
            }
            InferenceResult::Json(..) => panic!("Expected Chat inference result"),
        }
    }

    #[tokio::test]
    async fn test_first_of_n_timeout() {
        let setup = create_test_setup().await;
        let inference_models = InferenceModels {
            models: &setup.models,
            embedding_models: &setup.embedding_models,
        };
        let inference_config = setup.create_inference_config();
        let inference_clients = setup.create_inference_clients();

        let timeout_first_of_n_config = FirstOfNConfig {
            weight: Some(1.0),
            timeout_s: 0.1,
            candidates: vec!["slow_candidate".to_string()],
        };

        let result = timeout_first_of_n_config
            .infer(
                &setup.input,
                &inference_models,
                &setup.function,
                &inference_config,
                &inference_clients,
                InferenceParams::default(),
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_infer_first_of_n_stream() {
        use crate::variant::{VariantConfig, VariantInfo};
        use std::sync::Arc;

        let first_of_n_config = FirstOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec!["candidate1".to_string()],
        };

        let templates = get_test_template_config();
        let provider_types = ProviderTypesConfig::default();

        let models = ModelTable::new(
            HashMap::from([(
                "model1".into(),
                ModelConfig {
                    routing: vec!["model1".into()],
                    providers: HashMap::from([(
                        "model1".into(),
                        ModelProvider {
                            name: "model1".into(),
                            config: ProviderConfig::Dummy(DummyProvider {
                                model_name: "model1".into(),
                                ..Default::default()
                            }),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            timeouts: Default::default(),
                            discard_unknown_chunks: false,
                        },
                    )]),
                    timeouts: Default::default(),
                },
            )]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
        )
        .expect("Failed to create model table");

        let mut variants = HashMap::new();
        variants.insert(
            "candidate1".to_string(),
            Arc::new(VariantInfo {
                inner: VariantConfig::ChatCompletion(
                    UninitializedChatCompletionConfig {
                        model: "model1".into(),
                        weight: None,
                        ..Default::default()
                    }
                    .load(&SchemaData::default(), &ErrorContext::new_test())
                    .unwrap(),
                ),
                timeouts: Default::default(),
            }),
        );

        let function = FunctionConfig::Chat(FunctionConfigChat {
            variants,
            ..Default::default()
        });

        let embedding_models = EmbeddingModelTable::default();
        let inference_models = InferenceModels {
            models: &models,
            embedding_models: &embedding_models,
        };
        let input = LazyResolvedInput {
            system: None,
            messages: vec![],
        };

        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
            function_name: "test_function",
            variant_name: "first_of_n_variant",
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };

        let client = TensorzeroHttpClient::new().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let postgres_connection_info = PostgresConnectionInfo::Disabled;
        let api_keys = InferenceCredentials::default();
        let inference_clients = InferenceClients {
            http_client: &client,
            clickhouse_connection_info: &clickhouse_connection_info,
            postgres_connection_info: &postgres_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: &Default::default(),
            rate_limiting_config: &Default::default(),
            otlp_config: &Default::default(),
        };

        let result = first_of_n_config
            .infer_stream(
                &input,
                &inference_models,
                &function,
                &inference_config,
                &inference_clients,
                InferenceParams::default(),
            )
            .await;

        assert!(result.is_ok());
        let (_stream, model_used_info) = result.unwrap();

        assert!(!model_used_info.model_name.is_empty());
        assert!(!model_used_info.model_provider_name.is_empty());
    }
}
