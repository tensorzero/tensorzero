use std::collections::HashSet;

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

    pub fn timeout_s(&self) -> f64 {
        self.timeout_s
    }

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
    async fn infer<'a: 'request, 'request>(
        &self,
        input: &LazyResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        _inference_params: InferenceParams, // we ignore and use defaults
    ) -> Result<InferenceResult, Error> {
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
                        function,
                        inference_config,
                        clients,
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

    async fn infer_stream<'request>(
        &self,
        input: &LazyResolvedInput,
        models: &'request InferenceModels<'_>,
        function: &FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        _inference_params: InferenceParams, // we ignore and use defaults
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

    async fn validate(
        &self,
        function: &FunctionConfig,
        models: &mut ModelTable,
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
                function,
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

    async fn start_batch_inference<'a>(
        &'a self,
        _input: &[LazyResolvedInput],
        _models: &'a InferenceModels<'a>,
        _function: &'a FunctionConfig,
        _inference_configs: &'a [InferenceConfig<'a>],
        _clients: &'a InferenceClients<'a>,
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
