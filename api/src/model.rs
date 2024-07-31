use reqwest::Client;
use secrecy::SecretString;
use std::collections::HashMap;
use std::env;

#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyProvider;
use crate::{
    error::Error,
    inference::{
        providers::{
            anthropic::AnthropicProvider,
            openai::{FireworksProvider, OpenAIProvider, TogetherProvider},
            provider_trait::InferenceProvider,
        },
        types::{
            InferenceResponseStream, ModelInferenceRequest, ModelInferenceResponse,
            ModelInferenceResponseChunk,
        },
    },
};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub routing: Vec<String>, // [provider name A, provider name B, ...]
    pub providers: HashMap<String, ProviderConfig>, // provider name => provider config
}

impl ModelConfig {
    pub async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let mut provider_errors: Vec<Error> = Vec::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.infer(request, client).await;
            match response {
                Ok(response) => {
                    for error in &provider_errors {
                        error.log();
                    }
                    return Ok(response);
                }
                Err(error) => provider_errors.push(error),
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }

    pub async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let mut provider_errors = Vec::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.infer_stream(request, client).await;
            match response {
                Ok(response) => return Ok(response),
                Err(error) => provider_errors.push(error),
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }
}

// TODO: think about how we can manage typing here so we don't have to check every time this is passed that it is the correct type.
#[derive(Clone, Debug)]
pub enum ProviderConfig {
    Anthropic {
        model_name: String,
        api_key: Option<SecretString>,
    },
    Azure {
        model_name: String,
        api_base: String,
        api_key: Option<SecretString>,
    },
    Fireworks {
        model_name: String,
        api_key: Option<SecretString>,
    },
    OpenAI {
        model_name: String,
        api_base: Option<String>,
        api_key: Option<SecretString>,
    },
    Together {
        model_name: String,
        api_key: Option<SecretString>,
    },
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy { model_name: String },
}

impl<'de> Deserialize<'de> for ProviderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Helper struct for deserializing the ProviderConfig.
        /// This is necessary because we want to load environment variables as we deserialize
        /// and we need to be able to deserialize the correct one based on the "type" field.
        /// Use the ProviderConfig struct for all post-initialization logic.
        #[derive(Deserialize)]
        #[serde(tag = "type")]
        #[serde(rename_all = "lowercase")]
        #[serde(deny_unknown_fields)]
        enum ProviderConfigHelper {
            Anthropic {
                model_name: String,
            },
            Azure {
                model_name: String,
                api_base: String,
            },
            Fireworks {
                model_name: String,
            },
            OpenAI {
                model_name: String,
                api_base: Option<String>,
            },
            Together {
                model_name: String,
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            Dummy {
                model_name: String,
            },
        }

        let helper = ProviderConfigHelper::deserialize(deserializer)?;

        Ok(match helper {
            ProviderConfigHelper::Anthropic { model_name } => ProviderConfig::Anthropic {
                model_name,
                api_key: env::var("ANTHROPIC_API_KEY").ok().map(SecretString::new),
            },
            ProviderConfigHelper::Azure {
                model_name,
                api_base,
            } => ProviderConfig::Azure {
                model_name,
                api_base,
                api_key: env::var("AZURE_OPENAI_API_KEY").ok().map(SecretString::new),
            },
            ProviderConfigHelper::Fireworks { model_name } => ProviderConfig::Fireworks {
                model_name,
                api_key: env::var("FIREWORKS_API_KEY").ok().map(SecretString::new),
            },
            ProviderConfigHelper::OpenAI {
                model_name,
                api_base,
            } => ProviderConfig::OpenAI {
                model_name,
                api_base,
                api_key: env::var("OPENAI_API_KEY").ok().map(SecretString::new),
            },
            ProviderConfigHelper::Together { model_name } => ProviderConfig::Together {
                model_name,
                api_key: env::var("TOGETHER_API_KEY").ok().map(SecretString::new),
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfigHelper::Dummy { model_name } => ProviderConfig::Dummy { model_name },
        })
    }
}

impl ProviderConfig {
    pub async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<ModelInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic { .. } => {
                AnthropicProvider::infer(request, self, client).await
            }
            ProviderConfig::Azure { .. } => {
                todo!()
            }
            ProviderConfig::Fireworks { .. } => {
                FireworksProvider::infer(request, self, client).await
            }
            ProviderConfig::OpenAI { .. } => OpenAIProvider::infer(request, self, client).await,
            ProviderConfig::Together { .. } => TogetherProvider::infer(request, self, client).await,
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy { .. } => DummyProvider::infer(request, self, client).await,
        }
    }

    pub async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        match self {
            ProviderConfig::Anthropic { .. } => {
                AnthropicProvider::infer_stream(request, self, client).await
            }
            ProviderConfig::Azure { .. } => {
                todo!()
            }
            ProviderConfig::Fireworks { .. } => {
                FireworksProvider::infer_stream(request, self, client).await
            }
            ProviderConfig::OpenAI { .. } => {
                OpenAIProvider::infer_stream(request, self, client).await
            }
            ProviderConfig::Together { .. } => {
                TogetherProvider::infer_stream(request, self, client).await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy { .. } => {
                DummyProvider::infer_stream(request, self, client).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::{
        providers::dummy::{
            DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW, DUMMY_INFER_USAGE,
            DUMMY_STREAMING_RESPONSE,
        },
        types::FunctionType,
    };
    use tokio_stream::StreamExt;

    use super::*;

    #[tokio::test]
    async fn test_model_config_infer_routing() {
        let good_provider_config = ProviderConfig::Dummy {
            model_name: "good".to_string(),
        };
        let bad_provider_config = ProviderConfig::Dummy {
            model_name: "error".to_string(),
        };
        let model_config = ModelConfig {
            routing: vec!["good".to_string()],
            providers: HashMap::from([("good".to_string(), good_provider_config.clone())]),
        };

        // Try inferring the good model only
        let request = ModelInferenceRequest {
            messages: vec![],
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: false,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let response = model_config.infer(&request, &Client::new()).await.unwrap();
        let content = response.content.unwrap();
        assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
        let raw = response.raw;
        assert_eq!(raw, DUMMY_INFER_RESPONSE_RAW);
        let usage = response.usage;
        assert_eq!(usage, DUMMY_INFER_USAGE);

        // Try inferring the bad model
        let model_config = ModelConfig {
            routing: vec!["error".to_string()],
            providers: HashMap::from([("error".to_string(), bad_provider_config.clone())]),
        };
        let response = model_config
            .infer(&request, &Client::new())
            .await
            .unwrap_err();
        assert_eq!(
            response,
            Error::ModelProvidersExhausted {
                provider_errors: vec![Error::InferenceClient {
                    message: "Error sending request to Dummy provider.".to_string()
                }]
            }
        );

        // Try inferring the good model as a fallback
        let model_config = ModelConfig {
            routing: vec!["error".to_string(), "good".to_string()],
            providers: HashMap::from([
                ("error".to_string(), bad_provider_config),
                ("good".to_string(), good_provider_config),
            ]),
        };
        let response = model_config.infer(&request, &Client::new()).await.unwrap();
        let content = response.content.unwrap();
        assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
        let raw = response.raw;
        assert_eq!(raw, DUMMY_INFER_RESPONSE_RAW);
        let usage = response.usage;
        assert_eq!(usage, DUMMY_INFER_USAGE);
        // TODO: assert that an error was logged then do it in the other order and assert that one was not.
    }

    #[tokio::test]
    async fn test_model_config_infer_stream_routing() {
        let good_provider_config = ProviderConfig::Dummy {
            model_name: "good".to_string(),
        };
        let bad_provider_config = ProviderConfig::Dummy {
            model_name: "error".to_string(),
        };

        let request = ModelInferenceRequest {
            messages: vec![],
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: true,
            json_mode: false,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        // Test good model
        let model_config = ModelConfig {
            routing: vec!["good".to_string()],
            providers: HashMap::from([("good".to_string(), good_provider_config.clone())]),
        };
        let (initial_chunk, stream) = model_config
            .infer_stream(&request, &Client::new())
            .await
            .unwrap();
        assert_eq!(
            initial_chunk.content,
            Some(DUMMY_STREAMING_RESPONSE[0].to_string())
        );

        let mut collected_content = initial_chunk.content.unwrap_or_default();
        let mut stream = Box::pin(stream);
        while let Some(Ok(chunk)) = stream.next().await {
            if let Some(content) = chunk.content {
                collected_content.push_str(&content);
            }
        }
        assert_eq!(collected_content, DUMMY_STREAMING_RESPONSE.join(""));

        // Test bad model
        let model_config = ModelConfig {
            routing: vec!["error".to_string()],
            providers: HashMap::from([("error".to_string(), bad_provider_config.clone())]),
        };
        let response = model_config.infer_stream(&request, &Client::new()).await;
        assert!(response.is_err());
        let error = match response {
            Err(error) => error,
            Ok(_) => panic!("Expected error, got Ok(_)"),
        };
        assert_eq!(
            error,
            Error::ModelProvidersExhausted {
                provider_errors: vec![Error::InferenceClient {
                    message: "Error sending request to Dummy provider.".to_string()
                }]
            }
        );

        // Test fallback
        let model_config = ModelConfig {
            routing: vec!["error".to_string(), "good".to_string()],
            providers: HashMap::from([
                ("error".to_string(), bad_provider_config),
                ("good".to_string(), good_provider_config),
            ]),
        };
        let (initial_chunk, stream) = model_config
            .infer_stream(&request, &Client::new())
            .await
            .unwrap();
        assert_eq!(
            initial_chunk.content,
            Some(DUMMY_STREAMING_RESPONSE[0].to_string())
        );

        let mut collected_content = initial_chunk.content.unwrap_or_default();
        let mut stream = Box::pin(stream);
        while let Some(Ok(chunk)) = stream.next().await {
            if let Some(content) = chunk.content {
                collected_content.push_str(&content);
            }
        }
        assert_eq!(collected_content, DUMMY_STREAMING_RESPONSE.join(""));
        // TODO: assert that an error was logged then do it in the other order and assert that one was not.
    }
}
