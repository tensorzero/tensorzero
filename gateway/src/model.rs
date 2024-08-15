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
            aws_bedrock::AWSBedrockProvider,
            azure::AzureProvider,
            fireworks::FireworksProvider,
            gcp_vertex::{GCPCredentials, GCPVertexGeminiProvider},
            openai::OpenAIProvider,
            provider_trait::InferenceProvider,
            together::TogetherProvider,
        },
        types::{
            InferenceResultStream, ModelInferenceRequest, ModelInferenceResponse,
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
    ) -> Result<(ModelInferenceResponseChunk, InferenceResultStream), Error> {
        let mut provider_errors: Vec<Error> = Vec::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.infer_stream(request, client).await;
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
}

#[derive(Clone, Debug)]
pub enum ProviderConfig {
    Anthropic(AnthropicProvider),
    AWSBedrock(AWSBedrockProvider),
    Azure(AzureProvider),
    Fireworks(FireworksProvider),
    GCPVertexGemini(GCPVertexGeminiProvider),
    OpenAI(OpenAIProvider),
    Together(TogetherProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
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
            #[serde(rename = "aws_bedrock")]
            AWSBedrock {
                model_id: String,
                region: Option<String>,
            },
            Azure {
                model_name: String,
                api_base: String,
                deployment_id: String,
            },
            #[serde(rename = "gcp_vertex_gemini")]
            GCPVertexGemini {
                model_id: String,
                location: String,
                project_id: String,
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
            ProviderConfigHelper::Anthropic { model_name } => {
                ProviderConfig::Anthropic(AnthropicProvider {
                    model_name,
                    api_key: env::var("ANTHROPIC_API_KEY").ok().map(SecretString::new),
                })
            }
            ProviderConfigHelper::AWSBedrock { model_id, region } => {
                let region = region.map(aws_types::region::Region::new);
                ProviderConfig::AWSBedrock(AWSBedrockProvider { model_id, region })
            }
            ProviderConfigHelper::Azure {
                model_name,
                api_base,
                deployment_id,
            } => ProviderConfig::Azure(AzureProvider {
                model_name,
                api_base,
                deployment_id,
                api_key: env::var("AZURE_OPENAI_API_KEY").ok().map(SecretString::new),
            }),
            ProviderConfigHelper::Fireworks { model_name } => {
                ProviderConfig::Fireworks(FireworksProvider {
                    model_name,
                    api_key: env::var("FIREWORKS_API_KEY").ok().map(SecretString::new),
                })
            }
            ProviderConfigHelper::GCPVertexGemini {
                model_id,
                location,
                project_id,
            } => {
                // If the environment variable is not set, we will simply have None as our credentials.
                let credentials_path = env::var("GCP_VERTEX_CREDENTIALS_PATH").ok();
                // If the environment variable is set, we will load and validate (as much as possible)
                // the credentials from the path. If this fails, we will throw an error and stop the startup.
                let credentials = match credentials_path {
                    Some(path) => Some(GCPCredentials::from_env(path.as_str()).map_err(|e| {
                        serde::de::Error::custom(format!("Failed to load GCP credentials: {}", e))
                    })?),
                    None => None,
                };
                let request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent");
                let streaming_request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:streamGenerateContent?alt=sse");
                let audience = format!("https://{location}-aiplatform.googleapis.com/");

                ProviderConfig::GCPVertexGemini(GCPVertexGeminiProvider {
                    request_url,
                    streaming_request_url,
                    audience,
                    credentials,
                })
            }
            ProviderConfigHelper::OpenAI {
                model_name,
                api_base,
            } => ProviderConfig::OpenAI(OpenAIProvider {
                model_name,
                api_base,
                api_key: env::var("OPENAI_API_KEY").ok().map(SecretString::new),
            }),
            ProviderConfigHelper::Together { model_name } => {
                ProviderConfig::Together(TogetherProvider {
                    model_name,
                    api_key: env::var("TOGETHER_API_KEY").ok().map(SecretString::new),
                })
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfigHelper::Dummy { model_name } => {
                ProviderConfig::Dummy(DummyProvider { model_name })
            }
        })
    }
}

impl InferenceProvider for ProviderConfig {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<ModelInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic(provider) => provider.infer(request, client).await,
            ProviderConfig::AWSBedrock(provider) => provider.infer(request, client).await,
            ProviderConfig::Azure(provider) => provider.infer(request, client).await,
            ProviderConfig::Fireworks(provider) => provider.infer(request, client).await,
            ProviderConfig::GCPVertexGemini(provider) => provider.infer(request, client).await,
            ProviderConfig::OpenAI(provider) => provider.infer(request, client).await,
            ProviderConfig::Together(provider) => provider.infer(request, client).await,
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => provider.infer(request, client).await,
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResultStream), Error> {
        match self {
            ProviderConfig::Anthropic(provider) => provider.infer_stream(request, client).await,
            ProviderConfig::AWSBedrock(provider) => provider.infer_stream(request, client).await,
            ProviderConfig::Azure(provider) => provider.infer_stream(request, client).await,
            ProviderConfig::Fireworks(provider) => provider.infer_stream(request, client).await,
            ProviderConfig::GCPVertexGemini(provider) => {
                provider.infer_stream(request, client).await
            }
            ProviderConfig::OpenAI(provider) => provider.infer_stream(request, client).await,
            ProviderConfig::Together(provider) => provider.infer_stream(request, client).await,
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => provider.infer_stream(request, client).await,
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
        types::{ContentBlockChunk, FunctionType, JSONMode, TextChunk},
    };
    use crate::tool::{ToolCallConfig, ToolChoice};
    use tokio_stream::StreamExt;
    use tracing_test::traced_test;

    use super::*;

    #[tokio::test]
    async fn test_model_config_infer_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
        });
        let model_config = ModelConfig {
            routing: vec!["good".to_string()],
            providers: HashMap::from([("good".to_string(), good_provider_config.clone())]),
        };
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: &ToolChoice::Auto,
            parallel_tool_calls: false,
        };

        // Try inferring the good model only
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: Some(&tool_config),
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let response = model_config.infer(&request, &Client::new()).await.unwrap();
        let content = response.content;
        assert_eq!(
            content,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
        let raw = response.raw_response;
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
    }

    #[tokio::test]
    #[traced_test]
    async fn test_model_config_infer_routing_fallback() {
        // Test that fallback works with bad --> good model provider

        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
        });

        // Try inferring the good model only
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let model_config = ModelConfig {
            routing: vec!["error".to_string(), "good".to_string()],
            providers: HashMap::from([
                ("error".to_string(), bad_provider_config),
                ("good".to_string(), good_provider_config),
            ]),
        };

        let response = model_config.infer(&request, &Client::new()).await.unwrap();
        // Ensure that the error for the bad provider was logged, but the request worked nonetheless
        assert!(logs_contain("Error sending request to Dummy provider"));
        let content = response.content;
        assert_eq!(
            content,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
        let raw = response.raw_response;
        assert_eq!(raw, DUMMY_INFER_RESPONSE_RAW);
        let usage = response.usage;
        assert_eq!(usage, DUMMY_INFER_USAGE);
    }

    #[tokio::test]
    async fn test_model_config_infer_stream_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
        });

        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            stream: true,
            json_mode: JSONMode::Off,
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
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })],
        );

        let mut collected_content: Vec<ContentBlockChunk> =
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })];
        let mut stream = Box::pin(stream);
        while let Some(Ok(chunk)) = stream.next().await {
            let mut content = chunk.content;
            assert!(content.len() <= 1);
            if content.len() == 1 {
                collected_content.push(content.pop().unwrap());
            }
        }
        let mut collected_content_str = String::new();
        for content in collected_content {
            match content {
                ContentBlockChunk::Text(text) => collected_content_str.push_str(&text.text),
                _ => unreachable!(),
            }
        }
        assert_eq!(collected_content_str, DUMMY_STREAMING_RESPONSE.join(""));

        // Test bad model
        let model_config = ModelConfig {
            routing: vec!["error".to_string()],
            providers: HashMap::from([("error".to_string(), bad_provider_config.clone())]),
        };
        let response = model_config.infer_stream(&request, &Client::new()).await;
        assert!(response.is_err());
        let error = match response {
            Err(error) => error,
            Ok(_) => unreachable!("Expected error, got Ok(_)"),
        };
        assert_eq!(
            error,
            Error::ModelProvidersExhausted {
                provider_errors: vec![Error::InferenceClient {
                    message: "Error sending request to Dummy provider.".to_string()
                }]
            }
        );
    }

    #[tokio::test]
    #[traced_test]
    async fn test_model_config_infer_stream_routing_fallback() {
        // Test that fallback works with bad --> good model provider (streaming)

        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
        });

        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            stream: true,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

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

        // Ensure that the error for the bad provider was logged, but the request worked nonetheless
        assert!(logs_contain("Error sending request to Dummy provider"));

        assert_eq!(
            initial_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })],
        );

        let mut collected_content = initial_chunk.content;
        let mut stream = Box::pin(stream);
        while let Some(Ok(chunk)) = stream.next().await {
            let mut content = chunk.content;
            assert!(content.len() <= 1);
            if content.len() == 1 {
                collected_content.push(content.pop().unwrap());
            }
        }
        let mut collected_content_str = String::new();
        for content in collected_content {
            match content {
                ContentBlockChunk::Text(text) => collected_content_str.push_str(&text.text),
                _ => unreachable!(),
            }
        }
        assert_eq!(collected_content_str, DUMMY_STREAMING_RESPONSE.join(""));
    }
}
