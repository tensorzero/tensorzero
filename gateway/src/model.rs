use reqwest::Client;
use secrecy::SecretString;
use std::env;
use std::{borrow::Cow, collections::HashMap};
use url::Url;

#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyProvider;
use crate::{
    endpoints::inference::InferenceApiKeys,
    error::Error,
    inference::{
        providers::{
            anthropic::AnthropicProvider,
            aws_bedrock::AWSBedrockProvider,
            azure::AzureProvider,
            fireworks::FireworksProvider,
            gcp_vertex_anthropic::GCPVertexAnthropicProvider,
            gcp_vertex_gemini::{GCPCredentials, GCPVertexGeminiProvider},
            mistral::MistralProvider,
            openai::OpenAIProvider,
            provider_trait::{HasCredentials, InferenceProvider},
            together::TogetherProvider,
            vllm::VLLMProvider,
        },
        types::{
            ModelInferenceRequest, ModelInferenceResponse, ProviderInferenceResponse,
            ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
        },
    },
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub routing: Vec<String>, // [provider name A, provider name B, ...]
    pub providers: HashMap<String, ProviderConfig>, // provider name => provider config
}

impl ModelConfig {
    pub async fn infer<'a, 'request>(
        &'a self,
        request: &'request ModelInferenceRequest<'request>,
        client: &'request Client,
        api_keys: &'request InferenceApiKeys,
    ) -> Result<ModelInferenceResponse<'a>, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.infer(request, client, api_keys).await;
            match response {
                Ok(response) => {
                    for error in provider_errors.values() {
                        error.log();
                    }
                    let model_inference_response =
                        ModelInferenceResponse::new(response, provider_name);
                    return Ok(model_inference_response);
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }

    pub async fn infer_stream<'a, 'request>(
        &'a self,
        request: &'request ModelInferenceRequest<'request>,
        client: &'request Client,
        api_keys: &'request InferenceApiKeys,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
            &'a str,
        ),
        Error,
    > {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config
                .infer_stream(request, client, api_keys)
                .await;
            match response {
                Ok(response) => {
                    for error in provider_errors.values() {
                        error.log();
                    }
                    let (chunk, stream, raw_request) = response;
                    return Ok((chunk, stream, raw_request, provider_name));
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }

    pub fn validate(&self) -> Result<(), Error> {
        // Ensure that all providers have credentials
        if !self
            .providers
            .values()
            .all(|provider| provider.has_credentials())
        {
            return Err(Error::ModelValidation {
                message: "At least one provider lacks credentials".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ProviderConfig {
    Anthropic(AnthropicProvider),
    AWSBedrock(AWSBedrockProvider),
    Azure(AzureProvider),
    Fireworks(FireworksProvider),
    GCPVertexAnthropic(GCPVertexAnthropicProvider),
    GCPVertexGemini(GCPVertexGeminiProvider),
    Mistral(MistralProvider),
    OpenAI(OpenAIProvider),
    Together(TogetherProvider),
    VLLM(VLLMProvider),
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
                deployment_id: String,
                endpoint: Url,
            },
            #[serde(rename = "gcp_vertex_anthropic")]
            GCPVertexAnthropic {
                model_id: String,
                location: String,
                project_id: String,
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
            Mistral {
                model_name: String,
            },
            OpenAI {
                model_name: String,
                api_base: Option<Url>,
            },
            Together {
                model_name: String,
            },
            #[allow(clippy::upper_case_acronyms)]
            VLLM {
                model_name: String,
                api_base: Url,
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
                    api_key: env::var("ANTHROPIC_API_KEY").ok().map(SecretString::from),
                })
            }
            ProviderConfigHelper::AWSBedrock { model_id, region } => {
                let region = region.map(aws_types::region::Region::new);

                // NB: We need to make an async call here to initialize the AWS Bedrock client.

                let provider = tokio::task::block_in_place(move || {
                    tokio::runtime::Handle::current()
                        .block_on(async { AWSBedrockProvider::new(model_id, region).await })
                        .map_err(|e| {
                            serde::de::Error::custom(format!(
                                "Failed to initialize AWS Bedrock provider: {e}"
                            ))
                        })
                })?;

                ProviderConfig::AWSBedrock(provider)
            }
            ProviderConfigHelper::Azure {
                deployment_id,
                endpoint,
            } => ProviderConfig::Azure(AzureProvider {
                deployment_id,
                endpoint,
                api_key: env::var("AZURE_OPENAI_API_KEY")
                    .ok()
                    .map(SecretString::from),
            }),
            ProviderConfigHelper::Fireworks { model_name } => {
                ProviderConfig::Fireworks(FireworksProvider {
                    model_name,
                    api_key: env::var("FIREWORKS_API_KEY").ok().map(SecretString::from),
                })
            }
            ProviderConfigHelper::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
            } => {
                // If the environment variable is not set or is empty, we will have None as our credentials.
                let credentials_path = env::var("GCP_VERTEX_CREDENTIALS_PATH")
                    .ok()
                    .filter(|s| !s.is_empty());
                // If the environment variable is set and non-empty, we will load and validate (as much as possible)
                // the credentials from the path. If this fails, we will throw an error and stop the startup.
                let credentials = match credentials_path {
                    Some(path) => Some(GCPCredentials::from_env(path.as_str()).map_err(|e| {
                        serde::de::Error::custom(format!("Failed to load GCP credentials: {}", e))
                    })?),
                    None => None,
                };
                let request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:rawPredict");
                let streaming_request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:streamRawPredict");
                let audience = format!("https://{location}-aiplatform.googleapis.com/");

                ProviderConfig::GCPVertexAnthropic(GCPVertexAnthropicProvider {
                    request_url,
                    streaming_request_url,
                    audience,
                    credentials,
                    model_id,
                })
            }
            ProviderConfigHelper::GCPVertexGemini {
                model_id,
                location,
                project_id,
            } => {
                // If the environment variable is not set or is empty, we will have None as our credentials.
                let credentials_path = env::var("GCP_VERTEX_CREDENTIALS_PATH")
                    .ok()
                    .filter(|s| !s.is_empty());
                // If the environment variable is set and non-empty, we will load and validate (as much as possible)
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
                    model_id,
                })
            }
            ProviderConfigHelper::Mistral { model_name } => {
                ProviderConfig::Mistral(MistralProvider {
                    model_name,
                    api_key: env::var("MISTRAL_API_KEY").ok().map(SecretString::from),
                })
            }
            ProviderConfigHelper::OpenAI {
                model_name,
                api_base,
            } => ProviderConfig::OpenAI(OpenAIProvider {
                model_name,
                api_base,
                api_key: env::var("OPENAI_API_KEY").ok().map(SecretString::from),
            }),
            ProviderConfigHelper::Together { model_name } => {
                ProviderConfig::Together(TogetherProvider {
                    model_name,
                    api_key: env::var("TOGETHER_API_KEY").ok().map(SecretString::from),
                })
            }
            ProviderConfigHelper::VLLM {
                model_name,
                api_base,
            } => ProviderConfig::VLLM(VLLMProvider {
                model_name,
                api_base,
                api_key: env::var("VLLM_API_KEY").ok().map(SecretString::from),
            }),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfigHelper::Dummy { model_name } => {
                ProviderConfig::Dummy(DummyProvider { model_name })
            }
        })
    }
}

impl ProviderConfig {
    async fn infer(
        &self,
        request: &ModelInferenceRequest<'_>,
        client: &Client,
        api_keys: &InferenceApiKeys,
    ) -> Result<ProviderInferenceResponse, Error> {
        let api_key = self.get_api_key(api_keys)?;
        match self {
            ProviderConfig::Anthropic(provider) => provider.infer(request, client, api_key).await,
            ProviderConfig::AWSBedrock(provider) => provider.infer(request, client, api_key).await,
            ProviderConfig::Azure(provider) => provider.infer(request, client, api_key).await,
            ProviderConfig::Fireworks(provider) => provider.infer(request, client, api_key).await,
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider.infer(request, client, api_key).await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider.infer(request, client, api_key).await
            }
            ProviderConfig::Mistral(provider) => provider.infer(request, client, api_key).await,
            ProviderConfig::OpenAI(provider) => provider.infer(request, client, api_key).await,
            ProviderConfig::Together(provider) => provider.infer(request, client, api_key).await,
            ProviderConfig::VLLM(provider) => provider.infer(request, client, api_key).await,
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => provider.infer(request, client, api_key).await,
        }
    }

    async fn infer_stream(
        &self,
        request: &ModelInferenceRequest<'_>,
        client: &Client,
        api_keys: &InferenceApiKeys,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        let api_key = self.get_api_key(api_keys)?;
        match self {
            ProviderConfig::Anthropic(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::Azure(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::Fireworks(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::Mistral(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::OpenAI(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::Together(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
            ProviderConfig::VLLM(provider) => provider.infer_stream(request, client, api_key).await,
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider.infer_stream(request, client, api_key).await
            }
        }
    }
}

impl HasCredentials for ProviderConfig {
    fn has_credentials(&self) -> bool {
        match self {
            ProviderConfig::Anthropic(provider) => provider.has_credentials(),
            ProviderConfig::AWSBedrock(provider) => provider.has_credentials(),
            ProviderConfig::Azure(provider) => provider.has_credentials(),
            ProviderConfig::Fireworks(provider) => provider.has_credentials(),
            ProviderConfig::GCPVertexAnthropic(provider) => provider.has_credentials(),
            ProviderConfig::GCPVertexGemini(provider) => provider.has_credentials(),
            ProviderConfig::Mistral(provider) => provider.has_credentials(),
            ProviderConfig::OpenAI(provider) => provider.has_credentials(),
            ProviderConfig::Together(provider) => provider.has_credentials(),
            ProviderConfig::VLLM(provider) => provider.has_credentials(),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => provider.has_credentials(),
        }
    }

    fn get_api_key<'a>(
        &'a self,
        api_keys: &'a InferenceApiKeys,
    ) -> Result<Cow<'a, SecretString>, Error> {
        match self {
            ProviderConfig::Anthropic(provider) => provider.get_api_key(api_keys),
            ProviderConfig::AWSBedrock(provider) => provider.get_api_key(api_keys),
            ProviderConfig::Azure(provider) => provider.get_api_key(api_keys),
            ProviderConfig::Fireworks(provider) => provider.get_api_key(api_keys),
            ProviderConfig::GCPVertexAnthropic(provider) => provider.get_api_key(api_keys),
            ProviderConfig::GCPVertexGemini(provider) => provider.get_api_key(api_keys),
            ProviderConfig::Mistral(provider) => provider.get_api_key(api_keys),
            ProviderConfig::OpenAI(provider) => provider.get_api_key(api_keys),
            ProviderConfig::Together(provider) => provider.get_api_key(api_keys),
            ProviderConfig::VLLM(provider) => provider.get_api_key(api_keys),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => provider.get_api_key(api_keys),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::inference::{
        providers::dummy::{
            DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW, DUMMY_INFER_USAGE,
            DUMMY_STREAMING_RESPONSE,
        },
        types::{ContentBlockChunk, FunctionType, ModelInferenceRequestJsonMode, TextChunk},
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
            routing: vec!["good_provider".to_string()],
            providers: HashMap::from([("good_provider".to_string(), good_provider_config)]),
        };
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
        };
        let api_keys = InferenceApiKeys::default();

        // Try inferring the good model only
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let response = model_config
            .infer(&request, &Client::new(), &api_keys)
            .await
            .unwrap();
        let content = response.content;
        assert_eq!(
            content,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
        let raw = response.raw_response;
        assert_eq!(raw, DUMMY_INFER_RESPONSE_RAW);
        let usage = response.usage;
        assert_eq!(usage, DUMMY_INFER_USAGE);
        assert_eq!(response.model_provider_name, "good_provider");

        // Try inferring the bad model
        let model_config = ModelConfig {
            routing: vec!["error".to_string()],
            providers: HashMap::from([("error".to_string(), bad_provider_config)]),
        };
        let response = model_config
            .infer(&request, &Client::new(), &api_keys)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            Error::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "error".to_string(),
                    Error::InferenceClient {
                        message: "Error sending request to Dummy provider.".to_string()
                    }
                )])
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
        let api_keys = InferenceApiKeys::default();
        // Try inferring the good model only
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let model_config = ModelConfig {
            routing: vec!["error_provider".to_string(), "good_provider".to_string()],
            providers: HashMap::from([
                ("error_provider".to_string(), bad_provider_config),
                ("good_provider".to_string(), good_provider_config),
            ]),
        };

        let response = model_config
            .infer(&request, &Client::new(), &api_keys)
            .await
            .unwrap();
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
        assert_eq!(response.model_provider_name, "good_provider");
    }

    #[tokio::test]
    async fn test_model_config_infer_stream_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
        });
        let api_keys = InferenceApiKeys::default();
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        // Test good model
        let model_config = ModelConfig {
            routing: vec!["good_provider".to_string()],
            providers: HashMap::from([("good_provider".to_string(), good_provider_config)]),
        };
        let (initial_chunk, stream, raw_request, model_provider_name) = model_config
            .infer_stream(&request, &Client::new(), &api_keys)
            .await
            .unwrap();
        assert_eq!(
            initial_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })],
        );
        assert_eq!(raw_request, "raw request");
        assert_eq!(model_provider_name, "good_provider");
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
                _ => panic!("Expected a text content block"),
            }
        }
        assert_eq!(collected_content_str, DUMMY_STREAMING_RESPONSE.join(""));

        // Test bad model
        let model_config = ModelConfig {
            routing: vec!["error".to_string()],
            providers: HashMap::from([("error".to_string(), bad_provider_config)]),
        };
        let response = model_config
            .infer_stream(&request, &Client::new(), &api_keys)
            .await;
        assert!(response.is_err());
        let error = match response {
            Err(error) => error,
            Ok(_) => panic!("Expected error, got Ok(_)"),
        };
        assert_eq!(
            error,
            Error::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "error".to_string(),
                    Error::InferenceClient {
                        message: "Error sending request to Dummy provider.".to_string()
                    }
                )])
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
        let api_keys = InferenceApiKeys::default();
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        // Test fallback
        let model_config = ModelConfig {
            routing: vec!["error_provider".to_string(), "good_provider".to_string()],
            providers: HashMap::from([
                ("error_provider".to_string(), bad_provider_config),
                ("good_provider".to_string(), good_provider_config),
            ]),
        };
        let (initial_chunk, stream, raw_request, model_provider_name) = model_config
            .infer_stream(&request, &Client::new(), &api_keys)
            .await
            .unwrap();
        assert_eq!(model_provider_name, "good_provider");
        // Ensure that the error for the bad provider was logged, but the request worked nonetheless
        assert!(logs_contain("Error sending request to Dummy provider"));
        assert_eq!(raw_request, "raw request");

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
                _ => panic!("Expected a text content block"),
            }
        }
        assert_eq!(collected_content_str, DUMMY_STREAMING_RESPONSE.join(""));
    }

    #[tokio::test]
    async fn test_dynamic_api_keys() {
        let provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "test_key".to_string(),
        });
        let model_config = ModelConfig {
            routing: vec!["model".to_string()],
            providers: HashMap::from([("model".to_string(), provider_config)]),
        };
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
        };
        let api_keys = InferenceApiKeys::default();

        // Try inferring the good model only
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let error = model_config
            .infer(&request, &Client::new(), &api_keys)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            Error::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "model".to_string(),
                    Error::InferenceClient {
                        message: "Invalid API key for Dummy provider".to_string()
                    }
                )])
            }
        );

        let api_keys = InferenceApiKeys {
            dummy_api_key: Some(SecretString::from("good_key".to_string())),
            ..Default::default()
        };
        let response = model_config
            .infer(&request, &Client::new(), &api_keys)
            .await
            .unwrap();
        assert_eq!(
            response.content,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
    }
}
