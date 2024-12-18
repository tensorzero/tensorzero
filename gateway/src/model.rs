use reqwest::Client;
use serde::de::Error as SerdeError;
use std::collections::HashMap;
use tracing::instrument;
use url::Url;

#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyProvider;
use crate::inference::providers::google_ai_studio_gemini::GoogleAIStudioGeminiProvider;

use crate::inference::types::batch::{BatchModelInferenceResponse, BatchProviderInferenceResponse};
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        providers::{
            anthropic::AnthropicProvider, aws_bedrock::AWSBedrockProvider, azure::AzureProvider,
            fireworks::FireworksProvider, gcp_vertex_anthropic::GCPVertexAnthropicProvider,
            gcp_vertex_gemini::GCPVertexGeminiProvider, mistral::MistralProvider,
            openai::OpenAIProvider, provider_trait::InferenceProvider, together::TogetherProvider,
            vllm::VLLMProvider, xai::XAIProvider,
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
        api_keys: &'request InferenceCredentials,
    ) -> Result<ModelInferenceResponse<'a>, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider_config = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.clone(),
                })
            })?;
            let response = provider_config.infer(request, client, api_keys).await;
            match response {
                Ok(response) => {
                    let model_inference_response =
                        ModelInferenceResponse::new(response, provider_name);
                    return Ok(model_inference_response);
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::new(ErrorDetails::ModelProvidersExhausted {
            provider_errors,
        }))
    }

    #[instrument(skip_all)]
    pub async fn infer_stream<'a, 'request>(
        &'a self,
        request: &'request ModelInferenceRequest<'request>,
        client: &'request Client,
        api_keys: &'request InferenceCredentials,
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
            let provider_config = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.clone(),
                })
            })?;
            let response = provider_config
                .infer_stream(request, client, api_keys)
                .await;
            match response {
                Ok(response) => {
                    let (chunk, stream, raw_request) = response;
                    return Ok((chunk, stream, raw_request, provider_name));
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::new(ErrorDetails::ModelProvidersExhausted {
            provider_errors,
        }))
    }

    pub async fn start_batch_inference<'a, 'request>(
        &'a self,
        requests: &'request [ModelInferenceRequest<'request>],
        client: &'request Client,
        api_keys: &'request InferenceCredentials,
    ) -> Result<BatchModelInferenceResponse<'a>, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider_config = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.clone(),
                })
            })?;
            let response = provider_config
                .start_batch_inference(requests, client, api_keys)
                .await;
            match response {
                Ok(response) => {
                    return Ok(BatchModelInferenceResponse::new(response, provider_name));
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::new(ErrorDetails::ModelProvidersExhausted {
            provider_errors,
        }))
    }

    pub fn validate(&self) -> Result<(), Error> {
        // Placeholder in case we want to add validation in the future
        Ok(())
    }
}

// NOTE: if one adds a new provider, make sure to add it to the set of `BLACKLISTED_NAMES` in `config_parser.rs`
#[derive(Debug)]
pub enum ProviderConfig {
    Anthropic(AnthropicProvider),
    AWSBedrock(AWSBedrockProvider),
    Azure(AzureProvider),
    Fireworks(FireworksProvider),
    GCPVertexAnthropic(GCPVertexAnthropicProvider),
    GCPVertexGemini(GCPVertexGeminiProvider),
    GoogleAIStudioGemini(GoogleAIStudioGeminiProvider),
    Mistral(MistralProvider),
    OpenAI(OpenAIProvider),
    Together(TogetherProvider),
    VLLM(VLLMProvider),
    XAI(XAIProvider),
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
                api_key_location: Option<CredentialLocation>,
            },
            #[serde(rename = "aws_bedrock")]
            AWSBedrock {
                model_id: String,
                region: Option<String>,
            },
            Azure {
                deployment_id: String,
                endpoint: Url,
                api_key_location: Option<CredentialLocation>,
            },
            #[serde(rename = "gcp_vertex_anthropic")]
            GCPVertexAnthropic {
                model_id: String,
                location: String,
                project_id: String,
                credential_location: Option<CredentialLocation>,
            },
            #[serde(rename = "gcp_vertex_gemini")]
            GCPVertexGemini {
                model_id: String,
                location: String,
                project_id: String,
                credential_location: Option<CredentialLocation>,
            },
            #[serde(rename = "google_ai_studio_gemini")]
            GoogleAIStudioGemini {
                model_name: String,
                api_key_location: Option<CredentialLocation>,
            },
            Fireworks {
                model_name: String,
                api_key_location: Option<CredentialLocation>,
            },
            Mistral {
                model_name: String,
                api_key_location: Option<CredentialLocation>,
            },
            OpenAI {
                model_name: String,
                api_base: Option<Url>,
                api_key_location: Option<CredentialLocation>,
            },
            Together {
                model_name: String,
                api_key_location: Option<CredentialLocation>,
            },
            #[allow(clippy::upper_case_acronyms)]
            VLLM {
                model_name: String,
                api_base: Url,
                api_key_location: Option<CredentialLocation>,
            },
            #[allow(clippy::upper_case_acronyms)]
            XAI {
                model_name: String,
                api_key_location: Option<CredentialLocation>,
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            Dummy {
                model_name: String,
                api_key_location: Option<CredentialLocation>,
            },
        }

        let helper = ProviderConfigHelper::deserialize(deserializer)?;

        Ok(match helper {
            ProviderConfigHelper::Anthropic {
                model_name,
                api_key_location,
            } => ProviderConfig::Anthropic(
                AnthropicProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
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
                api_key_location,
            } => ProviderConfig::Azure(
                AzureProvider::new(deployment_id, endpoint, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::Fireworks {
                model_name,
                api_key_location,
            } => ProviderConfig::Fireworks(
                FireworksProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
                credential_location: api_key_location,
            } => ProviderConfig::GCPVertexAnthropic(
                GCPVertexAnthropicProvider::new(model_id, location, project_id, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::GCPVertexGemini {
                model_id,
                location,
                project_id,
                credential_location: api_key_location,
            } => ProviderConfig::GCPVertexGemini(
                GCPVertexGeminiProvider::new(model_id, location, project_id, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::GoogleAIStudioGemini {
                model_name,
                api_key_location,
            } => ProviderConfig::GoogleAIStudioGemini(
                GoogleAIStudioGeminiProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::Mistral {
                model_name,
                api_key_location,
            } => ProviderConfig::Mistral(
                MistralProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::OpenAI {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::OpenAI(
                OpenAIProvider::new(model_name, api_base, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::Together {
                model_name,
                api_key_location,
            } => ProviderConfig::Together(
                TogetherProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::VLLM {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::VLLM(
                VLLMProvider::new(model_name, api_base, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::XAI {
                model_name,
                api_key_location,
            } => ProviderConfig::XAI(
                XAIProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfigHelper::Dummy {
                model_name,
                api_key_location,
            } => ProviderConfig::Dummy(
                DummyProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
        })
    }
}

impl ProviderConfig {
    async fn infer(
        &self,
        request: &ModelInferenceRequest<'_>,
        client: &Client,
        api_keys: &InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::AWSBedrock(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::Azure(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::Fireworks(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider.infer(request, client, api_keys).await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider.infer(request, client, api_keys).await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider.infer(request, client, api_keys).await
            }
            ProviderConfig::Mistral(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::OpenAI(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::Together(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::VLLM(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::XAI(provider) => provider.infer(request, client, api_keys).await,
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => provider.infer(request, client, api_keys).await,
        }
    }

    async fn infer_stream(
        &self,
        request: &ModelInferenceRequest<'_>,
        client: &Client,
        api_keys: &InferenceCredentials,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        match self {
            ProviderConfig::Anthropic(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::Azure(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::Fireworks(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::Mistral(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::OpenAI(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::Together(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::XAI(provider) => provider.infer_stream(request, client, api_keys).await,
            ProviderConfig::VLLM(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
        }
    }

    async fn start_batch_inference<'a>(
        &self,
        requests: &'a [ModelInferenceRequest<'a>],
        client: &'a Client,
        api_keys: &'a InferenceCredentials,
    ) -> Result<BatchProviderInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
        }
    }
}

pub enum CredentialLocation {
    Env(String),
    Dynamic(String),
    Path(String),
    None,
}

impl<'de> Deserialize<'de> for CredentialLocation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if let Some(inner) = s.strip_prefix("env::") {
            Ok(CredentialLocation::Env(inner.to_string()))
        } else if let Some(inner) = s.strip_prefix("dynamic::") {
            Ok(CredentialLocation::Dynamic(inner.to_string()))
        } else if let Some(inner) = s.strip_prefix("path::") {
            Ok(CredentialLocation::Path(inner.to_string()))
        } else if s == "none" {
            Ok(CredentialLocation::None)
        } else {
            Err(serde::de::Error::custom(format!(
                "Invalid ApiKeyLocation format: {}",
                s
            )))
        }
    }
}

const RESERVED_MODEL_PREFIXES: &[&str] = &[
    "anthropic::",
    "aws_bedrock::",
    "azure::",
    "fireworks::",
    "gcp_vertex_anthropic::",
    "gcp_vertex_gemini::",
    "google_ai_studio_gemini::",
    "mistral::",
    "openai::",
    "together::",
    "vllm::",
    "xai::",
    "dummy::",
];

const SHORTHAND_MODEL_PREFIXES: &[&str] = &[
    "anthropic::",
    "fireworks::",
    "google_ai_studio_gemini::",
    "mistral::",
    "openai::",
    "together::",
    "xai::",
    "dummy::",
];

#[derive(Debug, Default, Deserialize)]
#[serde(try_from = "HashMap<String, ModelConfig>")]
pub struct ModelTable(HashMap<String, ModelConfig>);

impl TryFrom<HashMap<String, ModelConfig>> for ModelTable {
    type Error = String;

    fn try_from(map: HashMap<String, ModelConfig>) -> Result<Self, Self::Error> {
        for key in map.keys() {
            if RESERVED_MODEL_PREFIXES
                .iter()
                .any(|&name| key.starts_with(name))
            {
                return Err(format!("Model name '{}' contains a reserved prefix", key));
            }
        }
        Ok(ModelTable(map))
    }
}

impl std::ops::Deref for ModelTable {
    type Target = HashMap<String, ModelConfig>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ModelTable {
    pub fn validate_or_create(&mut self, key: &str) -> Result<(), Error> {
        // Try matching shorthand prefixes
        if let Some(prefix) = SHORTHAND_MODEL_PREFIXES
            .iter()
            .find(|&&prefix| key.starts_with(prefix))
        {
            let model_name = match key.strip_prefix(prefix) {
                Some(name) => name,
                None => {
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "Failed to strip prefix '{}' from model name '{}' This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/issues/new",
                            prefix, key
                        ),
                    }
                    .into());
                }
            };
            // Remove the last two characters of the prefix to get the provider type
            let provider_type = &prefix[..prefix.len() - 2];
            let model_config = model_config_from_shorthand(provider_type, model_name)?;
            model_config.validate()?;
            self.0.insert(key.to_string(), model_config);
            return Ok(());
        }

        // Try direct lookup (if it's blacklisted, it's not in the table)
        if let Some(model_config) = self.0.get(key) {
            model_config.validate()?;
            return Ok(());
        }

        Err(ErrorDetails::Config {
            message: format!("Model name '{}' not found in model table", key),
        }
        .into())
    }
}

fn model_config_from_shorthand(
    provider_type: &str,
    model_name: &str,
) -> Result<ModelConfig, Error> {
    let model_name = model_name.to_string();
    let provider_config = match provider_type {
        "anthropic" => ProviderConfig::Anthropic(AnthropicProvider::new(model_name, None)?),
        "fireworks" => ProviderConfig::Fireworks(FireworksProvider::new(model_name, None)?),
        "google_ai_studio_gemini" => ProviderConfig::GoogleAIStudioGemini(
            GoogleAIStudioGeminiProvider::new(model_name, None)?,
        ),
        "mistral" => ProviderConfig::Mistral(MistralProvider::new(model_name, None)?),
        "openai" => ProviderConfig::OpenAI(OpenAIProvider::new(model_name, None, None)?),
        "together" => ProviderConfig::Together(TogetherProvider::new(model_name, None)?),
        "xai" => ProviderConfig::XAI(XAIProvider::new(model_name, None)?),
        #[cfg(any(test, feature = "e2e_tests"))]
        "dummy" => ProviderConfig::Dummy(DummyProvider::new(model_name, None)?),
        _ => {
            return Err(ErrorDetails::Config {
                message: format!("Invalid provider type: {}", provider_type),
            }
            .into());
        }
    };
    Ok(ModelConfig {
        routing: vec![provider_type.to_string()],
        providers: HashMap::from([(provider_type.to_string(), provider_config)]),
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::inference::{
        providers::{
            anthropic::AnthropicCredentials,
            dummy::{
                DummyCredentials, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW,
                DUMMY_INFER_USAGE, DUMMY_STREAMING_RESPONSE,
            },
        },
        types::{ContentBlockChunk, FunctionType, ModelInferenceRequestJsonMode, TextChunk},
    };
    use crate::tool::{ToolCallConfig, ToolChoice};
    use secrecy::SecretString;
    use tokio_stream::StreamExt;
    use tracing_test::traced_test;

    use super::*;

    #[tokio::test]
    async fn test_model_config_infer_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
            credentials: DummyCredentials::None,
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
        let api_keys = InferenceCredentials::default();

        // Try inferring the good model only
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
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
        let content = response.output;
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
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "error".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Error sending request to Dummy provider.".to_string()
                    }
                    .into()
                )])
            }
            .into()
        );
    }

    #[tokio::test]
    #[traced_test]
    async fn test_model_config_infer_routing_fallback() {
        // Test that fallback works with bad --> good model provider

        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        // Try inferring the good model only
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
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
        let content = response.output;
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
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
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
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "error".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Error sending request to Dummy provider.".to_string()
                    }
                    .into()
                )])
            }
            .into()
        );
    }

    #[tokio::test]
    #[traced_test]
    async fn test_model_config_infer_stream_routing_fallback() {
        // Test that fallback works with bad --> good model provider (streaming)

        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
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
            credentials: DummyCredentials::Dynamic("TEST_KEY".to_string()),
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
        let api_keys = InferenceCredentials::default();

        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
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
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "model".to_string(),
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Dummy".to_string()
                    }
                    .into()
                )])
            }
            .into()
        );

        let api_keys = HashMap::from([(
            "TEST_KEY".to_string(),
            SecretString::from("notgoodkey".to_string()),
        )]);
        let response = model_config
            .infer(&request, &Client::new(), &api_keys)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "model".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Invalid API key for Dummy provider".to_string()
                    }
                    .into()
                )])
            }
            .into()
        );

        let provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "test_key".to_string(),
            credentials: DummyCredentials::Dynamic("TEST_KEY".to_string()),
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
        let api_keys = InferenceCredentials::default();

        let request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
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
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "model".to_string(),
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Dummy".to_string()
                    }
                    .into()
                )])
            }
            .into()
        );

        let api_keys = HashMap::from([(
            "TEST_KEY".to_string(),
            SecretString::from("good_key".to_string()),
        )]);
        let response = model_config
            .infer(&request, &Client::new(), &api_keys)
            .await
            .unwrap();
        assert_eq!(
            response.output,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
    }

    #[test]
    fn test_get_or_create_model_config() {
        let mut model_table = ModelTable::default();
        // Test that we can get or create a model config
        model_table.validate_or_create("openai::gpt-4o").unwrap();
        assert_eq!(model_table.len(), 1);
        let model_config = model_table.get("openai::gpt-4o").unwrap();
        assert_eq!(model_config.routing, vec!["openai".to_string()]);
        let provider_config = model_config.providers.get("openai").unwrap();
        match provider_config {
            ProviderConfig::OpenAI(provider) => assert_eq!(provider.model_name, "gpt-4o"),
            _ => panic!("Expected OpenAI provider"),
        }

        // Test that it is idempotent
        model_table.validate_or_create("openai::gpt-4o").unwrap();
        assert_eq!(model_table.len(), 1);
        let model_config = model_table.get("openai::gpt-4o").unwrap();
        assert_eq!(model_config.routing, vec!["openai".to_string()]);
        let provider_config = model_config.providers.get("openai").unwrap();
        match provider_config {
            ProviderConfig::OpenAI(provider) => assert_eq!(provider.model_name, "gpt-4o"),
            _ => panic!("Expected OpenAI provider"),
        }

        // Test that it fails if the model is not well-formed
        let model_config = model_table.validate_or_create("foo::bar");
        assert!(model_config.is_err());
        assert_eq!(
            model_config.unwrap_err(),
            ErrorDetails::Config {
                message: "Model name 'foo::bar' not found in model table".to_string()
            }
            .into()
        );
        // Test that it works with an initialized model
        let anthropic_provider_config = ProviderConfig::Anthropic(AnthropicProvider {
            model_name: "claude".to_string(),
            credentials: AnthropicCredentials::Static("".to_string().into()),
        });
        let anthropic_model_config = ModelConfig {
            routing: vec!["anthropic".to_string()],
            providers: HashMap::from([("anthropic".to_string(), anthropic_provider_config)]),
        };
        let mut model_table: ModelTable =
            HashMap::from([("claude".to_string(), anthropic_model_config)])
                .try_into()
                .unwrap();
        model_table.validate_or_create("claude").unwrap();
    }
}
