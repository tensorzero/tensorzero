use reqwest::Client;
use serde::de::Error as SerdeError;
use std::collections::HashMap;
use std::env;
use tracing::instrument;
use url::Url;

use crate::inference::providers;
use crate::inference::providers::anthropic::AnthropicCredentials;
use crate::inference::providers::azure::AzureCredentials;
#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyCredentials;
#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyProvider;
use crate::inference::providers::fireworks::FireworksCredentials;
use crate::inference::providers::gcp_vertex_gemini::GCPVertexCredentials;
use crate::inference::providers::google_ai_studio_gemini::{
    GoogleAIStudioCredentials, GoogleAIStudioGeminiProvider,
};

use crate::inference::providers::mistral::MistralCredentials;
use crate::inference::providers::openai::OpenAICredentials;
use crate::inference::providers::together::TogetherCredentials;
use crate::inference::providers::vllm::VLLMCredentials;
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        providers::{
            anthropic::AnthropicProvider,
            aws_bedrock::AWSBedrockProvider,
            azure::AzureProvider,
            fireworks::FireworksProvider,
            gcp_vertex_anthropic::GCPVertexAnthropicProvider,
            gcp_vertex_gemini::{GCPServiceAccountCredentials, GCPVertexGeminiProvider},
            mistral::MistralProvider,
            openai::OpenAIProvider,
            provider_trait::InferenceProvider,
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

    pub fn validate(&self) -> Result<(), Error> {
        // Ensure that all providers have credentials
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
    GoogleAIStudioGemini(GoogleAIStudioGeminiProvider),
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
                #[serde(default = "providers::anthropic::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
            #[serde(rename = "aws_bedrock")]
            AWSBedrock {
                model_id: String,
                region: Option<String>,
            },
            Azure {
                deployment_id: String,
                endpoint: Url,
                #[serde(default = "providers::azure::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
            #[serde(rename = "gcp_vertex_anthropic")]
            GCPVertexAnthropic {
                model_id: String,
                location: String,
                project_id: String,
                #[serde(default = "providers::gcp_vertex_gemini::default_api_key_location")]
                credential_location: CredentialLocation,
            },
            #[serde(rename = "gcp_vertex_gemini")]
            GCPVertexGemini {
                model_id: String,
                location: String,
                project_id: String,
                #[serde(default = "providers::gcp_vertex_gemini::default_api_key_location")]
                credential_location: CredentialLocation,
            },
            #[serde(rename = "google_ai_studio_gemini")]
            GoogleAIStudioGemini {
                model_name: String,
                #[serde(default = "providers::google_ai_studio_gemini::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
            Fireworks {
                model_name: String,
                #[serde(default = "providers::fireworks::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
            Mistral {
                model_name: String,
                #[serde(default = "providers::mistral::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
            OpenAI {
                model_name: String,
                api_base: Option<Url>,
                #[serde(default = "providers::openai::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
            Together {
                model_name: String,
                #[serde(default = "providers::together::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
            #[allow(clippy::upper_case_acronyms)]
            VLLM {
                model_name: String,
                api_base: Url,
                #[serde(default = "providers::vllm::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            Dummy {
                model_name: String,
                #[serde(default = "providers::dummy::default_api_key_location")]
                api_key_location: CredentialLocation,
            },
        }

        let helper = ProviderConfigHelper::deserialize(deserializer)?;

        Ok(match helper {
            ProviderConfigHelper::Anthropic {
                model_name,
                api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let api_key = env::var(key_name)
                            .map_err(|_| {
                                serde::de::Error::custom(
                                    Error::new(ErrorDetails::ApiKeyMissing {
                                        provider_name: "Anthropic".to_string(),
                                    })
                                    .to_string(),
                                )
                            })?
                            .into();
                        AnthropicCredentials::Static(api_key)
                    }
                    CredentialLocation::Dynamic(key_name) => {
                        AnthropicCredentials::Dynamic(key_name)
                    }
                    _ => Err(serde::de::Error::custom(
                        "Invalid api_key_location for Anthropic provider".to_string(),
                    ))?,
                };
                ProviderConfig::Anthropic(AnthropicProvider {
                    model_name,
                    credentials,
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
                api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let api_key = env::var(key_name)
                            .map_err(|_| {
                                serde::de::Error::custom(
                                    ErrorDetails::ApiKeyMissing {
                                        provider_name: "Azure".to_string(),
                                    }
                                    .to_string(),
                                )
                            })?
                            .into();
                        AzureCredentials::Static(api_key)
                    }
                    CredentialLocation::Dynamic(key_name) => AzureCredentials::Dynamic(key_name),
                    _ => Err(serde::de::Error::custom(
                        "Invalid api_key_location for Azure provider".to_string(),
                    ))?,
                };
                ProviderConfig::Azure(AzureProvider {
                    deployment_id,
                    endpoint,
                    credentials,
                })
            }
            ProviderConfigHelper::Fireworks {
                model_name,
                api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let api_key = env::var(key_name)
                            .map_err(|_| {
                                serde::de::Error::custom(
                                    ErrorDetails::ApiKeyMissing {
                                        provider_name: "Fireworks".to_string(),
                                    }
                                    .to_string(),
                                )
                            })?
                            .into();
                        FireworksCredentials::Static(api_key)
                    }
                    CredentialLocation::Dynamic(key_name) => {
                        FireworksCredentials::Dynamic(key_name)
                    }
                    _ => Err(serde::de::Error::custom(
                        "Invalid api_key_location for Fireworks provider".to_string(),
                    ))?,
                };
                ProviderConfig::Fireworks(FireworksProvider {
                    model_name,
                    credentials,
                })
            }
            ProviderConfigHelper::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
                credential_location: api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let path = env::var(key_name).map_err(|e| {
                            serde::de::Error::custom(format!(
                                "Failed to load GCP credentials from environment variable: {}",
                                e
                            ))
                        })?;
                        GCPVertexCredentials::Static(
                            GCPServiceAccountCredentials::from_path(path.as_str()).map_err(
                                |e| {
                                    serde::de::Error::custom(format!(
                                        "Failed to load GCP credentials: {}",
                                        e
                                    ))
                                },
                            )?,
                        )
                    }
                    CredentialLocation::Path(path) => GCPVertexCredentials::Static(
                        GCPServiceAccountCredentials::from_path(path.as_str()).map_err(|e| {
                            serde::de::Error::custom(format!(
                                "Failed to load GCP credentials: {}",
                                e
                            ))
                        })?,
                    ),
                    CredentialLocation::Dynamic(key_name) => {
                        GCPVertexCredentials::Dynamic(key_name)
                    }
                    _ => Err(serde::de::Error::custom(
                        "Invalid credential_location for GCPVertexAnthropic provider".to_string(),
                    ))?,
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
                credential_location: api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let path = env::var(key_name).map_err(|e| {
                            serde::de::Error::custom(format!(
                                "Failed to load GCP credentials from environment variable: {}",
                                e
                            ))
                        })?;
                        GCPVertexCredentials::Static(
                            GCPServiceAccountCredentials::from_path(path.as_str()).map_err(
                                |e| {
                                    serde::de::Error::custom(format!(
                                        "Failed to load GCP credentials: {}",
                                        e
                                    ))
                                },
                            )?,
                        )
                    }
                    CredentialLocation::Path(path) => GCPVertexCredentials::Static(
                        GCPServiceAccountCredentials::from_path(path.as_str()).map_err(|e| {
                            serde::de::Error::custom(format!(
                                "Failed to load GCP credentials: {}",
                                e
                            ))
                        })?,
                    ),
                    CredentialLocation::Dynamic(key_name) => {
                        GCPVertexCredentials::Dynamic(key_name)
                    }
                    _ => Err(serde::de::Error::custom(
                        "Invalid credential_location for GCPVertexGemini provider".to_string(),
                    ))?,
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
            ProviderConfigHelper::GoogleAIStudioGemini {
                model_name,
                api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => GoogleAIStudioCredentials::Static(
                        env::var(key_name)
                            .map_err(|_| {
                                serde::de::Error::custom(
                                    ErrorDetails::ApiKeyMissing {
                                        provider_name: "Google AI Studio Gemini".to_string(),
                                    }
                                    .to_string(),
                                )
                            })?
                            .into(),
                    ),
                    CredentialLocation::Dynamic(key_name) => {
                        GoogleAIStudioCredentials::Dynamic(key_name)
                    }
                    _ => Err(serde::de::Error::custom(
                        "Invalid api_key_location for Google AI Studio Gemini provider".to_string(),
                    ))?,
                };
                let request_url = Url::parse(&format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                )).map_err(|e| D::Error::custom(format!("Failed to parse request URL: {}", e)))?;
                let streaming_request_url = Url::parse(&format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse",
                )).map_err(|e| D::Error::custom(format!("Failed to parse streaming request URL: {}", e)))?;
                ProviderConfig::GoogleAIStudioGemini(GoogleAIStudioGeminiProvider {
                    request_url,
                    streaming_request_url,
                    credentials,
                })
            }
            ProviderConfigHelper::Mistral {
                model_name,
                api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let api_key = env::var(key_name)
                            .map_err(|_| {
                                serde::de::Error::custom(
                                    ErrorDetails::ApiKeyMissing {
                                        provider_name: "Mistral".to_string(),
                                    }
                                    .to_string(),
                                )
                            })?
                            .into();
                        MistralCredentials::Static(api_key)
                    }
                    CredentialLocation::Dynamic(key_name) => MistralCredentials::Dynamic(key_name),
                    _ => Err(serde::de::Error::custom(
                        "Invalid api_key_location for Mistral provider".to_string(),
                    ))?,
                };
                ProviderConfig::Mistral(MistralProvider {
                    model_name,
                    credentials,
                })
            }
            ProviderConfigHelper::OpenAI {
                model_name,
                api_base,
                api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let api_key = env::var(key_name)
                            .map_err(|_| {
                                serde::de::Error::custom(
                                    ErrorDetails::ApiKeyMissing {
                                        provider_name: "OpenAI".to_string(),
                                    }
                                    .to_string(),
                                )
                            })?
                            .into();
                        OpenAICredentials::Static(api_key)
                    }
                    CredentialLocation::Dynamic(key_name) => OpenAICredentials::Dynamic(key_name),
                    _ => Err(serde::de::Error::custom(
                        "Invalid api_key_location for OpenAI provider".to_string(),
                    ))?,
                };
                ProviderConfig::OpenAI(OpenAIProvider {
                    model_name,
                    api_base,
                    credentials,
                })
            }
            ProviderConfigHelper::Together {
                model_name,
                api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let api_key = env::var(key_name)
                            .map_err(|_| {
                                serde::de::Error::custom(
                                    ErrorDetails::ApiKeyMissing {
                                        provider_name: "Together".to_string(),
                                    }
                                    .to_string(),
                                )
                            })?
                            .into();
                        TogetherCredentials::Static(api_key)
                    }
                    CredentialLocation::Dynamic(key_name) => TogetherCredentials::Dynamic(key_name),
                    _ => Err(serde::de::Error::custom(
                        "Invalid api_key_location for Together provider".to_string(),
                    ))?,
                };
                ProviderConfig::Together(TogetherProvider {
                    model_name,
                    credentials,
                })
            }
            ProviderConfigHelper::VLLM {
                model_name,
                api_base,
                api_key_location,
            } => {
                let credentials = match api_key_location {
                    CredentialLocation::Env(key_name) => {
                        let api_key = env::var(key_name)
                            .map_err(|_| {
                                serde::de::Error::custom(
                                    ErrorDetails::ApiKeyMissing {
                                        provider_name: "VLLM".to_string(),
                                    }
                                    .to_string(),
                                )
                            })?
                            .into();
                        VLLMCredentials::Static(api_key)
                    }
                    CredentialLocation::Dynamic(key_name) => VLLMCredentials::Dynamic(key_name),
                    CredentialLocation::None => VLLMCredentials::None,
                    _ => Err(serde::de::Error::custom(
                        "Invalid api_key_location for VLLM provider".to_string(),
                    ))?,
                };
                ProviderConfig::VLLM(VLLMProvider {
                    model_name,
                    api_base,
                    credentials,
                })
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfigHelper::Dummy {
                model_name,
                api_key_location,
            } => match api_key_location {
                CredentialLocation::Dynamic(key_name) => ProviderConfig::Dummy(DummyProvider {
                    model_name,
                    credentials: DummyCredentials::Dynamic(key_name),
                }),
                CredentialLocation::None => ProviderConfig::Dummy(DummyProvider {
                    model_name,
                    credentials: DummyCredentials::None,
                }),
                _ => Err(serde::de::Error::custom(
                    "Invalid api_key_location for Dummy provider".to_string(),
                ))?,
            },
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
            ProviderConfig::VLLM(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider.infer_stream(request, client, api_keys).await
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::inference::{
        providers::dummy::{
            DummyCredentials, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW,
            DUMMY_INFER_USAGE, DUMMY_STREAMING_RESPONSE,
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
}
