use lazy_static::lazy_static;
use reqwest::Client;
use secrecy::SecretString;
use serde::de::Error as SerdeError;
use std::collections::HashMap;
use std::sync::Arc;
use std::{env, fs};
use strum::VariantNames;
#[allow(unused_imports)]
use tracing::{span, warn, Instrument, Level};
use url::Url;

use crate::cache::{cache_lookup, ModelProviderRequest};
use crate::endpoints::inference::InferenceClients;
#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyProvider;
use crate::inference::providers::google_ai_studio_gemini::GoogleAIStudioGeminiProvider;

use crate::inference::providers::hyperbolic::HyperbolicProvider;
use crate::inference::providers::sglang::SGLangProvider;
use crate::inference::providers::tgi::TGIProvider;
use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchModelInferenceResponse,
    StartBatchProviderInferenceResponse,
};
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
    pub routing: Vec<Arc<str>>, // [provider name A, provider name B, ...]
    pub providers: HashMap<Arc<str>, ProviderConfig>, // provider name => provider config
}

impl ModelConfig {
    pub async fn infer<'request>(
        &self,
        request: &'request ModelInferenceRequest<'request>,
        client: &'request InferenceClients<'request>,
        api_keys: &'request InferenceCredentials,
        model_name: &'request str,
    ) -> Result<ModelInferenceResponse, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let cache_lookup = cache_lookup(
                &client.clickhouse_connection_info,
                ModelProviderRequest {
                    request,
                    model_name,
                    provider_name,
                },
            )
            .await
            .ok()
            .flatten();
            if let Some(cache_lookup) = cache_lookup {
                return Ok(cache_lookup);
            }
            let provider_config = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                })
            })?;
            let response = provider_config
                .infer(request, client.http_client, api_keys)
                .instrument(span!(
                    Level::INFO,
                    "infer",
                    provider_name = &**provider_name
                ))
                .await;
            match response {
                Ok(response) => {
                    let model_inference_response =
                        ModelInferenceResponse::new(response, provider_name.clone());
                    return Ok(model_inference_response);
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        let err = Error::new(ErrorDetails::ModelProvidersExhausted { provider_errors });
        Err(err)
    }

    pub async fn infer_stream<'request>(
        &self,
        request: &'request ModelInferenceRequest<'request>,
        client: &'request Client,
        api_keys: &'request InferenceCredentials,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
            Arc<str>,
        ),
        Error,
    > {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider_config = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                })
            })?;
            let response = provider_config
                .infer_stream(request, client, api_keys)
                .instrument(span!(
                    Level::INFO,
                    "infer_stream",
                    provider_name = &**provider_name
                ))
                .await;
            match response {
                Ok(response) => {
                    let (chunk, stream, raw_request) = response;
                    return Ok((chunk, stream, raw_request, provider_name.clone()));
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
    ) -> Result<StartBatchModelInferenceResponse<'a>, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider_config = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                })
            })?;
            let response = provider_config
                .start_batch_inference(requests, client, api_keys)
                .instrument(span!(
                    Level::INFO,
                    "start_batch_inference",
                    provider_name = &**provider_name
                ))
                .await;
            match response {
                Ok(response) => {
                    return Ok(StartBatchModelInferenceResponse::new(
                        response,
                        provider_name,
                    ));
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

#[derive(Debug)]
pub enum ProviderConfig {
    Anthropic(AnthropicProvider),
    AWSBedrock(AWSBedrockProvider),
    Azure(AzureProvider),
    Fireworks(FireworksProvider),
    GCPVertexAnthropic(GCPVertexAnthropicProvider),
    GCPVertexGemini(GCPVertexGeminiProvider),
    GoogleAIStudioGemini(GoogleAIStudioGeminiProvider),
    Hyperbolic(HyperbolicProvider),
    Mistral(MistralProvider),
    OpenAI(OpenAIProvider),
    Together(TogetherProvider),
    VLLM(VLLMProvider),
    XAI(XAIProvider),
    TGI(TGIProvider),
    SGLang(SGLangProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

/// Helper struct for deserializing the ProviderConfig.
/// This is necessary because we want to load environment variables as we deserialize
/// and we need to be able to deserialize the correct one based on the "type" field.
/// Use the ProviderConfig struct for all post-initialization logic.
#[derive(Deserialize, VariantNames)]
#[strum(serialize_all = "lowercase")]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
enum ProviderConfigHelper {
    Anthropic {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "aws_bedrock")]
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
    #[strum(serialize = "gcp_vertex_anthropic")]
    #[serde(rename = "gcp_vertex_anthropic")]
    GCPVertexAnthropic {
        model_id: String,
        location: String,
        project_id: String,
        credential_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "gcp_vertex_gemini")]
    #[serde(rename = "gcp_vertex_gemini")]
    GCPVertexGemini {
        model_id: String,
        location: String,
        project_id: String,
        credential_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "google_ai_studio_gemini")]
    #[serde(rename = "google_ai_studio_gemini")]
    GoogleAIStudioGemini {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    Hyperbolic {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "fireworks")]
    #[serde(rename = "fireworks")]
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
    #[allow(clippy::upper_case_acronyms)]
    TGI {
        api_base: Url,
        api_key_location: Option<CredentialLocation>,
    },
    #[allow(clippy::upper_case_acronyms)]
    SGLang {
        model_name: String,
        api_base: Url,
        api_key_location: Option<CredentialLocation>,
    },
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
}

impl<'de> Deserialize<'de> for ProviderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
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
            ProviderConfigHelper::Hyperbolic {
                model_name,
                api_key_location,
            } => ProviderConfig::Hyperbolic(
                HyperbolicProvider::new(model_name, api_key_location)
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
            ProviderConfigHelper::SGLang {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::SGLang(
                SGLangProvider::new(model_name, api_base, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::TGI {
                api_base,
                api_key_location,
            } => ProviderConfig::TGI(
                TGIProvider::new(api_base, api_key_location)
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
            ProviderConfig::Hyperbolic(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::Mistral(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::OpenAI(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::Together(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::SGLang(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::VLLM(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::XAI(provider) => provider.infer(request, client, api_keys).await,
            ProviderConfig::TGI(provider) => provider.infer(request, client, api_keys).await,
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
            ProviderConfig::Hyperbolic(provider) => {
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
            ProviderConfig::SGLang(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::XAI(provider) => provider.infer_stream(request, client, api_keys).await,
            ProviderConfig::VLLM(provider) => {
                provider.infer_stream(request, client, api_keys).await
            }
            ProviderConfig::TGI(provider) => provider.infer_stream(request, client, api_keys).await,
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
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
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
            ProviderConfig::Hyperbolic(provider) => {
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
            ProviderConfig::SGLang(provider) => {
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
            ProviderConfig::TGI(provider) => {
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

    pub async fn poll_batch_inference<'a>(
        &self,
        batch_request: &'a BatchRequestRow<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
        }
    }
}

pub enum CredentialLocation {
    /// Environment variable containing the actual credential
    Env(String),
    /// Environment variable containing the path to a credential file
    PathFromEnv(String),
    /// For dynamic credential resolution
    Dynamic(String),
    /// Direct path to a credential file
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
        } else if let Some(inner) = s.strip_prefix("path_from_env::") {
            Ok(CredentialLocation::PathFromEnv(inner.to_string()))
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

pub enum Credential {
    Static(SecretString),
    FileContents(SecretString),
    Dynamic(String),
    None,
    #[cfg(any(test, feature = "e2e_tests"))]
    Missing,
}

impl TryFrom<(CredentialLocation, &str)> for Credential {
    type Error = Error;
    #[allow(unused_variables)]
    fn try_from(
        (location, provider_type): (CredentialLocation, &str),
    ) -> Result<Self, Self::Error> {
        match location {
            CredentialLocation::Env(key_name) => match env::var(key_name) {
                Ok(value) => Ok(Credential::Static(SecretString::from(value))),
                Err(_) => {
                    #[cfg(any(test, feature = "e2e_tests"))]
                    {
                        warn!(
                            "You are missing the credentials required for a model provider of type {}, so the associated tests will likely fail.",
                            provider_type
                        );
                        Ok(Credential::Missing)
                    }
                    #[cfg(not(any(test, feature = "e2e_tests")))]
                    {
                        Err(Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: provider_type.to_string(),
                        }))
                    }
                }
            },
            CredentialLocation::PathFromEnv(env_key) => {
                // First get the path from environment variable
                let path = match env::var(&env_key) {
                    Ok(path) => path,
                    Err(_) => {
                        #[cfg(any(test, feature = "e2e_tests"))]
                        {
                            warn!(
                                "Environment variable {} is required for a model provider of type {} but is missing, so the associated tests will likely fail.",
                                env_key, provider_type
                            );
                            return Ok(Credential::Missing);
                        }
                        #[cfg(not(any(test, feature = "e2e_tests")))]
                        {
                            return Err(Error::new(ErrorDetails::ApiKeyMissing {
                                provider_name: format!(
                                    "{}: Environment variable {} for credentials path is missing",
                                    provider_type, env_key
                                ),
                            }));
                        }
                    }
                };
                // Then read the file contents
                match fs::read_to_string(path) {
                    Ok(contents) => Ok(Credential::FileContents(SecretString::from(contents))),
                    Err(e) => {
                        #[cfg(any(test, feature = "e2e_tests"))]
                        {
                            warn!(
                                "Failed to read credentials file for a model provider of type {}, so the associated tests will likely fail: {}",
                                provider_type, e
                            );
                            Ok(Credential::Missing)
                        }
                        #[cfg(not(any(test, feature = "e2e_tests")))]
                        {
                            Err(Error::new(ErrorDetails::ApiKeyMissing {
                                provider_name: format!(
                                    "{}: Failed to read credentials file - {}",
                                    provider_type, e
                                ),
                            }))
                        }
                    }
                }
            }
            CredentialLocation::Path(path) => match fs::read_to_string(path) {
                Ok(contents) => Ok(Credential::FileContents(SecretString::from(contents))),
                Err(e) => {
                    #[cfg(any(test, feature = "e2e_tests"))]
                    {
                        warn!(
                            "Failed to read credentials file for a model provider of type {}, so the associated tests will likely fail: {}",
                            provider_type, e
                        );
                        Ok(Credential::Missing)
                    }
                    #[cfg(not(any(test, feature = "e2e_tests")))]
                    {
                        Err(Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: format!(
                                "{}: Failed to read credentials file - {}",
                                provider_type, e
                            ),
                        }))
                    }
                }
            },
            CredentialLocation::Dynamic(key_name) => Ok(Credential::Dynamic(key_name.clone())),
            CredentialLocation::None => Ok(Credential::None),
        }
    }
}

lazy_static! {
    static ref RESERVED_MODEL_PREFIXES: Vec<String> = ProviderConfigHelper::VARIANTS
        .iter()
        .map(|&v| format!("{}::", v))
        .collect();
}

const SHORTHAND_MODEL_PREFIXES: &[&str] = &[
    "anthropic::",
    "fireworks::",
    "google_ai_studio_gemini::",
    "hyperbolic::",
    "mistral::",
    "openai::",
    "together::",
    "xai::",
    "dummy::",
];

#[derive(Debug, Default, Deserialize)]
#[serde(try_from = "HashMap<Arc<str>, ModelConfig>")]
pub struct ModelTable(HashMap<Arc<str>, ModelConfig>);

impl TryFrom<HashMap<Arc<str>, ModelConfig>> for ModelTable {
    type Error = String;

    fn try_from(map: HashMap<Arc<str>, ModelConfig>) -> Result<Self, Self::Error> {
        for key in map.keys() {
            if RESERVED_MODEL_PREFIXES
                .iter()
                .any(|name| key.starts_with(name))
            {
                return Err(format!("Model name '{}' contains a reserved prefix", key));
            }
        }
        Ok(ModelTable(map))
    }
}

impl std::ops::Deref for ModelTable {
    type Target = HashMap<Arc<str>, ModelConfig>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ModelTable {
    /// Check that a model name is valid
    /// This is either true because it's in the table, or because it's a valid shorthand name
    /// In the latter case, we actually create a new model config in the table corresponding to the shorthand
    pub fn validate_or_create(&mut self, key: &str) -> Result<(), Error> {
        // Try direct lookup (if it's blacklisted, it's not in the table)
        // If it's shorthand and already in the table, it's valid
        if let Some(model_config) = self.0.get(key) {
            model_config.validate()?;
            return Ok(());
        }

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
            self.0.insert(key.into(), model_config);
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
        "hyperbolic" => ProviderConfig::Hyperbolic(HyperbolicProvider::new(model_name, None)?),
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
        routing: vec![provider_type.to_string().into()],
        providers: HashMap::from([(provider_type.to_string().into(), provider_config)]),
    })
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
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let model_config = ModelConfig {
            routing: vec!["good_provider".into()],
            providers: HashMap::from([("good_provider".into(), good_provider_config)]),
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
        let model_name = "test model";
        let response = model_config
            .infer(&request, &Client::new(), &api_keys, model_name)
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
        assert_eq!(&*response.model_provider_name, "good_provider");

        // Try inferring the bad model
        let model_config = ModelConfig {
            routing: vec!["error".into()],
            providers: HashMap::from([("error".into(), bad_provider_config)]),
        };
        let response = model_config
            .infer(&request, &Client::new(), &api_keys, model_name)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "error".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Error sending request to Dummy provider.".to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
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
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
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
            routing: vec![
                "error_provider".to_string().into(),
                "good_provider".to_string().into(),
            ],
            providers: HashMap::from([
                ("error_provider".to_string().into(), bad_provider_config),
                ("good_provider".to_string().into(), good_provider_config),
            ]),
        };

        let model_name = "test model";
        let response = model_config
            .infer(&request, &Client::new(), &api_keys, model_name)
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
        assert_eq!(&*response.model_provider_name, "good_provider");
    }

    #[tokio::test]
    async fn test_model_config_infer_stream_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
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
            routing: vec!["good_provider".to_string().into()],
            providers: HashMap::from([("good_provider".to_string().into(), good_provider_config)]),
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
        assert_eq!(&*model_provider_name, "good_provider");
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
            routing: vec!["error".to_string().into()],
            providers: HashMap::from([("error".to_string().into(), bad_provider_config)]),
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
                        message: "Error sending request to Dummy provider.".to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
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
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
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
            routing: vec!["error_provider".into(), "good_provider".into()],
            providers: HashMap::from([
                ("error_provider".into(), bad_provider_config),
                ("good_provider".into(), good_provider_config),
            ]),
        };
        let (initial_chunk, stream, raw_request, model_provider_name) = model_config
            .infer_stream(&request, &Client::new(), &api_keys)
            .await
            .unwrap();
        assert_eq!(&*model_provider_name, "good_provider");
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
            model_name: "test_key".into(),
            credentials: DummyCredentials::Dynamic("TEST_KEY".to_string()),
        });
        let model_config = ModelConfig {
            routing: vec!["model".into()],
            providers: HashMap::from([("model".into(), provider_config)]),
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
        let model_name = "test model";
        let error = model_config
            .infer(&request, &Client::new(), &api_keys, model_name)
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
            .infer(&request, &Client::new(), &api_keys, model_name)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "model".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Invalid API key for Dummy provider".to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                    }
                    .into()
                )])
            }
            .into()
        );

        let provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "test_key".into(),
            credentials: DummyCredentials::Dynamic("TEST_KEY".to_string()),
        });
        let model_config = ModelConfig {
            routing: vec!["model".to_string().into()],
            providers: HashMap::from([("model".to_string().into(), provider_config)]),
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
            .infer(&request, &Client::new(), &api_keys, model_name)
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
            .infer(&request, &Client::new(), &api_keys, model_name)
            .await
            .unwrap();
        assert_eq!(
            response.output,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
    }

    #[test]
    fn test_validate_or_create_model_config() {
        let mut model_table = ModelTable::default();
        // Test that we can get or create a model config
        model_table.validate_or_create("dummy::gpt-4o").unwrap();
        assert_eq!(model_table.len(), 1);
        let model_config = model_table.get("dummy::gpt-4o").unwrap();
        assert_eq!(model_config.routing, vec!["dummy".into()]);
        let provider_config = model_config.providers.get("dummy").unwrap();
        match provider_config {
            ProviderConfig::Dummy(provider) => assert_eq!(&*provider.model_name, "gpt-4o"),
            _ => panic!("Expected Dummy provider"),
        }

        // Test that it is idempotent
        model_table.validate_or_create("dummy::gpt-4o").unwrap();
        assert_eq!(model_table.len(), 1);
        let model_config = model_table.get("dummy::gpt-4o").unwrap();
        assert_eq!(model_config.routing, vec!["dummy".into()]);
        let provider_config = model_config.providers.get("dummy").unwrap();
        match provider_config {
            ProviderConfig::Dummy(provider) => assert_eq!(&*provider.model_name, "gpt-4o"),
            _ => panic!("Expected Dummy provider"),
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
        let anthropic_provider_config =
            ProviderConfig::Anthropic(AnthropicProvider::new("claude".to_string(), None).unwrap());
        let anthropic_model_config = ModelConfig {
            routing: vec!["anthropic".into()],
            providers: HashMap::from([("anthropic".into(), anthropic_provider_config)]),
        };
        let mut model_table: ModelTable =
            HashMap::from([("claude".into(), anthropic_model_config)])
                .try_into()
                .unwrap();

        model_table.validate_or_create("dummy::claude").unwrap();
    }

    #[test]
    fn test_shorthand_prefixes_subset_of_reserved() {
        for &shorthand in SHORTHAND_MODEL_PREFIXES {
            assert!(
                RESERVED_MODEL_PREFIXES.contains(&shorthand.to_string()),
                "Shorthand prefix '{}' is not in RESERVED_MODEL_PREFIXES",
                shorthand
            );
        }
    }
}
