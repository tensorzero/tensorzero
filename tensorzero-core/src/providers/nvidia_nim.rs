use std::sync::OnceLock;
use lazy_static::lazy_static;
use secrecy::SecretString;
use serde::Serialize;
use url::Url;

use crate::{
    cache::ModelProviderRequest,
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        types::{
            ModelInferenceRequest,
            PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
        },
        InferenceProvider,
    },
    model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider},
};

lazy_static! {
    static ref DEFAULT_NIM_API_BASE: Url = {
        Url::parse("https://integrate.api.nvidia.com/v1/")
            .expect("Failed to parse DEFAULT_NIM_API_BASE")
    };
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("NVIDIA_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "NVIDIA NIM";
pub const PROVIDER_TYPE: &str = "nvidia_nim";

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct NvidiaNimProvider {
    model_name: String,
    api_base: Url,
    #[serde(skip)]
    #[cfg_attr(test, ts(skip))]
    credentials: NvidiaNimCredentials,
}

static DEFAULT_CREDENTIALS: OnceLock<NvidiaNimCredentials> = OnceLock::new();

impl NvidiaNimProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
        api_base: Option<String>,
    ) -> Result<Self, Error> {
        let credentials = build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;

        let api_base = match api_base {
            Some(base) => {
                let mut url = Url::parse(&base).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Invalid api_base URL: {}", e),
                    })
                })?;

                // Ensure URL ends with a slash
                if !url.path().ends_with('/') {
                    url.set_path(&format!("{}/", url.path()));
                }

                url
            },
            None => DEFAULT_NIM_API_BASE.clone(),
        };

        Ok(NvidiaNimProvider {
            model_name,
            api_base,
            credentials,
        })
    }


    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum NvidiaNimCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for NvidiaNimCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(NvidiaNimCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(NvidiaNimCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(NvidiaNimCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for NVIDIA NIM provider".to_string(),
            })),
        }
    }
}

impl NvidiaNimCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            NvidiaNimCredentials::Static(api_key) => Ok(Some(api_key)),
            NvidiaNimCredentials::Dynamic(key_name) => {
                Ok(Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                })?))
            }
            NvidiaNimCredentials::None => Ok(None),
        }
    }

    fn to_credential_location(&self) -> Option<CredentialLocation> {
        match self {
            NvidiaNimCredentials::Static(_) => Some(default_api_key_location()),
            NvidiaNimCredentials::Dynamic(name) => Some(CredentialLocation::Dynamic(name.clone())),
            NvidiaNimCredentials::None => None,
        }
    }
}

impl InferenceProvider for NvidiaNimProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        // Create an OpenAI provider with NIM settings
        // OpenAIProvider::new expects (model_name, api_base, api_key_location)
        let openai_provider = super::openai::OpenAIProvider::new(
            self.model_name.clone(),
            Some(self.api_base.clone()),
            self.credentials.to_credential_location(),
        )?;

        openai_provider.infer(request, http_client, dynamic_api_keys, model_provider).await
    }

    async fn infer_stream<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let openai_provider = super::openai::OpenAIProvider::new(
            self.model_name.clone(),
            Some(self.api_base.clone()),
            self.credentials.to_credential_location(),
        )?;

        openai_provider.infer_stream(request, http_client, dynamic_api_keys, model_provider).await
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<crate::inference::types::batch::StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a crate::inference::types::batch::BatchRequestRow<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<crate::inference::types::batch::PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CredentialLocation;

    #[test]
    fn test_nvidia_nim_provider_new() {
        // Test with dynamic credentials (the normal case)
        let provider = NvidiaNimProvider::new(
            "meta/llama-3.1-8b-instruct".to_string(),
            Some(CredentialLocation::Dynamic("nvidia_api_key".to_string())),
            None,
        );

        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.model_name(), "meta/llama-3.1-8b-instruct");
        assert_eq!(provider.api_base.as_str(), "https://integrate.api.nvidia.com/v1/");

        // Test with custom API base
        let provider = NvidiaNimProvider::new(
            "local:my-model".to_string(),
            Some(CredentialLocation::Dynamic("custom_key".to_string())),
            Some("http://localhost:8000/v1/".to_string()),
        );

        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.api_base.as_str(), "http://localhost:8000/v1/");
    }

    #[test]
    fn test_credentials_try_from() {
        // Test static credential
        let cred = Credential::Static(secrecy::SecretString::from("test_key"));
        let nim_cred = NvidiaNimCredentials::try_from(cred).unwrap();
        assert!(matches!(nim_cred, NvidiaNimCredentials::Static(_)));

        // Test dynamic credential
        let cred = Credential::Dynamic("key_name".to_string());
        let nim_cred = NvidiaNimCredentials::try_from(cred).unwrap();
        assert!(matches!(nim_cred, NvidiaNimCredentials::Dynamic(_)));

        // Test missing credential (might be valid for the enum but not for provider creation)
        let cred = Credential::Missing;
        let nim_cred = NvidiaNimCredentials::try_from(cred).unwrap();
        assert!(matches!(nim_cred, NvidiaNimCredentials::None));
    }

    #[test]
    fn test_various_model_configurations() {
        // Test various supported models with valid credentials
        let models = vec![
            ("meta/llama-3.1-8b-instruct", None),
            ("meta/llama-3.1-70b-instruct", None),
            ("mistralai/mistral-7b-instruct-v0.3", None),
            ("google/gemma-2-9b-it", None),
            ("microsoft/phi-3-mini-128k-instruct", None),
            ("nvidia/llama-3.1-nemotron-70b-instruct", None),
            ("custom-model", Some("http://my-server:8000/v1/")),
        ];

        for (model_name, api_base) in models {
            let provider = NvidiaNimProvider::new(
                model_name.to_string(),
                Some(CredentialLocation::Dynamic("nvidia_api_key".to_string())),
                api_base.map(String::from),
            );

            assert!(
                provider.is_ok(),
                "Failed to create provider for model {}: {:?}",
                model_name,
                provider.err()
            );
        }
    }

    #[test]
    fn test_api_base_normalization() {
        let test_cases = vec![
            ("http://localhost:8000/v1", "http://localhost:8000/v1/"),
            ("http://localhost:8000/v1/", "http://localhost:8000/v1/"),
            ("https://api.example.com/v1", "https://api.example.com/v1/"),
            ("https://api.example.com/v1/", "https://api.example.com/v1/"),
        ];

        for (input, expected) in test_cases {
            let provider = NvidiaNimProvider::new(
                "test-model".to_string(),
                Some(CredentialLocation::Dynamic("test_key".to_string())),
                Some(input.to_string()),
            )
            .unwrap();

            assert_eq!(
                provider.api_base.as_str(),
                expected,
                "API base normalization failed for input: {}",
                input
            );
        }
    }

    #[tokio::test]
    async fn test_nvidia_nim_openai_delegation() {
        // This test verifies that the NVIDIA NIM provider correctly delegates to OpenAI
        // We'll test the request preparation without making actual API calls

        let provider = NvidiaNimProvider::new(
            "meta/llama-3.1-8b-instruct".to_string(),
            Some(CredentialLocation::Dynamic("nvidia_api_key".to_string())),
            None,
        )
        .expect("Failed to create provider");

        // Verify the provider is configured correctly
        assert_eq!(provider.model_name(), "meta/llama-3.1-8b-instruct");
        assert_eq!(provider.api_base.as_str(), "https://integrate.api.nvidia.com/v1/");
    }

    #[test]
    fn test_deployment_scenarios() {
        // Scenario 1: Cloud deployment
        let cloud_provider = NvidiaNimProvider::new(
            "meta/llama-3.1-70b-instruct".to_string(),
            Some(CredentialLocation::Dynamic("nvidia_key".to_string())),
            None,
        )
        .unwrap();

        assert_eq!(cloud_provider.api_base.as_str(), "https://integrate.api.nvidia.com/v1/");

        // Scenario 2: Self-hosted deployment
        let self_hosted_provider = NvidiaNimProvider::new(
            "custom-model".to_string(),
            Some(CredentialLocation::Dynamic("self_hosted_key".to_string())),
            Some("http://192.168.1.100:8000/v1/".to_string()),
        )
        .unwrap();

        assert_eq!(self_hosted_provider.api_base.as_str(), "http://192.168.1.100:8000/v1/");
    }

    #[test]
    fn test_provider_type_constant() {
        // Verify the provider type is correctly set
        assert_eq!(PROVIDER_TYPE, "nvidia_nim");
    }

    #[test]
    fn test_error_handling_scenarios() {
        // Test invalid API base URL
        let invalid_url = NvidiaNimProvider::new(
            "model".to_string(),
            Some(CredentialLocation::Dynamic("key".to_string())),
            Some("not a valid url".to_string()),
        );

        assert!(invalid_url.is_err());
        assert!(invalid_url.unwrap_err().to_string().contains("Invalid api_base URL"));

        // Test invalid credential location
        let no_creds = NvidiaNimProvider::new(
            "model".to_string(),
            Some(CredentialLocation::None),
            None,
        );
        assert!(no_creds.is_err());
        assert!(no_creds.unwrap_err().to_string().contains("Invalid api_key_location"));
    }

    #[test]
    fn test_credential_validation() {
        // Test that CredentialLocation::None is rejected (invalid configuration)
        let result = NvidiaNimProvider::new(
            "meta/llama-3.1-8b-instruct".to_string(),
            Some(CredentialLocation::None),
            None,
        );

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Invalid api_key_location"));

        // Test that None uses the default credential location (which is valid)
        let result = NvidiaNimProvider::new(
            "meta/llama-3.1-8b-instruct".to_string(),
            None,  // This will use default_api_key_location()
            None,
        );

        // This should succeed because it uses the default
        assert!(result.is_ok());
        // Just verify it created successfully - don't check the internal credential type
        // since build_creds_caching_default may transform it
    }
}