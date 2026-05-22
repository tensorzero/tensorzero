//! AWS Bedrock model provider using direct HTTP calls.
//!
//! Uses the Bedrock runtime endpoint for for Converse API,

use aws_types::region::Region;
use serde::Serialize;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::InferenceProvider;
use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
};
use crate::inference::types::{
    ApiType, ModelInferenceRequest, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse,
};
use crate::model::{
    CredentialLocation, CredentialLocationOrHardcoded, ModelProviderRequestInfo,
    ProviderInferenceRequest,
};

use super::aws_common::{
    AWSBedrockCredentials, AWSEndpointUrl, AWSRegion, parse_aws_region,
    resolve_request_credentials, sign_request, warn_if_credential_exfiltration_risk,
};

pub const PROVIDER_TYPE: &str = "aws_bedrock";

/// AWS Bedrock provider using direct HTTP calls.
#[derive(ts_rs::TS, Debug, Serialize)]
#[ts(export)]
pub struct AWSBedrockProvider {
    model_id: String,
    #[serde(skip)]
    region: AWSRegion,
    #[serde(skip)]
    endpoint_url: Option<AWSEndpointUrl>,
    #[serde(skip)]
    credentials: AWSBedrockCredentials,
}

/// Build AWS Bedrock provider configuration from config fields.
///
/// Handles region resolution (including deprecated `allow_auto_detect_region`),
/// endpoint URL parsing, and authentication (api_key or IAM credentials).
///
/// Returns `(region, endpoint_url, auth)`.
pub async fn build_aws_bedrock_provider_config(
    region: Option<CredentialLocationOrHardcoded>,
    allow_auto_detect_region: bool,
    endpoint_url: Option<CredentialLocationOrHardcoded>,
    api_key: Option<CredentialLocation>,
    access_key_id: Option<CredentialLocation>,
    secret_access_key: Option<CredentialLocation>,
    session_token: Option<CredentialLocation>,
) -> Result<(AWSRegion, Option<AWSEndpointUrl>, AWSBedrockCredentials), Error> {
    let aws_region = parse_aws_region(region, allow_auto_detect_region, PROVIDER_TYPE)?;

    let endpoint_url = endpoint_url
        .map(|loc| AWSEndpointUrl::from_credential_location(loc, PROVIDER_TYPE))
        .transpose()?
        .flatten();

    // Convert credential fields to AWSBedrockCredentials (handles api_key for bearer auth)
    let (auth, resolved_sdk_region) = AWSBedrockCredentials::from_fields(
        api_key,
        access_key_id,
        secret_access_key,
        session_token,
        &aws_region,
        PROVIDER_TYPE,
    )
    .await?;

    // For bearer auth with region = "sdk", use the resolved region
    let aws_region = match resolved_sdk_region {
        Some(resolved) => AWSRegion::Static(resolved),
        None => aws_region,
    };

    // Warn about credential exfiltration risk with dynamic endpoint
    let has_dynamic_endpoint = endpoint_url
        .as_ref()
        .is_some_and(|ep| matches!(ep, AWSEndpointUrl::Dynamic(_)));
    match &auth {
        AWSBedrockCredentials::IAM { credentials, .. } => {
            warn_if_credential_exfiltration_risk(&endpoint_url, credentials, PROVIDER_TYPE);
        }
        AWSBedrockCredentials::ApiKey(_) if has_dynamic_endpoint => {
            // Static API key with dynamic endpoint is also a risk
            tracing::warn!(
                "You configured a dynamic `endpoint_url` with a static API key for `{PROVIDER_TYPE}`. \
                 A malicious client could exfiltrate your API key via a malicious endpoint."
            );
        }
        AWSBedrockCredentials::ApiKey(_) | AWSBedrockCredentials::DynamicApiKey(_) => {
            // Static endpoint or dynamic API key - no exfiltration risk
        }
    }

    Ok((aws_region, endpoint_url, auth))
}

impl AWSBedrockProvider {
    pub fn new(
        model_id: String,
        region: AWSRegion,
        endpoint_url: Option<AWSEndpointUrl>,
        credentials: AWSBedrockCredentials,
    ) -> Self {
        Self {
            model_id,
            region,
            endpoint_url,
            credentials,
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the base URL for AWS Bedrock requests.
    pub(super) fn get_base_url(
        &self,
        dynamic_api_keys: &InferenceCredentials,
        api_type: ApiType,
    ) -> Result<String, Error> {
        if let Some(endpoint_url) = &self.endpoint_url {
            let url = endpoint_url.resolve(dynamic_api_keys)?;
            Ok(url.to_string().trim_end_matches('/').to_string())
        } else {
            let region = self.get_region(dynamic_api_keys, api_type)?;
            Ok(format!(
                "https://bedrock-runtime.{}.amazonaws.com",
                region.as_ref()
            ))
        }
    }

    /// Get the region for this request.
    fn get_region(
        &self,
        dynamic_api_keys: &InferenceCredentials,
        api_type: ApiType,
    ) -> Result<Region, Error> {
        // Extract SDK config from IAM credentials if available
        let sdk_config = match &self.credentials {
            AWSBedrockCredentials::IAM { sdk_config, .. } => Some(sdk_config.as_ref()),
            _ => None,
        };
        self.region
            .resolve_with_sdk_config(dynamic_api_keys, sdk_config, PROVIDER_TYPE, api_type)
    }

    pub(super) async fn build_request_headers(
        &self,
        dynamic_api_keys: &InferenceCredentials,
        url: &str,
        http_extra_headers: http::HeaderMap,
        body_bytes: &[u8],
        provider_name: &str,
        add_api_headers_fn: fn(
            http::HeaderMap,
            Option<&secrecy::SecretString>,
        ) -> Result<http::HeaderMap, Error>,
    ) -> Result<http::HeaderMap, Error> {
        match &self.credentials {
            AWSBedrockCredentials::ApiKey(api_key) => {
                add_api_headers_fn(http_extra_headers, Some(api_key))
            }
            AWSBedrockCredentials::DynamicApiKey(key_name) => {
                let api_key = dynamic_api_keys.get(key_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: provider_name.to_string(),
                        message: format!("Dynamic `api_key` with key `{key_name}` is missing"),
                    })
                })?;
                add_api_headers_fn(http_extra_headers, Some(api_key))
            }
            AWSBedrockCredentials::IAM {
                credentials,
                sdk_config,
            } => {
                let headers = add_api_headers_fn(http_extra_headers, None)?;

                // Use SigV4 signing with IAM credentials
                let resolved_credentials = resolve_request_credentials(
                    credentials,
                    sdk_config,
                    dynamic_api_keys,
                    PROVIDER_TYPE,
                    ApiType::ChatCompletions,
                )
                .await?;
                let region = self.get_region(dynamic_api_keys, ApiType::ChatCompletions)?;
                sign_request(
                    "POST",
                    url,
                    &headers,
                    body_bytes,
                    &resolved_credentials,
                    region.as_ref(),
                    "bedrock",
                    PROVIDER_TYPE,
                    ApiType::ChatCompletions,
                )
            }
        }
    }
}

impl InferenceProvider for AWSBedrockProvider {
    async fn infer<'a>(
        &'a self,
        request: ProviderInferenceRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProviderRequestInfo,
    ) -> Result<ProviderInferenceResponse, Error> {
        self.infer_converse(request, http_client, dynamic_api_keys, model_provider)
            .await
    }

    async fn infer_stream<'a>(
        &'a self,
        request: ProviderInferenceRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProviderRequestInfo,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        self.infer_stream_converse(request, http_client, dynamic_api_keys, model_provider)
            .await
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::testing::reset_capture_logs;
    use googletest::prelude::*;
    use tensorzero_types::Usage;

    #[tokio::test]
    async fn test_get_aws_bedrock_client_no_aws_credentials() {
        // Clear bearer token env var so we test the SDK credential chain path
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_BEARER_TOKEN_BEDROCK");

        let logs_contain = crate::utils::testing::capture_logs();

        // Every call should trigger client creation since each provider has its own AWS Bedrock client
        // SDK config is now loaded in AWSBedrockCredentials::from_fields()
        let region = AWSRegion::Static(Region::new("uk-hogwarts-1"));
        let (auth, _) = AWSBedrockCredentials::from_fields(
            None, // api_key
            None, // access_key_id
            None, // secret_access_key
            None, // session_token
            &region,
            PROVIDER_TYPE,
        )
        .await
        .unwrap();

        let _provider = AWSBedrockProvider::new("test".to_string(), region.clone(), None, auth);

        assert!(logs_contain(
            "Creating new AWS config for region: uk-hogwarts-1"
        ));

        reset_capture_logs();

        let region = AWSRegion::Static(Region::new("uk-hogwarts-1"));
        let (auth, _) =
            AWSBedrockCredentials::from_fields(None, None, None, None, &region, PROVIDER_TYPE)
                .await
                .unwrap();

        let _provider = AWSBedrockProvider::new("test".to_string(), region, None, auth);

        assert!(logs_contain(
            "Creating new AWS config for region: uk-hogwarts-1"
        ));

        reset_capture_logs();

        // We want auto-detection to fail, so we clear these environment variables.
        // We use 'nextest' as our runner, so each test runs in its own process
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_REGION");
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_DEFAULT_REGION");

        let region = AWSRegion::Sdk;
        let err =
            AWSBedrockCredentials::from_fields(None, None, None, None, &region, PROVIDER_TYPE)
                .await
                .expect_err("AWS Bedrock credentials should fail when it cannot detect region");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Failed to determine AWS region."),
            "Unexpected error message: {err_msg}"
        );

        assert!(logs_contain("Failed to determine AWS region."));

        reset_capture_logs();

        let region = AWSRegion::Static(Region::new("me-shire-2"));
        let (auth, _) =
            AWSBedrockCredentials::from_fields(None, None, None, None, &region, PROVIDER_TYPE)
                .await
                .unwrap();

        let _provider = AWSBedrockProvider::new("test".to_string(), region, None, auth);

        assert!(logs_contain(
            "Creating new AWS config for region: me-shire-2"
        ));
    }

    #[gtest]
    fn test_aws_bedrock_usage_with_cache_tokens() {
        use tensorzero_types_providers::aws_bedrock;

        let bedrock_usage = aws_bedrock::Usage {
            input_tokens: 50,
            output_tokens: 30,
            total_tokens: Some(80),
            cache_read_input_tokens: Some(40),
            cache_write_input_tokens: Some(10),
        };

        // Replicate the conversion logic from the provider:
        // total_input_tokens includes cache tokens since Bedrock reports them separately
        let total_input_tokens = bedrock_usage.input_tokens as u32
            + bedrock_usage.cache_read_input_tokens.unwrap_or(0) as u32
            + bedrock_usage.cache_write_input_tokens.unwrap_or(0) as u32;
        let usage = Usage {
            input_tokens: Some(total_input_tokens),
            output_tokens: Some(bedrock_usage.output_tokens as u32),
            provider_cache_read_input_tokens: bedrock_usage
                .cache_read_input_tokens
                .map(|v| v as u32),
            provider_cache_write_input_tokens: bedrock_usage
                .cache_write_input_tokens
                .map(|v| v as u32),
            cost: None,
        };

        expect_that!(usage.input_tokens, eq(Some(100)));
        expect_that!(usage.output_tokens, eq(Some(30)));
        expect_that!(usage.provider_cache_read_input_tokens, eq(Some(40)));
        expect_that!(usage.provider_cache_write_input_tokens, eq(Some(10)));
    }
}
