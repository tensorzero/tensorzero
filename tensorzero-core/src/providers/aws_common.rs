use std::sync::{Arc, Mutex};

use aws_config::{Region, meta::region::RegionProviderChain};
use aws_smithy_runtime_api::client::interceptors::Intercept;
use aws_smithy_runtime_api::client::interceptors::context::AfterDeserializationInterceptorContextRef;
use aws_smithy_runtime_api::client::interceptors::context::BeforeTransmitInterceptorContextMut;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_runtime_api::client::stalled_stream_protection::StalledStreamProtectionConfig;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::config_bag::ConfigBag;
use aws_types::SdkConfig;
use reqwest::StatusCode;
use secrecy::ExposeSecret;
use url::Url;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    inference::types::{
        ModelInferenceRequest, extra_body::FullExtraBodyConfig,
        extra_headers::FullExtraHeadersConfig,
    },
    model::{EndpointLocation, ModelProvider, ModelProviderRequestInfo},
};

use super::helpers::inject_extra_request_data;

/// AWS endpoint configuration supporting static, env, and dynamic resolution.
#[derive(Clone, Debug)]
pub enum AWSEndpoint {
    Static(Url),
    Dynamic(String),
}

impl AWSEndpoint {
    /// Create AWSEndpoint from EndpointLocation, resolving env vars at startup.
    pub fn from_location(location: EndpointLocation, provider_type: &str) -> Result<Self, Error> {
        match location {
            EndpointLocation::Static(url_str) => {
                let url = parse_and_warn_endpoint(&url_str, provider_type)?;
                Ok(AWSEndpoint::Static(url))
            }
            EndpointLocation::Env(env_var) => {
                let url_str = std::env::var(&env_var).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Environment variable `{env_var}` not found. \
                             Your configuration for a `{provider_type}` provider requires this variable for `endpoint_url`."
                        ),
                    })
                })?;
                let url = parse_and_warn_endpoint(&url_str, provider_type)?;
                Ok(AWSEndpoint::Static(url))
            }
            EndpointLocation::Dynamic(key_name) => {
                tracing::warn!(
                    "You configured a dynamic `endpoint_url` for a `{provider_type}` provider. \
                     Only use this setting with trusted clients. \
                     An untrusted client can exfiltrate your AWS credentials with a malicious endpoint."
                );
                Ok(AWSEndpoint::Dynamic(key_name))
            }
        }
    }

    /// Get static URL if available (for use at construction time).
    pub fn get_static_url(&self) -> Option<&Url> {
        match self {
            AWSEndpoint::Static(url) => Some(url),
            AWSEndpoint::Dynamic(_) => None,
        }
    }

    /// Resolve endpoint URL at runtime (for dynamic endpoints).
    pub fn resolve(&self, credentials: &InferenceCredentials) -> Result<Url, Error> {
        match self {
            AWSEndpoint::Static(url) => Ok(url.clone()),
            AWSEndpoint::Dynamic(key_name) => {
                let url_str = credentials.get(key_name).ok_or_else(|| {
                    Error::new(ErrorDetails::DynamicEndpointNotFound {
                        key_name: key_name.clone(),
                    })
                })?;
                let url = Url::parse(url_str.expose_secret()).map_err(|_| {
                    Error::new(ErrorDetails::InvalidDynamicEndpoint {
                        url: url_str.expose_secret().to_string(),
                    })
                })?;
                warn_if_not_aws_domain(&url);
                Ok(url)
            }
        }
    }
}

fn parse_and_warn_endpoint(url_str: &str, provider_type: &str) -> Result<Url, Error> {
    let url = Url::parse(url_str).map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!("Invalid endpoint URL `{url_str}` for {provider_type}: {e}"),
        })
    })?;
    warn_if_not_aws_domain(&url);
    Ok(url)
}

fn warn_if_not_aws_domain(url: &Url) {
    if let Some(host) = url.host_str() {
        let host_lower = host.to_lowercase();
        if !host_lower.ends_with(".amazonaws.com") && !host_lower.ends_with(".api.aws") {
            tracing::warn!(
                "AWS endpoint URL `{url}` does not appear to be an AWS domain \
                 (expected *.amazonaws.com or *.api.aws). \
                 Requests will be sent to this endpoint."
            );
        }
    }
}

pub async fn config_with_region(
    provider_type: &str,
    region: Option<Region>,
) -> Result<SdkConfig, Error> {
    // If no region is provided, we will use the default region.
    // Decide which AWS region to use. We try the following in order:
    // - The provided `region` argument
    // - The region defined by the credentials (e.g. `AWS_REGION` environment variable)
    // - The default region (us-east-1)
    let region = RegionProviderChain::first_try(region)
        .or_default_provider()
        .region()
        .await
        .ok_or_else(|| {
            Error::new(ErrorDetails::InferenceClient {
                raw_request: None,
                raw_response: None,
                status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
                message: "Failed to determine AWS region.".to_string(),
                provider_type: provider_type.to_string(),
            })
        })?;

    tracing::trace!("Creating new AWS config for region: {region}",);

    let config = aws_config::from_env()
        .region(region)
        // Using a custom HTTP client seems to break stalled stream protection, so disable it:
        // https://github.com/awslabs/aws-sdk-rust/issues/1287
        // We shouldn't actually need it, since we have user-configurable timeouts for
        // both streaming and non-streaming requests.
        .stalled_stream_protection(StalledStreamProtectionConfig::disabled())
        .load()
        .await;
    Ok(config)
}

#[derive(Debug)]
pub struct TensorZeroInterceptor {
    /// Captures the raw request from `modify_before_signing`.
    /// After the request is executed, we use this to retrieve the raw request.
    raw_request: Arc<Mutex<Option<String>>>,
    /// Captures the raw response from `read_after_deserialization`.
    /// After the request is executed, we use this to retrieve the raw response.
    raw_response: Arc<Mutex<Option<String>>>,
    extra_body: FullExtraBodyConfig,
    extra_headers: FullExtraHeadersConfig,
    model_provider_info: ModelProviderRequestInfo,
    model_name: String,
}

pub struct InterceptorAndRawBody<P: Fn() -> Result<String, Error>, Q: Fn() -> Result<String, Error>>
{
    pub interceptor: TensorZeroInterceptor,
    pub get_raw_request: P,
    pub get_raw_response: Q,
}

impl Intercept for TensorZeroInterceptor {
    fn name(&self) -> &'static str {
        "TensorZeroInterceptor"
    }
    // This interceptor injects our 'extra_body' parameters into the request body,
    // and captures the raw request.
    fn modify_before_signing(
        &self,
        context: &mut BeforeTransmitInterceptorContextMut<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let http_request = context.request_mut();
        let bytes = http_request.body().bytes().ok_or_else(|| {
            Error::new(ErrorDetails::Serialization {
                message: "Failed to get body from AWS request".to_string(),
            })
        })?;
        let mut body_json: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Failed to deserialize AWS request body: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let headers = inject_extra_request_data(
            &self.extra_body,
            &self.extra_headers,
            self.model_provider_info.clone(),
            &self.model_name,
            &mut body_json,
        )?;
        // Get a consistent order for use with the provider-proxy cache
        if cfg!(feature = "e2e_tests") {
            body_json.sort_all_objects();
        }

        let raw_request = serde_json::to_string(&body_json).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Failed to serialize AWS request body: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        // AWS inexplicably sets this header before calling this interceptor, so we need to update
        // it ourselves (in case the body length changed)
        http_request
            .headers_mut()
            .insert("content-length", raw_request.len().to_string());
        *http_request.body_mut() = SdkBody::from(raw_request.clone());

        // Capture the raw request for later use. Note that `modify_before_signing` may be
        // called multiple times (due to internal aws sdk retries), so this will overwrite
        // the Mutex to contain the latest raw request (which is what we want).
        let body = self.raw_request.lock();
        // Ignore poisoned lock, since we're overwriting it.
        let mut body = match body {
            Ok(body) => body,
            Err(e) => e.into_inner(),
        };
        *body = Some(raw_request);

        // We iterate over a reference and clone, since `header.into_iter()`
        // produces (Option<HeaderName>, HeaderValue)
        for (name, value) in &headers {
            http_request
                .headers_mut()
                .insert(name.clone(), value.clone());
        }
        Ok(())
    }

    fn read_after_deserialization(
        &self,
        context: &AfterDeserializationInterceptorContextRef<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
        let bytes = context.response().body().bytes();
        if let Some(bytes) = bytes {
            let raw_response = self.raw_response.lock();
            // Ignore poisoned lock, since we're overwriting it.
            let mut body = match raw_response {
                Ok(body) => body,
                Err(e) => e.into_inner(),
            };
            *body = Some(String::from_utf8_lossy(bytes).into_owned());
        }
        Ok(())
    }
}

/// Builds our custom interceptor to the request builder, which injects our 'extra_body' parameters into
/// the request body.
/// Returns the interceptor, and a function to retrieve the raw request.
/// This awkward signature is due to the fact that we cannot call `send()` from a generic
/// function, as one of the needed traits is private: https://github.com/awslabs/aws-sdk-rust/issues/987
pub fn build_interceptor(
    request: &ModelInferenceRequest<'_>,
    model_provider: &ModelProvider,
    model_name: String,
) -> InterceptorAndRawBody<impl Fn() -> Result<String, Error>, impl Fn() -> Result<String, Error>> {
    let raw_request = Arc::new(Mutex::new(None));
    let raw_response = Arc::new(Mutex::new(None));
    let extra_body = request.extra_body.clone();
    let extra_headers = request.extra_headers.clone();

    let interceptor = TensorZeroInterceptor {
        raw_request: raw_request.clone(),
        raw_response: raw_response.clone(),
        extra_body,
        extra_headers,
        model_provider_info: model_provider.into(),
        model_name,
    };

    InterceptorAndRawBody {
        interceptor,
        get_raw_request: move || {
            let raw_request = raw_request
                .lock()
                .map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Poisoned raw_request mutex for AWS request: {e:?}"),
                    })
                })?
                .clone()
                .ok_or_else(|| {
                    Error::new(ErrorDetails::Serialization {
                        message: "Failed to get serialized AWS request".to_string(),
                    })
                })?;
            Ok(raw_request)
        },
        get_raw_response: move || {
            let raw_response = raw_response
                .lock()
                .map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Poisoned raw_response mutex for AWS request: {e:?}"),
                    })
                })?
                .clone()
                .ok_or_else(|| {
                    Error::new(ErrorDetails::Serialization {
                        message: "Failed to get serialized AWS response".to_string(),
                    })
                })?;
            Ok(raw_response)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_endpoint_from_static_valid_amazonaws() {
        let endpoint = AWSEndpoint::from_location(
            EndpointLocation::Static("https://bedrock-runtime.us-east-1.amazonaws.com".to_string()),
            "aws_bedrock",
        )
        .unwrap();
        assert!(matches!(endpoint, AWSEndpoint::Static(_)));
        assert_eq!(
            endpoint.get_static_url().unwrap().as_str(),
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
        );
    }

    #[test]
    fn test_aws_endpoint_from_static_valid_api_aws() {
        let endpoint = AWSEndpoint::from_location(
            EndpointLocation::Static("https://bedrock.us-east-1.api.aws".to_string()),
            "aws_bedrock",
        )
        .unwrap();
        assert!(matches!(endpoint, AWSEndpoint::Static(_)));
    }

    #[test]
    fn test_aws_endpoint_from_static_case_insensitive() {
        // Should not panic or error - domain check is case-insensitive
        let endpoint = AWSEndpoint::from_location(
            EndpointLocation::Static("https://bedrock-runtime.us-east-1.AMAZONAWS.COM".to_string()),
            "aws_bedrock",
        )
        .unwrap();
        assert!(matches!(endpoint, AWSEndpoint::Static(_)));
    }

    #[test]
    fn test_aws_endpoint_from_static_non_aws_warns_but_succeeds() {
        // Non-AWS domains should succeed (with a warning logged)
        let endpoint = AWSEndpoint::from_location(
            EndpointLocation::Static("http://localhost:4566".to_string()),
            "aws_bedrock",
        )
        .unwrap();
        assert!(matches!(endpoint, AWSEndpoint::Static(_)));
    }

    #[test]
    fn test_aws_endpoint_from_static_invalid_url_fails() {
        let result = AWSEndpoint::from_location(
            EndpointLocation::Static("not-a-valid-url".to_string()),
            "aws_bedrock",
        );
        assert!(result.is_err(), "Invalid URL should return an error");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Invalid endpoint URL"),
            "Error should mention invalid endpoint URL, but got: {err}"
        );
    }

    #[test]
    fn test_aws_endpoint_from_dynamic() {
        let endpoint = AWSEndpoint::from_location(
            EndpointLocation::Dynamic("my_endpoint_key".to_string()),
            "aws_bedrock",
        )
        .unwrap();
        assert!(matches!(endpoint, AWSEndpoint::Dynamic(_)));
        // Dynamic endpoints don't have a static URL
        assert!(endpoint.get_static_url().is_none());
    }

    #[test]
    fn test_aws_endpoint_from_env_missing_fails() {
        let result = AWSEndpoint::from_location(
            EndpointLocation::Env("NONEXISTENT_AWS_ENDPOINT_VAR_12345".to_string()),
            "aws_bedrock",
        );
        assert!(result.is_err(), "Missing env var should return an error");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Environment variable"),
            "Error should mention environment variable, but got: {err}"
        );
    }
}
