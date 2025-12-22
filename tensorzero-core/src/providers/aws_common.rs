use std::sync::{Arc, Mutex};

use aws_config::{Region, meta::region::RegionProviderChain};
use aws_credential_types::Credentials;
use aws_smithy_runtime_api::client::interceptors::Intercept;
use aws_smithy_runtime_api::client::interceptors::context::AfterDeserializationInterceptorContextRef;
use aws_smithy_runtime_api::client::interceptors::context::BeforeTransmitInterceptorContextMut;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_runtime_api::client::stalled_stream_protection::StalledStreamProtectionConfig;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::config_bag::ConfigBag;
use aws_types::SdkConfig;
use reqwest::StatusCode;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    inference::types::{
        ModelInferenceRequest, extra_body::FullExtraBodyConfig,
        extra_headers::FullExtraHeadersConfig,
    },
    model::{CredentialLocation, ModelProvider, ModelProviderRequestInfo},
};

use super::helpers::inject_extra_request_data;

// ============================================================================
// AWS Region Type
// ============================================================================

/// AWS region configuration with support for static, environment, and dynamic resolution.
#[derive(Clone, Debug, ts_rs::TS)]
#[ts(export)]
pub enum AWSRegion {
    /// A static region string (e.g., "us-east-1")
    Static(String),
    /// Region loaded from an environment variable (e.g., "env::AWS_DEFAULT_REGION")
    Env(String),
    /// Region provided dynamically at request time (e.g., "dynamic::region_key")
    Dynamic(String),
}

impl Serialize for AWSRegion {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            AWSRegion::Static(value) => serializer.serialize_str(value),
            AWSRegion::Env(var_name) => serializer.serialize_str(&format!("env::{var_name}")),
            AWSRegion::Dynamic(key_name) => {
                serializer.serialize_str(&format!("dynamic::{key_name}"))
            }
        }
    }
}

impl<'de> Deserialize<'de> for AWSRegion {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        if let Some(var_name) = s.strip_prefix("env::") {
            Ok(AWSRegion::Env(var_name.to_string()))
        } else if let Some(key_name) = s.strip_prefix("dynamic::") {
            Ok(AWSRegion::Dynamic(key_name.to_string()))
        } else {
            Ok(AWSRegion::Static(s))
        }
    }
}

// ============================================================================
// AWS Credentials Type
// ============================================================================

/// AWS credentials configuration for access key and secret key.
#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AWSCredentials {
    #[ts(type = "string")]
    pub access_key: CredentialLocation,
    #[ts(type = "string")]
    pub secret_key: CredentialLocation,
}

// ============================================================================
// Resolution Functions
// ============================================================================

/// Resolves AWS region from config and dynamic credentials.
///
/// Returns `None` if the SDK should auto-detect the region.
pub fn resolve_aws_region(
    region: Option<&AWSRegion>,
    dynamic_credentials: &InferenceCredentials,
) -> Result<Option<Region>, Error> {
    match region {
        Some(AWSRegion::Static(region_str)) => Ok(Some(Region::new(region_str.clone()))),
        Some(AWSRegion::Env(env_var)) => {
            let region_str = std::env::var(env_var).map_err(|_| {
                Error::new(ErrorDetails::Config {
                    message: format!("Environment variable '{env_var}' not found for AWS region"),
                })
            })?;
            Ok(Some(Region::new(region_str)))
        }
        Some(AWSRegion::Dynamic(key_name)) => {
            let region_str = dynamic_credentials.get(key_name).ok_or_else(|| {
                Error::new(ErrorDetails::ApiKeyMissing {
                    provider_name: "AWS".to_string(),
                    message: format!("Dynamic credential '{key_name}' not found for AWS region"),
                })
            })?;
            Ok(Some(Region::new(region_str.expose_secret().to_string())))
        }
        None => Ok(None),
    }
}

/// Resolves AWS credentials from config and dynamic credentials.
///
/// Returns `None` if the SDK should use its default credential chain.
pub fn resolve_aws_credentials(
    credentials: Option<&AWSCredentials>,
    dynamic_credentials: &InferenceCredentials,
) -> Result<Option<(String, SecretString)>, Error> {
    let Some(creds) = credentials else {
        return Ok(None);
    };

    let access_key =
        resolve_credential_location(&creds.access_key, dynamic_credentials, "access_key")?;
    let secret_key =
        resolve_credential_location(&creds.secret_key, dynamic_credentials, "secret_key")?;

    Ok(Some((
        access_key,
        SecretString::new(secret_key.into_boxed_str()),
    )))
}

/// Resolves a single credential location to its actual value.
fn resolve_credential_location(
    location: &CredentialLocation,
    dynamic_credentials: &InferenceCredentials,
    credential_name: &str,
) -> Result<String, Error> {
    match location {
        CredentialLocation::Env(env_var) => std::env::var(env_var).map_err(|_| {
            Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: "AWS".to_string(),
                message: format!(
                    "Environment variable '{env_var}' not found for AWS {credential_name}"
                ),
            })
        }),
        CredentialLocation::Dynamic(key_name) => {
            let value = dynamic_credentials.get(key_name).ok_or_else(|| {
                Error::new(ErrorDetails::ApiKeyMissing {
                    provider_name: "AWS".to_string(),
                    message: format!(
                        "Dynamic credential '{key_name}' not found for AWS {credential_name}"
                    ),
                })
            })?;
            Ok(value.expose_secret().to_string())
        }
        CredentialLocation::None => Err(Error::new(ErrorDetails::ApiKeyMissing {
            provider_name: "AWS".to_string(),
            message: format!("AWS {credential_name} is required but was set to 'none'"),
        })),
        CredentialLocation::Sdk => Err(Error::new(ErrorDetails::Config {
            message: format!(
                "CredentialLocation::Sdk is not supported for AWS {credential_name}. \
                 Omit the credentials config entirely to use SDK default credential chain."
            ),
        })),
        CredentialLocation::Path(_) | CredentialLocation::PathFromEnv(_) => {
            Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "File-based credentials are not supported for AWS {credential_name}. \
                     Use 'env::VAR_NAME' or 'dynamic::key_name' instead."
                ),
            }))
        }
    }
}

/// Checks if credentials require dynamic resolution (i.e., cannot be resolved at init time).
pub fn credentials_need_dynamic_resolution(credentials: Option<&AWSCredentials>) -> bool {
    credentials.is_some_and(|creds| {
        matches!(creds.access_key, CredentialLocation::Dynamic(_))
            || matches!(creds.secret_key, CredentialLocation::Dynamic(_))
    })
}

/// Checks if region requires dynamic resolution.
pub fn region_needs_dynamic_resolution(region: Option<&AWSRegion>) -> bool {
    matches!(region, Some(AWSRegion::Dynamic(_)))
}

pub async fn config_with_region(
    provider_type: &str,
    region: Option<Region>,
) -> Result<SdkConfig, Error> {
    config_with_region_and_credentials(provider_type, region, None, None).await
}

pub async fn config_with_region_and_credentials(
    provider_type: &str,
    region: Option<Region>,
    access_key_id: Option<String>,
    secret_access_key: Option<SecretString>,
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

    let mut config_builder = aws_config::from_env()
        .region(region)
        // Using a custom HTTP client seems to break stalled stream protection, so disable it:
        // https://github.com/awslabs/aws-sdk-rust/issues/1287
        // We shouldn't actually need it, since we have user-configurable timeouts for
        // both streaming and non-streaming requests.
        .stalled_stream_protection(StalledStreamProtectionConfig::disabled());

    // Set credentials if provided
    if let (Some(access_key), Some(secret_key)) = (access_key_id, secret_access_key) {
        let credentials = Credentials::new(
            access_key,
            secret_key.expose_secret().to_string(),
            None,
            None,
            "tensorzero",
        );
        config_builder = config_builder.credentials_provider(credentials);
    }

    let config = config_builder.load().await;
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

    // =========================================================================
    // AWSCredentials deserialization tests
    // =========================================================================

    #[test]
    fn test_aws_credentials_deserialize() {
        let json =
            r#"{"access_key": "env::AWS_ACCESS_KEY_ID", "secret_key": "dynamic::my_secret"}"#;
        let creds: AWSCredentials = serde_json::from_str(json).unwrap();
        assert!(
            matches!(creds.access_key, CredentialLocation::Env(ref e) if e == "AWS_ACCESS_KEY_ID")
        );
        assert!(matches!(creds.secret_key, CredentialLocation::Dynamic(ref d) if d == "my_secret"));
    }

    // =========================================================================
    // AWSRegion deserialization tests
    // =========================================================================

    #[test]
    fn test_aws_region_deserialize_static() {
        let json = r#""us-east-1""#;
        let region: AWSRegion = serde_json::from_str(json).unwrap();
        assert!(matches!(region, AWSRegion::Static(ref s) if s == "us-east-1"));
    }

    #[test]
    fn test_aws_region_deserialize_env() {
        let json = r#""env::AWS_DEFAULT_REGION""#;
        let region: AWSRegion = serde_json::from_str(json).unwrap();
        assert!(matches!(region, AWSRegion::Env(ref e) if e == "AWS_DEFAULT_REGION"));
    }

    #[test]
    fn test_aws_region_deserialize_dynamic() {
        let json = r#""dynamic::region_key""#;
        let region: AWSRegion = serde_json::from_str(json).unwrap();
        assert!(matches!(region, AWSRegion::Dynamic(ref d) if d == "region_key"));
    }

    #[test]
    fn test_aws_region_serialize() {
        let static_region = AWSRegion::Static("us-west-2".to_string());
        assert_eq!(
            serde_json::to_string(&static_region).unwrap(),
            r#""us-west-2""#
        );

        let env_region = AWSRegion::Env("MY_REGION".to_string());
        assert_eq!(
            serde_json::to_string(&env_region).unwrap(),
            r#""env::MY_REGION""#
        );

        let dynamic_region = AWSRegion::Dynamic("dyn_key".to_string());
        assert_eq!(
            serde_json::to_string(&dynamic_region).unwrap(),
            r#""dynamic::dyn_key""#
        );
    }

    // =========================================================================
    // resolve_aws_region tests
    // =========================================================================

    #[test]
    fn test_resolve_aws_region_none() {
        let result = resolve_aws_region(None, &InferenceCredentials::default()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_aws_region_static() {
        let region = AWSRegion::Static("ap-southeast-1".to_string());
        let result = resolve_aws_region(Some(&region), &InferenceCredentials::default()).unwrap();
        assert_eq!(result.unwrap().as_ref(), "ap-southeast-1");
    }

    #[test]
    fn test_resolve_aws_region_env_present() {
        tensorzero_unsafe_helpers::set_env_var_tests_only("TEST_AWS_REGION", "eu-central-1");

        let region = AWSRegion::Env("TEST_AWS_REGION".to_string());
        let result = resolve_aws_region(Some(&region), &InferenceCredentials::default());

        tensorzero_unsafe_helpers::remove_env_var_tests_only("TEST_AWS_REGION");

        assert_eq!(result.unwrap().unwrap().as_ref(), "eu-central-1");
    }

    #[test]
    fn test_resolve_aws_region_env_missing() {
        tensorzero_unsafe_helpers::remove_env_var_tests_only("NONEXISTENT_REGION_VAR");

        let region = AWSRegion::Env("NONEXISTENT_REGION_VAR".to_string());
        let result = resolve_aws_region(Some(&region), &InferenceCredentials::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("NONEXISTENT_REGION_VAR"));
    }

    #[test]
    fn test_resolve_aws_region_dynamic_present() {
        let region = AWSRegion::Dynamic("region_key".to_string());
        let mut creds = InferenceCredentials::default();
        creds.insert(
            "region_key".into(),
            secrecy::SecretString::new("us-west-2".into()),
        );
        let result = resolve_aws_region(Some(&region), &creds).unwrap();
        assert_eq!(result.unwrap().as_ref(), "us-west-2");
    }

    #[test]
    fn test_resolve_aws_region_dynamic_missing() {
        let region = AWSRegion::Dynamic("missing_key".to_string());
        let result = resolve_aws_region(Some(&region), &InferenceCredentials::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("missing_key"));
    }

    // =========================================================================
    // resolve_aws_credentials tests
    // =========================================================================

    #[test]
    fn test_resolve_aws_credentials_none() {
        let result = resolve_aws_credentials(None, &InferenceCredentials::default()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_aws_credentials_env_present() {
        // Set env vars for this test
        tensorzero_unsafe_helpers::set_env_var_tests_only(
            "TEST_AWS_ACCESS_KEY",
            "test_access_key_123",
        );
        tensorzero_unsafe_helpers::set_env_var_tests_only(
            "TEST_AWS_SECRET_KEY",
            "test_secret_key_456",
        );

        let creds = AWSCredentials {
            access_key: CredentialLocation::Env("TEST_AWS_ACCESS_KEY".to_string()),
            secret_key: CredentialLocation::Env("TEST_AWS_SECRET_KEY".to_string()),
        };
        let result = resolve_aws_credentials(Some(&creds), &InferenceCredentials::default());

        // Clean up
        tensorzero_unsafe_helpers::remove_env_var_tests_only("TEST_AWS_ACCESS_KEY");
        tensorzero_unsafe_helpers::remove_env_var_tests_only("TEST_AWS_SECRET_KEY");

        let (access_key, secret_key) = result.unwrap().unwrap();
        assert_eq!(access_key, "test_access_key_123");
        assert_eq!(secret_key.expose_secret(), "test_secret_key_456");
    }

    #[test]
    fn test_resolve_aws_credentials_env_missing() {
        tensorzero_unsafe_helpers::remove_env_var_tests_only("NONEXISTENT_AWS_KEY");

        let creds = AWSCredentials {
            access_key: CredentialLocation::Env("NONEXISTENT_AWS_KEY".to_string()),
            secret_key: CredentialLocation::Env("NONEXISTENT_AWS_SECRET".to_string()),
        };
        let result = resolve_aws_credentials(Some(&creds), &InferenceCredentials::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("NONEXISTENT_AWS_KEY"));
    }

    #[test]
    fn test_resolve_aws_credentials_dynamic_present() {
        let creds = AWSCredentials {
            access_key: CredentialLocation::Dynamic("ak_key".to_string()),
            secret_key: CredentialLocation::Dynamic("sk_key".to_string()),
        };
        let mut dynamic_creds = InferenceCredentials::default();
        dynamic_creds.insert("ak_key".into(), secrecy::SecretString::new("dyn_ak".into()));
        dynamic_creds.insert("sk_key".into(), secrecy::SecretString::new("dyn_sk".into()));

        let result = resolve_aws_credentials(Some(&creds), &dynamic_creds).unwrap();
        let (access_key, secret_key) = result.unwrap();
        assert_eq!(access_key, "dyn_ak");
        assert_eq!(secret_key.expose_secret(), "dyn_sk");
    }

    #[test]
    fn test_resolve_aws_credentials_dynamic_missing() {
        let creds = AWSCredentials {
            access_key: CredentialLocation::Dynamic("missing_ak".to_string()),
            secret_key: CredentialLocation::Dynamic("missing_sk".to_string()),
        };
        let result = resolve_aws_credentials(Some(&creds), &InferenceCredentials::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("missing_ak"));
    }

    #[test]
    fn test_resolve_aws_credentials_unsupported_sdk() {
        let creds = AWSCredentials {
            access_key: CredentialLocation::Sdk,
            secret_key: CredentialLocation::Sdk,
        };
        let result = resolve_aws_credentials(Some(&creds), &InferenceCredentials::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Sdk is not supported"));
    }

    #[test]
    fn test_resolve_aws_credentials_unsupported_none() {
        let creds = AWSCredentials {
            access_key: CredentialLocation::None,
            secret_key: CredentialLocation::None,
        };
        let result = resolve_aws_credentials(Some(&creds), &InferenceCredentials::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("set to 'none'"));
    }

    // =========================================================================
    // needs_dynamic_resolution tests
    // =========================================================================

    #[test]
    fn test_credentials_need_dynamic_resolution_none() {
        assert!(!credentials_need_dynamic_resolution(None));
    }

    #[test]
    fn test_credentials_need_dynamic_resolution_env_only() {
        let creds = AWSCredentials {
            access_key: CredentialLocation::Env("X".to_string()),
            secret_key: CredentialLocation::Env("Y".to_string()),
        };
        assert!(!credentials_need_dynamic_resolution(Some(&creds)));
    }

    #[test]
    fn test_credentials_need_dynamic_resolution_dynamic_access_key() {
        let creds = AWSCredentials {
            access_key: CredentialLocation::Dynamic("X".to_string()),
            secret_key: CredentialLocation::Env("Y".to_string()),
        };
        assert!(credentials_need_dynamic_resolution(Some(&creds)));
    }

    #[test]
    fn test_credentials_need_dynamic_resolution_dynamic_secret_key() {
        let creds = AWSCredentials {
            access_key: CredentialLocation::Env("X".to_string()),
            secret_key: CredentialLocation::Dynamic("Y".to_string()),
        };
        assert!(credentials_need_dynamic_resolution(Some(&creds)));
    }

    #[test]
    fn test_region_needs_dynamic_resolution_none() {
        assert!(!region_needs_dynamic_resolution(None));
    }

    #[test]
    fn test_region_needs_dynamic_resolution_static() {
        let region = AWSRegion::Static("us-east-1".to_string());
        assert!(!region_needs_dynamic_resolution(Some(&region)));
    }

    #[test]
    fn test_region_needs_dynamic_resolution_env() {
        let region = AWSRegion::Env("AWS_REGION".to_string());
        assert!(!region_needs_dynamic_resolution(Some(&region)));
    }

    #[test]
    fn test_region_needs_dynamic_resolution_dynamic() {
        let region = AWSRegion::Dynamic("key".to_string());
        assert!(region_needs_dynamic_resolution(Some(&region)));
    }
}
