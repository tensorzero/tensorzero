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
use url::Url;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    inference::types::{
        ModelInferenceRequest, extra_body::FullExtraBodyConfig,
        extra_headers::FullExtraHeadersConfig,
    },
    model::{
        CredentialLocation, CredentialLocationOrHardcoded, ModelProvider, ModelProviderRequestInfo,
    },
};

use super::helpers::inject_extra_request_data;

/// AWS endpoint configuration supporting static, env, and dynamic resolution.
#[derive(Clone, Debug)]
pub enum AWSEndpointUrl {
    Static(Url),
    Dynamic(String),
}

impl AWSEndpointUrl {
    /// Create AWSEndpointUrl from CredentialLocationOrHardcoded.
    /// Returns None for None/Sdk variants since they don't apply to endpoints.
    pub fn from_credential_location(
        location: CredentialLocationOrHardcoded,
        provider_type: &str,
    ) -> Result<Option<Self>, Error> {
        match location {
            CredentialLocationOrHardcoded::Hardcoded(url_str) => {
                let url = parse_and_warn_endpoint(&url_str, provider_type)?;
                Ok(Some(AWSEndpointUrl::Static(url)))
            }
            CredentialLocationOrHardcoded::Location(loc) => match loc {
                CredentialLocation::Env(env_var) => {
                    let url_str = std::env::var(&env_var).map_err(|_| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "Environment variable `{env_var}` not found. \
                                 Your configuration for a `{provider_type}` provider requires this variable for `endpoint_url`."
                            ),
                        })
                    })?;
                    let url = parse_and_warn_endpoint(&url_str, provider_type)?;
                    Ok(Some(AWSEndpointUrl::Static(url)))
                }
                CredentialLocation::PathFromEnv(env_var) => {
                    let path = std::env::var(&env_var).map_err(|_| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "Environment variable `{env_var}` not found. \
                                 Your configuration for a `{provider_type}` provider requires this variable for `endpoint_url` path."
                            ),
                        })
                    })?;
                    let url_str = std::fs::read_to_string(&path).map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "Failed to read endpoint URL from file `{path}` for `{provider_type}`: {e}"
                            ),
                        })
                    })?;
                    let url = parse_and_warn_endpoint(url_str.trim(), provider_type)?;
                    Ok(Some(AWSEndpointUrl::Static(url)))
                }
                CredentialLocation::Dynamic(key_name) => {
                    tracing::warn!(
                        "You configured a dynamic `endpoint_url` for a `{provider_type}` provider. \
                         Only use this setting with trusted clients. \
                         An untrusted client can exfiltrate your AWS credentials with a malicious endpoint."
                    );
                    Ok(Some(AWSEndpointUrl::Dynamic(key_name)))
                }
                CredentialLocation::Path(path) => {
                    let url_str = std::fs::read_to_string(&path).map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "Failed to read endpoint URL from file `{path}` for `{provider_type}`: {e}"
                            ),
                        })
                    })?;
                    let url = parse_and_warn_endpoint(url_str.trim(), provider_type)?;
                    Ok(Some(AWSEndpointUrl::Static(url)))
                }
                CredentialLocation::Sdk => {
                    // SDK doesn't make sense for endpoint_url
                    Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "`endpoint_url = \"sdk\"` is not supported for `{provider_type}`. \
                             Use a static URL, `env::`, `path::`, or `dynamic::` instead."
                        ),
                    }))
                }
                CredentialLocation::None => Ok(None),
            },
        }
    }

    /// Get static URL if available (for use at construction time).
    pub fn get_static_url(&self) -> Option<&Url> {
        match self {
            AWSEndpointUrl::Static(url) => Some(url),
            AWSEndpointUrl::Dynamic(_) => None,
        }
    }

    /// Resolve endpoint URL at runtime (for dynamic endpoints).
    pub fn resolve(&self, credentials: &InferenceCredentials) -> Result<Url, Error> {
        match self {
            AWSEndpointUrl::Static(url) => Ok(url.clone()),
            AWSEndpointUrl::Dynamic(key_name) => {
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

/// AWS region configuration supporting static, env, dynamic, and sdk resolution.
#[derive(Clone, Debug)]
pub enum AWSRegion {
    Static(Region),
    Dynamic(String),
    /// Use AWS SDK to auto-detect region.
    Sdk,
}

impl AWSRegion {
    /// Create AWSRegion from CredentialLocationOrHardcoded.
    /// Returns None for None variant.
    pub fn from_credential_location(
        location: CredentialLocationOrHardcoded,
        provider_type: &str,
    ) -> Result<Option<Self>, Error> {
        match location {
            CredentialLocationOrHardcoded::Hardcoded(region_str) => {
                Ok(Some(AWSRegion::Static(Region::new(region_str))))
            }
            CredentialLocationOrHardcoded::Location(loc) => match loc {
                CredentialLocation::Env(env_var) => {
                    let region_str = std::env::var(&env_var).map_err(|_| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "Environment variable `{env_var}` not found. \
                                 Your configuration for a `{provider_type}` provider requires this variable for `region`."
                            ),
                        })
                    })?;
                    Ok(Some(AWSRegion::Static(Region::new(region_str))))
                }
                CredentialLocation::PathFromEnv(env_var) => {
                    let path = std::env::var(&env_var).map_err(|_| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "Environment variable `{env_var}` not found. \
                                 Your configuration for a `{provider_type}` provider requires this variable for `region` path."
                            ),
                        })
                    })?;
                    let region_str = std::fs::read_to_string(&path).map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "Failed to read region from file `{path}` for `{provider_type}`: {e}"
                            ),
                        })
                    })?;
                    Ok(Some(AWSRegion::Static(Region::new(
                        region_str.trim().to_string(),
                    ))))
                }
                CredentialLocation::Dynamic(key_name) => Ok(Some(AWSRegion::Dynamic(key_name))),
                CredentialLocation::Path(path) => {
                    let region_str = std::fs::read_to_string(&path).map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!(
                                "Failed to read region from file `{path}` for `{provider_type}`: {e}"
                            ),
                        })
                    })?;
                    Ok(Some(AWSRegion::Static(Region::new(
                        region_str.trim().to_string(),
                    ))))
                }
                CredentialLocation::Sdk => Ok(Some(AWSRegion::Sdk)),
                CredentialLocation::None => Ok(None),
            },
        }
    }

    /// Get static region if available (for use at construction time).
    pub fn get_static_region(&self) -> Option<&Region> {
        match self {
            AWSRegion::Static(region) => Some(region),
            AWSRegion::Dynamic(_) | AWSRegion::Sdk => None,
        }
    }

    /// Returns true if this region requires dynamic resolution at request time.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, AWSRegion::Dynamic(_))
    }

    /// Resolve region at runtime (for dynamic regions).
    pub fn resolve(&self, credentials: &InferenceCredentials) -> Result<Region, Error> {
        match self {
            AWSRegion::Static(region) => Ok(region.clone()),
            AWSRegion::Dynamic(key_name) => {
                let region_str = credentials.get(key_name).ok_or_else(|| {
                    Error::new(ErrorDetails::DynamicRegionNotFound {
                        key_name: key_name.clone(),
                    })
                })?;
                Ok(Region::new(region_str.expose_secret().to_string()))
            }
            AWSRegion::Sdk => {
                // This should not be called at runtime - Sdk regions are resolved at construction time
                Err(Error::new(ErrorDetails::InternalError {
                    message: "AWSRegion::Sdk should be resolved at construction time, not at request time".to_string(),
                }))
            }
        }
    }
}

/// AWS credentials configuration supporting static (env), dynamic, and sdk resolution.
#[derive(Clone, Debug)]
pub enum AWSCredentials {
    /// Credentials resolved from env vars at startup
    Static {
        access_key_id: String,
        secret_access_key: SecretString,
        session_token: Option<SecretString>,
    },
    /// Credentials resolved dynamically at request time
    Dynamic {
        access_key_id_key: String,
        secret_access_key_key: String,
        session_token_key: Option<String>,
    },
    /// Use AWS SDK credential chain (default behavior)
    Sdk,
}

impl AWSCredentials {
    /// Create AWSCredentials from flattened credential location fields.
    /// Returns None if no credentials are specified (use SDK default).
    pub fn from_fields(
        access_key_id: Option<CredentialLocation>,
        secret_access_key: Option<CredentialLocation>,
        session_token: Option<CredentialLocation>,
        provider_type: &str,
    ) -> Result<Option<Self>, Error> {
        // Validate: both access_key_id and secret_access_key must be provided together
        match (access_key_id, secret_access_key) {
            (None, None) => {
                // No credentials specified - also validate session_token is None
                if session_token.is_some() {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "`session_token` cannot be specified without `access_key_id` and `secret_access_key` for `{provider_type}`."
                        ),
                    }));
                }
                Ok(None)
            }
            (Some(_), None) => Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "`access_key_id` requires `secret_access_key` to also be specified for `{provider_type}`."
                ),
            })),
            (None, Some(_)) => Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "`secret_access_key` requires `access_key_id` to also be specified for `{provider_type}`."
                ),
            })),
            (Some(access_key_id), Some(secret_access_key)) => {
                // Both provided - convert to AWSCredentials
                Self::from_locations(
                    access_key_id,
                    secret_access_key,
                    session_token,
                    provider_type,
                )
                .map(Some)
            }
        }
    }

    fn from_locations(
        access_key_id: CredentialLocation,
        secret_access_key: CredentialLocation,
        session_token: Option<CredentialLocation>,
        provider_type: &str,
    ) -> Result<Self, Error> {
        // Validate all locations are allowed types
        validate_aws_credential_location(&access_key_id, "access_key_id", provider_type)?;
        validate_aws_credential_location(&secret_access_key, "secret_access_key", provider_type)?;
        if let Some(ref st) = session_token {
            validate_aws_credential_location(st, "session_token", provider_type)?;
        }

        // Convert based on location types
        match (&access_key_id, &secret_access_key) {
            (CredentialLocation::Sdk, CredentialLocation::Sdk) => {
                // session_token should also be Sdk or None
                if let Some(ref st) = session_token
                    && !matches!(st, CredentialLocation::Sdk)
                {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "When using `sdk` for credentials, `session_token` must also be `sdk` or omitted for `{provider_type}`."
                        ),
                    }));
                }
                Ok(AWSCredentials::Sdk)
            }
            (CredentialLocation::Env(ak_var), CredentialLocation::Env(sk_var)) => {
                // Static credentials from environment
                let ak = std::env::var(ak_var).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Environment variable `{ak_var}` not found for `access_key_id` in `{provider_type}`."
                        ),
                    })
                })?;
                let sk = std::env::var(sk_var).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Environment variable `{sk_var}` not found for `secret_access_key` in `{provider_type}`."
                        ),
                    })
                })?;
                let st = match session_token {
                    None => None,
                    Some(CredentialLocation::Env(st_var)) => {
                        let st = std::env::var(&st_var).map_err(|_| {
                            Error::new(ErrorDetails::Config {
                                message: format!(
                                    "Environment variable `{st_var}` not found for `session_token` in `{provider_type}`."
                                ),
                            })
                        })?;
                        Some(SecretString::new(st.into()))
                    }
                    Some(CredentialLocation::Sdk) => None, // Sdk means use SDK, which doesn't provide session token separately
                    _ => {
                        return Err(Error::new(ErrorDetails::Config {
                            message: format!(
                                "When using `env::` for `access_key_id` and `secret_access_key`, `session_token` must also use `env::` or be omitted for `{provider_type}`."
                            ),
                        }));
                    }
                };
                Ok(AWSCredentials::Static {
                    access_key_id: ak,
                    secret_access_key: SecretString::new(sk.into()),
                    session_token: st,
                })
            }
            (CredentialLocation::Dynamic(ak_key), CredentialLocation::Dynamic(sk_key)) => {
                // Dynamic credentials from request
                let st_key = match session_token {
                    None => None,
                    Some(CredentialLocation::Dynamic(st_key)) => Some(st_key),
                    Some(CredentialLocation::Sdk) => None,
                    _ => {
                        return Err(Error::new(ErrorDetails::Config {
                            message: format!(
                                "When using `dynamic::` for `access_key_id` and `secret_access_key`, `session_token` must also use `dynamic::` or be omitted for `{provider_type}`."
                            ),
                        }));
                    }
                };
                Ok(AWSCredentials::Dynamic {
                    access_key_id_key: ak_key.clone(),
                    secret_access_key_key: sk_key.clone(),
                    session_token_key: st_key,
                })
            }
            _ => {
                // Mismatched types (e.g., one env:: and one dynamic::)
                Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "`access_key_id` and `secret_access_key` must use the same source type (both `env::`, both `dynamic::`, or both `sdk`) for `{provider_type}`."
                    ),
                }))
            }
        }
    }

    /// Get static credentials if available (for use at construction time).
    pub fn get_static_credentials(&self) -> Option<Credentials> {
        match self {
            AWSCredentials::Static {
                access_key_id,
                secret_access_key,
                session_token,
            } => Some(Credentials::new(
                access_key_id.clone(),
                secret_access_key.expose_secret().to_string(),
                session_token
                    .as_ref()
                    .map(|st| st.expose_secret().to_string()),
                None, // expiration
                "tensorzero",
            )),
            AWSCredentials::Dynamic { .. } | AWSCredentials::Sdk => None,
        }
    }

    /// Returns true if this credential requires dynamic resolution at request time.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, AWSCredentials::Dynamic { .. })
    }

    /// Resolve credentials at runtime (for dynamic credentials).
    pub fn resolve(&self, credentials: &InferenceCredentials) -> Result<Credentials, Error> {
        match self {
            AWSCredentials::Static {
                access_key_id,
                secret_access_key,
                session_token,
            } => Ok(Credentials::new(
                access_key_id.clone(),
                secret_access_key.expose_secret().to_string(),
                session_token
                    .as_ref()
                    .map(|st| st.expose_secret().to_string()),
                None,
                "tensorzero",
            )),
            AWSCredentials::Dynamic {
                access_key_id_key,
                secret_access_key_key,
                session_token_key,
            } => {
                let ak = credentials.get(access_key_id_key).ok_or_else(|| {
                    Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: "aws".to_string(),
                        message: format!(
                            "Dynamic `access_key_id` with key `{access_key_id_key}` is missing"
                        ),
                    })
                })?;
                let sk = credentials.get(secret_access_key_key).ok_or_else(|| {
                    Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: "aws".to_string(),
                        message: format!("Dynamic `secret_access_key` with key `{secret_access_key_key}` is missing"),
                    })
                })?;
                let st = session_token_key
                    .as_ref()
                    .map(|key| {
                        credentials.get(key).ok_or_else(|| {
                            Error::new(ErrorDetails::ApiKeyMissing {
                                provider_name: "aws".to_string(),
                                message: format!(
                                    "Dynamic `session_token` with key `{key}` is missing"
                                ),
                            })
                        })
                    })
                    .transpose()?;

                Ok(Credentials::new(
                    ak.expose_secret().to_string(),
                    sk.expose_secret().to_string(),
                    st.map(|s| s.expose_secret().to_string()),
                    None,
                    "tensorzero",
                ))
            }
            AWSCredentials::Sdk => {
                // This should not be called at runtime - Sdk credentials are resolved at construction time
                Err(Error::new(ErrorDetails::InternalError {
                    message: "AWSCredentials::Sdk should be resolved at construction time, not at request time".to_string(),
                }))
            }
        }
    }
}

/// Validate that a credential location is allowed for AWS credentials.
/// Only env::, dynamic::, and sdk are allowed.
fn validate_aws_credential_location(
    location: &CredentialLocation,
    field_name: &str,
    provider_type: &str,
) -> Result<(), Error> {
    match location {
        CredentialLocation::Env(_) | CredentialLocation::Dynamic(_) | CredentialLocation::Sdk => {
            Ok(())
        }
        CredentialLocation::Path(_)
        | CredentialLocation::PathFromEnv(_)
        | CredentialLocation::None => Err(Error::new(ErrorDetails::Config {
            message: format!(
                "Invalid `{field_name}` for `{provider_type}` provider: \
                 only `env::`, `dynamic::`, and `sdk` are supported."
            ),
        })),
    }
}

/// Warn if there's a potential credential exfiltration risk.
/// This occurs when dynamic endpoint_url is configured with static or SDK credentials.
pub fn warn_if_credential_exfiltration_risk(
    endpoint_url: &Option<AWSEndpointUrl>,
    credentials: &Option<AWSCredentials>,
    provider_type: &str,
) {
    let has_dynamic_endpoint = endpoint_url
        .as_ref()
        .is_some_and(|ep| matches!(ep, AWSEndpointUrl::Dynamic(_)));

    // Warn if there are static or SDK credentials that could be exfiltrated via dynamic endpoint.
    // If credentials are also dynamic, there's no exfiltration risk (client controls all credentials).
    // When credentials is None, the SDK default credential chain is used, which is also exfiltrable.
    let has_exfiltrable_credentials = !credentials
        .as_ref()
        .is_some_and(|c| matches!(c, AWSCredentials::Dynamic { .. }));

    if has_dynamic_endpoint && has_exfiltrable_credentials {
        tracing::warn!(
            "You configured a dynamic `endpoint_url` with static or SDK credentials for a `{provider_type}` provider. \
             This is a potential security risk: a malicious client could exfiltrate your credentials via a malicious endpoint. \
             Only use this configuration with fully trusted clients."
        );
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
    use secrecy::SecretString;
    use std::collections::HashMap;

    fn make_credentials(map: HashMap<&str, &str>) -> InferenceCredentials {
        map.into_iter()
            .map(|(k, v)| (k.to_string(), SecretString::new(v.to_string().into())))
            .collect()
    }

    // ===== AWSEndpointUrl::resolve tests =====

    #[test]
    fn test_aws_endpoint_url_resolve_static() {
        let url = Url::parse("https://bedrock.us-east-1.amazonaws.com").unwrap();
        let endpoint = AWSEndpointUrl::Static(url.clone());
        let credentials = make_credentials(HashMap::new());

        let result = endpoint
            .resolve(&credentials)
            .expect("resolve should succeed");
        assert_eq!(
            result, url,
            "Static endpoint should return the URL directly"
        );
    }

    #[test]
    fn test_aws_endpoint_url_resolve_dynamic_found() {
        let endpoint = AWSEndpointUrl::Dynamic("my_endpoint".to_string());
        let credentials = make_credentials(HashMap::from([(
            "my_endpoint",
            "https://custom.endpoint.com",
        )]));

        let result = endpoint
            .resolve(&credentials)
            .expect("resolve should succeed");
        assert_eq!(
            result.as_str(),
            "https://custom.endpoint.com/",
            "Dynamic endpoint should resolve from credentials"
        );
    }

    #[test]
    fn test_aws_endpoint_url_resolve_dynamic_not_found() {
        let endpoint = AWSEndpointUrl::Dynamic("missing_key".to_string());
        let credentials = make_credentials(HashMap::new());

        let result = endpoint.resolve(&credentials);
        assert!(result.is_err(), "Should error when dynamic key is missing");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("missing_key"),
            "Error should mention the missing key: {err}"
        );
    }

    #[test]
    fn test_aws_endpoint_url_resolve_dynamic_invalid_url() {
        let endpoint = AWSEndpointUrl::Dynamic("my_endpoint".to_string());
        let credentials = make_credentials(HashMap::from([("my_endpoint", "not-a-valid-url")]));

        let result = endpoint.resolve(&credentials);
        assert!(result.is_err(), "Should error for invalid URL");
    }

    // ===== AWSRegion::resolve tests =====

    #[test]
    fn test_aws_region_resolve_static() {
        let region = AWSRegion::Static(Region::new("us-west-2"));
        let credentials = make_credentials(HashMap::new());

        let result = region
            .resolve(&credentials)
            .expect("resolve should succeed");
        assert_eq!(
            result.as_ref(),
            "us-west-2",
            "Static region should return the region directly"
        );
    }

    #[test]
    fn test_aws_region_resolve_dynamic_found() {
        let region = AWSRegion::Dynamic("aws_region".to_string());
        let credentials = make_credentials(HashMap::from([("aws_region", "eu-central-1")]));

        let result = region
            .resolve(&credentials)
            .expect("resolve should succeed");
        assert_eq!(
            result.as_ref(),
            "eu-central-1",
            "Dynamic region should resolve from credentials"
        );
    }

    #[test]
    fn test_aws_region_resolve_dynamic_not_found() {
        let region = AWSRegion::Dynamic("missing_region".to_string());
        let credentials = make_credentials(HashMap::new());

        let result = region.resolve(&credentials);
        assert!(result.is_err(), "Should error when dynamic key is missing");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("missing_region"),
            "Error should mention the missing key: {err}"
        );
    }

    #[test]
    fn test_aws_region_resolve_sdk_errors() {
        let region = AWSRegion::Sdk;
        let credentials = make_credentials(HashMap::new());

        let result = region.resolve(&credentials);
        assert!(
            result.is_err(),
            "Sdk region should error when resolved at runtime"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("construction time"),
            "Error should explain Sdk is resolved at construction: {err}"
        );
    }

    // ===== AWSCredentials::resolve tests =====

    #[test]
    fn test_aws_credentials_resolve_static() {
        let creds = AWSCredentials::Static {
            access_key_id: "AKIATEST".to_string(),
            secret_access_key: SecretString::new("secretkey".to_string().into()),
            session_token: None,
        };
        let credentials = make_credentials(HashMap::new());

        let result = creds.resolve(&credentials).expect("resolve should succeed");
        assert_eq!(
            result.access_key_id(),
            "AKIATEST",
            "Static credentials should return access_key_id"
        );
    }

    #[test]
    fn test_aws_credentials_resolve_static_with_session_token() {
        let creds = AWSCredentials::Static {
            access_key_id: "AKIATEST".to_string(),
            secret_access_key: SecretString::new("secretkey".to_string().into()),
            session_token: Some(SecretString::new("mysessiontoken".to_string().into())),
        };
        let credentials = make_credentials(HashMap::new());

        let result = creds.resolve(&credentials).expect("resolve should succeed");
        assert_eq!(
            result.session_token(),
            Some("mysessiontoken"),
            "Static credentials should include session_token"
        );
    }

    #[test]
    fn test_aws_credentials_resolve_dynamic_found() {
        let creds = AWSCredentials::Dynamic {
            access_key_id_key: "aws_access_key".to_string(),
            secret_access_key_key: "aws_secret_key".to_string(),
            session_token_key: None,
        };
        let credentials = make_credentials(HashMap::from([
            ("aws_access_key", "AKIADYNAMIC"),
            ("aws_secret_key", "dynamicsecret"),
        ]));

        let result = creds.resolve(&credentials).expect("resolve should succeed");
        assert_eq!(
            result.access_key_id(),
            "AKIADYNAMIC",
            "Dynamic credentials should resolve from request"
        );
        assert_eq!(
            result.secret_access_key(),
            "dynamicsecret",
            "Dynamic secret key should resolve from request"
        );
    }

    #[test]
    fn test_aws_credentials_resolve_dynamic_with_session_token() {
        let creds = AWSCredentials::Dynamic {
            access_key_id_key: "aws_access_key".to_string(),
            secret_access_key_key: "aws_secret_key".to_string(),
            session_token_key: Some("aws_session_token".to_string()),
        };
        let credentials = make_credentials(HashMap::from([
            ("aws_access_key", "AKIADYNAMIC"),
            ("aws_secret_key", "dynamicsecret"),
            ("aws_session_token", "dynamicsession"),
        ]));

        let result = creds.resolve(&credentials).expect("resolve should succeed");
        assert_eq!(
            result.session_token(),
            Some("dynamicsession"),
            "Dynamic session token should resolve from request"
        );
    }

    #[test]
    fn test_aws_credentials_resolve_dynamic_missing_access_key() {
        let creds = AWSCredentials::Dynamic {
            access_key_id_key: "missing_access_key".to_string(),
            secret_access_key_key: "aws_secret_key".to_string(),
            session_token_key: None,
        };
        let credentials = make_credentials(HashMap::from([("aws_secret_key", "dynamicsecret")]));

        let result = creds.resolve(&credentials);
        assert!(
            result.is_err(),
            "Should error when access_key_id is missing"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("missing_access_key"),
            "Error should mention the missing key: {err}"
        );
    }

    #[test]
    fn test_aws_credentials_resolve_dynamic_missing_secret_key() {
        let creds = AWSCredentials::Dynamic {
            access_key_id_key: "aws_access_key".to_string(),
            secret_access_key_key: "missing_secret_key".to_string(),
            session_token_key: None,
        };
        let credentials = make_credentials(HashMap::from([("aws_access_key", "AKIADYNAMIC")]));

        let result = creds.resolve(&credentials);
        assert!(
            result.is_err(),
            "Should error when secret_access_key is missing"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("missing_secret_key"),
            "Error should mention the missing key: {err}"
        );
    }

    #[test]
    fn test_aws_credentials_resolve_dynamic_missing_session_token() {
        let creds = AWSCredentials::Dynamic {
            access_key_id_key: "aws_access_key".to_string(),
            secret_access_key_key: "aws_secret_key".to_string(),
            session_token_key: Some("missing_session".to_string()),
        };
        let credentials = make_credentials(HashMap::from([
            ("aws_access_key", "AKIADYNAMIC"),
            ("aws_secret_key", "dynamicsecret"),
        ]));

        let result = creds.resolve(&credentials);
        assert!(
            result.is_err(),
            "Should error when session_token is missing"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("missing_session"),
            "Error should mention the missing key: {err}"
        );
    }

    #[test]
    fn test_aws_credentials_resolve_sdk_errors() {
        let creds = AWSCredentials::Sdk;
        let credentials = make_credentials(HashMap::new());

        let result = creds.resolve(&credentials);
        assert!(
            result.is_err(),
            "Sdk credentials should error when resolved at runtime"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("construction time"),
            "Error should explain Sdk is resolved at construction: {err}"
        );
    }

    // ===== AWSCredentials::from_fields validation tests =====

    #[test]
    fn test_aws_credentials_from_fields_none_returns_none() {
        let result =
            AWSCredentials::from_fields(None, None, None, "test_provider").expect("should succeed");
        assert!(result.is_none(), "No credentials should return None");
    }

    #[test]
    fn test_aws_credentials_from_fields_session_token_without_credentials_errors() {
        let result = AWSCredentials::from_fields(
            None,
            None,
            Some(CredentialLocation::Dynamic("token".to_string())),
            "test_provider",
        );
        assert!(
            result.is_err(),
            "session_token without credentials should error"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("session_token"),
            "Error should mention session_token: {err}"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_access_key_without_secret_errors() {
        let result = AWSCredentials::from_fields(
            Some(CredentialLocation::Dynamic("ak".to_string())),
            None,
            None,
            "test_provider",
        );
        assert!(
            result.is_err(),
            "access_key without secret_key should error"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("secret_access_key"),
            "Error should mention secret_access_key: {err}"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_secret_key_without_access_key_errors() {
        let result = AWSCredentials::from_fields(
            None,
            Some(CredentialLocation::Dynamic("sk".to_string())),
            None,
            "test_provider",
        );
        assert!(
            result.is_err(),
            "secret_key without access_key should error"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("access_key_id"),
            "Error should mention access_key_id: {err}"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_all_dynamic_succeeds() {
        let result = AWSCredentials::from_fields(
            Some(CredentialLocation::Dynamic("ak".to_string())),
            Some(CredentialLocation::Dynamic("sk".to_string())),
            Some(CredentialLocation::Dynamic("st".to_string())),
            "test_provider",
        )
        .expect("should succeed");

        assert!(result.is_some(), "Should return Some");
        let creds = result.unwrap();
        assert!(
            matches!(creds, AWSCredentials::Dynamic { .. }),
            "Should be Dynamic credentials"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_all_sdk_succeeds() {
        let result = AWSCredentials::from_fields(
            Some(CredentialLocation::Sdk),
            Some(CredentialLocation::Sdk),
            Some(CredentialLocation::Sdk),
            "test_provider",
        )
        .expect("should succeed");

        assert!(result.is_some(), "Should return Some");
        let creds = result.unwrap();
        assert!(
            matches!(creds, AWSCredentials::Sdk),
            "Should be Sdk credentials"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_mixed_dynamic_env_errors() {
        let result = AWSCredentials::from_fields(
            Some(CredentialLocation::Dynamic("ak".to_string())),
            Some(CredentialLocation::Env("SECRET_KEY".to_string())),
            None,
            "test_provider",
        );
        assert!(result.is_err(), "Mixed dynamic and env should error");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("same source type"),
            "Error should mention same source type: {err}"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_session_token_mismatch_errors() {
        let result = AWSCredentials::from_fields(
            Some(CredentialLocation::Dynamic("ak".to_string())),
            Some(CredentialLocation::Dynamic("sk".to_string())),
            Some(CredentialLocation::Env("SESSION_TOKEN".to_string())),
            "test_provider",
        );
        assert!(
            result.is_err(),
            "Session token source mismatch should error"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("session_token"),
            "Error should mention session_token: {err}"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_path_not_allowed() {
        let result = AWSCredentials::from_fields(
            Some(CredentialLocation::Path("/path/to/key".to_string())),
            Some(CredentialLocation::Path("/path/to/secret".to_string())),
            None,
            "test_provider",
        );
        assert!(result.is_err(), "Path credentials should not be allowed");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("only `env::`"),
            "Error should explain allowed types: {err}"
        );
    }

    // ===== warn_if_credential_exfiltration_risk tests =====
    // Note: These tests verify the logic but can't easily verify tracing output.
    // We test the conditions that would trigger/not trigger warnings.

    #[test]
    fn test_exfiltration_risk_conditions() {
        // Dynamic endpoint with static creds - should warn (we can't verify tracing here,
        // but we verify the conditions)
        let dynamic_endpoint = Some(AWSEndpointUrl::Dynamic("ep".to_string()));
        let static_creds = Some(AWSCredentials::Static {
            access_key_id: "ak".to_string(),
            secret_access_key: SecretString::new("sk".to_string().into()),
            session_token: None,
        });

        // This should trigger warning internally (we can't verify without tracing subscriber)
        warn_if_credential_exfiltration_risk(&dynamic_endpoint, &static_creds, "test");

        // Dynamic endpoint with SDK creds - should warn
        let sdk_creds = Some(AWSCredentials::Sdk);
        warn_if_credential_exfiltration_risk(&dynamic_endpoint, &sdk_creds, "test");

        // Dynamic endpoint with None creds (defaults to SDK) - should warn
        warn_if_credential_exfiltration_risk(&dynamic_endpoint, &None, "test");

        // Dynamic endpoint with dynamic creds - should NOT warn
        let dynamic_creds = Some(AWSCredentials::Dynamic {
            access_key_id_key: "ak".to_string(),
            secret_access_key_key: "sk".to_string(),
            session_token_key: None,
        });
        warn_if_credential_exfiltration_risk(&dynamic_endpoint, &dynamic_creds, "test");

        // Static endpoint with static creds - should NOT warn
        let static_endpoint = Some(AWSEndpointUrl::Static(
            Url::parse("https://bedrock.amazonaws.com").unwrap(),
        ));
        warn_if_credential_exfiltration_risk(&static_endpoint, &static_creds, "test");

        // No endpoint - should NOT warn
        warn_if_credential_exfiltration_risk(&None, &static_creds, "test");
    }

    // ===== AWSRegion::is_dynamic tests =====

    #[test]
    fn test_aws_region_is_dynamic() {
        assert!(
            AWSRegion::Dynamic("key".to_string()).is_dynamic(),
            "Dynamic region should return true"
        );
        assert!(
            !AWSRegion::Static(Region::new("us-east-1")).is_dynamic(),
            "Static region should return false"
        );
        assert!(
            !AWSRegion::Sdk.is_dynamic(),
            "Sdk region should return false"
        );
    }

    // ===== AWSCredentials::is_dynamic tests =====

    #[test]
    fn test_aws_credentials_is_dynamic() {
        assert!(
            AWSCredentials::Dynamic {
                access_key_id_key: "ak".to_string(),
                secret_access_key_key: "sk".to_string(),
                session_token_key: None,
            }
            .is_dynamic(),
            "Dynamic credentials should return true"
        );
        assert!(
            !AWSCredentials::Static {
                access_key_id: "ak".to_string(),
                secret_access_key: SecretString::new("sk".to_string().into()),
                session_token: None,
            }
            .is_dynamic(),
            "Static credentials should return false"
        );
        assert!(
            !AWSCredentials::Sdk.is_dynamic(),
            "Sdk credentials should return false"
        );
    }
}
