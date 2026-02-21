use std::time::{Duration, Instant, SystemTime};

use aws_config::{Region, meta::region::RegionProviderChain};
use aws_credential_types::Credentials;
use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use aws_smithy_runtime_api::client::stalled_stream_protection::StalledStreamProtectionConfig;
use aws_smithy_types::event_stream::{Header, Message};
use aws_types::SdkConfig;
use reqwest::StatusCode;
use secrecy::{ExposeSecret, SecretString};
use url::Url;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    http::TensorzeroHttpClient,
    inference::types::ApiType,
    model::{CredentialLocation, CredentialLocationOrHardcoded},
};

/// Parse and validate AWS region configuration.
///
/// Handles:
/// - Deprecation warning for `allow_auto_detect_region`
/// - Region parsing from `CredentialLocationOrHardcoded` to `AWSRegion`
/// - Region validation (requiring a region is present)
///
/// Returns the resolved `AWSRegion` or an error if region is required but missing.
pub fn parse_aws_region(
    region: Option<CredentialLocationOrHardcoded>,
    allow_auto_detect_region: bool,
    provider_type: &str,
) -> Result<AWSRegion, Error> {
    // Emit deprecation warning if allow_auto_detect_region is used
    if allow_auto_detect_region {
        crate::utils::deprecation_warning(&format!(
            "The `allow_auto_detect_region` field is deprecated for `{provider_type}`. \
             Use `region = \"sdk\"` instead to enable auto-detection. (#5596)"
        ));
    }

    // Convert CredentialLocationOrHardcoded to AWSRegion
    let aws_region = region
        .map(|loc| AWSRegion::from_credential_location(loc, provider_type))
        .transpose()?
        .flatten();

    // If no region specified and allow_auto_detect_region (deprecated) is set, use region = "sdk"
    let aws_region = if aws_region.is_none() && allow_auto_detect_region {
        Some(AWSRegion::Sdk)
    } else {
        aws_region
    };

    // Check if we have a region or need to error
    aws_region.ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!(
                "AWS {provider_type} provider requires a region. \
                 Use `region = \"sdk\"` to enable auto-detection, \
                 or specify a region like `region = \"us-east-1\"`."
            ),
        })
    })
}

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

    /// Get the static region to use when initializing the AWS SDK config.
    ///
    /// - `Static(r)`: Use the configured region
    /// - `Sdk`: None (let SDK auto-detect from environment)
    /// - `Dynamic`: Use a fallback region (the actual request region comes dynamically)
    pub fn static_region_for_sdk_config(&self) -> Option<Region> {
        match self {
            AWSRegion::Static(r) => Some(r.clone()),
            AWSRegion::Sdk => None,
            AWSRegion::Dynamic(_) => Some(Region::new("us-east-1")),
        }
    }

    /// Resolve region at runtime with optional SDK config for Sdk variant.
    ///
    /// - `Static(r)`: Returns the configured region directly
    /// - `Dynamic(key)`: Resolves from request credentials
    /// - `Sdk`: Extracts region from the provided SDK config
    ///
    /// For `Sdk` variant, `sdk_config` must be provided, otherwise an error is returned.
    pub fn resolve_with_sdk_config(
        &self,
        credentials: &InferenceCredentials,
        sdk_config: Option<&SdkConfig>,
        provider_type: &str,
        api_type: ApiType,
    ) -> Result<Region, Error> {
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
            AWSRegion::Sdk => sdk_config
                .and_then(|config| config.region().cloned())
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InferenceClient {
                        raw_request: None,
                        raw_response: None,
                        status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
                        message: "No region configured".to_string(),
                        provider_type: provider_type.to_string(),
                        api_type,
                    })
                }),
        }
    }
}

/// AWS credentials configuration supporting static (env), dynamic, and sdk resolution.
#[derive(Clone, Debug)]
pub enum AWSIAMCredentials {
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

impl AWSIAMCredentials {
    /// Create AWSCredentials from flattened credential location fields.
    /// Returns `Sdk` if no credentials are specified (uses SDK default credential chain).
    pub fn from_fields(
        access_key_id: Option<CredentialLocation>,
        secret_access_key: Option<CredentialLocation>,
        session_token: Option<CredentialLocation>,
        provider_type: &str,
    ) -> Result<Self, Error> {
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
                Ok(AWSIAMCredentials::Sdk)
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
                Ok(AWSIAMCredentials::Sdk)
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
                Ok(AWSIAMCredentials::Static {
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
                Ok(AWSIAMCredentials::Dynamic {
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

/// AWS authentication method - either API key (bearer token) or IAM credentials (SigV4).
/// Used by AWS Bedrock to support both bearer token auth and SigV4 signing.
#[derive(Debug)]
pub enum AWSBedrockCredentials {
    /// Bearer token authentication (Authorization: Bearer <token>).
    /// Used with AWS Bedrock API keys.
    ApiKey(SecretString),
    /// Dynamic bearer token resolved at request time.
    DynamicApiKey(String),
    /// IAM credentials for SigV4 signing, with loaded SDK config.
    IAM {
        credentials: AWSIAMCredentials,
        sdk_config: Box<SdkConfig>,
    },
}

impl AWSBedrockCredentials {
    /// Create AWSBedrockCredentials from config fields.
    ///
    /// Priority:
    /// 1. Explicit api_key → bearer auth
    /// 2. Explicit IAM credentials → SigV4
    /// 3. AWS_BEARER_TOKEN_BEDROCK env var → bearer auth
    /// 4. SDK credential chain → SigV4
    ///
    /// For IAM auth, loads the AWS SDK config with the given region.
    ///
    /// Returns `(credentials, resolved_sdk_region)` where `resolved_sdk_region` is `Some`
    /// when bearer auth is used with `region = "sdk"`. In this case, the caller should
    /// replace `AWSRegion::Sdk` with `AWSRegion::Static(resolved_sdk_region)`.
    pub async fn from_fields(
        api_key: Option<CredentialLocation>,
        access_key_id: Option<CredentialLocation>,
        secret_access_key: Option<CredentialLocation>,
        session_token: Option<CredentialLocation>,
        region: &AWSRegion,
        provider_type: &str,
    ) -> Result<(Self, Option<Region>), Error> {
        // Validate: cannot specify both api_key and IAM credentials
        let has_iam_creds =
            access_key_id.is_some() || secret_access_key.is_some() || session_token.is_some();

        if api_key.is_some() && has_iam_creds {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Cannot specify both `api_key` and IAM credentials (`access_key_id`/`secret_access_key`) for `{provider_type}`. Use one or the other."
                ),
            }));
        }

        // 1. Explicit api_key takes priority
        if let Some(key_loc) = api_key {
            let creds = Self::from_api_key_location(key_loc, provider_type)?;
            // For bearer auth with region = "sdk", resolve the region now
            let resolved_region =
                Self::resolve_sdk_region_for_bearer_auth(region, provider_type).await?;
            return Ok((creds, resolved_region));
        }

        // Compute the static region for SDK config initialization
        let static_region = region.static_region_for_sdk_config();

        // 2. Explicit IAM credentials (or session_token alone, which will error in from_fields)
        if has_iam_creds {
            let credentials = AWSIAMCredentials::from_fields(
                access_key_id,
                secret_access_key,
                session_token,
                provider_type,
            )?;
            let sdk_config = config_with_region(provider_type, static_region).await?;
            return Ok((
                AWSBedrockCredentials::IAM {
                    credentials,
                    sdk_config: Box::new(sdk_config),
                },
                None,
            ));
        }

        // 3. Nothing configured - try AWS_BEARER_TOKEN_BEDROCK env var first, then SDK
        if let Some(token) = std::env::var("AWS_BEARER_TOKEN_BEDROCK")
            .ok()
            .filter(|t| !t.is_empty())
        {
            tracing::debug!(
                "Using environment variable `AWS_BEARER_TOKEN_BEDROCK` for `{provider_type}` authentication"
            );
            // For bearer auth with region = "sdk", resolve the region now
            let resolved_region =
                Self::resolve_sdk_region_for_bearer_auth(region, provider_type).await?;
            return Ok((
                AWSBedrockCredentials::ApiKey(SecretString::new(token.into())),
                resolved_region,
            ));
        }

        // 4. Fall back to SDK credential chain (SigV4)
        tracing::debug!("Using AWS SDK credential chain for `{provider_type}` authentication");
        let sdk_config = config_with_region(provider_type, static_region).await?;
        Ok((
            AWSBedrockCredentials::IAM {
                credentials: AWSIAMCredentials::Sdk,
                sdk_config: Box::new(sdk_config),
            },
            None,
        ))
    }

    /// For bearer auth with `region = "sdk"`, load SDK config to resolve the region.
    /// Returns `Some(region)` if region was `Sdk`, `None` otherwise.
    async fn resolve_sdk_region_for_bearer_auth(
        region: &AWSRegion,
        provider_type: &str,
    ) -> Result<Option<Region>, Error> {
        if !matches!(region, AWSRegion::Sdk) {
            return Ok(None);
        }
        // Load SDK config just to resolve the region
        let sdk_config = config_with_region(provider_type, None).await?;
        let resolved = sdk_config.region().cloned().ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Failed to auto-detect AWS region for `{provider_type}`. \
                     Please set `AWS_REGION` environment variable or specify an explicit region."
                ),
            })
        })?;
        tracing::debug!(
            "Resolved SDK region `{}` for bearer auth in `{provider_type}`",
            resolved.as_ref()
        );
        Ok(Some(resolved))
    }

    fn from_api_key_location(loc: CredentialLocation, provider_type: &str) -> Result<Self, Error> {
        match loc {
            CredentialLocation::Env(var) => {
                let token = std::env::var(&var).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Environment variable `{var}` not found for `api_key` in `{provider_type}`."
                        ),
                    })
                })?;
                Ok(AWSBedrockCredentials::ApiKey(SecretString::new(
                    token.into(),
                )))
            }
            CredentialLocation::Dynamic(key) => Ok(AWSBedrockCredentials::DynamicApiKey(key)),
            _ => Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "Unsupported credential location for `api_key` in `{provider_type}`. Use `env::` or `dynamic::`."
                ),
            })),
        }
    }

    /// Returns true if this auth method uses bearer token authentication.
    pub fn is_bearer_auth(&self) -> bool {
        matches!(
            self,
            AWSBedrockCredentials::ApiKey(_) | AWSBedrockCredentials::DynamicApiKey(_)
        )
    }
}

/// Warn if there's a potential credential exfiltration risk.
/// This occurs when dynamic endpoint_url is configured with static or SDK credentials.
pub fn warn_if_credential_exfiltration_risk(
    endpoint_url: &Option<AWSEndpointUrl>,
    credentials: &AWSIAMCredentials,
    provider_type: &str,
) {
    let has_dynamic_endpoint = endpoint_url
        .as_ref()
        .is_some_and(|ep| matches!(ep, AWSEndpointUrl::Dynamic(_)));

    // Warn if there are static or SDK credentials that could be exfiltrated via dynamic endpoint.
    // If credentials are also dynamic, there's no exfiltration risk (client controls all credentials).
    let has_exfiltrable_credentials = !matches!(credentials, AWSIAMCredentials::Dynamic { .. });

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
        // Check for all known AWS partition domain suffixes:
        // - amazonaws.com (standard AWS)
        // - amazonaws.com.cn (AWS China: cn-north-1, cn-northwest-1)
        // - api.aws (newer AWS API endpoints)
        // - c2s.ic.gov (AWS US ISO)
        // - sc2s.sgov.gov (AWS US ISOB)
        let is_aws_domain = host_lower.ends_with(".amazonaws.com")
            || host_lower.ends_with(".amazonaws.com.cn")
            || host_lower.ends_with(".api.aws")
            || host_lower.ends_with(".c2s.ic.gov")
            || host_lower.ends_with(".sc2s.sgov.gov");
        if !is_aws_domain {
            tracing::warn!(
                "AWS endpoint URL `{url}` does not appear to be an AWS domain (e.g. *.amazonaws.com). \
                 TensorZero will route requests to this endpoint, but be careful: a malicious endpoint can exfiltrate data or credentials."
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
                api_type: ApiType::ChatCompletions,
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

/// Resolve credentials for an AWS request.
///
/// Handles static, dynamic, and SDK credential types. This is the shared
/// implementation used by both Bedrock and SageMaker providers.
pub async fn resolve_request_credentials(
    credentials: &AWSIAMCredentials,
    sdk_config: &SdkConfig,
    dynamic_api_keys: &InferenceCredentials,
    provider_type: &str,
    api_type: ApiType,
) -> Result<Credentials, Error> {
    match credentials {
        AWSIAMCredentials::Static {
            access_key_id,
            secret_access_key,
            session_token,
        } => {
            // Use static credentials directly
            Ok(Credentials::new(
                access_key_id.clone(),
                secret_access_key.expose_secret().to_string(),
                session_token
                    .as_ref()
                    .map(|st| st.expose_secret().to_string()),
                None,
                "tensorzero",
            ))
        }
        AWSIAMCredentials::Dynamic {
            access_key_id_key,
            secret_access_key_key,
            session_token_key,
        } => {
            // Resolve dynamic credentials from the request
            let ak = dynamic_api_keys.get(access_key_id_key).ok_or_else(|| {
                Error::new(ErrorDetails::ApiKeyMissing {
                    provider_name: "aws".to_string(),
                    message: format!(
                        "Dynamic `access_key_id` with key `{access_key_id_key}` is missing"
                    ),
                })
            })?;
            let sk = dynamic_api_keys.get(secret_access_key_key).ok_or_else(|| {
                Error::new(ErrorDetails::ApiKeyMissing {
                    provider_name: "aws".to_string(),
                    message: format!(
                        "Dynamic `secret_access_key` with key `{secret_access_key_key}` is missing"
                    ),
                })
            })?;
            let st = session_token_key
                .as_ref()
                .map(|key| {
                    dynamic_api_keys.get(key).ok_or_else(|| {
                        Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: "aws".to_string(),
                            message: format!("Dynamic `session_token` with key `{key}` is missing"),
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
        AWSIAMCredentials::Sdk => get_credentials(sdk_config, provider_type, api_type).await,
    }
}

/// Get fresh credentials from the SDK config.
/// This is used to handle credential rotation for long-running processes.
pub async fn get_credentials(
    config: &SdkConfig,
    provider_type: &str,
    api_type: ApiType,
) -> Result<Credentials, Error> {
    let provider = config.credentials_provider().ok_or_else(|| {
        Error::new(ErrorDetails::InferenceClient {
            raw_request: None,
            raw_response: None,
            status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
            message: "No credentials provider configured".to_string(),
            provider_type: provider_type.to_string(),
            api_type,
        })
    })?;

    provider.provide_credentials().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            raw_request: None,
            raw_response: None,
            status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
            message: format!("Failed to get AWS credentials: {e}"),
            provider_type: provider_type.to_string(),
            api_type,
        })
    })
}

/// Sign an HTTP request using AWS SigV4.
///
/// This function signs the request in-place by adding the required AWS authentication headers.
#[expect(clippy::too_many_arguments)]
pub fn sign_request(
    method: &str,
    uri: &str,
    headers: &reqwest::header::HeaderMap,
    body: &[u8],
    credentials: &Credentials,
    region: &str,
    service: &str,
    provider_type: &str,
    api_type: ApiType,
) -> Result<reqwest::header::HeaderMap, Error> {
    let identity: Identity = credentials.clone().into();

    let signing_params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(service)
        .time(SystemTime::now())
        .settings(SigningSettings::default())
        .build()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                raw_request: None,
                raw_response: None,
                status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
                message: format!("Failed to build signing params: {e}"),
                provider_type: provider_type.to_string(),
                api_type,
            })
        })?;

    let header_pairs: Vec<(&str, &str)> = headers
        .iter()
        .map(|(k, v)| {
            v.to_str().map(|v_str| (k.as_str(), v_str)).map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    raw_request: None,
                    raw_response: None,
                    status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
                    message: format!("Invalid header value: {e}"),
                    provider_type: provider_type.to_string(),
                    api_type,
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let signable_request = SignableRequest::new(
        method,
        uri.to_string(),
        header_pairs.into_iter(),
        SignableBody::Bytes(body),
    )
    .map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            raw_request: None,
            raw_response: None,
            status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
            message: format!("Failed to create signable request: {e}"),
            provider_type: provider_type.to_string(),
            api_type,
        })
    })?;

    let (signing_instructions, _signature) = sign(signable_request, &signing_params.into())
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                raw_request: None,
                raw_response: None,
                status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
                message: format!("Failed to sign request: {e}"),
                provider_type: provider_type.to_string(),
                api_type,
            })
        })?
        .into_parts();

    // Build a new header map with the signing headers.
    // We preserve existing headers (e.g., user-provided extra_headers) and only add
    // signing headers that aren't already present. This allows users to override
    // headers like x-amz-security-token for testing purposes.
    let mut signed_headers = headers.clone();
    for (name, value) in signing_instructions.headers() {
        let header_name =
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    raw_request: None,
                    raw_response: None,
                    status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
                    message: format!("Invalid header name from signing: {e}"),
                    provider_type: provider_type.to_string(),
                    api_type,
                })
            })?;
        let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                raw_request: None,
                raw_response: None,
                status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
                message: format!("Invalid header value from signing: {e}"),
                provider_type: provider_type.to_string(),
                api_type,
            })
        })?;
        // Only insert if the header isn't already present (preserve user-provided headers)
        if let reqwest::header::Entry::Vacant(entry) = signed_headers.entry(header_name) {
            entry.insert(header_value);
        }
    }

    Ok(signed_headers)
}

/// Response from sending an AWS request.
pub struct AwsRequestResponse {
    pub raw_response: String,
    pub response_time: Duration,
}

/// Send a signed AWS HTTP request and handle common error patterns.
///
/// This handles: header setup, request signing, sending, reading response,
/// and status validation. Returns the raw response text on success.
#[expect(clippy::too_many_arguments)]
pub async fn send_aws_request(
    http_client: &TensorzeroHttpClient,
    url: &str,
    extra_headers: http::HeaderMap,
    body_bytes: Vec<u8>,
    credentials: &Credentials,
    region: &str,
    service: &str,
    provider_type: &str,
    raw_request: &str,
    api_type: ApiType,
) -> Result<AwsRequestResponse, Error> {
    // Build headers with content-type and accept
    let mut headers = extra_headers;
    headers.insert(
        http::header::CONTENT_TYPE,
        http::header::HeaderValue::from_static("application/json"),
    );
    headers.insert(
        http::header::ACCEPT,
        http::header::HeaderValue::from_static("application/json"),
    );

    // Sign the request
    let signed_headers = sign_request(
        "POST",
        url,
        &headers,
        &body_bytes,
        credentials,
        region,
        service,
        provider_type,
        api_type,
    )?;

    // Send request
    let start_time = Instant::now();
    let response = http_client
        .post(url)
        .headers(signed_headers)
        .body(body_bytes)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error sending request to AWS {service}: {e}"),
                raw_request: Some(raw_request.to_string()),
                raw_response: None,
                provider_type: provider_type.to_string(),
                api_type,
            })
        })?;

    let response_time = start_time.elapsed();
    let status = response.status();
    let raw_response = response.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error reading response from AWS {service}: {e}"),
            raw_request: Some(raw_request.to_string()),
            raw_response: None,
            provider_type: provider_type.to_string(),
            api_type,
        })
    })?;

    if !status.is_success() {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: format!("AWS {service} returned error status {status}: {raw_response}"),
            raw_request: Some(raw_request.to_string()),
            raw_response: Some(raw_response),
            provider_type: provider_type.to_string(),
            api_type,
        }));
    }

    Ok(AwsRequestResponse {
        raw_response,
        response_time,
    })
}

/// Send an AWS Bedrock request with API key (bearer token) authentication.
///
/// This is used when an API key is configured instead of IAM credentials.
/// Uses `Authorization: Bearer <token>` header instead of SigV4 signing.
#[expect(clippy::too_many_arguments)]
pub async fn send_aws_request_with_api_key(
    http_client: &TensorzeroHttpClient,
    url: &str,
    extra_headers: http::HeaderMap,
    body_bytes: Vec<u8>,
    api_key: &SecretString,
    provider_type: &str,
    raw_request: &str,
    api_type: ApiType,
) -> Result<AwsRequestResponse, Error> {
    // Build headers with content-type, accept, and authorization
    let mut headers = extra_headers;
    headers.insert(
        http::header::CONTENT_TYPE,
        http::header::HeaderValue::from_static("application/json"),
    );
    headers.insert(
        http::header::ACCEPT,
        http::header::HeaderValue::from_static("application/json"),
    );
    headers.insert(
        http::header::AUTHORIZATION,
        http::header::HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
            .map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Invalid API key format: {e}"),
                })
            })?,
    );

    // Send request (no signing needed for bearer auth)
    let start_time = Instant::now();
    let response = http_client
        .post(url)
        .headers(headers)
        .body(body_bytes)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error sending request to AWS Bedrock: {e}"),
                raw_request: Some(raw_request.to_string()),
                raw_response: None,
                provider_type: provider_type.to_string(),
                api_type,
            })
        })?;

    let response_time = start_time.elapsed();
    let status = response.status();
    let raw_response = response.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error reading response from AWS Bedrock: {e}"),
            raw_request: Some(raw_request.to_string()),
            raw_response: None,
            provider_type: provider_type.to_string(),
            api_type,
        })
    })?;

    if !status.is_success() {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: format!("AWS Bedrock returned error status {status}: {raw_response}"),
            raw_request: Some(raw_request.to_string()),
            raw_response: Some(raw_response),
            provider_type: provider_type.to_string(),
            api_type,
        }));
    }

    Ok(AwsRequestResponse {
        raw_response,
        response_time,
    })
}

/// Check if a Smithy EventStream message is an exception.
///
/// AWS Smithy event streams can contain exception messages indicated by the
/// `:message-type` header being set to "exception". This function checks for
/// such exceptions and extracts the exception type and error message.
///
/// Returns `Some((exception_type, error_message))` if the message is an exception,
/// `None` otherwise.
///
/// See https://smithy.io/2.0/aws/amazon-eventstream.html for details on the format.
pub fn check_eventstream_exception(message: &Message) -> Option<(String, String)> {
    let message_type = message
        .headers()
        .iter()
        .find(|h: &&Header| h.name().as_str() == ":message-type")
        .and_then(|h: &Header| h.value().as_string().ok())
        .map(|s: &aws_smithy_types::str_bytes::StrBytes| s.as_str().to_owned());

    if message_type.as_deref() != Some("exception") {
        return None;
    }

    let exception_type = message
        .headers()
        .iter()
        .find(|h: &&Header| h.name().as_str() == ":exception-type")
        .and_then(|h: &Header| h.value().as_string().ok())
        .map(|s: &aws_smithy_types::str_bytes::StrBytes| s.as_str().to_owned())
        .unwrap_or_else(|| "unknown".to_string());

    let error_message = String::from_utf8_lossy(message.payload()).to_string();

    Some((exception_type, error_message))
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

    // ===== resolve_request_credentials tests =====

    #[tokio::test]
    async fn test_resolve_request_credentials_static() {
        let creds = AWSIAMCredentials::Static {
            access_key_id: "AKIATEST".to_string(),
            secret_access_key: SecretString::new("secretkey".to_string().into()),
            session_token: None,
        };
        let sdk_config = SdkConfig::builder().build();
        let dynamic_api_keys = make_credentials(HashMap::new());

        let result = resolve_request_credentials(
            &creds,
            &sdk_config,
            &dynamic_api_keys,
            "test",
            ApiType::ChatCompletions,
        )
        .await
        .expect("resolve should succeed");
        assert_eq!(
            result.access_key_id(),
            "AKIATEST",
            "Static credentials should return access_key_id"
        );
    }

    #[tokio::test]
    async fn test_resolve_request_credentials_static_with_session_token() {
        let creds = AWSIAMCredentials::Static {
            access_key_id: "AKIATEST".to_string(),
            secret_access_key: SecretString::new("secretkey".to_string().into()),
            session_token: Some(SecretString::new("mysessiontoken".to_string().into())),
        };
        let sdk_config = SdkConfig::builder().build();
        let dynamic_api_keys = make_credentials(HashMap::new());

        let result = resolve_request_credentials(
            &creds,
            &sdk_config,
            &dynamic_api_keys,
            "test",
            ApiType::ChatCompletions,
        )
        .await
        .expect("resolve should succeed");
        assert_eq!(
            result.session_token(),
            Some("mysessiontoken"),
            "Static credentials should include session_token"
        );
    }

    #[tokio::test]
    async fn test_resolve_request_credentials_dynamic_found() {
        let creds = AWSIAMCredentials::Dynamic {
            access_key_id_key: "aws_access_key".to_string(),
            secret_access_key_key: "aws_secret_key".to_string(),
            session_token_key: None,
        };
        let sdk_config = SdkConfig::builder().build();
        let dynamic_api_keys = make_credentials(HashMap::from([
            ("aws_access_key", "AKIADYNAMIC"),
            ("aws_secret_key", "dynamicsecret"),
        ]));

        let result = resolve_request_credentials(
            &creds,
            &sdk_config,
            &dynamic_api_keys,
            "test",
            ApiType::ChatCompletions,
        )
        .await
        .expect("resolve should succeed");
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

    #[tokio::test]
    async fn test_resolve_request_credentials_dynamic_with_session_token() {
        let creds = AWSIAMCredentials::Dynamic {
            access_key_id_key: "aws_access_key".to_string(),
            secret_access_key_key: "aws_secret_key".to_string(),
            session_token_key: Some("aws_session_token".to_string()),
        };
        let sdk_config = SdkConfig::builder().build();
        let dynamic_api_keys = make_credentials(HashMap::from([
            ("aws_access_key", "AKIADYNAMIC"),
            ("aws_secret_key", "dynamicsecret"),
            ("aws_session_token", "dynamicsession"),
        ]));

        let result = resolve_request_credentials(
            &creds,
            &sdk_config,
            &dynamic_api_keys,
            "test",
            ApiType::ChatCompletions,
        )
        .await
        .expect("resolve should succeed");
        assert_eq!(
            result.session_token(),
            Some("dynamicsession"),
            "Dynamic session token should resolve from request"
        );
    }

    #[tokio::test]
    async fn test_resolve_request_credentials_dynamic_missing_access_key() {
        let creds = AWSIAMCredentials::Dynamic {
            access_key_id_key: "missing_access_key".to_string(),
            secret_access_key_key: "aws_secret_key".to_string(),
            session_token_key: None,
        };
        let sdk_config = SdkConfig::builder().build();
        let dynamic_api_keys =
            make_credentials(HashMap::from([("aws_secret_key", "dynamicsecret")]));

        let result = resolve_request_credentials(
            &creds,
            &sdk_config,
            &dynamic_api_keys,
            "test",
            ApiType::ChatCompletions,
        )
        .await;
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

    #[tokio::test]
    async fn test_resolve_request_credentials_dynamic_missing_secret_key() {
        let creds = AWSIAMCredentials::Dynamic {
            access_key_id_key: "aws_access_key".to_string(),
            secret_access_key_key: "missing_secret_key".to_string(),
            session_token_key: None,
        };
        let sdk_config = SdkConfig::builder().build();
        let dynamic_api_keys = make_credentials(HashMap::from([("aws_access_key", "AKIADYNAMIC")]));

        let result = resolve_request_credentials(
            &creds,
            &sdk_config,
            &dynamic_api_keys,
            "test",
            ApiType::ChatCompletions,
        )
        .await;
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

    #[tokio::test]
    async fn test_resolve_request_credentials_dynamic_missing_session_token() {
        let creds = AWSIAMCredentials::Dynamic {
            access_key_id_key: "aws_access_key".to_string(),
            secret_access_key_key: "aws_secret_key".to_string(),
            session_token_key: Some("missing_session".to_string()),
        };
        let sdk_config = SdkConfig::builder().build();
        let dynamic_api_keys = make_credentials(HashMap::from([
            ("aws_access_key", "AKIADYNAMIC"),
            ("aws_secret_key", "dynamicsecret"),
        ]));

        let result = resolve_request_credentials(
            &creds,
            &sdk_config,
            &dynamic_api_keys,
            "test",
            ApiType::ChatCompletions,
        )
        .await;
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

    #[tokio::test]
    async fn test_resolve_request_credentials_sdk_no_provider() {
        let creds = AWSIAMCredentials::Sdk;
        let sdk_config = SdkConfig::builder().build();
        let dynamic_api_keys = make_credentials(HashMap::new());

        let result = resolve_request_credentials(
            &creds,
            &sdk_config,
            &dynamic_api_keys,
            "test",
            ApiType::ChatCompletions,
        )
        .await;
        assert!(
            result.is_err(),
            "Sdk credentials should error when no credentials provider is configured"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("No credentials provider"),
            "Error should explain no provider is configured: {err}"
        );
    }

    // ===== AWSCredentials::from_fields validation tests =====

    #[test]
    fn test_aws_credentials_from_fields_none_returns_sdk() {
        let result = AWSIAMCredentials::from_fields(None, None, None, "test_provider")
            .expect("should succeed");
        assert!(
            matches!(result, AWSIAMCredentials::Sdk),
            "No credentials should return Sdk"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_session_token_without_credentials_errors() {
        let result = AWSIAMCredentials::from_fields(
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
        let result = AWSIAMCredentials::from_fields(
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
        let result = AWSIAMCredentials::from_fields(
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
        let creds = AWSIAMCredentials::from_fields(
            Some(CredentialLocation::Dynamic("ak".to_string())),
            Some(CredentialLocation::Dynamic("sk".to_string())),
            Some(CredentialLocation::Dynamic("st".to_string())),
            "test_provider",
        )
        .expect("should succeed");

        assert!(
            matches!(creds, AWSIAMCredentials::Dynamic { .. }),
            "Should be Dynamic credentials"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_all_sdk_succeeds() {
        let creds = AWSIAMCredentials::from_fields(
            Some(CredentialLocation::Sdk),
            Some(CredentialLocation::Sdk),
            Some(CredentialLocation::Sdk),
            "test_provider",
        )
        .expect("should succeed");

        assert!(
            matches!(creds, AWSIAMCredentials::Sdk),
            "Should be Sdk credentials"
        );
    }

    #[test]
    fn test_aws_credentials_from_fields_mixed_dynamic_env_errors() {
        let result = AWSIAMCredentials::from_fields(
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
        let result = AWSIAMCredentials::from_fields(
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
        let result = AWSIAMCredentials::from_fields(
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
        let static_creds = AWSIAMCredentials::Static {
            access_key_id: "ak".to_string(),
            secret_access_key: SecretString::new("sk".to_string().into()),
            session_token: None,
        };

        // This should trigger warning internally (we can't verify without tracing subscriber)
        warn_if_credential_exfiltration_risk(&dynamic_endpoint, &static_creds, "test");

        // Dynamic endpoint with SDK creds - should warn
        let sdk_creds = AWSIAMCredentials::Sdk;
        warn_if_credential_exfiltration_risk(&dynamic_endpoint, &sdk_creds, "test");

        // Dynamic endpoint with dynamic creds - should NOT warn
        let dynamic_creds = AWSIAMCredentials::Dynamic {
            access_key_id_key: "ak".to_string(),
            secret_access_key_key: "sk".to_string(),
            session_token_key: None,
        };
        warn_if_credential_exfiltration_risk(&dynamic_endpoint, &dynamic_creds, "test");

        // Static endpoint with static creds - should NOT warn
        let static_endpoint = Some(AWSEndpointUrl::Static(
            Url::parse("https://bedrock.amazonaws.com").unwrap(),
        ));
        warn_if_credential_exfiltration_risk(&static_endpoint, &static_creds, "test");

        // No endpoint - should NOT warn
        warn_if_credential_exfiltration_risk(&None, &static_creds, "test");
    }

    // ===== AWSBedrockCredentials::from_fields priority tests =====

    #[tokio::test]
    async fn test_bedrock_credentials_priority_explicit_api_key_over_env_var() {
        // Set env var that would otherwise be used
        tensorzero_unsafe_helpers::set_env_var_tests_only(
            "AWS_BEARER_TOKEN_BEDROCK",
            "env_token_should_not_be_used",
        );

        let region = AWSRegion::Static(Region::new("us-east-1"));

        // Set an env var for the explicit api_key to use
        tensorzero_unsafe_helpers::set_env_var_tests_only(
            "TEST_EXPLICIT_API_KEY",
            "explicit_api_key_value",
        );

        let (creds, _) = AWSBedrockCredentials::from_fields(
            Some(CredentialLocation::Env("TEST_EXPLICIT_API_KEY".to_string())),
            None,
            None,
            None,
            &region,
            "test",
        )
        .await
        .expect("Explicit api_key should succeed");

        assert!(
            matches!(creds, AWSBedrockCredentials::ApiKey(_)),
            "Explicit api_key should take priority and return ApiKey variant"
        );

        // Clean up
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_BEARER_TOKEN_BEDROCK");
        tensorzero_unsafe_helpers::remove_env_var_tests_only("TEST_EXPLICIT_API_KEY");
    }

    #[tokio::test]
    async fn test_bedrock_credentials_priority_explicit_iam_over_env_var() {
        // Set env var that would otherwise be used
        tensorzero_unsafe_helpers::set_env_var_tests_only(
            "AWS_BEARER_TOKEN_BEDROCK",
            "env_token_should_not_be_used",
        );

        let region = AWSRegion::Static(Region::new("us-east-1"));

        // Set env vars for explicit IAM credentials
        tensorzero_unsafe_helpers::set_env_var_tests_only("TEST_AWS_ACCESS_KEY", "AKIATEST");
        tensorzero_unsafe_helpers::set_env_var_tests_only("TEST_AWS_SECRET_KEY", "secrettest");

        let (creds, _) = AWSBedrockCredentials::from_fields(
            None,
            Some(CredentialLocation::Env("TEST_AWS_ACCESS_KEY".to_string())),
            Some(CredentialLocation::Env("TEST_AWS_SECRET_KEY".to_string())),
            None,
            &region,
            "test",
        )
        .await
        .expect("Explicit IAM credentials should succeed");

        assert!(
            matches!(creds, AWSBedrockCredentials::IAM { .. }),
            "Explicit IAM credentials should take priority and return IAM variant"
        );

        // Clean up
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_BEARER_TOKEN_BEDROCK");
        tensorzero_unsafe_helpers::remove_env_var_tests_only("TEST_AWS_ACCESS_KEY");
        tensorzero_unsafe_helpers::remove_env_var_tests_only("TEST_AWS_SECRET_KEY");
    }

    #[tokio::test]
    async fn test_bedrock_credentials_priority_env_var_when_nothing_configured() {
        // Clear any explicit config and set the env var
        tensorzero_unsafe_helpers::set_env_var_tests_only(
            "AWS_BEARER_TOKEN_BEDROCK",
            "bearer_token_from_env",
        );

        let region = AWSRegion::Static(Region::new("us-east-1"));

        let (creds, _) = AWSBedrockCredentials::from_fields(
            None, // no api_key
            None, // no access_key_id
            None, // no secret_access_key
            None, // no session_token
            &region, "test",
        )
        .await
        .expect("Env var AWS_BEARER_TOKEN_BEDROCK should be used");

        assert!(
            matches!(creds, AWSBedrockCredentials::ApiKey(_)),
            "AWS_BEARER_TOKEN_BEDROCK env var should be used when nothing else is configured"
        );

        // Clean up
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_BEARER_TOKEN_BEDROCK");
    }

    #[tokio::test]
    async fn test_bedrock_credentials_priority_empty_env_var_falls_back_to_sdk() {
        // Set env var to empty string - should be ignored
        tensorzero_unsafe_helpers::set_env_var_tests_only("AWS_BEARER_TOKEN_BEDROCK", "");

        let region = AWSRegion::Static(Region::new("us-east-1"));

        let (creds, _) = AWSBedrockCredentials::from_fields(
            None, // no api_key
            None, // no access_key_id
            None, // no secret_access_key
            None, // no session_token
            &region, "test",
        )
        .await
        .expect("Empty env var should fall back to SDK");

        assert!(
            matches!(
                creds,
                AWSBedrockCredentials::IAM {
                    credentials: AWSIAMCredentials::Sdk,
                    ..
                }
            ),
            "Empty AWS_BEARER_TOKEN_BEDROCK should fall back to SDK credential chain"
        );

        // Clean up
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_BEARER_TOKEN_BEDROCK");
    }

    #[tokio::test]
    async fn test_bedrock_credentials_priority_no_env_var_falls_back_to_sdk() {
        // Ensure env var is not set
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_BEARER_TOKEN_BEDROCK");

        let region = AWSRegion::Static(Region::new("us-east-1"));

        let (creds, _) = AWSBedrockCredentials::from_fields(
            None, // no api_key
            None, // no access_key_id
            None, // no secret_access_key
            None, // no session_token
            &region, "test",
        )
        .await
        .expect("No credentials should fall back to SDK");

        assert!(
            matches!(
                creds,
                AWSBedrockCredentials::IAM {
                    credentials: AWSIAMCredentials::Sdk,
                    ..
                }
            ),
            "No credentials configured should fall back to SDK credential chain"
        );
    }
}
