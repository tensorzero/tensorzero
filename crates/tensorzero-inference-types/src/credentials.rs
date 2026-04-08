use std::sync::Arc;

use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::ModelInferenceRequest;
use crate::extra_body::ExtraBodyConfig;
use crate::extra_headers::ExtraHeadersConfig;

// =============================================================================
// CredentialLocation
// =============================================================================

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum CredentialLocation {
    /// Environment variable containing the actual credential
    Env(String),
    /// Environment variable containing the path to a credential file
    PathFromEnv(String),
    /// For dynamic credential resolution
    Dynamic(String),
    /// Direct path to a credential file
    Path(String),
    /// Use a provider-specific SDK to determine credentials
    Sdk,
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
        } else if s == "sdk" {
            Ok(CredentialLocation::Sdk)
        } else if s == "none" {
            Ok(CredentialLocation::None)
        } else {
            Err(serde::de::Error::custom(format!(
                "Invalid credential location format: `{s}`. \
                 Use `env::VAR_NAME`, `path::FILE_PATH`, `dynamic::KEY_NAME`, or `sdk`."
            )))
        }
    }
}

impl Serialize for CredentialLocation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            CredentialLocation::Env(inner) => format!("env::{inner}"),
            CredentialLocation::PathFromEnv(inner) => format!("path_from_env::{inner}"),
            CredentialLocation::Dynamic(inner) => format!("dynamic::{inner}"),
            CredentialLocation::Path(inner) => format!("path::{inner}"),
            CredentialLocation::Sdk => "sdk".to_string(),
            CredentialLocation::None => "none".to_string(),
        };
        serializer.serialize_str(&s)
    }
}

// =============================================================================
// CredentialLocationWithFallback
// =============================================================================

/// Credential location with optional fallback support
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, PartialEq, Clone, Serialize)]
#[serde(untagged)]
pub enum CredentialLocationWithFallback {
    /// Single credential location (backward compatible)
    Single(#[cfg_attr(feature = "ts-bindings", ts(type = "string"))] CredentialLocation),
    /// Credential location with fallback
    WithFallback {
        #[cfg_attr(feature = "ts-bindings", ts(type = "string"))]
        default: CredentialLocation,
        #[cfg_attr(feature = "ts-bindings", ts(type = "string"))]
        fallback: CredentialLocation,
    },
}

impl CredentialLocationWithFallback {
    /// Get the default (primary) credential location
    pub fn default_location(&self) -> &CredentialLocation {
        match self {
            CredentialLocationWithFallback::Single(loc) => loc,
            CredentialLocationWithFallback::WithFallback { default, .. } => default,
        }
    }

    /// Get the fallback credential location if present
    pub fn fallback_location(&self) -> Option<&CredentialLocation> {
        match self {
            CredentialLocationWithFallback::Single(_) => None,
            CredentialLocationWithFallback::WithFallback { fallback, .. } => Some(fallback),
        }
    }
}

impl<'de> Deserialize<'de> for CredentialLocationWithFallback {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, MapAccess, Visitor};
        use std::fmt;

        struct CredentialLocationWithFallbackVisitor;

        impl<'de> Visitor<'de> for CredentialLocationWithFallbackVisitor {
            type Value = CredentialLocationWithFallback;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string or an object with 'default' and 'fallback' fields")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                // Parse as single CredentialLocation
                let location =
                    CredentialLocation::deserialize(serde::de::value::StrDeserializer::new(value))?;
                Ok(CredentialLocationWithFallback::Single(location))
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut default = None;
                let mut fallback = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "default" => {
                            if default.is_some() {
                                return Err(Error::duplicate_field("default"));
                            }
                            let value: String = map.next_value()?;
                            default = Some(CredentialLocation::deserialize(
                                serde::de::value::StrDeserializer::new(&value),
                            )?);
                        }
                        "fallback" => {
                            if fallback.is_some() {
                                return Err(Error::duplicate_field("fallback"));
                            }
                            let value: String = map.next_value()?;
                            fallback = Some(CredentialLocation::deserialize(
                                serde::de::value::StrDeserializer::new(&value),
                            )?);
                        }
                        _ => {
                            return Err(Error::unknown_field(&key, &["default", "fallback"]));
                        }
                    }
                }

                let default = default.ok_or_else(|| Error::missing_field("default"))?;
                let fallback = fallback.ok_or_else(|| Error::missing_field("fallback"))?;

                Ok(CredentialLocationWithFallback::WithFallback { default, fallback })
            }
        }

        deserializer.deserialize_any(CredentialLocationWithFallbackVisitor)
    }
}

// =============================================================================
// CredentialLocationOrHardcoded
// =============================================================================

/// Credential location that also allows hardcoded string values.
/// Used for non-sensitive fields like AWS region and endpoint_url.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum CredentialLocationOrHardcoded {
    /// Hardcoded value (e.g., region = "us-east-1")
    Hardcoded(String),
    /// Standard credential location (env::, dynamic::, sdk, etc.)
    #[cfg_attr(feature = "ts-bindings", ts(type = "string"))]
    Location(CredentialLocation),
}

impl<'de> Deserialize<'de> for CredentialLocationOrHardcoded {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        // Try to parse as CredentialLocation first
        if let Some(inner) = s.strip_prefix("env::") {
            Ok(CredentialLocationOrHardcoded::Location(
                CredentialLocation::Env(inner.to_string()),
            ))
        } else if let Some(inner) = s.strip_prefix("path_from_env::") {
            Ok(CredentialLocationOrHardcoded::Location(
                CredentialLocation::PathFromEnv(inner.to_string()),
            ))
        } else if let Some(inner) = s.strip_prefix("dynamic::") {
            Ok(CredentialLocationOrHardcoded::Location(
                CredentialLocation::Dynamic(inner.to_string()),
            ))
        } else if let Some(inner) = s.strip_prefix("path::") {
            Ok(CredentialLocationOrHardcoded::Location(
                CredentialLocation::Path(inner.to_string()),
            ))
        } else if s == "sdk" {
            Ok(CredentialLocationOrHardcoded::Location(
                CredentialLocation::Sdk,
            ))
        } else if s == "none" {
            Ok(CredentialLocationOrHardcoded::Location(
                CredentialLocation::None,
            ))
        } else {
            // Treat as hardcoded value
            Ok(CredentialLocationOrHardcoded::Hardcoded(s))
        }
    }
}

impl Serialize for CredentialLocationOrHardcoded {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            CredentialLocationOrHardcoded::Hardcoded(inner) => serializer.serialize_str(inner),
            CredentialLocationOrHardcoded::Location(loc) => loc.serialize(serializer),
        }
    }
}

// =============================================================================
// EndpointLocation
// =============================================================================

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum EndpointLocation {
    /// Environment variable containing the actual endpoint URL
    Env(String),
    /// For dynamic endpoint resolution
    Dynamic(String),
    /// Direct endpoint URL
    Static(String),
}

impl<'de> Deserialize<'de> for EndpointLocation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if let Some(inner) = s.strip_prefix("env::") {
            Ok(EndpointLocation::Env(inner.to_string()))
        } else if let Some(inner) = s.strip_prefix("dynamic::") {
            Ok(EndpointLocation::Dynamic(inner.to_string()))
        } else {
            // Default to static endpoint
            Ok(EndpointLocation::Static(s))
        }
    }
}

impl Serialize for EndpointLocation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            EndpointLocation::Env(inner) => format!("env::{inner}"),
            EndpointLocation::Dynamic(inner) => format!("dynamic::{inner}"),
            EndpointLocation::Static(inner) => inner.clone(),
        };
        serializer.serialize_str(&s)
    }
}

// =============================================================================
// Credential
// =============================================================================

#[derive(Clone, Debug)]
pub enum Credential {
    Static(SecretString),
    FileContents(SecretString),
    Dynamic(String),
    Sdk,
    None,
    Missing,
    WithFallback {
        default: Box<Credential>,
        fallback: Box<Credential>,
    },
}

// =============================================================================
// ModelProviderRequestInfo
// =============================================================================

#[derive(Clone, Debug)]
pub struct ModelProviderRequestInfo {
    pub provider_name: Arc<str>,
    pub extra_headers: Option<ExtraHeadersConfig>,
    pub extra_body: Option<ExtraBodyConfig>,
    pub discard_unknown_chunks: bool,
}

impl From<&ModelProviderRequestInfo> for ModelProviderRequestInfo {
    fn from(val: &ModelProviderRequestInfo) -> Self {
        val.clone()
    }
}

// =============================================================================
// ProviderInferenceRequest
// =============================================================================

/// Provider-facing inference request without core-internal fields like `otlp_config`.
#[derive(Debug)]
pub struct ProviderInferenceRequest<'request> {
    pub request: &'request ModelInferenceRequest<'request>,
    pub model_name: &'request str,
    pub provider_name: &'request str,
    pub model_inference_id: Uuid,
}

// We need a manual impl to avoid adding a bound on the lifetime parameter
impl Copy for ProviderInferenceRequest<'_> {}
impl Clone for ProviderInferenceRequest<'_> {
    fn clone(&self) -> Self {
        *self
    }
}
