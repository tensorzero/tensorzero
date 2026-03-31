use serde::{Deserialize, Serialize};

/// Stored version of `CredentialLocation`.
///
/// Internally tagged with `"type"` so the JSON shape is explicit and typed,
/// e.g. `{"type": "env", "value": "MY_API_KEY"}` or `{"type": "sdk"}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredCredentialLocation {
    Env { value: String },
    PathFromEnv { value: String },
    Dynamic { value: String },
    Path { value: String },
    Sdk,
    None,
}

/// Stored version of `CredentialLocationOrHardcoded`.
///
/// A credential that can either be a hardcoded string value or a standard
/// credential location (env var, path, etc.).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredCredentialLocationOrHardcoded {
    Hardcoded { value: String },
    Location { location: StoredCredentialLocation },
}

/// Stored version of `EndpointLocation`.
///
/// An endpoint URL that can come from an env var, dynamic resolution, or a static string.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredEndpointLocation {
    Env { value: String },
    Dynamic { value: String },
    Static { value: String },
}

/// Stored version of `CredentialLocationWithFallback`.
///
/// Either a single credential location or an object with
/// `default` and `fallback` fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredCredentialLocationWithFallback {
    Single {
        location: StoredCredentialLocation,
    },
    WithFallback {
        default: StoredCredentialLocation,
        fallback: StoredCredentialLocation,
    },
}
