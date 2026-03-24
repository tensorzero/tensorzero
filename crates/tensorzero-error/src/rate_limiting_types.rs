use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize, Serializer};

/// 1 dollar = 1,000,000,000 nano-dollars
pub const NANO_DOLLARS_PER_DOLLAR: u64 = 1_000_000_000;

/// Convert nano-dollars (u64) back to dollars (f64).
pub fn nano_cost_to_cost(nano_cost: u64) -> f64 {
    nano_cost as f64 / NANO_DOLLARS_PER_DOLLAR as f64
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum RateLimitResource {
    ModelInference,
    Token,
    Cost,
}

impl RateLimitResource {
    /// Returns the snake_case string representation matching the serde serialization
    pub fn as_str(&self) -> &'static str {
        match self {
            RateLimitResource::ModelInference => "model_inference",
            RateLimitResource::Token => "token",
            RateLimitResource::Cost => "cost",
        }
    }
}

/// Type that lists the different ways a Scope + matching ScopeInfo can be
/// serialized into a key.
/// We need this struct to have stable serialization behavior because we want rate limits to be stable
/// across releases.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum RateLimitingScopeKey {
    TagTotal { key: String },
    TagEach { key: String, value: String },
    TagConcrete { key: String, value: String },
    ApiKeyPublicIdEach { api_key_public_id: Arc<str> },
    ApiKeyPublicIdConcrete { api_key_public_id: Arc<str> },
}

impl RateLimitingScopeKey {
    /// Returns a representation that matches the config format for easy lookup
    pub fn to_config_representation(&self) -> String {
        match self {
            RateLimitingScopeKey::TagTotal { key } => {
                format!(r#"tag_key="{key}", tag_value="tensorzero::total""#)
            }
            RateLimitingScopeKey::TagEach { key, value } => {
                format!(r#"tag_key="{key}", tag_value="tensorzero::each" (matched: "{value}")"#)
            }
            RateLimitingScopeKey::TagConcrete { key, value } => {
                format!(r#"tag_key="{key}", tag_value="{value}""#)
            }
            RateLimitingScopeKey::ApiKeyPublicIdEach { api_key_public_id } => {
                format!(r#"api_key_public_id="tensorzero::each" (matched: "{api_key_public_id}")"#)
            }
            RateLimitingScopeKey::ApiKeyPublicIdConcrete { api_key_public_id } => {
                format!(r#"api_key_public_id="{api_key_public_id}""#)
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Eq, Hash)]
pub struct ActiveRateLimitKey(pub String);

impl ActiveRateLimitKey {
    pub fn new(key: String) -> Self {
        ActiveRateLimitKey(key)
    }
}

impl std::fmt::Display for ActiveRateLimitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, PartialEq)]
pub struct FailedRateLimit {
    pub key: ActiveRateLimitKey,
    /// Raw value in internal units (nano-dollars for Cost, count for others)
    pub requested: u64,
    /// Raw value in internal units (nano-dollars for Cost, count for others)
    pub available: u64,
    pub resource: RateLimitResource,
    pub scope_key: Vec<RateLimitingScopeKey>,
}

impl Serialize for FailedRateLimit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("FailedRateLimit", 5)?;
        state.serialize_field("key", &self.key)?;
        // For Cost resources, convert nano-dollars to dollars for display
        match self.resource {
            RateLimitResource::Cost => {
                state.serialize_field("requested", &nano_cost_to_cost(self.requested))?;
                state.serialize_field("available", &nano_cost_to_cost(self.available))?;
            }
            _ => {
                state.serialize_field("requested", &self.requested)?;
                state.serialize_field("available", &self.available)?;
            }
        }
        state.serialize_field("resource", &self.resource)?;
        state.serialize_field("scope_key", &self.scope_key)?;
        state.end()
    }
}

/// Wrapper type for rate limiting scopes.
/// Forces them to be sorted on construction
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(try_from = "Vec<RateLimitingConfigScope>")]
pub struct RateLimitingConfigScopes(Vec<RateLimitingConfigScope>);

impl TryFrom<Vec<RateLimitingConfigScope>> for RateLimitingConfigScopes {
    type Error = &'static str;

    fn try_from(scopes: Vec<RateLimitingConfigScope>) -> Result<Self, Self::Error> {
        Self::new(scopes)
    }
}

impl RateLimitingConfigScopes {
    /// Creates a new instance of `RateLimitingConfigScopes`.
    /// Ensures that there are no duplicate scopes and sorts them.
    pub fn new(mut scopes: Vec<RateLimitingConfigScope>) -> Result<Self, &'static str> {
        // First, we check to make sure there are no duplicate scopes
        if scopes.len() != scopes.iter().collect::<HashSet<_>>().len() {
            return Err("duplicate scopes are not allowed in the same rule");
        }
        // We sort the scopes so we are guaranteed a
        // stable order when generating the key
        scopes.sort();
        Ok(RateLimitingConfigScopes(scopes))
    }

    /// Creates an empty `RateLimitingConfigScopes`.
    pub fn empty() -> Self {
        RateLimitingConfigScopes(vec![])
    }

    /// Returns a slice of the inner scopes.
    pub fn as_slice(&self) -> &[RateLimitingConfigScope] {
        &self.0
    }

    /// Returns the number of scopes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if there are no scopes.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a reference to the scope at the given index, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&RateLimitingConfigScope> {
        self.0.get(index)
    }
}

impl std::ops::Index<usize> for RateLimitingConfigScopes {
    type Output = RateLimitingConfigScope;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

// IMPORTANT: the types below are used to set up scopes and keys for rate limiting.
// We need the keys that have already been used in production to remain stable.
// So, we cannot change the sort order.
// As scope types are added, please append new ones at the end to maintain a stable sort order,
// and add a test each time that ensures the sort order is maintained as further changes are made.
//
// Note to reviewer:  what else could we do to ensure the sort order is maintained across future changes?
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(untagged)]
pub enum RateLimitingConfigScope {
    Tag(TagRateLimitingConfigScope),
    ApiKeyPublicId(ApiKeyPublicIdConfigScope),
    // model_name = "my_model"
    // function_name = "my_function"
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct TagRateLimitingConfigScope {
    tag_key: String,
    tag_value: TagValueScope,
}

impl TagRateLimitingConfigScope {
    /// Creates a new `TagRateLimitingConfigScope`.
    pub fn new(tag_key: String, tag_value: TagValueScope) -> Self {
        Self { tag_key, tag_value }
    }

    pub fn tag_key(&self) -> &str {
        &self.tag_key
    }

    pub fn tag_value(&self) -> &TagValueScope {
        &self.tag_value
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ApiKeyPublicIdConfigScope {
    api_key_public_id: ApiKeyPublicIdValueScope,
}

impl ApiKeyPublicIdConfigScope {
    /// Creates a new `ApiKeyPublicIdConfigScope`.
    pub fn new(api_key_public_id: ApiKeyPublicIdValueScope) -> Self {
        Self { api_key_public_id }
    }

    pub fn api_key_public_id(&self) -> &ApiKeyPublicIdValueScope {
        &self.api_key_public_id
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum TagValueScope {
    Concrete(String),
    Each,
    Total,
}

impl Serialize for TagValueScope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            TagValueScope::Concrete(s) => serializer.serialize_str(s),
            TagValueScope::Each => serializer.serialize_str("tensorzero::each"),
            TagValueScope::Total => serializer.serialize_str("tensorzero::total"),
        }
    }
}

impl<'de> Deserialize<'de> for TagValueScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "tensorzero::each" => Ok(TagValueScope::Each),
            "tensorzero::total" => Ok(TagValueScope::Total),
            _ if s.starts_with("tensorzero::") => Err(serde::de::Error::custom(
                r#"Tag values in rate limiting scopes besides tensorzero::each and tensorzero::total may not start with "tensorzero::"."#,
            )),
            _ => Ok(TagValueScope::Concrete(s)),
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum ApiKeyPublicIdValueScope {
    Concrete(String),
    Each,
}

/// Length of the public ID portion of a TensorZero API key.
const PUBLIC_ID_LENGTH: usize = 12;

impl Serialize for ApiKeyPublicIdValueScope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ApiKeyPublicIdValueScope::Concrete(s) => serializer.serialize_str(s),
            ApiKeyPublicIdValueScope::Each => serializer.serialize_str("tensorzero::each"),
        }
    }
}

impl<'de> Deserialize<'de> for ApiKeyPublicIdValueScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s == "tensorzero::each" {
            Ok(ApiKeyPublicIdValueScope::Each)
        } else if s.starts_with("tensorzero::") {
            Err(serde::de::Error::custom(
                r#"API key public ID values in rate limiting scopes besides tensorzero::each may not start with "tensorzero::"."#,
            ))
        } else if s.len() != PUBLIC_ID_LENGTH {
            Err(serde::de::Error::custom(format!(
                "API key public ID `{s}` must be {PUBLIC_ID_LENGTH} characters long. Check that this is a TensorZero API key public ID."
            )))
        } else {
            Ok(ApiKeyPublicIdValueScope::Concrete(s))
        }
    }
}
