use serde::{Deserialize, Serialize};
use tensorzero_types::rate_limiting_types::{
    ApiKeyPublicIdConfigScope, ApiKeyPublicIdValueScope, RateLimitResource,
    RateLimitingConfigPriority, RateLimitingConfigScope, RateLimitingConfigScopes,
    TagRateLimitingConfigScope, TagValueScope,
};

pub const STORED_RATE_LIMITING_CONFIG_SCHEMA_REVISION: i32 = 1;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRateLimitingConfig {
    pub rules: Option<Vec<StoredRateLimitingRule>>,
    pub enabled: Option<bool>,
    pub backend: Option<StoredRateLimitingBackend>,
    pub default_nano_cost: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredRateLimitingBackend {
    Auto,
    Postgres,
    Valkey,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRateLimitingRule {
    pub limits: Vec<StoredRateLimit>,
    pub scope: StoredRateLimitingConfigScopes,
    pub priority: StoredRateLimitingConfigPriority,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRateLimitingConfigScopes {
    pub scopes: Vec<StoredRateLimitingConfigScope>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredRateLimitingConfigScope {
    Tag(StoredTagRateLimitingConfigScope),
    ApiKeyPublicId(StoredApiKeyPublicIdConfigScope),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredTagRateLimitingConfigScope {
    pub tag_key: String,
    pub tag_value: StoredTagValueScope,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredTagValueScope {
    Concrete { value: String },
    Each,
    Total,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredApiKeyPublicIdConfigScope {
    pub api_key_public_id: StoredApiKeyPublicIdValueScope,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredApiKeyPublicIdValueScope {
    Concrete { value: String },
    Each,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredRateLimitingConfigPriority {
    Priority { value: usize },
    Always,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRateLimit {
    pub resource: StoredRateLimitResource,
    pub interval: StoredRateLimitInterval,
    pub capacity: u64,
    pub refill_rate: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredRateLimitResource {
    ModelInference,
    Token,
    Cost,
}

// `RateLimitResource` lives in `tensorzero-error`, and `StoredRateLimitResource`
// lives here, so this `From` impl has to live in this crate to satisfy the
// orphan rule.
impl From<StoredRateLimitResource> for RateLimitResource {
    fn from(stored: StoredRateLimitResource) -> Self {
        match stored {
            StoredRateLimitResource::ModelInference => RateLimitResource::ModelInference,
            StoredRateLimitResource::Token => RateLimitResource::Token,
            StoredRateLimitResource::Cost => RateLimitResource::Cost,
        }
    }
}

impl From<RateLimitResource> for StoredRateLimitResource {
    fn from(resource: RateLimitResource) -> Self {
        match resource {
            RateLimitResource::ModelInference => Self::ModelInference,
            RateLimitResource::Token => Self::Token,
            RateLimitResource::Cost => Self::Cost,
        }
    }
}

// The core rate limiting scope/priority types live in `tensorzero-error`, and
// their stored counterparts live here, so the conversion impls must live in
// this crate to satisfy the orphan rule.

impl From<&RateLimitingConfigScopes> for StoredRateLimitingConfigScopes {
    fn from(scopes: &RateLimitingConfigScopes) -> Self {
        StoredRateLimitingConfigScopes {
            scopes: scopes.as_slice().iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<StoredRateLimitingConfigScopes> for RateLimitingConfigScopes {
    type Error = &'static str;

    fn try_from(stored: StoredRateLimitingConfigScopes) -> Result<Self, Self::Error> {
        let scopes = stored.scopes.into_iter().map(Into::into).collect();
        RateLimitingConfigScopes::new(scopes)
    }
}

impl From<&RateLimitingConfigScope> for StoredRateLimitingConfigScope {
    fn from(scope: &RateLimitingConfigScope) -> Self {
        match scope {
            RateLimitingConfigScope::Tag(tag) => {
                StoredRateLimitingConfigScope::Tag(StoredTagRateLimitingConfigScope {
                    tag_key: tag.tag_key().to_string(),
                    tag_value: match tag.tag_value() {
                        TagValueScope::Concrete(s) => {
                            StoredTagValueScope::Concrete { value: s.clone() }
                        }
                        TagValueScope::Each => StoredTagValueScope::Each,
                        TagValueScope::Total => StoredTagValueScope::Total,
                    },
                })
            }
            RateLimitingConfigScope::ApiKeyPublicId(api_key) => {
                StoredRateLimitingConfigScope::ApiKeyPublicId(StoredApiKeyPublicIdConfigScope {
                    api_key_public_id: match api_key.api_key_public_id() {
                        ApiKeyPublicIdValueScope::Concrete(s) => {
                            StoredApiKeyPublicIdValueScope::Concrete { value: s.clone() }
                        }
                        ApiKeyPublicIdValueScope::Each => StoredApiKeyPublicIdValueScope::Each,
                    },
                })
            }
        }
    }
}

impl From<StoredRateLimitingConfigScope> for RateLimitingConfigScope {
    fn from(stored: StoredRateLimitingConfigScope) -> Self {
        match stored {
            StoredRateLimitingConfigScope::Tag(tag) => {
                let tag_value = match tag.tag_value {
                    StoredTagValueScope::Concrete { value } => TagValueScope::Concrete(value),
                    StoredTagValueScope::Each => TagValueScope::Each,
                    StoredTagValueScope::Total => TagValueScope::Total,
                };
                RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
                    tag.tag_key,
                    tag_value,
                ))
            }
            StoredRateLimitingConfigScope::ApiKeyPublicId(api_key) => {
                let value = match api_key.api_key_public_id {
                    StoredApiKeyPublicIdValueScope::Concrete { value } => {
                        ApiKeyPublicIdValueScope::Concrete(value)
                    }
                    StoredApiKeyPublicIdValueScope::Each => ApiKeyPublicIdValueScope::Each,
                };
                RateLimitingConfigScope::ApiKeyPublicId(ApiKeyPublicIdConfigScope::new(value))
            }
        }
    }
}

impl From<&RateLimitingConfigPriority> for StoredRateLimitingConfigPriority {
    fn from(priority: &RateLimitingConfigPriority) -> Self {
        match priority {
            RateLimitingConfigPriority::Priority(value) => {
                StoredRateLimitingConfigPriority::Priority { value: *value }
            }
            RateLimitingConfigPriority::Always => StoredRateLimitingConfigPriority::Always,
        }
    }
}

impl From<StoredRateLimitingConfigPriority> for RateLimitingConfigPriority {
    fn from(stored: StoredRateLimitingConfigPriority) -> Self {
        match stored {
            StoredRateLimitingConfigPriority::Priority { value } => {
                RateLimitingConfigPriority::Priority(value)
            }
            StoredRateLimitingConfigPriority::Always => RateLimitingConfigPriority::Always,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredRateLimitInterval {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
}
