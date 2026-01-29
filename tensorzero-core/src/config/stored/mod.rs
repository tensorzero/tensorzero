//! Stored config types for backward-compatible deserialization of historical snapshots.
//!
//! When deprecating a config field:
//! 1. Remove it from the `Uninitialized*` type (fresh configs will reject it)
//! 2. Keep it in the `Stored*` type (snapshots can still load)
//! 3. Implement migration in `From<Stored*> for Uninitialized*`
//!
//! The `From` implementations use explicit destructuring to ensure compile-time errors
//! when fields are added or removed from either the stored or uninitialized types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::gateway::UninitializedGatewayConfig;
use crate::config::provider_types::ProviderTypesConfig;
use crate::config::{
    MetricConfig, PostgresConfig, TimeoutsConfig, UninitializedConfig, UninitializedFunctionConfig,
    UninitializedToolConfig,
};
use crate::embeddings::{UninitializedEmbeddingModelConfig, UninitializedEmbeddingProviderConfig};
use crate::evaluations::UninitializedEvaluationConfig;
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::inference::types::storage::StorageKind;
use crate::model::UninitializedModelConfig;
use crate::model::UninitializedProviderConfig;
use crate::optimization::UninitializedOptimizerInfo;
use crate::rate_limiting::{
    ApiKeyPublicIdConfigScope, ApiKeyPublicIdValueScope, RateLimit, RateLimitingBackend,
    RateLimitingConfigPriority, RateLimitingConfigRule, RateLimitingConfigScope,
    RateLimitingConfigScopes, TagRateLimitingConfigScope, TagValueScope,
    UninitializedRateLimitingConfig,
};

// ============================================================================
// Rate Limiting Stored Types
// ============================================================================
//
// These types handle deserialization of rate limiting config from JSON format.
// The runtime types use custom Deserialize impls for TOML shorthand format,
// but when serialized to JSON for storage, they produce a different format
// that needs these stored types to deserialize.

/// Stored version of `RateLimitingConfigPriority`.
///
/// The runtime type has a custom deserializer expecting `always`/`priority` fields,
/// but serializes as `{"Priority": n}` or `"Always"`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum StoredRateLimitingConfigPriority {
    Priority(usize),
    Always,
}

impl From<StoredRateLimitingConfigPriority> for RateLimitingConfigPriority {
    fn from(stored: StoredRateLimitingConfigPriority) -> Self {
        match stored {
            StoredRateLimitingConfigPriority::Priority(p) => {
                RateLimitingConfigPriority::Priority(p)
            }
            StoredRateLimitingConfigPriority::Always => RateLimitingConfigPriority::Always,
        }
    }
}

impl From<RateLimitingConfigPriority> for StoredRateLimitingConfigPriority {
    fn from(priority: RateLimitingConfigPriority) -> Self {
        match priority {
            RateLimitingConfigPriority::Priority(p) => {
                StoredRateLimitingConfigPriority::Priority(p)
            }
            RateLimitingConfigPriority::Always => StoredRateLimitingConfigPriority::Always,
        }
    }
}

/// Stored version of `TagValueScope`.
///
/// The runtime type has a custom serializer that converts to strings like "tensorzero::each",
/// so we need a matching deserializer.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(untagged)]
pub enum StoredTagValueScope {
    Special(StoredTagValueScopeSpecial),
    Concrete(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StoredTagValueScopeSpecial {
    #[serde(rename = "tensorzero::each")]
    Each,
    #[serde(rename = "tensorzero::total")]
    Total,
}

impl From<StoredTagValueScope> for TagValueScope {
    fn from(stored: StoredTagValueScope) -> Self {
        match stored {
            StoredTagValueScope::Special(StoredTagValueScopeSpecial::Each) => TagValueScope::Each,
            StoredTagValueScope::Special(StoredTagValueScopeSpecial::Total) => TagValueScope::Total,
            StoredTagValueScope::Concrete(s) => TagValueScope::Concrete(s),
        }
    }
}

impl From<TagValueScope> for StoredTagValueScope {
    fn from(scope: TagValueScope) -> Self {
        match scope {
            TagValueScope::Each => StoredTagValueScope::Special(StoredTagValueScopeSpecial::Each),
            TagValueScope::Total => StoredTagValueScope::Special(StoredTagValueScopeSpecial::Total),
            TagValueScope::Concrete(s) => StoredTagValueScope::Concrete(s),
        }
    }
}

/// Stored version of `ApiKeyPublicIdValueScope`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(untagged)]
pub enum StoredApiKeyPublicIdValueScope {
    Special(StoredApiKeyPublicIdValueScopeSpecial),
    Concrete(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StoredApiKeyPublicIdValueScopeSpecial {
    #[serde(rename = "tensorzero::each")]
    Each,
}

impl From<StoredApiKeyPublicIdValueScope> for ApiKeyPublicIdValueScope {
    fn from(stored: StoredApiKeyPublicIdValueScope) -> Self {
        match stored {
            StoredApiKeyPublicIdValueScope::Special(
                StoredApiKeyPublicIdValueScopeSpecial::Each,
            ) => ApiKeyPublicIdValueScope::Each,
            StoredApiKeyPublicIdValueScope::Concrete(s) => ApiKeyPublicIdValueScope::Concrete(s),
        }
    }
}

impl From<ApiKeyPublicIdValueScope> for StoredApiKeyPublicIdValueScope {
    fn from(scope: ApiKeyPublicIdValueScope) -> Self {
        match scope {
            ApiKeyPublicIdValueScope::Each => {
                StoredApiKeyPublicIdValueScope::Special(StoredApiKeyPublicIdValueScopeSpecial::Each)
            }
            ApiKeyPublicIdValueScope::Concrete(s) => StoredApiKeyPublicIdValueScope::Concrete(s),
        }
    }
}

/// Stored version of `TagRateLimitingConfigScope`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoredTagRateLimitingConfigScope {
    pub tag_key: String,
    pub tag_value: StoredTagValueScope,
}

impl From<StoredTagRateLimitingConfigScope> for TagRateLimitingConfigScope {
    fn from(stored: StoredTagRateLimitingConfigScope) -> Self {
        let StoredTagRateLimitingConfigScope { tag_key, tag_value } = stored;
        Self::new(tag_key, tag_value.into())
    }
}

impl From<TagRateLimitingConfigScope> for StoredTagRateLimitingConfigScope {
    fn from(scope: TagRateLimitingConfigScope) -> Self {
        let (tag_key, tag_value) = scope.into_parts();
        Self {
            tag_key,
            tag_value: tag_value.into(),
        }
    }
}

/// Stored version of `ApiKeyPublicIdConfigScope`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoredApiKeyPublicIdConfigScope {
    pub api_key_public_id: StoredApiKeyPublicIdValueScope,
}

impl From<StoredApiKeyPublicIdConfigScope> for ApiKeyPublicIdConfigScope {
    fn from(stored: StoredApiKeyPublicIdConfigScope) -> Self {
        let StoredApiKeyPublicIdConfigScope { api_key_public_id } = stored;
        Self::new(api_key_public_id.into())
    }
}

impl From<ApiKeyPublicIdConfigScope> for StoredApiKeyPublicIdConfigScope {
    fn from(scope: ApiKeyPublicIdConfigScope) -> Self {
        Self {
            api_key_public_id: scope.into_inner().into(),
        }
    }
}

/// Stored version of `RateLimitingConfigScope`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(untagged)]
pub enum StoredRateLimitingConfigScope {
    Tag(StoredTagRateLimitingConfigScope),
    ApiKeyPublicId(StoredApiKeyPublicIdConfigScope),
}

impl From<StoredRateLimitingConfigScope> for RateLimitingConfigScope {
    fn from(stored: StoredRateLimitingConfigScope) -> Self {
        match stored {
            StoredRateLimitingConfigScope::Tag(t) => RateLimitingConfigScope::Tag(t.into()),
            StoredRateLimitingConfigScope::ApiKeyPublicId(a) => {
                RateLimitingConfigScope::ApiKeyPublicId(a.into())
            }
        }
    }
}

impl From<RateLimitingConfigScope> for StoredRateLimitingConfigScope {
    fn from(scope: RateLimitingConfigScope) -> Self {
        match scope {
            RateLimitingConfigScope::Tag(t) => StoredRateLimitingConfigScope::Tag(t.into()),
            RateLimitingConfigScope::ApiKeyPublicId(a) => {
                StoredRateLimitingConfigScope::ApiKeyPublicId(a.into())
            }
        }
    }
}

/// Stored version of `RateLimitingConfigScopes`.
///
/// The runtime type only has `Serialize` (not `Deserialize`), so we need this
/// wrapper that can deserialize from the JSON array format.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct StoredRateLimitingConfigScopes(pub Vec<StoredRateLimitingConfigScope>);

impl TryFrom<StoredRateLimitingConfigScopes> for RateLimitingConfigScopes {
    type Error = &'static str;

    fn try_from(stored: StoredRateLimitingConfigScopes) -> Result<Self, Self::Error> {
        let scopes: Vec<RateLimitingConfigScope> = stored.0.into_iter().map(|s| s.into()).collect();
        RateLimitingConfigScopes::new(scopes)
    }
}

impl From<RateLimitingConfigScopes> for StoredRateLimitingConfigScopes {
    fn from(scopes: RateLimitingConfigScopes) -> Self {
        Self(scopes.into_inner().into_iter().map(|s| s.into()).collect())
    }
}

/// Stored version of `RateLimit`.
///
/// The runtime type doesn't implement `Deserialize`, so we need this stored version.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredRateLimit {
    pub resource: crate::rate_limiting::RateLimitResource,
    pub interval: crate::rate_limiting::RateLimitInterval,
    pub capacity: u64,
    pub refill_rate: u64,
}

impl From<StoredRateLimit> for RateLimit {
    fn from(stored: StoredRateLimit) -> Self {
        let StoredRateLimit {
            resource,
            interval,
            capacity,
            refill_rate,
        } = stored;
        Self {
            resource,
            interval,
            capacity,
            refill_rate,
        }
    }
}

impl From<&RateLimit> for StoredRateLimit {
    fn from(rate_limit: &RateLimit) -> Self {
        Self {
            resource: rate_limit.resource,
            interval: rate_limit.interval,
            capacity: rate_limit.capacity,
            refill_rate: rate_limit.refill_rate,
        }
    }
}

/// Stored version of `RateLimitingConfigRule`.
///
/// The runtime type has a custom deserializer for TOML shorthand format,
/// but serializes with explicit `limits`, `scope`, and `priority` fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredRateLimitingConfigRule {
    pub limits: Vec<StoredRateLimit>,
    pub scope: StoredRateLimitingConfigScopes,
    pub priority: StoredRateLimitingConfigPriority,
}

impl TryFrom<StoredRateLimitingConfigRule> for RateLimitingConfigRule {
    type Error = &'static str;

    fn try_from(stored: StoredRateLimitingConfigRule) -> Result<Self, Self::Error> {
        let StoredRateLimitingConfigRule {
            limits,
            scope,
            priority,
        } = stored;
        Ok(Self {
            limits: limits.into_iter().map(|l| Arc::new(l.into())).collect(),
            scope: scope.try_into()?,
            priority: priority.into(),
        })
    }
}

impl From<RateLimitingConfigRule> for StoredRateLimitingConfigRule {
    fn from(rule: RateLimitingConfigRule) -> Self {
        let RateLimitingConfigRule {
            limits,
            scope,
            priority,
        } = rule;
        Self {
            limits: limits.iter().map(|l| l.as_ref().into()).collect(),
            scope: scope.into(),
            priority: priority.into(),
        }
    }
}

/// Stored version of `UninitializedRateLimitingConfig`.
///
/// Note: We implement `Default` manually to match `UninitializedRateLimitingConfig::default()`,
/// which sets `enabled: true`. Using `#[derive(Default)]` would set `enabled: false` (bool's default),
/// causing historical config snapshots without a `rate_limiting` field to unexpectedly disable rate limiting.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredUninitializedRateLimitingConfig {
    #[serde(default)]
    pub rules: Vec<StoredRateLimitingConfigRule>,
    #[serde(default = "default_rate_limiting_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub backend: RateLimitingBackend,
}

impl Default for StoredUninitializedRateLimitingConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            enabled: true,
            backend: RateLimitingBackend::default(),
        }
    }
}

fn default_rate_limiting_enabled() -> bool {
    true
}

impl TryFrom<StoredUninitializedRateLimitingConfig> for UninitializedRateLimitingConfig {
    type Error = &'static str;

    fn try_from(stored: StoredUninitializedRateLimitingConfig) -> Result<Self, Self::Error> {
        let StoredUninitializedRateLimitingConfig {
            rules,
            enabled,
            backend,
        } = stored;
        let rules: Result<Vec<_>, _> = rules.into_iter().map(|r| r.try_into()).collect();
        Ok(Self {
            rules: rules?,
            enabled,
            backend,
        })
    }
}

impl From<UninitializedRateLimitingConfig> for StoredUninitializedRateLimitingConfig {
    fn from(config: UninitializedRateLimitingConfig) -> Self {
        let UninitializedRateLimitingConfig {
            rules,
            enabled,
            backend,
        } = config;
        Self {
            rules: rules.into_iter().map(|r| r.into()).collect(),
            enabled,
            backend,
        }
    }
}

// ============================================================================
// Embedding Model Stored Types
// ============================================================================

/// Stored version of `UninitializedEmbeddingModelConfig`.
///
/// Accepts the deprecated `timeouts` field for backward compatibility with
/// historical config snapshots stored in ClickHouse.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StoredEmbeddingModelConfig {
    pub routing: Vec<Arc<str>>,
    pub providers: HashMap<Arc<str>, StoredEmbeddingProviderConfig>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// DEPRECATED: Use `timeout_ms` instead.
    /// Kept for backward compatibility with stored snapshots.
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
}

impl From<StoredEmbeddingModelConfig> for UninitializedEmbeddingModelConfig {
    fn from(stored: StoredEmbeddingModelConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let StoredEmbeddingModelConfig {
            routing,
            providers,
            timeout_ms,
            timeouts,
        } = stored;

        Self {
            routing,
            providers: providers.into_iter().map(|(k, v)| (k, v.into())).collect(),
            // Migration: prefer new field, fall back to deprecated
            timeout_ms: timeout_ms.or(timeouts.non_streaming.total_ms),
        }
    }
}

/// Stored version of `UninitializedEmbeddingProviderConfig`.
///
/// Accepts the deprecated `timeouts` field for backward compatibility.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredEmbeddingProviderConfig {
    #[serde(flatten)]
    pub config: UninitializedProviderConfig,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// DEPRECATED: Use `timeout_ms` instead.
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    #[serde(default)]
    pub extra_body: Option<ExtraBodyConfig>,
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub extra_headers: Option<ExtraHeadersConfig>,
}

impl From<StoredEmbeddingProviderConfig> for UninitializedEmbeddingProviderConfig {
    fn from(stored: StoredEmbeddingProviderConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let StoredEmbeddingProviderConfig {
            config,
            timeout_ms,
            timeouts,
            extra_body,
            extra_headers,
        } = stored;

        Self {
            config,
            // Migration: prefer new field, fall back to deprecated
            timeout_ms: timeout_ms.or(timeouts.non_streaming.total_ms),
            extra_body,
            extra_headers,
        }
    }
}

impl From<UninitializedEmbeddingProviderConfig> for StoredEmbeddingProviderConfig {
    fn from(uninitialized: UninitializedEmbeddingProviderConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let UninitializedEmbeddingProviderConfig {
            config,
            timeout_ms,
            extra_body,
            extra_headers,
        } = uninitialized;

        Self {
            config,
            timeout_ms,
            timeouts: TimeoutsConfig::default(),
            extra_body,
            extra_headers,
        }
    }
}

impl From<UninitializedEmbeddingModelConfig> for StoredEmbeddingModelConfig {
    fn from(uninitialized: UninitializedEmbeddingModelConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let UninitializedEmbeddingModelConfig {
            routing,
            providers,
            timeout_ms,
        } = uninitialized;

        Self {
            routing,
            providers: providers.into_iter().map(|(k, v)| (k, v.into())).collect(),
            timeout_ms,
            timeouts: TimeoutsConfig::default(),
        }
    }
}

/// Top-level stored config type.
///
/// Only fields with deprecations in their subtree use `Stored*` types.
/// Other fields re-use `Uninitialized*` types directly.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StoredConfig {
    // Fields WITHOUT deprecations - reuse Uninitialized* types
    #[serde(default)]
    pub gateway: UninitializedGatewayConfig,
    #[serde(default)]
    pub postgres: PostgresConfig,
    pub object_storage: Option<StorageKind>,
    #[serde(default)]
    pub models: HashMap<Arc<str>, UninitializedModelConfig>,
    #[serde(default)]
    pub functions: HashMap<String, UninitializedFunctionConfig>,
    #[serde(default)]
    pub metrics: HashMap<String, MetricConfig>,
    #[serde(default)]
    pub tools: HashMap<String, UninitializedToolConfig>,
    #[serde(default)]
    pub evaluations: HashMap<String, UninitializedEvaluationConfig>,
    #[serde(default)]
    pub provider_types: ProviderTypesConfig,
    #[serde(default)]
    pub optimizers: HashMap<String, UninitializedOptimizerInfo>,

    // Fields WITH deprecations or custom serde - use Stored* types
    #[serde(default)]
    pub rate_limiting: StoredUninitializedRateLimitingConfig,
    #[serde(default)]
    pub embedding_models: HashMap<Arc<str>, StoredEmbeddingModelConfig>,
}

impl From<UninitializedConfig> for StoredConfig {
    fn from(config: UninitializedConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let UninitializedConfig {
            gateway,
            postgres,
            rate_limiting,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            embedding_models,
        } = config;

        Self {
            gateway,
            postgres,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            rate_limiting: rate_limiting.into(),
            embedding_models: embedding_models
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        }
    }
}

impl TryFrom<StoredConfig> for UninitializedConfig {
    type Error = &'static str;

    fn try_from(stored: StoredConfig) -> Result<Self, Self::Error> {
        // Explicit destructuring ensures compile error if fields are added/removed
        let StoredConfig {
            gateway,
            postgres,
            rate_limiting,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            embedding_models,
        } = stored;

        Ok(Self {
            gateway,
            postgres,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            rate_limiting: rate_limiting.try_into()?,
            embedding_models: embedding_models
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that we can deserialize JSON in the format that RateLimitingConfigRule serializes to.
    /// This is the format stored in the database.
    #[test]
    fn test_deserialize_stored_rate_limiting_config() {
        let json = r#"{
            "rules": [
                {
                    "limits": [
                        {
                            "resource": "token",
                            "interval": "minute",
                            "capacity": 1000,
                            "refill_rate": 100
                        },
                        {
                            "resource": "model_inference",
                            "interval": "second",
                            "capacity": 10,
                            "refill_rate": 10
                        }
                    ],
                    "scope": [
                        {
                            "tag_key": "user_id",
                            "tag_value": "tensorzero::each"
                        }
                    ],
                    "priority": {"Priority": 5}
                }
            ],
            "enabled": true,
            "backend": "auto"
        }"#;

        let stored: StoredUninitializedRateLimitingConfig =
            serde_json::from_str(json).expect("should deserialize stored rate limiting config");

        assert!(stored.enabled, "enabled should be true");
        assert_eq!(stored.rules.len(), 1, "should have 1 rule");

        let rule = &stored.rules[0];
        assert_eq!(rule.limits.len(), 2, "rule should have 2 limits");
        assert_eq!(
            rule.priority,
            StoredRateLimitingConfigPriority::Priority(5),
            "priority should be Priority(5)"
        );

        // Convert to runtime type
        let config: UninitializedRateLimitingConfig =
            stored.try_into().expect("should convert to runtime type");
        assert!(config.enabled, "converted config should be enabled");
        assert_eq!(config.rules.len(), 1, "converted config should have 1 rule");
    }

    /// Test deserializing with "Always" priority
    #[test]
    fn test_deserialize_always_priority() {
        let json = r#"{
            "rules": [
                {
                    "limits": [],
                    "scope": [],
                    "priority": "Always"
                }
            ],
            "enabled": true,
            "backend": "auto"
        }"#;

        let stored: StoredUninitializedRateLimitingConfig =
            serde_json::from_str(json).expect("should deserialize");

        assert_eq!(stored.rules.len(), 1, "should have 1 rule");
        assert_eq!(
            stored.rules[0].priority,
            StoredRateLimitingConfigPriority::Always,
            "priority should be Always"
        );
    }

    /// Test deserializing various tag value scopes
    #[test]
    fn test_deserialize_tag_value_scopes() {
        let json = r#"{
            "rules": [
                {
                    "limits": [],
                    "scope": [
                        {"tag_key": "user_id", "tag_value": "tensorzero::each"},
                        {"tag_key": "app_id", "tag_value": "tensorzero::total"},
                        {"tag_key": "org_id", "tag_value": "concrete_value_123"}
                    ],
                    "priority": {"Priority": 1}
                }
            ],
            "enabled": true,
            "backend": "auto"
        }"#;

        let stored: StoredUninitializedRateLimitingConfig =
            serde_json::from_str(json).expect("should deserialize");

        let scopes = &stored.rules[0].scope.0;
        assert_eq!(scopes.len(), 3, "should have 3 scopes");

        // Check each scope type
        match &scopes[0] {
            StoredRateLimitingConfigScope::Tag(t) => {
                assert_eq!(t.tag_key, "user_id");
                assert_eq!(
                    t.tag_value,
                    StoredTagValueScope::Special(StoredTagValueScopeSpecial::Each)
                );
            }
            StoredRateLimitingConfigScope::ApiKeyPublicId(_) => panic!("expected Tag scope"),
        }

        match &scopes[1] {
            StoredRateLimitingConfigScope::Tag(t) => {
                assert_eq!(t.tag_key, "app_id");
                assert_eq!(
                    t.tag_value,
                    StoredTagValueScope::Special(StoredTagValueScopeSpecial::Total)
                );
            }
            StoredRateLimitingConfigScope::ApiKeyPublicId(_) => panic!("expected Tag scope"),
        }

        match &scopes[2] {
            StoredRateLimitingConfigScope::Tag(t) => {
                assert_eq!(t.tag_key, "org_id");
                assert_eq!(
                    t.tag_value,
                    StoredTagValueScope::Concrete("concrete_value_123".to_string())
                );
            }
            StoredRateLimitingConfigScope::ApiKeyPublicId(_) => panic!("expected Tag scope"),
        }
    }

    /// Test deserializing api_key_public_id scope
    #[test]
    fn test_deserialize_api_key_scope() {
        let json = r#"{
            "rules": [
                {
                    "limits": [],
                    "scope": [
                        {"api_key_public_id": "tensorzero::each"},
                        {"api_key_public_id": "abc123def456"}
                    ],
                    "priority": {"Priority": 1}
                }
            ],
            "enabled": true,
            "backend": "auto"
        }"#;

        let stored: StoredUninitializedRateLimitingConfig =
            serde_json::from_str(json).expect("should deserialize");

        let scopes = &stored.rules[0].scope.0;
        assert_eq!(scopes.len(), 2, "should have 2 scopes");

        match &scopes[0] {
            StoredRateLimitingConfigScope::ApiKeyPublicId(a) => {
                assert_eq!(
                    a.api_key_public_id,
                    StoredApiKeyPublicIdValueScope::Special(
                        StoredApiKeyPublicIdValueScopeSpecial::Each
                    )
                );
            }
            StoredRateLimitingConfigScope::Tag(_) => panic!("expected ApiKeyPublicId scope"),
        }

        match &scopes[1] {
            StoredRateLimitingConfigScope::ApiKeyPublicId(a) => {
                assert_eq!(
                    a.api_key_public_id,
                    StoredApiKeyPublicIdValueScope::Concrete("abc123def456".to_string())
                );
            }
            StoredRateLimitingConfigScope::Tag(_) => panic!("expected ApiKeyPublicId scope"),
        }
    }

    /// Test roundtrip: UninitializedRateLimitingConfig -> StoredUninitializedRateLimitingConfig -> JSON -> StoredUninitializedRateLimitingConfig -> UninitializedRateLimitingConfig
    #[test]
    fn test_rate_limiting_roundtrip() {
        use crate::rate_limiting::{RateLimitInterval, RateLimitResource};

        // Create a runtime config by parsing TOML (the normal path)
        let toml_str = r#"
            [[rules]]
            tokens_per_minute = 1000
            model_inferences_per_second = 10
            priority = 5
            scope = [
                { tag_key = "user_id", tag_value = "tensorzero::each" }
            ]

            [[rules]]
            tokens_per_hour = 50000
            always = true
        "#;

        let uninitialized: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).expect("should parse TOML");

        // Convert to stored type
        let stored: StoredUninitializedRateLimitingConfig = uninitialized.into();

        // Serialize to JSON (simulating database storage)
        let json = serde_json::to_string(&stored).expect("should serialize to JSON");

        // Deserialize back (simulating database retrieval)
        let stored_again: StoredUninitializedRateLimitingConfig =
            serde_json::from_str(&json).expect("should deserialize from JSON");

        // Convert back to runtime type
        let uninitialized_again: UninitializedRateLimitingConfig = stored_again
            .try_into()
            .expect("should convert back to runtime type");

        // Verify the config is equivalent
        assert!(
            uninitialized_again.enabled,
            "enabled should be preserved through roundtrip"
        );
        assert_eq!(
            uninitialized_again.rules.len(),
            2,
            "should have 2 rules after roundtrip"
        );

        // Check first rule
        let rule1 = &uninitialized_again.rules[0];
        assert_eq!(
            rule1.priority,
            RateLimitingConfigPriority::Priority(5),
            "first rule priority should be Priority(5)"
        );
        assert_eq!(rule1.limits.len(), 2, "first rule should have 2 limits");

        // Check that limits have correct values
        let has_token_limit = rule1.limits.iter().any(|l| {
            l.resource == RateLimitResource::Token
                && l.interval == RateLimitInterval::Minute
                && l.capacity == 1000
        });
        assert!(
            has_token_limit,
            "first rule should have token per minute limit"
        );

        // Check second rule
        let rule2 = &uninitialized_again.rules[1];
        assert_eq!(
            rule2.priority,
            RateLimitingConfigPriority::Always,
            "second rule priority should be Always"
        );
    }

    /// Test deserializing empty/default config
    #[test]
    fn test_deserialize_empty_config() {
        let json = r"{}";

        let stored: StoredUninitializedRateLimitingConfig =
            serde_json::from_str(json).expect("should deserialize empty config");

        assert!(stored.enabled, "default enabled should be true");
        assert!(stored.rules.is_empty(), "default rules should be empty");
        assert_eq!(
            stored.backend,
            RateLimitingBackend::Auto,
            "default backend should be Auto"
        );
    }

    /// Test deserializing with different backends
    #[test]
    fn test_deserialize_backends() {
        for (backend_str, expected) in [
            ("auto", RateLimitingBackend::Auto),
            ("postgres", RateLimitingBackend::Postgres),
            ("valkey", RateLimitingBackend::Valkey),
        ] {
            let json = format!(r#"{{"backend": "{backend_str}", "enabled": true, "rules": []}}"#);
            let stored: StoredUninitializedRateLimitingConfig =
                serde_json::from_str(&json).expect("should deserialize");
            assert_eq!(stored.backend, expected, "backend should be {expected:?}");
        }
    }

    /// Test that StoredConfig without a rate_limiting field defaults to enabled: true.
    ///
    /// This is a regression test: the struct-level Default (used when the entire field is missing)
    /// must match UninitializedRateLimitingConfig::default() which has enabled: true.
    /// A derived Default would incorrectly set enabled: false (bool's default).
    #[test]
    fn test_stored_config_missing_rate_limiting_defaults_to_enabled() {
        // Minimal StoredConfig JSON without rate_limiting field
        let json = r#"{
            "gateway": {},
            "postgres": {},
            "models": {},
            "functions": {},
            "metrics": {},
            "tools": {},
            "evaluations": {},
            "provider_types": {},
            "optimizers": {},
            "embedding_models": {}
        }"#;

        let stored: StoredConfig = serde_json::from_str(json)
            .expect("should deserialize StoredConfig without rate_limiting");

        assert!(
            stored.rate_limiting.enabled,
            "rate_limiting.enabled should default to true when rate_limiting field is missing"
        );
    }
}
