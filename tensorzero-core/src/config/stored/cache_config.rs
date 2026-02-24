use serde::{Deserialize, Serialize};

use crate::config::gateway::{
    InferenceCacheBackend, ModelInferenceCacheConfig, ValkeyModelInferenceCacheConfig,
};

/// Stored version of `ModelInferenceCacheConfig`.
///
/// Omits `deny_unknown_fields` so that snapshots from newer gateways
/// (with additional fields) don't break deserialization in rolled-back gateways.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StoredCacheConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub backend: InferenceCacheBackend,
    #[serde(default)]
    pub valkey: ValkeyModelInferenceCacheConfig,
}

impl From<ModelInferenceCacheConfig> for StoredCacheConfig {
    fn from(config: ModelInferenceCacheConfig) -> Self {
        let ModelInferenceCacheConfig {
            enabled,
            backend,
            valkey,
        } = config;
        Self {
            enabled,
            backend,
            valkey,
        }
    }
}

impl From<StoredCacheConfig> for ModelInferenceCacheConfig {
    fn from(stored: StoredCacheConfig) -> Self {
        let StoredCacheConfig {
            enabled,
            backend,
            valkey,
        } = stored;
        Self {
            enabled,
            backend,
            valkey,
        }
    }
}
