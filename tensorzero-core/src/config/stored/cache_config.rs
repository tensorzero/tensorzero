use serde::{Deserialize, Serialize};

use crate::config::gateway::{ModelInferenceCacheConfig, ValkeyModelInferenceCacheConfig};

/// Stored version of `ModelInferenceCacheConfig`.
///
/// Omits `deny_unknown_fields` so that future fields (e.g. `enabled`, `backend`)
/// don't break deserialization in rolled-back gateways.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StoredCacheConfig {
    #[serde(default)]
    pub valkey: ValkeyModelInferenceCacheConfig,
}

impl From<ModelInferenceCacheConfig> for StoredCacheConfig {
    fn from(config: ModelInferenceCacheConfig) -> Self {
        let ModelInferenceCacheConfig { valkey } = config;
        Self { valkey }
    }
}

impl From<StoredCacheConfig> for ModelInferenceCacheConfig {
    fn from(stored: StoredCacheConfig) -> Self {
        let StoredCacheConfig { valkey } = stored;
        Self { valkey }
    }
}
