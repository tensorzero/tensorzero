use async_trait::async_trait;

use crate::cache::CacheKey;
use crate::error::Error;

/// Trait abstracting cache read/write operations at the raw JSON string level.
/// Operates on serialized `CacheData<T>` JSON to avoid generic method signatures
/// (which would prevent trait object usage).
#[async_trait]
pub trait CacheQueries: Send + Sync {
    /// Returns serialized `CacheData<T>` JSON if found, or `None`.
    async fn cache_lookup(
        &self,
        cache_key: &CacheKey,
        max_age_s: Option<u32>,
    ) -> Result<Option<String>, Error>;

    /// Writes serialized `CacheData<T>` JSON to the cache.
    async fn cache_write(&self, cache_key: &CacheKey, data: &str) -> Result<(), Error>;
}
