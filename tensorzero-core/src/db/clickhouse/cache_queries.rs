use std::collections::HashMap;

use async_trait::async_trait;

use crate::cache::CacheKey;
use crate::db::cache::CacheQueries;
use crate::error::{Error, ErrorDetails};

use super::{ClickHouseConnectionInfo, TableName};

#[async_trait]
impl CacheQueries for ClickHouseConnectionInfo {
    async fn cache_lookup(
        &self,
        cache_key: &CacheKey,
        max_age_s: Option<u32>,
    ) -> Result<Option<String>, Error> {
        // NOTE: the short cache key is just so the ClickHouse index can be as efficient as possible
        // but we always check against the long cache key before returning a result
        let short_cache_key = cache_key.get_short_key()?.to_string();
        let long_cache_key = cache_key.get_long_key();
        // The clickhouse query args look like rust format string args, but they're not.
        let query = if max_age_s.is_some() {
            r"
                SELECT
                    output,
                    raw_request,
                    raw_response,
                    input_tokens,
                    output_tokens,
                    finish_reason
                FROM ModelInferenceCache
                WHERE short_cache_key = {short_cache_key:UInt64}
                    AND long_cache_key = {long_cache_key:String}
                    AND timestamp > subtractSeconds(now(), {lookback_s:UInt32})
                ORDER BY timestamp DESC
                LIMIT 1
                FORMAT JSONEachRow
            "
        } else {
            r"
                SELECT
                    output,
                    raw_request,
                    raw_response,
                    input_tokens,
                    output_tokens,
                    finish_reason
                FROM ModelInferenceCache
                WHERE short_cache_key = {short_cache_key:UInt64}
                    AND long_cache_key = {long_cache_key:String}
                ORDER BY timestamp DESC
                LIMIT 1
                FORMAT JSONEachRow
            "
        };
        let mut query_params = HashMap::from([
            ("short_cache_key", short_cache_key.as_str()),
            ("long_cache_key", long_cache_key.as_str()),
        ]);
        let lookback_str;
        if let Some(lookback) = max_age_s {
            lookback_str = lookback.to_string();
            query_params.insert("lookback_s", lookback_str.as_str());
        }
        let result = self
            .run_query_synchronous(query.to_string(), &query_params)
            .await?;
        if result.response.is_empty() {
            return Ok(None);
        }
        Ok(Some(result.response))
    }

    async fn cache_write(&self, cache_key: &CacheKey, data: &str) -> Result<(), Error> {
        let short_cache_key = cache_key.get_short_key()?;
        let long_cache_key = cache_key.get_long_key();
        // Parse the CacheData JSON and add the cache key fields for the ClickHouse row
        let mut row: serde_json::Map<String, serde_json::Value> = serde_json::from_str(data)
            .map_err(|e| {
                Error::new(ErrorDetails::Cache {
                    message: format!("Failed to parse cache data for ClickHouse write: {e}"),
                })
            })?;
        row.insert(
            "short_cache_key".to_string(),
            serde_json::Value::Number(short_cache_key.into()),
        );
        row.insert(
            "long_cache_key".to_string(),
            serde_json::Value::String(long_cache_key),
        );
        let row_json = serde_json::to_string(&row).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize cache row: {e}"),
            })
        })?;
        self.write_batched_raw(vec![row_json], TableName::ModelInferenceCache)
            .await
    }
}
