use async_trait::async_trait;
use redis::aio::ConnectionLike;
use serde::{Deserialize, Serialize};

use crate::cache::CacheKey;
use crate::db::cache::CacheQueries;
use crate::error::{Error, ErrorDetails};

const CACHE_KEY_PREFIX: &str = "tensorzero_cache:";

/// Wrapper stored in Valkey that includes a write timestamp
/// so we can enforce `max_age_s` filtering at read time
/// (matching ClickHouse's read-time filtering semantics).
#[derive(Serialize, Deserialize)]
struct ValkeyCacheEntry {
    /// Unix timestamp (seconds) when this entry was written
    written_at: i64,
    /// The serialized `CacheData<T>` JSON
    data: serde_json::Value,
}

fn make_valkey_key(cache_key: &CacheKey) -> String {
    format!("{CACHE_KEY_PREFIX}{}", cache_key.get_long_key())
}

/// Execute a cache lookup against any async Redis-compatible connection.
async fn execute_cache_lookup<C: ConnectionLike>(
    conn: &mut C,
    cache_key: &CacheKey,
    max_age_s: Option<u32>,
) -> Result<Option<String>, Error> {
    let key = make_valkey_key(cache_key);
    let result: Option<String> = redis::cmd("GET")
        .arg(&key)
        .query_async(conn)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::Cache {
                message: format!("Valkey GET failed: {e}"),
            })
        })?;
    let Some(raw_json) = result else {
        return Ok(None);
    };
    let entry: ValkeyCacheEntry = serde_json::from_str(&raw_json).map_err(|e| {
        Error::new(ErrorDetails::Cache {
            message: format!("Failed to deserialize Valkey cache entry: {e}"),
        })
    })?;
    // Check max_age_s against the stored timestamp
    if let Some(max_age) = max_age_s {
        let now = chrono::Utc::now().timestamp();
        if now - entry.written_at >= max_age as i64 {
            return Ok(None);
        }
    }
    let data_str = serde_json::to_string(&entry.data).map_err(|e| {
        Error::new(ErrorDetails::Cache {
            message: format!("Failed to re-serialize Valkey cache data: {e}"),
        })
    })?;
    Ok(Some(data_str))
}

/// Execute a cache write against any async Redis-compatible connection.
async fn execute_cache_write<C: ConnectionLike>(
    conn: &mut C,
    cache_key: &CacheKey,
    data: &str,
    cache_ttl_s: u64,
    written_at: i64,
) -> Result<(), Error> {
    let key = make_valkey_key(cache_key);
    let data_value: serde_json::Value = serde_json::from_str(data).map_err(|e| {
        Error::new(ErrorDetails::Cache {
            message: format!("Failed to parse cache data for Valkey write: {e}"),
        })
    })?;
    let entry = ValkeyCacheEntry {
        written_at,
        data: data_value,
    };
    let entry_json = serde_json::to_string(&entry).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize Valkey cache entry: {e}"),
        })
    })?;
    redis::cmd("SETEX")
        .arg(&key)
        .arg(cache_ttl_s as i64)
        .arg(&entry_json)
        .query_async::<()>(conn)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::Cache {
                message: format!("Valkey SET EX failed: {e}"),
            })
        })?;
    Ok(())
}

/// Valkey-backed cache client that pairs a connection with a TTL config.
#[derive(Clone)]
pub struct ValkeyCacheClient {
    connection: Box<redis::aio::ConnectionManager>,
    cache_ttl_s: u64,
}

impl ValkeyCacheClient {
    pub fn new(connection: Box<redis::aio::ConnectionManager>, cache_ttl_s: u64) -> Self {
        Self {
            connection,
            cache_ttl_s,
        }
    }
}

#[async_trait]
impl CacheQueries for ValkeyCacheClient {
    async fn cache_lookup(
        &self,
        cache_key: &CacheKey,
        max_age_s: Option<u32>,
    ) -> Result<Option<String>, Error> {
        let mut conn = self.connection.clone();
        execute_cache_lookup(&mut *conn, cache_key, max_age_s).await
    }

    async fn cache_write(&self, cache_key: &CacheKey, data: &str) -> Result<(), Error> {
        let mut conn = self.connection.clone();
        let written_at = chrono::Utc::now().timestamp();
        execute_cache_write(&mut *conn, cache_key, data, self.cache_ttl_s, written_at).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        CacheData, CacheOutput, EmbeddingCacheData, NonStreamingCacheData, StreamingCacheData,
    };
    use crate::embeddings::Embedding;
    use crate::inference::types::{ContentBlockOutput, FinishReason};
    use redis_test::{MockCmd, MockRedisConnection};
    use serde::de::DeserializeOwned;

    #[test]
    fn test_make_valkey_key() {
        let cache_key = CacheKey::from(blake3::hash(b"test"));
        let key = make_valkey_key(&cache_key);
        assert!(
            key.starts_with(CACHE_KEY_PREFIX),
            "key should start with prefix"
        );
        assert!(
            key.len() > CACHE_KEY_PREFIX.len(),
            "key should have content after prefix"
        );
    }

    #[test]
    fn test_valkey_cache_entry_serde_roundtrip() {
        let entry = ValkeyCacheEntry {
            written_at: 1700000000,
            data: serde_json::json!({"output": "test", "raw_request": "req"}),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: ValkeyCacheEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.written_at, entry.written_at,
            "written_at should roundtrip"
        );
        assert_eq!(parsed.data, entry.data, "data should roundtrip");
    }

    #[test]
    fn test_max_age_s_logic() {
        let now = chrono::Utc::now().timestamp();

        // Entry written now - should pass any max_age check
        let recent_entry = ValkeyCacheEntry {
            written_at: now,
            data: serde_json::json!({}),
        };
        assert!(
            now - recent_entry.written_at <= 60,
            "recent entry should be within 60s"
        );

        // Entry written 10 minutes ago
        let old_entry = ValkeyCacheEntry {
            written_at: now - 600,
            data: serde_json::json!({}),
        };
        // Should pass with max_age of 1 hour
        assert!(
            now - old_entry.written_at <= 3600,
            "10-min-old entry should be within 1 hour"
        );
        // Should fail with max_age of 60 seconds
        assert!(
            now - old_entry.written_at > 60,
            "10-min-old entry should exceed 60s"
        );
    }

    /// Helper to verify that CacheData<T> roundtrips through JSON serialization
    /// (which is how it flows through the Valkey cache).
    fn assert_cache_data_serde_roundtrip<
        T: CacheOutput + Serialize + DeserializeOwned + std::fmt::Debug,
    >(
        cache_data: &CacheData<T>,
    ) {
        let json = serde_json::to_string(cache_data)
            .expect("should serialize CacheData for roundtrip test");
        let parsed: CacheData<T> =
            serde_json::from_str(&json).expect("should deserialize CacheData from JSON");
        assert_eq!(
            parsed.raw_request, cache_data.raw_request,
            "raw_request should roundtrip"
        );
        assert_eq!(
            parsed.raw_response, cache_data.raw_response,
            "raw_response should roundtrip"
        );
        assert_eq!(
            parsed.input_tokens, cache_data.input_tokens,
            "input_tokens should roundtrip"
        );
        assert_eq!(
            parsed.output_tokens, cache_data.output_tokens,
            "output_tokens should roundtrip"
        );
    }

    #[test]
    fn test_serde_roundtrip_non_streaming() {
        let cache_data = CacheData {
            output: NonStreamingCacheData {
                blocks: vec![ContentBlockOutput::Text(crate::inference::types::Text {
                    text: "hello world".to_string(),
                })],
            },
            raw_request: "test request".to_string(),
            raw_response: "test response".to_string(),
            input_tokens: Some(10),
            output_tokens: Some(20),
            finish_reason: Some(FinishReason::Stop),
        };
        assert_cache_data_serde_roundtrip(&cache_data);
    }

    #[test]
    fn test_serde_roundtrip_embedding() {
        let cache_data = CacheData {
            output: EmbeddingCacheData {
                embedding: Embedding::Float(vec![1.0, 2.0, 3.0]),
            },
            raw_request: "embed request".to_string(),
            raw_response: "embed response".to_string(),
            input_tokens: Some(5),
            output_tokens: None,
            finish_reason: None,
        };
        assert_cache_data_serde_roundtrip(&cache_data);
    }

    #[test]
    fn test_serde_roundtrip_streaming() {
        use crate::cache::CachedProviderInferenceResponseChunk;
        use crate::inference::types::ContentBlockChunk;

        let cache_data = CacheData {
            output: StreamingCacheData {
                chunks: vec![CachedProviderInferenceResponseChunk {
                    content: vec![ContentBlockChunk::Text(
                        crate::inference::types::TextChunk {
                            text: "streamed".to_string(),
                            id: "0".to_string(),
                        },
                    )],
                    usage: None,
                    raw_response: "chunk_raw".to_string(),
                }],
            },
            raw_request: "stream request".to_string(),
            raw_response: String::new(),
            input_tokens: Some(8),
            output_tokens: Some(12),
            finish_reason: Some(FinishReason::Stop),
        };
        assert_cache_data_serde_roundtrip(&cache_data);
    }

    /// Test that a ValkeyCacheEntry wrapping real CacheData roundtrips correctly.
    /// This simulates the full path: serialize CacheData -> wrap in ValkeyCacheEntry -> serialize entry -> deserialize entry -> extract CacheData.
    #[test]
    fn test_valkey_entry_wrapping_cache_data() {
        let cache_data = CacheData {
            output: NonStreamingCacheData {
                blocks: vec![ContentBlockOutput::Text(crate::inference::types::Text {
                    text: "wrapped".to_string(),
                })],
            },
            raw_request: "req".to_string(),
            raw_response: "resp".to_string(),
            input_tokens: Some(1),
            output_tokens: Some(2),
            finish_reason: None,
        };
        let data_json = serde_json::to_string(&cache_data).unwrap();
        let data_value: serde_json::Value = serde_json::from_str(&data_json).unwrap();
        let entry = ValkeyCacheEntry {
            written_at: 1700000000,
            data: data_value,
        };
        let entry_json = serde_json::to_string(&entry).unwrap();
        let parsed_entry: ValkeyCacheEntry = serde_json::from_str(&entry_json).unwrap();
        let extracted_json = serde_json::to_string(&parsed_entry.data).unwrap();
        let extracted_data: CacheData<NonStreamingCacheData> =
            serde_json::from_str(&extracted_json).unwrap();
        assert_eq!(
            extracted_data.raw_request, "req",
            "raw_request should survive the ValkeyCacheEntry wrapping"
        );
    }

    // ===== MOCK CONNECTION TESTS =====

    #[tokio::test]
    async fn test_execute_cache_lookup_miss() {
        let cache_key = CacheKey::from(blake3::hash(b"miss"));
        let valkey_key = make_valkey_key(&cache_key);
        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("GET").arg(&valkey_key),
            Ok(redis::Value::Nil),
        )])
        .assert_all_commands_consumed();

        let result = execute_cache_lookup(&mut mock, &cache_key, None)
            .await
            .expect("cache lookup should succeed");
        assert!(result.is_none(), "cache miss should return None");
    }

    #[tokio::test]
    async fn test_execute_cache_lookup_hit() {
        let cache_key = CacheKey::from(blake3::hash(b"hit"));
        let valkey_key = make_valkey_key(&cache_key);

        let inner_data = serde_json::json!({"output": "\"cached\"", "raw_request": "req", "raw_response": "resp", "input_tokens": 1, "output_tokens": 2, "finish_reason": null});
        let entry = ValkeyCacheEntry {
            written_at: chrono::Utc::now().timestamp(),
            data: inner_data.clone(),
        };
        let stored_json = serde_json::to_string(&entry).unwrap();

        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("GET").arg(&valkey_key),
            Ok(stored_json),
        )])
        .assert_all_commands_consumed();

        let result = execute_cache_lookup(&mut mock, &cache_key, None)
            .await
            .expect("cache lookup should succeed");
        assert!(result.is_some(), "cache hit should return Some");
        let returned: serde_json::Value =
            serde_json::from_str(&result.unwrap()).expect("returned data should be valid JSON");
        assert_eq!(
            returned, inner_data,
            "returned data should match stored data"
        );
    }

    #[tokio::test]
    async fn test_execute_cache_lookup_expired() {
        let cache_key = CacheKey::from(blake3::hash(b"expired"));
        let valkey_key = make_valkey_key(&cache_key);

        let inner_data = serde_json::json!({"output": "\"old\""});
        let entry = ValkeyCacheEntry {
            // Written 10 minutes ago
            written_at: chrono::Utc::now().timestamp() - 600,
            data: inner_data,
        };
        let stored_json = serde_json::to_string(&entry).unwrap();

        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("GET").arg(&valkey_key),
            Ok(stored_json),
        )])
        .assert_all_commands_consumed();

        // max_age_s = 60 means entries older than 60s are expired
        let result = execute_cache_lookup(&mut mock, &cache_key, Some(60))
            .await
            .expect("cache lookup should succeed");
        assert!(
            result.is_none(),
            "expired entry should return None when max_age_s is exceeded"
        );
    }

    #[tokio::test]
    async fn test_execute_cache_write_sends_setex() {
        let cache_key = CacheKey::from(blake3::hash(b"write"));
        let valkey_key = make_valkey_key(&cache_key);
        let data = r#"{"output":"\"test\"","raw_request":"req","raw_response":"resp","input_tokens":1,"output_tokens":2,"finish_reason":null}"#;
        let written_at = 1700000000i64;
        let cache_ttl_s = 86400u64;

        // Build the expected entry JSON to match what execute_cache_write will send
        let data_value: serde_json::Value = serde_json::from_str(data).unwrap();
        let expected_entry = ValkeyCacheEntry {
            written_at,
            data: data_value,
        };
        let expected_entry_json = serde_json::to_string(&expected_entry).unwrap();

        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("SETEX")
                .arg(&valkey_key)
                .arg(cache_ttl_s as i64)
                .arg(expected_entry_json),
            Ok("OK"),
        )])
        .assert_all_commands_consumed();

        execute_cache_write(&mut mock, &cache_key, data, cache_ttl_s, written_at)
            .await
            .expect("cache write should succeed");
    }

    #[tokio::test]
    async fn test_execute_cache_write_with_custom_ttl() {
        let cache_key = CacheKey::from(blake3::hash(b"custom_ttl"));
        let valkey_key = make_valkey_key(&cache_key);
        let data = r#"{"output":"\"test\""}"#;
        let written_at = 1700000000i64;
        let cache_ttl_s = 3600u64; // 1 hour

        let data_value: serde_json::Value = serde_json::from_str(data).unwrap();
        let expected_entry = ValkeyCacheEntry {
            written_at,
            data: data_value,
        };
        let expected_entry_json = serde_json::to_string(&expected_entry).unwrap();

        let mut mock = MockRedisConnection::new(vec![MockCmd::new(
            redis::cmd("SETEX")
                .arg(&valkey_key)
                .arg(cache_ttl_s as i64)
                .arg(expected_entry_json),
            Ok("OK"),
        )])
        .assert_all_commands_consumed();

        execute_cache_write(&mut mock, &cache_key, data, cache_ttl_s, written_at)
            .await
            .expect("cache write with custom TTL should succeed");
    }
}
