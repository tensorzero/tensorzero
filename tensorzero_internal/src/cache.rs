use std::collections::HashMap;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::batch::deserialize_json_string;
use crate::inference::types::{ContentBlock, ModelInferenceRequest, ModelInferenceResponse};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheEnabledMode {
    On,
    Off,
    ReadOnly,
    #[default]
    WriteOnly,
}

impl CacheEnabledMode {
    pub fn write(&self) -> bool {
        matches!(self, CacheEnabledMode::On | CacheEnabledMode::WriteOnly)
    }

    pub fn read(&self) -> bool {
        matches!(self, CacheEnabledMode::On | CacheEnabledMode::ReadOnly)
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct CacheParamsOptions {
    #[serde(default)]
    pub max_age_s: Option<u32>,
    #[serde(default)]
    pub enabled: CacheEnabledMode,
}

impl From<(CacheParamsOptions, bool)> for CacheOptions {
    fn from((options, dryrun): (CacheParamsOptions, bool)) -> Self {
        let enabled = match (options.enabled, dryrun) {
            (CacheEnabledMode::On, true) => CacheEnabledMode::ReadOnly,
            (CacheEnabledMode::On, false) => CacheEnabledMode::On,
            (CacheEnabledMode::WriteOnly, true) => CacheEnabledMode::Off,
            (CacheEnabledMode::WriteOnly, false) => CacheEnabledMode::WriteOnly,
            (CacheEnabledMode::ReadOnly, _) => CacheEnabledMode::ReadOnly,
            (CacheEnabledMode::Off, _) => CacheEnabledMode::Off,
        };
        Self {
            max_age_s: options.max_age_s,
            enabled,
        }
    }
}

#[derive(Debug)]
pub struct CacheOptions {
    pub max_age_s: Option<u32>,
    pub enabled: CacheEnabledMode,
}

#[derive(Debug, Clone)]
pub struct ModelProviderRequest<'request> {
    pub request: &'request ModelInferenceRequest<'request>,
    pub model_name: &'request str,
    pub provider_name: &'request str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheKey([u8; 32]);

impl CacheKey {
    pub fn get_short_key(&self) -> Result<u64, Error> {
        let bytes = self.0[..8].try_into().map_err(|e| {
            Error::new(ErrorDetails::Cache {
                message: format!("Failed to convert hash into u64 for short cache key: {e}"),
            })
        })?;
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn get_long_key(&self) -> String {
        hex::encode(self.0)
    }
}

impl ModelProviderRequest<'_> {
    pub fn get_cache_key(&self) -> Result<CacheKey, Error> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.model_name.as_bytes());
        hasher.update(&[0]); // null byte after model name to ensure data is prefix-free
        hasher.update(self.provider_name.as_bytes());
        hasher.update(&[0]); // null byte after provider name to ensure data is prefix-free
                             // Convert the request to a JSON Value, error if serialization fails
        let mut request_value = serde_json::to_value(self.request).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize request: {e}"),
            })
        })?;

        // Convert the Value to a mutable object and remove the inference_id field
        // We remove inference_id since it's unique per request and would prevent cache hits
        request_value
            .as_object_mut()
            .ok_or_else(|| {
                Error::new(ErrorDetails::Serialization {
                    message: "Failed to convert request to object".to_string(),
                })
            })?
            .remove("inference_id");

        // Convert the modified request back to a JSON string
        let serialized_request = serde_json::to_string(&request_value).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize request: {e}"),
            })
        })?;
        // Get the bytes of the serialized request to use in the hash
        let request_bytes = serialized_request.as_bytes();
        hasher.update(request_bytes);
        Ok(CacheKey(hasher.finalize().into()))
    }
}

#[derive(Debug, Serialize)]
struct ModelInferenceCacheRow {
    short_cache_key: u64,
    long_cache_key: String,
    output: Vec<ContentBlock>,
    raw_request: String,
    raw_response: String,
}

// This doesn't block
pub fn start_cache_write(
    clickhouse_client: &ClickHouseConnectionInfo,
    request: ModelProviderRequest<'_>,
    output: &[ContentBlock],
    raw_request: &str,
    raw_response: &str,
) -> Result<(), Error> {
    let cache_key = request.get_cache_key()?;
    let short_cache_key = cache_key.get_short_key()?;
    let long_cache_key = cache_key.get_long_key();
    let output = output.to_owned();
    let raw_request = raw_request.to_string();
    let raw_response = raw_response.to_string();
    let clickhouse_client = clickhouse_client.clone();
    tokio::spawn(async move {
        clickhouse_client
            .write(
                &[ModelInferenceCacheRow {
                    short_cache_key,
                    long_cache_key,
                    output,
                    raw_request,
                    raw_response,
                }],
                "ModelInferenceCache",
            )
            .await
    });
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheLookupResult {
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: Vec<ContentBlock>,
    pub raw_request: String,
    pub raw_response: String,
}

pub async fn cache_lookup(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    request: ModelProviderRequest<'_>,
    max_age_s: Option<u32>,
) -> Result<Option<ModelInferenceResponse>, Error> {
    let cache_key = request.get_cache_key()?;
    // NOTE: the short cache key is just so the ClickHouse index can be as efficient as possible
    // but we always check against the long cache key before returning a result
    let short_cache_key = cache_key.get_short_key()?.to_string();
    let long_cache_key = cache_key.get_long_key();
    let query = if max_age_s.is_some() {
        r#"
            SELECT
                output,
                raw_request,
                raw_response
            FROM ModelInferenceCache
            WHERE short_cache_key = {short_cache_key:UInt64}
                AND long_cache_key = {long_cache_key:String}
                AND timestamp > subtractSeconds(now(), {lookback_s:UInt32})
            ORDER BY timestamp DESC
            LIMIT 1
            FORMAT JSONEachRow
        "#
    } else {
        r#"
            SELECT
                output,
                raw_request,
                raw_response
            FROM ModelInferenceCache
            WHERE short_cache_key = {short_cache_key:UInt64}
                AND long_cache_key = {long_cache_key:String}
            ORDER BY timestamp DESC
            LIMIT 1
            FORMAT JSONEachRow
        "#
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
    let result = clickhouse_connection_info
        .run_query(query.to_string(), Some(&query_params))
        .await?;
    if result.is_empty() {
        return Ok(None);
    }
    let result: CacheLookupResult = serde_json::from_str(&result).map_err(|e| {
        Error::new(ErrorDetails::Cache {
            message: format!("Failed to deserialize output: {e}"),
        })
    })?;
    Ok(Some(ModelInferenceResponse::from_cache(
        result,
        request.request,
        request.provider_name,
    )))
}

#[cfg(test)]
mod tests {

    use uuid::Uuid;

    use crate::inference::types::{FunctionType, ModelInferenceRequestJsonMode};

    use super::*;

    /// This test ensures that if we make a small change to the ModelInferenceRequest,
    /// the cache key will change.
    #[test]
    fn test_get_cache_key() {
        let model_inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let model_provider_request = ModelProviderRequest {
            request: &model_inference_request,
            model_name: "test_model",
            provider_name: "test_provider",
        };
        let cache_key = model_provider_request.get_cache_key().unwrap();
        let model_inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let model_provider_request = ModelProviderRequest {
            request: &model_inference_request,
            model_name: "test_model",
            provider_name: "test_provider",
        };
        let new_cache_key = model_provider_request.get_cache_key().unwrap();
        // Make sure the first two get the same cache key (and that we ignore the inference_id)
        assert_eq!(cache_key, new_cache_key);
        let streaming_model_inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let model_provider_request = ModelProviderRequest {
            request: &streaming_model_inference_request,
            model_name: "test_model",
            provider_name: "test_provider",
        };
        let streaming_cache_key = model_provider_request.get_cache_key().unwrap();
        assert_ne!(cache_key, streaming_cache_key);
    }
}
