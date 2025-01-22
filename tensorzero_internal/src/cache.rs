use std::collections::HashMap;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::batch::deserialize_json_string;
use crate::inference::types::{ContentBlock, ModelInferenceRequest, ModelInferenceResponse};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CacheOptions {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub lookback_s: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ModelProviderRequest<'request> {
    pub request: &'request ModelInferenceRequest<'request>,
    pub model_name: &'request str,
    pub provider_name: &'request str,
}

impl ModelProviderRequest<'_> {
    pub fn get_cache_key(&self) -> Result<[u8; 32], Error> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.model_name.as_bytes());
        hasher.update(self.provider_name.as_bytes());
        let serialized_request = serde_json::to_string(self.request).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize request: {e}"),
            })
        })?;
        let request_bytes = serialized_request.as_bytes();
        hasher.update(request_bytes);
        Ok(*hasher.finalize().as_bytes())
    }
}

#[derive(Debug, Serialize)]
struct ModelInferenceCacheRow<'a> {
    short_cache_key: u64,
    long_cache_key: String,
    output: &'a Vec<ContentBlock>,
    raw_request: String,
    raw_response: String,
}

pub async fn cache_write(
    clickhouse_client: &ClickHouseConnectionInfo,
    request: ModelProviderRequest<'_>,
    output: &Vec<ContentBlock>,
    raw_request: &str,
    raw_response: &str,
) -> Result<(), Error> {
    let cache_key = request.get_cache_key()?;
    let short_cache_key = u64::from_le_bytes(cache_key[..8].try_into().map_err(|e| {
        Error::new(ErrorDetails::Cache {
            message: format!("failed to convert hash into u64 for short cache key: {e}"),
        })
    })?);
    let long_cache_key = hex::encode(cache_key);
    clickhouse_client
        .write(
            &[ModelInferenceCacheRow {
                short_cache_key,
                long_cache_key,
                output,
                raw_request: raw_request.to_string(),
                raw_response: raw_response.to_string(),
            }],
            "ModelInferenceCache",
        )
        .await
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
    lookback_s: Option<u32>,
) -> Result<Option<ModelInferenceResponse>, Error> {
    let cache_key = request.get_cache_key()?;
    let short_cache_key = u64::from_le_bytes(cache_key[..8].try_into().map_err(|e| {
        Error::new(ErrorDetails::Cache {
            message: format!("failed to convert hash into u64 for short cache key: {e}"),
        })
    })?);
    let long_cache_key = hex::encode(cache_key);
    let query = if lookback_s.is_some() {
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
        ("short_cache_key".to_string(), short_cache_key.to_string()),
        ("long_cache_key".to_string(), long_cache_key),
    ]);
    if let Some(lookback) = lookback_s {
        query_params.insert("lookback_s".to_string(), lookback.to_string());
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

    use crate::inference::types::{FunctionType, ModelInferenceRequestJsonMode};

    use super::*;

    /// This test ensures that if we make a small change to the ModelInferenceRequest,
    /// the cache key will change.
    #[test]
    fn test_get_cache_key() {
        let model_inference_request = ModelInferenceRequest {
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
        let streaming_model_inference_request = ModelInferenceRequest {
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
