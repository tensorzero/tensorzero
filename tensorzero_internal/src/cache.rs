use std::collections::HashMap;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{ContentBlock, ModelInferenceRequest, ModelInferenceResponse};
use hex;
use serde::{Deserialize, Serialize};

pub struct ModelProviderRequest<'request> {
    pub request: &'request ModelInferenceRequest<'request>,
    pub model_name: &'request str,
    pub provider_name: &'request str,
}

impl<'request> ModelProviderRequest<'request> {
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

// pub async fn cache_lookup(request: ModelProviderRequest<'_>) -> Option<String> {
//     let cache_key = match request.get_cache_key() {
//         Ok(cache_key) => cache_key,
//         Err(e) => {
//             return None;
//         }
//     };

//     todo!()
// }

#[derive(Debug, Serialize, Deserialize)]
struct ModelInferenceCacheRow {
    short_cache_key: u64,
    long_cache_key: String,
    output: String,
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
    let serialized_output = serde_json::to_string(output).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize output: {e}"),
        })
    })?;
    clickhouse_client
        .write(
            &[ModelInferenceCacheRow {
                short_cache_key,
                long_cache_key,
                output: serialized_output,
                raw_request: raw_request.to_string(),
                raw_response: raw_response.to_string(),
            }],
            "ModelInferenceCache",
        )
        .await
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheLookupResult {
    pub output: Vec<ContentBlock>,
    pub raw_request: String,
    pub raw_response: String,
}

pub async fn cache_lookup(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    request: ModelProviderRequest<'_>,
) -> Result<Option<ModelInferenceResponse>, Error> {
    let cache_key = request.get_cache_key()?;
    let short_cache_key = u64::from_le_bytes(cache_key[..8].try_into().map_err(|e| {
        Error::new(ErrorDetails::Cache {
            message: format!("failed to convert hash into u64 for short cache key: {e}"),
        })
    })?);
    let long_cache_key = hex::encode(cache_key);
    let query = r#"
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
    "#;
    let query_params = HashMap::from([
        ("short_cache_key".to_string(), short_cache_key.to_string()),
        ("long_cache_key".to_string(), long_cache_key.into()),
    ]);
    let result = clickhouse_connection_info
        .run_query(query.to_string(), Some(&query_params))
        .await?;
    // TODO: handle missing vs error
    println!("result: {result}");
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
