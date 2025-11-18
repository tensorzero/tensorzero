use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use crate::config::OtlpConfig;
use crate::db::clickhouse::{ClickHouseConnectionInfo, TableName};
use crate::embeddings::{Embedding, EmbeddingModelResponse, EmbeddingRequest};
use crate::error::{warn_discarded_cache_write, Error, ErrorDetails};
use crate::inference::types::{
    ContentBlockChunk, ContentBlockOutput, FinishReason, ModelInferenceRequest,
    ModelInferenceResponse, ProviderInferenceResponseChunk, Usage,
};
use crate::model::StreamResponse;
use crate::serde_util::{deserialize_json_string, serialize_json_string};
use crate::tool::{InferenceResponseToolCall, ToolCallConfig};
use crate::utils::spawn_ignoring_shutdown;
use blake3::Hash;
use clap::ValueEnum;
use serde::de::{DeserializeOwned, IgnoredAny};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ValueEnum, ts_rs::TS,
)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "snake_case")]
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

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
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

#[derive(Debug, Default, Clone)]
pub struct CacheOptions {
    pub max_age_s: Option<u32>,
    pub enabled: CacheEnabledMode,
}

#[derive(Debug)]
pub struct BaseModelProviderRequest<'request, T> {
    pub request: &'request T,
    pub model_name: &'request str,
    pub provider_name: &'request str,
    pub otlp_config: &'request OtlpConfig,
}

// We need a manual impl to avoid adding a 'T: Copy' bound
impl<T> Copy for BaseModelProviderRequest<'_, T> {}
impl<T> Clone for BaseModelProviderRequest<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

pub type ModelProviderRequest<'a> = BaseModelProviderRequest<'a, ModelInferenceRequest<'a>>;
pub type EmbeddingModelProviderRequest<'a> = BaseModelProviderRequest<'a, EmbeddingRequest>;

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

impl From<Hash> for CacheKey {
    fn from(hash: Hash) -> Self {
        Self(hash.into())
    }
}

impl EmbeddingModelProviderRequest<'_> {
    // Destructure EmbeddingModelProviderRequest so that we get a compiler error
    // if we add any new fields
    pub fn get_cache_key(&self) -> Result<CacheKey, Error> {
        let EmbeddingModelProviderRequest {
            model_name,
            provider_name,
            request,
            // The OTLP config is deliberately not included in the cache key,
            // since it's only used to construct the OTEL span.
            otlp_config: _,
        } = self;
        let mut hasher = blake3::Hasher::new();
        hasher.update(model_name.as_bytes());
        hasher.update(&[0]); // null byte after model name to ensure data is prefix-free
        hasher.update(provider_name.as_bytes());
        hasher.update(&[0]); // null byte after provider name to ensure data is prefix-free

        // Convert the request to a JSON Value, error if serialization fails
        let request_json = serde_json::to_string(request).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize request: {e}"),
            })
        })?;
        hasher.update(request_json.as_bytes());
        Ok(hasher.finalize().into())
    }
}

impl ModelProviderRequest<'_> {
    pub fn get_cache_key(&self) -> Result<CacheKey, Error> {
        // Destructure ModelProviderRequest so that we get a compiler error
        // if we add any new fields
        let ModelProviderRequest {
            model_name,
            provider_name,
            request,
            // The OTLP config is deliberately not included in the cache key,
            // since it's only used to construct the OTEL span.
            otlp_config: _,
        } = self;
        let mut hasher = blake3::Hasher::new();
        hasher.update(model_name.as_bytes());
        hasher.update(&[0]); // null byte after model name to ensure data is prefix-free
        hasher.update(provider_name.as_bytes());
        hasher.update(&[0]); // null byte after provider name to ensure data is prefix-free
                             // Convert the request to a JSON Value, error if serialization fails

        let mut request_value = serde_json::to_value(request).map_err(|e| {
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
        Ok(hasher.finalize().into())
    }
}

// The full row written to ClickHouse
#[derive(Debug, Serialize)]
struct FullCacheRow<T: CacheOutput> {
    short_cache_key: u64,
    long_cache_key: String,
    // We flatten this so that the fields map directly to columns in the ClickHouse table
    #[serde(flatten)]
    data: CacheData<T>,
}

/// The underlying cached input/output data. These are the fields that we actually retrieve from
/// ClickHouse when going a cache fetch
#[derive(Debug, Deserialize, Serialize)]
pub struct CacheData<T: CacheOutput> {
    pub output: T,
    pub raw_request: String,
    pub raw_response: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub finish_reason: Option<FinishReason>,
}

/// A marker trait for types that can be used in the 'output' field of `CacheData`
/// This ensures that we don't accidentally try to serialize/deserialize the wrong type
/// to/from ClickHouse
/// We use a marker trait rather than an enum so that the expected type can be enforced by the caller
/// (e.g. `infer_stream` will never try to deserialize a `NonStreamingCacheData`)
pub trait CacheOutput {
    /// If this return `false`, then we'll log a warning and skip writing this entry to the cache
    fn should_write_to_cache(
        &self,
        cache_validation_info: CacheValidationInfo,
    ) -> impl Future<Output = bool> + Send;
}

impl CacheOutput for StreamingCacheData {
    async fn should_write_to_cache(&self, _cache_validation_info: CacheValidationInfo) -> bool {
        // TODO - getting access to the tool calls would require re-running `collect_chunks`,
        // or refactoring it to make the parsed content blocks available when performing the cache write.
        // For now, we'll just always write to the cache.
        true
    }
}
impl CacheOutput for NonStreamingCacheData {
    async fn should_write_to_cache(&self, cache_validation_info: CacheValidationInfo) -> bool {
        for block in &self.blocks {
            if let ContentBlockOutput::ToolCall(tool_call) = block {
                if cache_validation_info.tool_config.is_some() {
                    // If we have a tool config, validate against the schema
                    let output = InferenceResponseToolCall::new(
                        tool_call.clone(),
                        cache_validation_info.tool_config.as_ref(),
                    )
                    .await;
                    if output.name.is_none() || output.arguments.is_none() {
                        return false;
                    }
                } else {
                    // If we don't have a tool config, then just check that the arguments are valid JSON
                    if serde_json::from_str::<IgnoredAny>(&tool_call.arguments).is_err() {
                        return false;
                    }
                }
            }
        }
        true
    }
}
impl CacheOutput for EmbeddingCacheData {
    async fn should_write_to_cache(&self, _cache_validation_info: CacheValidationInfo) -> bool {
        true
    }
}

/// Cache data for embeddings.
///
/// Note: Unlike `NonStreamingCacheData` and `StreamingCacheData`, this requires both `serialize_with` and
/// `deserialize_with` because `Embedding` is an untagged enum. Without `untagged`, OpenAI's API responses wouldn't
/// deserialize correctly (they send bare arrays/strings). But with `untagged`, the enum serializes as bare JSON
/// values ([1.0, 2.0] or "abc"), which breaks `deserialize_json_string` (which expects a JSON-encoded string). So we
/// need `serialize_json_string` to ensure the data is stored in the format that `deserialize_json_string` expects.
#[derive(Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct EmbeddingCacheData {
    #[serde(
        serialize_with = "serialize_json_string",
        deserialize_with = "deserialize_json_string"
    )]
    pub embedding: Embedding,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct NonStreamingCacheData {
    #[serde(deserialize_with = "deserialize_json_string")]
    pub blocks: Vec<ContentBlockOutput>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct StreamingCacheData {
    #[serde(deserialize_with = "deserialize_json_string")]
    pub chunks: Vec<CachedProviderInferenceResponseChunk>,
}

fn spawn_maybe_cache_write<T: Serialize + CacheOutput + Send + Sync + 'static>(
    row: FullCacheRow<T>,
    clickhouse_client: ClickHouseConnectionInfo,
    cache_validation_info: CacheValidationInfo,
) {
    spawn_ignoring_shutdown(async move {
        if row
            .data
            .output
            .should_write_to_cache(cache_validation_info)
            .await
        {
            if let Err(e) = clickhouse_client
                .write_batched(&[row], TableName::ModelInferenceCache)
                .await
            {
                tracing::warn!("Failed to write to cache: {e}");
            }
        } else {
            warn_discarded_cache_write(&row.data.raw_response);
        }
    });
}

/// Holds fields used to validate a `CacheData` before writing.
/// We use this to skip writing certain 'bad' cache entries
/// (e.g. when a tool call fails validation), while still allowing the
/// inference itself to succeed and return a response to the user
///
/// In the future, we may perform additional checks
/// (e.g. validating against the `output_schema`).
pub struct CacheValidationInfo {
    // The `ToolCallConfig` for the top-level inference request, if present
    // This is deliberately not part of the cache key - we only use it to
    // skip writing certain cache entries.
    pub tool_config: Option<ToolCallConfig>,
}

// This doesn't block
pub fn start_cache_write<T: Serialize + CacheOutput + Send + Sync + 'static>(
    clickhouse_client: &ClickHouseConnectionInfo,
    cache_key: CacheKey,
    cache_data: CacheData<T>,
    cache_validation_info: CacheValidationInfo,
) -> Result<(), Error> {
    let short_cache_key = cache_key.get_short_key()?;
    let long_cache_key = cache_key.get_long_key();
    let clickhouse_client = clickhouse_client.clone();
    spawn_maybe_cache_write(
        FullCacheRow {
            short_cache_key,
            long_cache_key,
            data: cache_data,
        },
        clickhouse_client,
        cache_validation_info,
    );
    Ok(())
}

/// A subset of `ProviderInferenceResponseChunk` containing only the fields we want to cache
/// For example, we exclude 'usage', and fill it in with 0 input/output tokens when we
/// return a cached chunk.
#[derive(Debug, Deserialize, Serialize)]
pub struct CachedProviderInferenceResponseChunk {
    pub content: Vec<ContentBlockChunk>,
    #[serde(default)]
    pub usage: Option<Usage>,
    pub raw_response: String,
}

// This starts a trailing write to the cache (without blocking the http response)
pub fn start_cache_write_streaming(
    clickhouse_client: &ClickHouseConnectionInfo,
    cache_key: CacheKey,
    chunks: Vec<ProviderInferenceResponseChunk>,
    raw_request: &str,
    usage: &Usage,
    tool_config: Option<ToolCallConfig>,
) -> Result<(), Error> {
    let short_cache_key = cache_key.get_short_key()?;
    let long_cache_key = cache_key.get_long_key();
    let input_tokens = usage.input_tokens;
    let output_tokens = usage.output_tokens;
    let mut finish_reason = None;
    for chunk in &chunks {
        if let Some(chunk_finish_reason) = &chunk.finish_reason {
            finish_reason = Some(chunk_finish_reason);
        }
    }
    let finish_reason = finish_reason.cloned();
    let output = StreamingCacheData {
        chunks: chunks
            .into_iter()
            .map(|c| CachedProviderInferenceResponseChunk {
                content: c.content,
                usage: c.usage,
                raw_response: c.raw_response,
            })
            .collect(),
    };
    let raw_request = raw_request.to_string();
    let clickhouse_client = clickhouse_client.clone();
    spawn_maybe_cache_write(
        FullCacheRow {
            short_cache_key,
            long_cache_key,
            data: CacheData {
                output,
                raw_request,
                raw_response: String::new(),
                input_tokens,
                output_tokens,
                finish_reason,
            },
        },
        clickhouse_client,
        CacheValidationInfo { tool_config },
    );
    Ok(())
}

pub async fn embedding_cache_lookup(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    request: &EmbeddingModelProviderRequest<'_>,
    max_age_s: Option<u32>,
) -> Result<Option<EmbeddingModelResponse>, Error> {
    let result = cache_lookup_inner::<EmbeddingCacheData>(
        clickhouse_connection_info,
        request.get_cache_key()?,
        max_age_s,
    )
    .await?;
    Ok(result.map(|result| EmbeddingModelResponse::from_cache(result, request)))
}

pub async fn cache_lookup(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    request: ModelProviderRequest<'_>,
    max_age_s: Option<u32>,
) -> Result<Option<ModelInferenceResponse>, Error> {
    let result = cache_lookup_inner::<NonStreamingCacheData>(
        clickhouse_connection_info,
        request.get_cache_key()?,
        max_age_s,
    )
    .await?;
    Ok(result.map(|result| {
        ModelInferenceResponse::from_cache(result, request.request, request.provider_name)
    }))
}

pub async fn cache_lookup_streaming(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    request: ModelProviderRequest<'_>,
    max_age_s: Option<u32>,
) -> Result<Option<StreamResponse>, Error> {
    let result = cache_lookup_inner(
        clickhouse_connection_info,
        request.get_cache_key()?,
        max_age_s,
    )
    .await?;
    Ok(result.map(|result| StreamResponse::from_cache(result, Arc::from(request.provider_name))))
}

pub async fn cache_lookup_inner<T: CacheOutput + DeserializeOwned>(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    cache_key: CacheKey,
    max_age_s: Option<u32>,
) -> Result<Option<CacheData<T>>, Error> {
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
    let result = clickhouse_connection_info
        .run_query_synchronous(query.to_string(), &query_params)
        .await?;
    if result.response.is_empty() {
        return Ok(None);
    }
    let result: CacheData<T> = serde_json::from_str(&result.response).map_err(|e| {
        Error::new(ErrorDetails::Cache {
            message: format!("Failed to deserialize output: {e}"),
        })
    })?;
    Ok(Some(result))
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
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            extra_cache_key: None,
            stop_sequences: None,
            ..Default::default()
        };
        let model_provider_request = ModelProviderRequest {
            request: &model_inference_request,
            model_name: "test_model",
            provider_name: "test_provider",
            otlp_config: &Default::default(),
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
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
            stop_sequences: None,
            ..Default::default()
        };
        let model_provider_request = ModelProviderRequest {
            request: &model_inference_request,
            model_name: "test_model",
            provider_name: "test_provider",
            otlp_config: &Default::default(),
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
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
            stop_sequences: None,
            ..Default::default()
        };
        let model_provider_request = ModelProviderRequest {
            request: &streaming_model_inference_request,
            model_name: "test_model",
            provider_name: "test_provider",
            otlp_config: &Default::default(),
        };
        let streaming_cache_key = model_provider_request.get_cache_key().unwrap();
        assert_ne!(cache_key, streaming_cache_key);
    }
}
