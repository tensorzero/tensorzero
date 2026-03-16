use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::OtlpConfig;
use crate::config::gateway::{InferenceCacheBackend, ModelInferenceCacheConfig};
use crate::db::cache::CacheQueries;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::delegating_connection::PrimaryDatastore;
use crate::db::valkey::ValkeyConnectionInfo;
use crate::db::valkey::cache::ValkeyCacheClient;
use crate::embeddings::{Embedding, EmbeddingModelResponse, EmbeddingRequest};
use crate::error::{Error, ErrorDetails, warn_discarded_cache_write};
use crate::inference::types::{
    ContentBlockChunk, ContentBlockOutput, FinishReason, ModelInferenceRequest,
    ModelInferenceResponse, ProviderInferenceResponseChunk, Usage,
};
use crate::model::StreamResponse;
use crate::serde_util::{deserialize_json_string, serialize_json_string};
use tensorzero_provider_types::ProviderToolCallConfig;

use crate::tool::{InferenceResponseToolCall, InferenceResponseToolCallExt};
use crate::utils::spawn_ignoring_shutdown;
use blake3::Hash;
use clap::ValueEnum;
use serde::de::{DeserializeOwned, IgnoredAny};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use uuid::Uuid;

/// Manager for cache operations, wrapping a `dyn CacheQueries` backend.
#[derive(Clone)]
pub struct CacheManager {
    client: Arc<dyn CacheQueries>,
}

impl CacheManager {
    pub fn new(client: Arc<dyn CacheQueries>) -> Self {
        Self { client }
    }

    /// Select the appropriate cache backend based on config and available connections.
    ///
    /// Respects `cache_config.enabled`:
    /// - `Some(false)` → disabled (no-op)
    /// - `Some(true)` → resolve backend, error if unavailable
    /// - `None` → resolve backend (see below)
    ///
    /// Backend resolution (`cache_config.backend`):
    /// - `Auto` + ClickHouse primary → ClickHouse
    /// - `Auto` + Postgres primary → Valkey (disabled if unavailable)
    /// - `ClickHouse` → ClickHouse (error if unavailable)
    /// - `Valkey` → Valkey (error if unavailable)
    pub fn new_from_connections(
        valkey_connection_info: &ValkeyConnectionInfo,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        cache_config: &ModelInferenceCacheConfig,
        primary_datastore: PrimaryDatastore,
    ) -> Result<Self, Error> {
        if cache_config.enabled == Some(false) {
            return Ok(Self::disabled());
        }

        let clickhouse_available =
            clickhouse_connection_info.client_type() != ClickHouseClientType::Disabled;
        let valkey_available =
            matches!(valkey_connection_info, ValkeyConnectionInfo::Enabled { .. });

        match cache_config.backend {
            InferenceCacheBackend::Auto => {
                let explicitly_enabled = cache_config.enabled == Some(true);
                if primary_datastore == PrimaryDatastore::ClickHouse {
                    if !clickhouse_available && explicitly_enabled {
                        return Err(ErrorDetails::AppState {
                            message: format!(
                                "`cache.enabled` is `true` but the cache backend (`{:?}`) is not available. \
                                 Ensure the required connection URL is set, or set `cache.enabled` to `false`.",
                                cache_config.backend,
                            ),
                        }
                        .into());
                    }
                    return Ok(Self::new(Arc::new(clickhouse_connection_info.clone())));
                }
                // Postgres primary: check if any backend is available
                if explicitly_enabled && !valkey_available && !clickhouse_available {
                    return Err(ErrorDetails::AppState {
                        message: format!(
                            "`cache.enabled` is `true` but the cache backend (`{:?}`) is not available. \
                             Ensure the required connection URL is set, or set `cache.enabled` to `false`.",
                            cache_config.backend,
                        ),
                    }
                    .into());
                }
                match valkey_connection_info {
                    ValkeyConnectionInfo::Enabled { connection } => Ok(Self::new(Arc::new(
                        ValkeyCacheClient::new(connection.clone(), cache_config.valkey.ttl_s),
                    ))),
                    ValkeyConnectionInfo::Disabled => Ok(Self::disabled()),
                }
            }
            InferenceCacheBackend::ClickHouse => {
                if !clickhouse_available {
                    return Err(ErrorDetails::AppState {
                        message: "`cache.backend` is set to `clickhouse` but ClickHouse is not available. \
                                  Ensure the required connection URL is set, or remove the `cache.backend` setting."
                            .to_string(),
                    }
                    .into());
                }
                Ok(Self::new(Arc::new(clickhouse_connection_info.clone())))
            }
            InferenceCacheBackend::Valkey => match valkey_connection_info {
                ValkeyConnectionInfo::Enabled { connection } => Ok(Self::new(Arc::new(
                    ValkeyCacheClient::new(connection.clone(), cache_config.valkey.ttl_s),
                ))),
                ValkeyConnectionInfo::Disabled => Err(ErrorDetails::AppState {
                    message: "`cache.backend` is set to `valkey` but Valkey is not available. \
                              Ensure the required connection URL is set, or remove the `cache.backend` setting."
                        .to_string(),
                }
                .into()),
            },
        }
    }

    /// Create a disabled cache manager (no-op: lookups return `None`, writes succeed immediately).
    pub fn disabled() -> Self {
        Self::new(Arc::new(DisabledCacheQueries))
    }
}

#[async_trait]
impl CacheQueries for CacheManager {
    async fn cache_lookup(
        &self,
        cache_key: &CacheKey,
        max_age_s: Option<u32>,
    ) -> Result<Option<String>, Error> {
        self.client.cache_lookup(cache_key, max_age_s).await
    }

    async fn cache_write(&self, cache_key: &CacheKey, data: &str) -> Result<(), Error> {
        self.client.cache_write(cache_key, data).await
    }
}

/// No-op cache backend: lookups return `None`, writes succeed immediately.
struct DisabledCacheQueries;

#[async_trait]
impl CacheQueries for DisabledCacheQueries {
    async fn cache_lookup(
        &self,
        _cache_key: &CacheKey,
        _max_age_s: Option<u32>,
    ) -> Result<Option<String>, Error> {
        Ok(None)
    }

    async fn cache_write(&self, _cache_key: &CacheKey, _data: &str) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Eq,
    PartialEq,
    Serialize,
    ValueEnum,
    schemars::JsonSchema,
)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "snake_case")]
pub enum CacheEnabledMode {
    On,
    #[default]
    Off,
    ReadOnly,
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
    pub model_inference_id: Uuid,
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
            model_inference_id: _,
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
            model_inference_id: _,
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

/// The underlying cached input/output data.
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
                    // If we have a tool config, validate that the tool name is known and arguments are valid JSON
                    let output = InferenceResponseToolCall::new_from_provider_tool_call(
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
    #[serde(
        serialize_with = "serialize_json_string",
        deserialize_with = "deserialize_json_string"
    )]
    pub blocks: Vec<ContentBlockOutput>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct StreamingCacheData {
    #[serde(
        serialize_with = "serialize_json_string",
        deserialize_with = "deserialize_json_string"
    )]
    pub chunks: Vec<CachedProviderInferenceResponseChunk>,
}

fn spawn_maybe_cache_write<
    T: Serialize + CacheOutput + Send + Sync + 'static,
    C: CacheQueries + Clone + 'static,
>(
    cache_key: CacheKey,
    cache_data: CacheData<T>,
    cache_manager: C,
    cache_validation_info: CacheValidationInfo,
) {
    spawn_ignoring_shutdown(async move {
        if cache_data
            .output
            .should_write_to_cache(cache_validation_info)
            .await
        {
            let data_json = match serde_json::to_string(&cache_data) {
                Ok(json) => json,
                Err(e) => {
                    tracing::warn!("Failed to serialize cache data: {e}");
                    return;
                }
            };
            if let Err(e) = cache_manager.cache_write(&cache_key, &data_json).await {
                tracing::warn!("Failed to write to cache: {e}");
            }
        } else {
            warn_discarded_cache_write(&cache_data.raw_response);
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
    // The `ProviderToolCallConfig` for the top-level inference request, if present
    // This is deliberately not part of the cache key - we only use it to
    // skip writing certain cache entries.
    pub tool_config: Option<ProviderToolCallConfig>,
}

// This doesn't block
pub fn start_cache_write<
    T: Serialize + CacheOutput + Send + Sync + 'static,
    C: CacheQueries + Clone + 'static,
>(
    cache_manager: &C,
    cache_key: CacheKey,
    cache_data: CacheData<T>,
    cache_validation_info: CacheValidationInfo,
) -> Result<(), Error> {
    let cache_manager = cache_manager.clone();
    spawn_maybe_cache_write(cache_key, cache_data, cache_manager, cache_validation_info);
    Ok(())
}

/// A subset of `ProviderInferenceResponseChunk` containing only the fields we want to cache.
/// We persist normalized usage but intentionally drop any raw usage entries.
#[derive(Debug, Deserialize, Serialize)]
pub struct CachedProviderInferenceResponseChunk {
    pub content: Vec<ContentBlockChunk>,
    #[serde(default)]
    pub usage: Option<Usage>,
    pub raw_response: String,
}

// This starts a trailing write to the cache (without blocking the http response)
pub fn start_cache_write_streaming<C: CacheQueries + Clone + 'static>(
    cache_manager: &C,
    cache_key: CacheKey,
    chunks: Vec<ProviderInferenceResponseChunk>,
    raw_request: &str,
    usage: &Usage,
    tool_config: Option<ProviderToolCallConfig>,
) -> Result<(), Error> {
    let input_tokens = usage.input_tokens;
    let output_tokens = usage.output_tokens;
    let mut finish_reason = None;
    for chunk in &chunks {
        if let Some(chunk_finish_reason) = &chunk.finish_reason {
            finish_reason = Some(chunk_finish_reason);
        }
    }
    let finish_reason = finish_reason.copied();
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
    let cache_manager = cache_manager.clone();
    spawn_maybe_cache_write(
        cache_key,
        CacheData {
            output,
            raw_request,
            raw_response: String::new(),
            input_tokens,
            output_tokens,
            finish_reason,
        },
        cache_manager,
        CacheValidationInfo { tool_config },
    );
    Ok(())
}

pub async fn embedding_cache_lookup(
    cache_manager: &impl CacheQueries,
    request: &EmbeddingModelProviderRequest<'_>,
    max_age_s: Option<u32>,
    provider_type: Arc<str>,
) -> Result<Option<EmbeddingModelResponse>, Error> {
    let result = cache_lookup_inner::<EmbeddingCacheData>(
        cache_manager,
        request.get_cache_key()?,
        max_age_s,
    )
    .await?;
    Ok(result.map(|result| EmbeddingModelResponse::from_cache(result, request, provider_type)))
}

pub async fn cache_lookup(
    cache_manager: &impl CacheQueries,
    request: ModelProviderRequest<'_>,
    max_age_s: Option<u32>,
    provider_type: Arc<str>,
) -> Result<Option<ModelInferenceResponse>, Error> {
    let result = cache_lookup_inner::<NonStreamingCacheData>(
        cache_manager,
        request.get_cache_key()?,
        max_age_s,
    )
    .await?;
    Ok(result.map(|result| {
        ModelInferenceResponse::from_cache(
            result,
            request.request,
            request.provider_name,
            provider_type,
        )
    }))
}

pub async fn cache_lookup_streaming(
    cache_manager: &impl CacheQueries,
    request: ModelProviderRequest<'_>,
    max_age_s: Option<u32>,
    provider_type: Arc<str>,
) -> Result<Option<StreamResponse>, Error> {
    let result = cache_lookup_inner(cache_manager, request.get_cache_key()?, max_age_s).await?;
    Ok(result.map(|result| {
        StreamResponse::from_cache(
            result,
            Arc::from(request.provider_name),
            provider_type,
            request.model_inference_id,
        )
    }))
}

pub async fn cache_lookup_inner<T: CacheOutput + DeserializeOwned>(
    cache_manager: &impl CacheQueries,
    cache_key: CacheKey,
    max_age_s: Option<u32>,
) -> Result<Option<CacheData<T>>, Error> {
    let response = cache_manager.cache_lookup(&cache_key, max_age_s).await?;
    match response {
        None => Ok(None),
        Some(json) => {
            let result: CacheData<T> = serde_json::from_str(&json).map_err(|e| {
                Error::new(ErrorDetails::Cache {
                    message: format!("Failed to deserialize output: {e}"),
                })
            })?;
            Ok(Some(result))
        }
    }
}

#[cfg(test)]
mod tests {

    use std::borrow::Cow;

    use std::panic::{RefUnwindSafe, UnwindSafe};

    use googletest::prelude::*;
    use uuid::Uuid;

    use crate::embeddings::{EmbeddingEncodingFormat, EmbeddingInput};
    use crate::inference::types::chat_completion_inference_params::ChatCompletionInferenceParamsV2;
    use crate::inference::types::extra_body::{
        ExtraBodyConfig, ExtraBodyReplacement, ExtraBodyReplacementKind, FullExtraBodyConfig,
    };
    use crate::inference::types::extra_headers::{
        ExtraHeader, ExtraHeaderKind, ExtraHeadersConfig, FullExtraHeadersConfig,
    };
    use crate::inference::types::{
        ContentBlock, FunctionType, ModelInferenceRequestJsonMode, RequestMessage,
    };
    use tensorzero_provider_types::ProviderToolCallConfig;
    use tensorzero_types::Role;

    use super::*;

    #[gtest]
    #[tokio::test]
    async fn test_disabled_returns_none() {
        let cache = CacheManager::disabled();
        let key = CacheKey::from(blake3::hash(b"test"));
        let result = cache.cache_lookup(&key, None).await.unwrap();
        expect_that!(result, none());
    }

    #[gtest]
    #[tokio::test]
    async fn test_disabled_write_succeeds() {
        let cache = CacheManager::disabled();
        let key = CacheKey::from(blake3::hash(b"test"));
        let result = cache.cache_write(&key, r#"{"output":"test"}"#).await;
        expect_that!(
            result,
            ok(anything()),
            "disabled backend write should succeed"
        );
    }

    /// Fixture providing a baseline `ModelInferenceRequest` with all cache-relevant
    /// fields set to non-default values, plus its precomputed cache key.
    ///
    /// The exhaustive destructure in `set_up` ensures a compile error when a new field
    /// is added to `ModelInferenceRequest`, forcing the author to decide whether it
    /// affects the cache key and add a corresponding test.
    struct CacheKeyFixture {
        request: ModelInferenceRequest<'static>,
        key: CacheKey,
    }

    // ModelInferenceRequest contains Vec/String which aren't UnwindSafe by default,
    // but our fixture is read-only so this is safe.
    impl UnwindSafe for CacheKeyFixture {}
    impl RefUnwindSafe for CacheKeyFixture {}

    impl Fixture for CacheKeyFixture {
        fn set_up() -> googletest::Result<Self> {
            static BASELINE_OUTPUT_SCHEMA: std::sync::LazyLock<serde_json::Value> =
                std::sync::LazyLock::new(|| serde_json::json!({"type": "object"}));

            let request = ModelInferenceRequest {
                inference_id: Uuid::now_v7(),
                messages: vec![RequestMessage {
                    role: Role::User,
                    content: vec![ContentBlock::from("hello".to_string())],
                }],
                system: Some("you are helpful".to_string()),
                tool_config: None,
                temperature: Some(0.5),
                top_p: Some(0.9),
                presence_penalty: Some(0.1),
                frequency_penalty: Some(0.2),
                max_tokens: Some(100),
                seed: Some(42),
                stream: false,
                json_mode: ModelInferenceRequestJsonMode::Off,
                function_type: FunctionType::Chat,
                output_schema: Some(&*BASELINE_OUTPUT_SCHEMA),
                extra_body: Default::default(),
                extra_headers: Default::default(),
                fetch_and_encode_input_files_before_inference: false,
                extra_cache_key: Some("baseline".to_string()),
                stop_sequences: Some(Cow::Owned(vec!["stop".to_string()])),
                inference_params_v2: ChatCompletionInferenceParamsV2 {
                    reasoning_effort: Some("high".to_string()),
                    service_tier: None,
                    thinking_budget_tokens: Some(1000),
                    verbosity: Some("verbose".to_string()),
                },
            };

            let key = cache_key_for(&request, "model", "provider");
            Ok(Self { request, key })
        }

        fn tear_down(self) -> googletest::Result<()> {
            Ok(())
        }
    }

    fn cache_key_for(
        request: &ModelInferenceRequest,
        model_name: &str,
        provider_name: &str,
    ) -> CacheKey {
        let otlp_config = OtlpConfig::default();
        ModelProviderRequest {
            request,
            model_name,
            provider_name,
            otlp_config: &otlp_config,
            model_inference_id: Uuid::now_v7(),
        }
        .get_cache_key()
        .expect("get_cache_key should not fail for a valid request")
    }

    #[gtest]
    fn test_all_fields_are_considered_for_cache(fixture: &CacheKeyFixture) {
        // Exhaustive destructure: a compile error here means a new field was added
        // and must be classified as cache-relevant or explicitly excluded.
        let ModelInferenceRequest {
            // Excluded from cache key (unique per request)
            inference_id: _,
            // All remaining fields participate via JSON serialization
            messages: _,
            system: _,
            tool_config: _,
            temperature: _,
            top_p: _,
            presence_penalty: _,
            frequency_penalty: _,
            max_tokens: _,
            seed: _,
            stream: _,
            json_mode: _,
            function_type: _,
            output_schema: _,
            extra_body: _,
            extra_headers: _,
            fetch_and_encode_input_files_before_inference: _,
            extra_cache_key: _,
            stop_sequences: _,
            inference_params_v2: _,
        } = fixture.request;

        // Expect a compile time error if this is incorrect.
    }

    #[gtest]
    fn test_cache_key_ignores_inference_id(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.inference_id = Uuid::now_v7();
        expect_that!(cache_key_for(&req, "model", "provider"), eq(fixture.key));
    }

    #[gtest]
    fn test_cache_key_ignores_model_inference_id(fixture: &CacheKeyFixture) {
        // cache_key_for uses a fresh model_inference_id each time
        let key = cache_key_for(&fixture.request, "model", "provider");
        expect_that!(key, eq(fixture.key));
    }

    #[gtest]
    fn test_cache_key_changes_with_model_name(fixture: &CacheKeyFixture) {
        let key = cache_key_for(&fixture.request, "other_model", "provider");
        expect_that!(key, not(eq(fixture.key)));
    }

    #[gtest]
    fn test_cache_key_changes_with_provider_name(fixture: &CacheKeyFixture) {
        let key = cache_key_for(&fixture.request, "model", "other_provider");
        expect_that!(key, not(eq(fixture.key)));
    }

    #[gtest]
    fn test_cache_key_prefix_free_encoding(fixture: &CacheKeyFixture) {
        let key = cache_key_for(&fixture.request, "modelp", "rovider");
        expect_that!(key, not(eq(fixture.key)));
    }

    #[gtest]
    fn test_cache_key_changes_with_messages(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.messages = vec![RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::from("goodbye".to_string())],
        }];
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_system(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.system = Some("different system".to_string());
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_tool_config(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.tool_config = Some(Cow::Owned(ProviderToolCallConfig::default()));
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_temperature(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.temperature = Some(0.9);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_top_p(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.top_p = Some(0.1);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_presence_penalty(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.presence_penalty = Some(0.9);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_frequency_penalty(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.frequency_penalty = Some(0.9);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_max_tokens(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.max_tokens = Some(200);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_seed(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.seed = Some(99);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_stream(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.stream = true;
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_json_mode(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.json_mode = ModelInferenceRequestJsonMode::On;
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_function_type(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.function_type = FunctionType::Json;
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_output_schema(fixture: &CacheKeyFixture) {
        let different_schema = serde_json::json!({"type": "string"});
        let mut req = fixture.request.clone();
        req.output_schema = Some(&different_schema);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_extra_cache_key(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.extra_cache_key = Some("different".to_string());
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_stop_sequences(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.stop_sequences = Some(Cow::Owned(vec!["different".to_string()]));
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_fetch_and_encode_input_files(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.fetch_and_encode_input_files_before_inference = true;
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_extra_body(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.extra_body = FullExtraBodyConfig {
            extra_body: Some(ExtraBodyConfig {
                data: vec![ExtraBodyReplacement {
                    pointer: "/custom_field".to_string(),
                    kind: ExtraBodyReplacementKind::Value(serde_json::json!("test")),
                }],
            }),
            inference_extra_body: Default::default(),
        };
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_extra_headers(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.extra_headers = FullExtraHeadersConfig {
            variant_extra_headers: Some(ExtraHeadersConfig {
                data: vec![ExtraHeader {
                    name: "X-Custom-Header".to_string(),
                    kind: ExtraHeaderKind::Value("test".to_string()),
                }],
            }),
            inference_extra_headers: Default::default(),
        };
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_reasoning_effort(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.inference_params_v2.reasoning_effort = Some("low".to_string());
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_service_tier(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.inference_params_v2.service_tier =
            Some(crate::inference::types::chat_completion_inference_params::ServiceTier::Flex);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_thinking_budget_tokens(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.inference_params_v2.thinking_budget_tokens = Some(500);
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    #[gtest]
    fn test_cache_key_changes_with_verbosity(fixture: &CacheKeyFixture) {
        let mut req = fixture.request.clone();
        req.inference_params_v2.verbosity = Some("concise".to_string());
        expect_that!(
            cache_key_for(&req, "model", "provider"),
            not(eq(fixture.key))
        );
    }

    // --- Embedding cache key tests ---

    fn baseline_embedding_request() -> EmbeddingRequest {
        EmbeddingRequest {
            input: EmbeddingInput::Single("hello world".to_string()),
            dimensions: Some(256),
            encoding_format: EmbeddingEncodingFormat::Float,
        }
    }

    fn embedding_cache_key_for(
        request: &EmbeddingRequest,
        model_name: &str,
        provider_name: &str,
    ) -> CacheKey {
        let otlp_config = OtlpConfig::default();
        EmbeddingModelProviderRequest {
            request,
            model_name,
            provider_name,
            otlp_config: &otlp_config,
            model_inference_id: Uuid::now_v7(),
        }
        .get_cache_key()
        .expect("get_cache_key should not fail for a valid embedding request")
    }

    #[gtest]
    fn test_embedding_cache_key_all_fields_considered() {
        let request = baseline_embedding_request();

        // Exhaustive destructure: a compile error here means a new field was added
        // and must be classified as cache-relevant or explicitly excluded.
        let EmbeddingRequest {
            input: _,
            dimensions: _,
            encoding_format: _,
        } = &request;
    }

    #[gtest]
    fn test_embedding_cache_key_deterministic() {
        let req = baseline_embedding_request();
        let key_a = embedding_cache_key_for(&req, "model", "provider");
        let key_b = embedding_cache_key_for(&req, "model", "provider");
        expect_that!(key_a, eq(key_b));
    }

    #[gtest]
    fn test_embedding_cache_key_ignores_model_inference_id() {
        let req = baseline_embedding_request();
        // embedding_cache_key_for uses a fresh model_inference_id each time
        let key_a = embedding_cache_key_for(&req, "model", "provider");
        let key_b = embedding_cache_key_for(&req, "model", "provider");
        expect_that!(key_a, eq(key_b));
    }

    #[gtest]
    fn test_embedding_cache_key_changes_with_model_name() {
        let req = baseline_embedding_request();
        let baseline_key = embedding_cache_key_for(&req, "model", "provider");
        let key = embedding_cache_key_for(&req, "other_model", "provider");
        expect_that!(key, not(eq(baseline_key)));
    }

    #[gtest]
    fn test_embedding_cache_key_changes_with_provider_name() {
        let req = baseline_embedding_request();
        let baseline_key = embedding_cache_key_for(&req, "model", "provider");
        let key = embedding_cache_key_for(&req, "model", "other_provider");
        expect_that!(key, not(eq(baseline_key)));
    }

    #[gtest]
    fn test_embedding_cache_key_prefix_free_encoding() {
        let req = baseline_embedding_request();
        let baseline_key = embedding_cache_key_for(&req, "model", "provider");
        let key = embedding_cache_key_for(&req, "modelp", "rovider");
        expect_that!(key, not(eq(baseline_key)));
    }

    #[gtest]
    fn test_embedding_cache_key_changes_with_input() {
        let req = baseline_embedding_request();
        let baseline_key = embedding_cache_key_for(&req, "model", "provider");

        let different = EmbeddingRequest {
            input: EmbeddingInput::Single("goodbye world".to_string()),
            ..baseline_embedding_request()
        };
        let key = embedding_cache_key_for(&different, "model", "provider");
        expect_that!(key, not(eq(baseline_key)));
    }

    #[gtest]
    fn test_embedding_cache_key_changes_with_dimensions() {
        let req = baseline_embedding_request();
        let baseline_key = embedding_cache_key_for(&req, "model", "provider");

        let different = EmbeddingRequest {
            dimensions: Some(512),
            ..baseline_embedding_request()
        };
        let key = embedding_cache_key_for(&different, "model", "provider");
        expect_that!(key, not(eq(baseline_key)));
    }

    #[gtest]
    fn test_embedding_cache_key_changes_with_encoding_format() {
        let req = baseline_embedding_request();
        let baseline_key = embedding_cache_key_for(&req, "model", "provider");

        let different = EmbeddingRequest {
            encoding_format: EmbeddingEncodingFormat::Base64,
            ..baseline_embedding_request()
        };
        let key = embedding_cache_key_for(&different, "model", "provider");
        expect_that!(key, not(eq(baseline_key)));
    }
}
