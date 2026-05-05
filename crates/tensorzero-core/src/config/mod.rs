use crate::experimentation::{
    ExperimentationConfig, ExperimentationConfigWithNamespaces,
    UninitializedExperimentationConfigWithNamespaces,
};
use crate::http::TensorzeroHttpClient;
use crate::rate_limiting::{RateLimitingConfig, UninitializedRateLimitingConfig};
use crate::relay::TensorzeroRelay;
use crate::utils::deprecation_warning;
use chrono::Duration;
/// IMPORTANT: THIS MODULE IS NOT STABLE.
///            IT IS MEANT FOR INTERNAL USE ONLY.
///            EXPECT FREQUENT, UNANNOUNCED BREAKING CHANGES.
///            USE AT YOUR OWN RISK.
use futures::future::try_join_all;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use provider_types::ProviderTypesConfig;
#[cfg(feature = "pyo3")]
use pyo3::IntoPyObjectExt;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyKeyError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use snapshot::SnapshotHash;
use sqlx::PgPool;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_stored_config::{
    StoredFileRef, StoredMetricConfig, StoredMetricLevel, StoredMetricOptimize, StoredMetricType,
    StoredNonStreamingTimeouts, StoredStreamingTimeouts, StoredTimeoutsConfig, StoredToolConfig,
};
use tracing::Span;
use tracing::instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use unwritten::UnwrittenConfig;
use url::Url;
use uuid::Uuid;

use crate::config::gateway::{GatewayConfig, UninitializedGatewayConfig};
use crate::config::path::{ResolvedTomlPathData, ResolvedTomlPathDirectory};
use crate::config::snapshot::ConfigSnapshot;
use crate::config::span_map::SpanMap;
use crate::embeddings::{EmbeddingModelTable, UninitializedEmbeddingModelConfig};
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::evaluations::{
    EvaluationConfig, EvaluatorConfig, UninitializedEvaluationConfig, UninitializedEvaluatorConfig,
    get_function_evaluator_metric_name, get_function_llm_judge_function_name,
};
use crate::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson, get_function};
#[cfg(feature = "pyo3")]
use crate::function::{FunctionConfigChatPyClass, FunctionConfigJsonPyClass};
use crate::inference::types::Usage;
use crate::inference::types::storage::StorageKind;
use crate::jsonschema_util::{JSONSchema, SchemaWithMetadata};
use crate::minijinja_util::TemplateConfig;
use crate::model::{
    CredentialLocationWithFallback, ModelConfig, ModelTable, UninitializedModelConfig,
};
use crate::model_table::{CowNoClone, ProviderTypeDefaultCredentials, ShorthandModelConfig};
use crate::optimization::{
    OptimizerInfo, UninitializedOptimizerConfig, UninitializedOptimizerInfo,
};
use crate::tool::{StaticToolConfig, ToolChoice, create_json_mode_tool_call_config};
use crate::variant::best_of_n_sampling::UninitializedBestOfNSamplingConfig;
use crate::variant::chain_of_thought::UninitializedChainOfThoughtConfig;
use crate::variant::chat_completion::UninitializedChatCompletionConfig;
use crate::variant::dicl::UninitializedDiclConfig;
use crate::variant::mixture_of_n::UninitializedMixtureOfNConfig;
use crate::variant::{Variant, VariantConfig, VariantInfo};
use std::error::Error as StdError;

pub mod built_in;
pub mod editable;
pub mod gateway;
pub mod namespace;
pub mod path;
pub mod provider_types;
pub mod rate_limiting;
pub mod rehydrate;
pub mod snapshot;
mod span_map;
#[cfg(test)]
mod tests;
pub mod unwritten;

pub use namespace::Namespace;

pub use tensorzero_inference_types::credential_validation::{
    skip_credential_validation, with_skip_credential_validation,
};

/// Configuration for the autopilot system.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutopilotConfig {
    /// Tools that are automatically approved without manual intervention.
    /// If unset, defaults to all nondestructive tools.
    /// If set, replaces the default list entirely.
    pub tool_whitelist: Option<Vec<String>>,
}

impl From<tensorzero_stored_config::StoredAutopilotConfig> for AutopilotConfig {
    fn from(stored: tensorzero_stored_config::StoredAutopilotConfig) -> Self {
        AutopilotConfig {
            tool_whitelist: stored.tool_whitelist,
        }
    }
}

impl From<&AutopilotConfig> for tensorzero_stored_config::StoredAutopilotConfig {
    fn from(config: &AutopilotConfig) -> Self {
        tensorzero_stored_config::StoredAutopilotConfig {
            tool_whitelist: config.tool_whitelist.clone(),
        }
    }
}

/// A per-item error encountered during config loading from the database.
/// Items with loading errors are skipped from the live config; this struct
/// preserves enough context for the UI to display the error and for operators
/// to copy/paste the raw config fragment for debugging.
#[serde_with::skip_serializing_none]
#[derive(ts_rs::TS, Clone, Debug, Serialize)]
#[ts(export, optional_fields)]
pub struct ConfigLoadingError {
    /// The type of config item (e.g. `"function"`, `"variant"`, `"model"`).
    pub kind: &'static str,
    /// The name of the broken item.
    pub name: String,
    /// For variants: the parent function name.
    pub parent: Option<String>,
    /// Human-readable description of what went wrong.
    pub error: String,
    /// Best-effort TOML fragment of the raw stored config, for copy/paste debugging.
    pub raw_toml: Option<String>,
}

// Note - the `Default` impl only exists for convenience in tests
// It might produce a completely broken config - if a test fails,
// use one of the public `Config` constructors instead.
#[derive(Debug)]
#[cfg_attr(any(test, feature = "e2e_tests"), derive(Default))]
pub struct Config {
    pub gateway: GatewayConfig,
    pub clickhouse: ClickHouseConfig,
    pub models: Arc<ModelTable>, // model name => model config
    pub embedding_models: Arc<EmbeddingModelTable>, // embedding model name => embedding model config
    pub functions: HashMap<String, Arc<FunctionConfig>>, // function name => function config
    pub metrics: HashMap<String, MetricConfig>,     // metric name => metric config
    pub tools: HashMap<String, Arc<StaticToolConfig>>, // tool name => tool config
    pub evaluations: HashMap<String, Arc<EvaluationConfig>>, // evaluation name => evaluation config
    pub templates: Arc<TemplateConfig<'static>>,
    pub object_store_info: Option<ObjectStoreInfo>,
    pub provider_types: ProviderTypesConfig,
    pub optimizers: HashMap<String, OptimizerInfo>,
    pub postgres: PostgresConfig,
    pub rate_limiting: RateLimitingConfig,
    pub http_client: TensorzeroHttpClient,
    pub autopilot: AutopilotConfig,
    pub hash: SnapshotHash,
    /// Errors encountered while loading config items from the database.
    /// Non-empty only in config-in-database mode; always empty for file configs.
    pub loading_errors: Vec<ConfigLoadingError>,
}

#[serde_with::skip_serializing_none]
#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export, optional_fields)]
#[serde(deny_unknown_fields)]
pub struct NonStreamingTimeouts {
    /// The total time allowed for the non-streaming request to complete.
    pub total_ms: Option<u64>,
}

#[serde_with::skip_serializing_none]
#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export, optional_fields)]
#[serde(deny_unknown_fields)]
pub struct StreamingTimeouts {
    /// The time allowed for the first token to be produced.
    pub ttft_ms: Option<u64>,
    /// The total time allowed for the entire streaming request to complete.
    pub total_ms: Option<u64>,
}

/// Configures the timeouts for both streaming and non-streaming requests.
/// This can be attached to various other configs (e.g. variants, models, model providers)
#[serde_with::skip_serializing_none]
#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export, optional_fields)]
#[serde(deny_unknown_fields)]
pub struct TimeoutsConfig {
    pub non_streaming: Option<NonStreamingTimeouts>,
    pub streaming: Option<StreamingTimeouts>,
}

impl From<&TimeoutsConfig> for StoredTimeoutsConfig {
    fn from(config: &TimeoutsConfig) -> Self {
        StoredTimeoutsConfig {
            non_streaming: config
                .non_streaming
                .as_ref()
                .map(|ns| StoredNonStreamingTimeouts {
                    total_ms: ns.total_ms,
                }),
            streaming: config.streaming.as_ref().map(|s| StoredStreamingTimeouts {
                ttft_ms: s.ttft_ms,
                total_ms: s.total_ms,
            }),
        }
    }
}

impl TimeoutsConfig {
    pub fn validate(&self, global_outbound_http_timeout: &Duration) -> Result<(), Error> {
        let total_ms = self.non_streaming.as_ref().and_then(|ns| ns.total_ms);
        let ttft_ms = self.streaming.as_ref().and_then(|s| s.ttft_ms);
        let streaming_total_ms = self.streaming.as_ref().and_then(|s| s.total_ms);

        let global_ms = global_outbound_http_timeout.num_milliseconds();

        if let Some(total_ms) = total_ms
            && Duration::milliseconds(total_ms as i64) > *global_outbound_http_timeout
        {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "The `timeouts.non_streaming.total_ms` value `{total_ms}` is greater than `gateway.global_outbound_http_timeout_ms`: `{global_ms}`"
                ),
            }));
        }
        if let Some(ttft_ms) = ttft_ms
            && Duration::milliseconds(ttft_ms as i64) > *global_outbound_http_timeout
        {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "The `timeouts.streaming.ttft_ms` value `{ttft_ms}` is greater than `gateway.global_outbound_http_timeout_ms`: `{global_ms}`"
                ),
            }));
        }
        if let Some(streaming_total_ms) = streaming_total_ms
            && Duration::milliseconds(streaming_total_ms as i64) > *global_outbound_http_timeout
        {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "The `timeouts.streaming.total_ms` value `{streaming_total_ms}` is greater than `gateway.global_outbound_http_timeout_ms`: `{global_ms}`"
                ),
            }));
        }
        if let Some(ttft_ms) = ttft_ms
            && let Some(streaming_total_ms) = streaming_total_ms
            && streaming_total_ms < ttft_ms
        {
            tracing::warn!(
                "The `timeouts.streaming.total_ms` value `{streaming_total_ms}` is less than `timeouts.streaming.ttft_ms` value `{ttft_ms}`. The `total_ms` timeout may fire before the `ttft_ms` timeout."
            );
        }

        Ok(())
    }
}

impl From<StoredTimeoutsConfig> for TimeoutsConfig {
    fn from(stored: StoredTimeoutsConfig) -> Self {
        TimeoutsConfig {
            non_streaming: stored.non_streaming.map(|ns| NonStreamingTimeouts {
                total_ms: ns.total_ms,
            }),
            streaming: stored.streaming.map(|s| StreamingTimeouts {
                ttft_ms: s.ttft_ms,
                total_ms: s.total_ms,
            }),
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TemplateFilesystemAccess {
    /// If `true`, allow minijinja to read from the filesystem (within the tree of the config file) for `{% include %}`
    /// Defaults to `false`
    enabled: Option<bool>,
    base_path: Option<ResolvedTomlPathDirectory>,
}

impl TemplateFilesystemAccess {
    /// Returns `true` if filesystem access is explicitly enabled or a base path is configured.
    pub fn is_active(&self) -> bool {
        self.enabled == Some(true) || self.base_path.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct ObjectStoreInfo {
    // This will be `None` if we have `StorageKind::Disabled`
    pub object_store: Option<Arc<dyn ObjectStore>>,
    pub kind: StorageKind,
}

impl ObjectStoreInfo {
    pub fn new(config: Option<StorageKind>) -> Result<Option<Self>, Error> {
        let Some(mut config) = config else {
            return Ok(None);
        };

        if let StorageKind::S3Compatible { endpoint, .. } = &mut config
            && let Some(endpoint_value) = endpoint.take()
        {
            *endpoint = Some(resolve_object_storage_endpoint(&endpoint_value)?);
        }

        let object_store: Option<Arc<dyn ObjectStore>> = match &config {
            StorageKind::Filesystem { path } => {
                Some(Arc::new(match LocalFileSystem::new_with_prefix(path) {
                    Ok(object_store) => object_store,
                    Err(e) =>
                    {
                        #[expect(clippy::if_not_else)]
                        if !std::fs::exists(path).unwrap_or(false) {
                            if skip_credential_validation() {
                                tracing::warn!(
                                    "Filesystem object store path does not exist: {path}. Treating object store as unconfigured"
                                );
                                return Ok(None);
                            }
                            return Err(Error::new(ErrorDetails::Config {
                                message: format!(
                                    "Failed to create filesystem object store: path does not exist: {path}"
                                ),
                            }));
                        } else {
                            return Err(Error::new(ErrorDetails::Config {
                                message: format!(
                                    "Failed to create filesystem object store for path: {path}: {e}"
                                ),
                            }));
                        }
                    }
                }))
            }
            StorageKind::S3Compatible {
                bucket_name,
                region,
                endpoint,
                allow_http,
                #[cfg(feature = "e2e_tests")]
                    prefix: _,
            } => {
                let mut builder = AmazonS3Builder::from_env()
                    // Uses the S3 'If-Match' and 'If-None-Match' headers to implement condition put
                    .with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch);

                // These env vars have the highest priority, overriding whatever was set from 'AmazonS3Builder::from_env()'
                if let Ok(s3_access_key) = std::env::var("S3_ACCESS_KEY_ID") {
                    let s3_secret_key = std::env::var("S3_SECRET_ACCESS_KEY").ok().ok_or_else(|| Error::new(ErrorDetails::Config {
                        message: "S3_ACCESS_KEY_ID is set but S3_SECRET_ACCESS_KEY is not. Please set neither or both.".to_string()
                    }))?;
                    builder = builder
                        .with_access_key_id(s3_access_key)
                        .with_secret_access_key(s3_secret_key);
                }

                if let Some(bucket_name) = bucket_name {
                    builder = builder.with_bucket_name(bucket_name);
                }
                if let Some(region) = region {
                    builder = builder.with_region(region);
                }
                if let Some(endpoint) = endpoint {
                    builder = builder.with_endpoint(endpoint);
                }
                if std::env::var("AWS_ALLOW_HTTP").as_deref() == Ok("true") {
                    tracing::warn!(
                        "`AWS_ALLOW_HTTP` is set to `true` - this is insecure, and should only be used when running a local S3-compatible object store"
                    );
                    if allow_http.is_some() {
                        tracing::info!(
                            "Config has `[object_storage.allow_http]` present - this takes precedence over `AWS_ALLOW_HTTP`"
                        );
                    }
                }
                if let Some(allow_http) = *allow_http {
                    if allow_http {
                        tracing::warn!(
                            "`[object_storage.allow_http]` is set to `true` - this is insecure, and should only be used when running a local S3-compatible object store"
                        );
                    }
                    builder = builder.with_allow_http(allow_http);
                }

                if let (Some(bucket_name), Some(endpoint)) = (bucket_name, endpoint)
                    && endpoint.ends_with(bucket_name)
                {
                    tracing::warn!(
                        "S3-compatible object endpoint `{endpoint}` ends with configured bucket_name `{bucket_name}`. This may be incorrect - if the gateway fails to start, consider setting `bucket_name = null`"
                    );
                }

                // This is used to speed up our unit tests - in the future,
                // we might want to expose more flexible options through the config
                #[cfg(test)]
                if std::env::var("TENSORZERO_E2E_DISABLE_S3_RETRY").is_ok() {
                    builder = builder.with_retry(object_store::RetryConfig {
                        max_retries: 0,
                        ..Default::default()
                    });
                }

                Some(Arc::new(builder.build()
                    .map_err(|e| Error::new(ErrorDetails::Config {
                        message: format!("Failed to create S3-compatible object store with config `{config:?}`: {e}"),
                    })
                )?),
            )
            }
            StorageKind::Disabled => None,
        };

        Ok(Some(Self {
            object_store,
            kind: config,
        }))
    }

    /// Verifies that the object store is configured correctly by writing an empty file to it.
    pub async fn verify(&self) -> Result<(), Error> {
        if let Some(store) = &self.object_store {
            tracing::info!(
                "Verifying that [object_storage] is configured correctly (writing .tensorzero-validate)"
            );
            store.put(&object_store::path::Path::from(".tensorzero-validate"), PutPayload::new())
                .await
                .map_err(|e| {
                    if contains_bad_scheme_err(&e) {
                        tracing::warn!("Consider setting `[object_storage.allow_http]` to `true` if you are using a non-HTTPs endpoint");
                    }
                    Error::new(ErrorDetails::Config {
                    message: format!("Failed to write `.tensorzero-validate` to object store. Check that your credentials are configured correctly: {e:?}"),
                })
            })?;
            tracing::info!("Successfully wrote .tensorzero-validate to object store");
        }
        Ok(())
    }
}

// Best-effort attempt to find a 'BadScheme' error by walking up
// the error 'source' chain. This should only be used for printing
// improved warning messages.
// We are attempting to find this error: `https://github.com/seanmonstar/reqwest/blob/c4a9fb060fb518f0053b98f78c7583071a760cf4/src/error.rs#L340`
fn contains_bad_scheme_err(e: &impl StdError) -> bool {
    format!("{e:?}").contains("BadScheme")
}

fn resolve_object_storage_endpoint(endpoint: &str) -> Result<String, Error> {
    if let Some(env_var) = endpoint.strip_prefix("env::") {
        return std::env::var(env_var).map_err(|_| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Environment variable `{env_var}` not found. Your configuration for `[object_storage]` requires this variable for `endpoint`."
                ),
            })
        });
    }

    if endpoint.starts_with("dynamic::")
        || endpoint.starts_with("path::")
        || endpoint.starts_with("path_from_env::")
        || matches!(endpoint, "sdk" | "none")
    {
        return Err(Error::new(ErrorDetails::Config {
            message: format!(
                "Invalid `[object_storage].endpoint`: `{endpoint}`. Use `env::ENVIRONMENT_VARIABLE` or a literal endpoint value."
            ),
        }));
    }

    Ok(endpoint.to_string())
}

/// Selects the primary datastore used for observability writes (inferences, feedback).
///
/// - `Auto` (default): prefers ClickHouse if available, falls back to Postgres.
/// - `ClickHouse`: explicitly use ClickHouse.
/// - `Postgres`: explicitly use Postgres.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ObservabilityBackend {
    Auto,
    ClickHouse,
    Postgres,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityConfig {
    /// Controls whether the gateway writes observability data (inferences, feedback).
    ///
    /// - `Some(true)`: observability is required — startup fails if the backend is unavailable.
    /// - `None` (default): observability is opportunistic — enabled if a backend is available, otherwise warns and continues.
    /// - `Some(false)`: observability is disabled — no data is written regardless of backend availability.
    pub enabled: Option<bool>,
    /// Selects the observability backend (and primary datastore).
    pub backend: Option<ObservabilityBackend>,
    pub async_writes: Option<bool>,
    pub batch_writes: Option<BatchWritesConfig>,
    #[deprecated(
        since = "2026.2.1",
        note = "Use `clickhouse.disable_automatic_migrations` instead"
    )]
    pub disable_automatic_migrations: Option<bool>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClickHouseConfig {
    /// When `true`, the gateway will not run ClickHouse schema migrations on startup.
    pub disable_automatic_migrations: Option<bool>,
}

impl From<tensorzero_stored_config::StoredClickHouseConfig> for ClickHouseConfig {
    fn from(stored: tensorzero_stored_config::StoredClickHouseConfig) -> Self {
        ClickHouseConfig {
            disable_automatic_migrations: stored.disable_automatic_migrations,
        }
    }
}

impl From<&ClickHouseConfig> for tensorzero_stored_config::StoredClickHouseConfig {
    fn from(config: &ClickHouseConfig) -> Self {
        tensorzero_stored_config::StoredClickHouseConfig {
            disable_automatic_migrations: config.disable_automatic_migrations,
        }
    }
}

impl ObservabilityConfig {
    /// Returns true when observability writes (inferences, feedback) should be persisted.
    /// Defaults to true when `enabled` is not explicitly set.
    pub fn writes_enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }

    /// Returns whether async writes are enabled.
    /// Defaults to `true` in production, `false` in e2e tests (where tests
    /// write data and immediately query for it).
    pub fn async_writes(&self) -> bool {
        self.async_writes.unwrap_or(!cfg!(feature = "e2e_tests"))
    }
}

pub fn default_flush_interval_ms() -> u64 {
    100
}

pub fn default_max_rows() -> usize {
    1000
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchWritesConfig {
    pub enabled: bool,
    // An internal flag to allow us to test batch writes in embedded gateway mode.
    // This can currently cause deadlocks, so we don't want normal embedded clients to use it.
    pub __force_allow_embedded_batch_writes: Option<bool>,
    pub flush_interval_ms: Option<u64>,
    pub max_rows: Option<usize>,
    /// Optional override for Postgres batch size. Defaults to `max_rows` when unset.
    pub max_rows_postgres: Option<usize>,
    /// Optional capacity for bounded batch writer channels per table type.
    /// When set, channels are bounded: if full, new rows are dropped and logged
    /// to protect against out-of-memory crashes.
    /// When unset (`None`), channels are unbounded (legacy behavior).
    pub write_queue_capacity: Option<usize>,
}

impl Default for BatchWritesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            __force_allow_embedded_batch_writes: Some(false),
            flush_interval_ms: Some(default_flush_interval_ms()),
            max_rows: Some(default_max_rows()),
            max_rows_postgres: None,
            write_queue_capacity: None,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExportConfig {
    pub otlp: Option<OtlpConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OtlpConfig {
    pub traces: Option<OtlpTracesConfig>,
}

impl OtlpConfig {
    /// Attaches usage inference to the model provider span (if traces are enabled).
    /// This is used for both streaming and non-streaming requests.
    pub fn apply_usage_to_model_provider_span(&self, span: &Span, usage: &Usage) {
        let traces = match &self.traces {
            Some(t) => t,
            None => return,
        };
        if traces.enabled.unwrap_or(false) {
            match traces.format {
                Some(OtlpTracesFormat::OpenInference) => {
                    if let Some(input_tokens) = usage.input_tokens {
                        span.set_attribute("llm.token_count.prompt", input_tokens as i64);
                    }
                    if let Some(output_tokens) = usage.output_tokens {
                        span.set_attribute("llm.token_count.completion", output_tokens as i64);
                    }
                    if let Some(total_tokens) = usage.total_tokens() {
                        span.set_attribute("llm.token_count.total", total_tokens as i64);
                    }
                }
                None | Some(OtlpTracesFormat::OpenTelemetry) => {
                    if let Some(input_tokens) = usage.input_tokens {
                        span.set_attribute("gen_ai.usage.input_tokens", input_tokens as i64);
                    }
                    if let Some(output_tokens) = usage.output_tokens {
                        span.set_attribute("gen_ai.usage.output_tokens", output_tokens as i64);
                    }
                    if let Some(total_tokens) = usage.total_tokens() {
                        span.set_attribute("gen_ai.usage.total_tokens", total_tokens as i64);
                    }
                }
            }
            // Emit computed cost only when the provider has a `cost` rate configured.
            if let Some(cost) = usage.cost {
                span.set_attribute("tensorzero.cost_usd", cost.to_string());
            }
        }
    }

    /// Whether GenAI content-capture attributes (`gen_ai.input.messages`,
    /// `gen_ai.output.messages`, `gen_ai.system_instructions`,
    /// `gen_ai.tool.definitions`) should be emitted: OTLP traces enabled,
    /// `include_content = true`, and `format = OpenTelemetry` (the default).
    pub fn genai_content_capture_enabled(&self) -> bool {
        let Some(traces) = &self.traces else {
            return false;
        };
        if !traces.enabled.unwrap_or(false) {
            return false;
        }
        if !traces.include_content.unwrap_or(false) {
            return false;
        }
        matches!(traces.format, None | Some(OtlpTracesFormat::OpenTelemetry))
    }

    /// Marks a span as being an OpenInference 'CHAIN' span.
    /// We use this for function/variant/model spans (but not model provider spans).
    /// At the moment, there doesn't seem to be a similar concept in the OpenTelemetry GenAI semantic conventions.
    pub fn mark_openinference_chain_span(&self, span: &Span) {
        let traces = match &self.traces {
            Some(t) => t,
            None => return,
        };
        if traces.enabled.unwrap_or(false) {
            match traces.format {
                Some(OtlpTracesFormat::OpenInference) => {
                    span.set_attribute("openinference.span.kind", "CHAIN");
                }
                None | Some(OtlpTracesFormat::OpenTelemetry) => {}
            }
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OtlpTracesConfig {
    /// Enable OpenTelemetry traces export to the configured OTLP endpoint (configured via OTLP environment variables)
    pub enabled: Option<bool>,
    pub format: Option<OtlpTracesFormat>,
    /// Extra headers to include in OTLP export requests (can be overridden by dynamic headers at request time)
    pub extra_headers: Option<HashMap<String, String>>,
    /// When `format = "opentelemetry"`, emit the content-carrying GenAI semantic-convention attributes
    /// (`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`,
    /// `gen_ai.tool.definitions`) on the `model_provider_inference` span. These contain full
    /// prompt/response content and are opt-in even when traces are enabled. Defaults to `false`.
    pub include_content: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "lowercase")]
pub enum OtlpTracesFormat {
    /// Sets 'gen_ai' attributes based on the OpenTelemetry GenAI semantic conventions:
    /// https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai
    OpenTelemetry,
    // Sets attributes based on the OpenInference semantic conventions:
    // https://github.com/Arize-ai/openinference/blob/main/spec/llm_spans.md
    OpenInference,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct MetricConfig {
    pub r#type: MetricConfigType,
    pub optimize: MetricConfigOptimize,
    pub level: MetricConfigLevel,
    pub description: Option<String>,
}

#[derive(Copy, Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum MetricConfigType {
    Boolean,
    Float,
}

impl From<MetricConfigType> for StoredMetricType {
    fn from(value: MetricConfigType) -> Self {
        match value {
            MetricConfigType::Boolean => Self::Boolean,
            MetricConfigType::Float => Self::Float,
        }
    }
}

impl MetricConfigType {
    pub fn to_clickhouse_table_name(&self) -> &'static str {
        match self {
            MetricConfigType::Boolean => "BooleanMetricFeedback",
            MetricConfigType::Float => "FloatMetricFeedback",
        }
    }

    /// Returns the Postgres table name for the given metric type.
    pub fn postgres_table_name(&self) -> &'static str {
        match self {
            MetricConfigType::Boolean => "tensorzero.boolean_metric_feedback",
            MetricConfigType::Float => "tensorzero.float_metric_feedback",
        }
    }
}

#[derive(ts_rs::TS, Copy, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[ts(export)]
pub enum MetricConfigOptimize {
    Min,
    Max,
}

impl From<MetricConfigOptimize> for StoredMetricOptimize {
    fn from(value: MetricConfigOptimize) -> Self {
        match value {
            MetricConfigOptimize::Min => Self::Min,
            MetricConfigOptimize::Max => Self::Max,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum MetricConfigLevel {
    Inference,
    Episode,
}

impl std::fmt::Display for MetricConfigLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let serialized = serde_json::to_string(self).map_err(|_| std::fmt::Error)?;
        // Remove the quotes around the string
        write!(f, "{}", serialized.trim_matches('"'))
    }
}

impl From<&MetricConfigLevel> for StoredMetricLevel {
    fn from(value: &MetricConfigLevel) -> Self {
        match value {
            MetricConfigLevel::Inference => Self::Inference,
            MetricConfigLevel::Episode => Self::Episode,
        }
    }
}

impl From<&MetricConfig> for StoredMetricConfig {
    fn from(config: &MetricConfig) -> Self {
        StoredMetricConfig {
            r#type: config.r#type.into(),
            optimize: config.optimize.into(),
            level: StoredMetricLevel::from(&config.level),
            description: config.description.clone(),
        }
    }
}

impl MetricConfigLevel {
    pub fn inference_column_name(&self) -> &'static str {
        match self {
            MetricConfigLevel::Inference => "id",
            MetricConfigLevel::Episode => "episode_id",
        }
    }
}

// ─── Stored → Uninitialized conversions for metric config ────────────────────

impl From<tensorzero_stored_config::StoredMetricType> for MetricConfigType {
    fn from(stored: tensorzero_stored_config::StoredMetricType) -> Self {
        match stored {
            tensorzero_stored_config::StoredMetricType::Boolean => MetricConfigType::Boolean,
            tensorzero_stored_config::StoredMetricType::Float => MetricConfigType::Float,
        }
    }
}

impl From<tensorzero_stored_config::StoredMetricOptimize> for MetricConfigOptimize {
    fn from(stored: tensorzero_stored_config::StoredMetricOptimize) -> Self {
        match stored {
            tensorzero_stored_config::StoredMetricOptimize::Min => MetricConfigOptimize::Min,
            tensorzero_stored_config::StoredMetricOptimize::Max => MetricConfigOptimize::Max,
        }
    }
}

impl From<tensorzero_stored_config::StoredMetricLevel> for MetricConfigLevel {
    fn from(stored: tensorzero_stored_config::StoredMetricLevel) -> Self {
        match stored {
            tensorzero_stored_config::StoredMetricLevel::Inference => MetricConfigLevel::Inference,
            tensorzero_stored_config::StoredMetricLevel::Episode => MetricConfigLevel::Episode,
        }
    }
}

impl From<tensorzero_stored_config::StoredMetricConfig> for MetricConfig {
    fn from(stored: tensorzero_stored_config::StoredMetricConfig) -> Self {
        MetricConfig {
            r#type: stored.r#type.into(),
            optimize: stored.optimize.into(),
            level: stored.level.into(),
            description: stored.description,
        }
    }
}

// Config file globbing:
// TensorZero supports loading from multiple config files (matching a unix-style glob pattern)
// This poses a number of challenges, mainly related to resolving paths specified inside config files
// (e.g. `user_template`) relative to the containing config file.
// The overall loading process is:
// 1. We call `ConfigFileGlob::new_from_path` to resolve the glob pattern into a list of paths,
//    This validates that we have at least one config file to load.
// 2. We build of a `SpanMap` from the glob using `SpanMap::from_glob`. This is responsible for
//    actually parsing each individual config file, and then merging the resulting `DeTable`s
//    into a single `DeTable`. Unfortunately, we cannot simple concatenate the files as parse
//    them as a single larger file, due to the possibility of files like this:
//    ```toml
//    [my_section]
//    my_section_key = "my_section_value"
//    ```
//
//    ```toml
//    top.level.path = "top_level_value"
//    ```
//
//    If we concatenated and parsed these files, we would incorrectly include `top.level.path`
//    underneath the `[my_section]`. Toml doesn't have a way of returning to the 'root' context,
//    so we're forced to manually parse each file, and then merge them ourselves in memory.
//
//    Each individually parsed `DeTable` tracks byte ranges (via `Spanned`) into the original
//    source string. To allow us to identify the original file by looking at `Spanned`
//    instances in the merge table, we insert extra whitespace into each file before individually
//    parsing that file. This ensures that all of our `Spanned` ranges are disjoint, allowing us
//    to map a `Spanned` back to the original TOML file.
//
// 3. We 'remap' our `DeTable using `resolve_toml_relative_paths`.
//    This allows us to deserialize types like `UninitializedChatCompletionConfig`
//    which store paths (e.g. `user_template`) relative to the containing config file.
//
//    Unfortunately, we cannot directly write `Spanned` types inside of these nested
//    structs. Due to a serde issue, using attributes like `#[serde(transparent)]`
//    or `#[serde(flatten)]` will cause Serde to deserialize into its own internal type,
//    and then deserialize the inner type with a different deserializer (not the original one).
//    The same problem shows up when constructing error messages - while our custom
//    `TensorZeroDeserialize` attempts to improve the situation, it's best-effort, and
//    we don't rely on it for correctness.
//
//    Instead, we 'remap' paths by walking the merged `DeTable`, and replacing entries
//    at known paths with their fully qualified paths (using the `SpanMap` to obtain
//    the base path for each entry). Each value is changed to a nested map that can be
//    deserialized into a `ResolvedTomlPath`. If we forget to remap any parts of the config
//    in `resolve_toml_relative_paths`, then deserializing the `ResolvedTomlPath` will fail
//    (rather than succeed with an incorrect path).
//
// At this point, we have a `DeTable` that can be successfully deserialized into an `UninitializedConfig`.
// From this point onward, none of the config-handling code needs to interact with globs, remapped config files,
// or even be aware of whether or not we have multiple config files. All path access goes through `ResolvedTomlPath`,
// which is self-contained (it stores the absolute path that we resolved earlier).

/// A glob pattern together with the resolved config file paths.
/// We eagerly resolve the glob pattern so that we can include all of the matched
/// config file paths in error messages.
#[derive(Debug)]
#[expect(clippy::manual_non_exhaustive)]
pub struct ConfigFileGlob {
    pub glob: String,
    pub paths: Vec<PathBuf>,
    _private: (),
}

impl ConfigFileGlob {
    /// Interprets a path as a glob pattern
    pub fn new_from_path(path: &Path) -> Result<Self, Error> {
        Self::new(path.display().to_string())
    }

    pub fn new_empty() -> Self {
        Self {
            glob: String::new(),
            paths: vec![],
            _private: (),
        }
    }

    pub fn new(glob: String) -> Result<Self, Error> {
        // Build a matcher from the glob pattern.
        // We enable `literal_separator` so that `*` and `?` do not match `/`.
        // Without `literal_separator`, a glob like `/app/config/*.toml` would
        // match both `/app/config/foo.toml` and `/app/config/somedir/foo.toml`.

        // This warning can be removed in 2026.5+.
        if glob.contains('*') && !glob.contains("**") {
            tracing::warn!(
                "Important: `--config-file {glob}` contains `*`. `*` no longer matches directory separators (e.g. `*.toml` will not match `subdir/foo.toml`). Use `**` for recursive matching (e.g. `**/*.toml` will match `subdir/foo.toml`)."
            );
        }

        let matcher = globset::GlobBuilder::new(&glob)
            .literal_separator(true)
            .build()
            .map_err(|e| {
                Error::new(ErrorDetails::Glob {
                    glob: glob.to_string(),
                    message: e.to_string(),
                })
            })?
            .compile_matcher();

        // Extract the base path to start walking from
        let base_path = extract_base_path_from_glob(&glob);

        // Find all files matching the glob pattern
        let mut glob_paths = find_matching_files(&base_path, &matcher);

        if glob_paths.is_empty() {
            return Err(Error::new(ErrorDetails::Glob {
                glob: glob.to_string(),
                message: "No files matched the glob pattern. Ensure that the path exists, and contains at least one file.".to_string(),
            }));
        }

        // Sort the paths to avoid depending on the filesystem iteration order
        // when we merge configs. This should only affect the precise error message we display,
        // not whether or not the config parses successfully (or the final `Config`
        // that we resolve)
        glob_paths.sort_by_key(|path| path.display().to_string());
        Ok(Self {
            glob,
            paths: glob_paths,
            _private: (),
        })
    }

    /// Returns the base path extracted from the glob pattern.
    /// This is the longest literal path prefix before any glob metacharacters.
    pub fn base_path(&self) -> PathBuf {
        extract_base_path_from_glob(&self.glob)
    }
}

/// Extract the base path from a glob pattern.
///
/// This finds the longest literal path prefix before any glob metacharacters,
/// following the same approach as the glob crate. It works at the path component
/// level (not character level) for better handling of path separators.
///
/// # Examples
/// - `/tmp/config/tensorzero.toml` → `/tmp/config/tensorzero.toml`
/// - `/tmp/config/**/*.toml` → `/tmp/config`
/// - `config/**/*.toml` → `config`
/// - `*.toml` → `.`
fn extract_base_path_from_glob(glob: &str) -> PathBuf {
    let path = Path::new(glob);
    let mut base_components = Vec::new();

    for component in path.components() {
        let component_str = component.as_os_str().to_string_lossy();
        // Stop at the first component containing glob metacharacters
        if component_str.contains(['*', '?', '[', ']', '{', '}']) {
            break;
        }
        base_components.push(component);
    }

    if base_components.is_empty() {
        PathBuf::from(".")
    } else {
        base_components.iter().collect()
    }
}
/// Check which files match the glob pattern.
/// If the base path is a file, check it directly against the matcher.
/// If the base path is a directory, walk it.
fn find_matching_files(base_path: &Path, matcher: &globset::GlobMatcher) -> Vec<PathBuf> {
    let mut matched_files = Vec::new();

    // If base_path is a file, check it directly against the matcher
    if base_path.is_file() {
        if matcher.is_match(base_path) {
            matched_files.push(base_path.to_path_buf());
        }
        return matched_files;
    }

    // If base_path is a directory, walk it
    for entry in walkdir::WalkDir::new(base_path).follow_links(false) {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                if path.is_file() && matcher.is_match(path) {
                    matched_files.push(path.to_path_buf());
                }
            }
            Err(e) => {
                let error_path = e
                    .path()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| base_path.to_string_lossy().into_owned());
                tracing::warn!(
                    "Skipping `{}` while scanning for configuration files: {e}",
                    error_path
                );
            }
        }
    }

    matched_files
}

/// Runtime configuration values to overlay onto snapshot config.
/// These reflect the current runtime environment rather than historical snapshot values.
///
/// When loading a config from a historical snapshot, infrastructure settings should
/// come from the current runtime environment, not the snapshot. This struct captures
/// those runtime values that need to be overlaid.
#[derive(Clone, Debug, Default)]
pub struct RuntimeOverlay {
    pub gateway: Option<UninitializedGatewayConfig>,
    pub clickhouse: Option<ClickHouseConfig>,
    pub postgres: Option<PostgresConfig>,
    pub rate_limiting: Option<UninitializedRateLimitingConfig>,
    pub object_store_info: Option<ObjectStoreInfo>,
}

impl RuntimeOverlay {
    /// Create a RuntimeOverlay by extracting runtime fields from an `UninitializedConfig`.
    ///
    /// This preserves `Option<T>` values exactly as parsed from the original config
    /// (None for omitted fields, Some for explicit values), avoiding the lossy
    /// `Config → UninitializedConfig` round-trip that would turn default values
    /// into explicit `Some(default)` and break snapshot hash stability.
    pub fn from_uninitialized_config(
        config: &UninitializedConfig,
        object_store_info: Option<ObjectStoreInfo>,
    ) -> Self {
        Self {
            gateway: config.gateway.clone(),
            clickhouse: config.clickhouse.clone(),
            postgres: config.postgres.clone(),
            rate_limiting: config.rate_limiting.clone(),
            object_store_info,
        }
    }
}

/// Result of processing the initial config input (Fresh table or Snapshot).
/// Contains the fields needed from UninitializedConfig after the branch-specific
/// processing (functions, gateway, object_storage) has been done.
struct ProcessedConfigInput {
    // Remaining UninitializedConfig fields (not consumed in branching)
    tools: HashMap<String, UninitializedToolConfig>,
    models: HashMap<Arc<str>, UninitializedModelConfig>,
    embedding_models: HashMap<Arc<str>, UninitializedEmbeddingModelConfig>,
    metrics: HashMap<String, MetricConfig>,
    evaluations: HashMap<String, UninitializedEvaluationConfig>,
    provider_types: ProviderTypesConfig,
    optimizers: HashMap<String, UninitializedOptimizerInfo>,
    clickhouse: ClickHouseConfig,
    postgres: PostgresConfig,
    rate_limiting: UninitializedRateLimitingConfig,
    autopilot: AutopilotConfig,
    snapshot: ConfigSnapshot,

    /// All functions (user-defined + built-in), loaded but with evaluator artifacts not yet extracted
    loaded_functions: HashMap<String, LoadedFunctionConfig>,
    gateway_config: GatewayConfig,
    object_store_info: Option<ObjectStoreInfo>,

    /// The original UninitializedConfig (with built-in functions injected), preserved for
    /// validation during config-in-db writes.
    uninitialized_config: UninitializedConfig,

    /// Runtime overlay captured from the UninitializedConfig before defaults are resolved.
    runtime_overlay: RuntimeOverlay,

    /// Errors collected during DB loading, to be stored on the final `Config`.
    /// Always empty for fresh TOML and snapshot inputs.
    loading_errors: Vec<ConfigLoadingError>,
}

pub(crate) fn validate_user_config_names(config: &UninitializedConfig) -> Result<(), Error> {
    for name in config.functions.as_ref().into_iter().flat_map(|m| m.keys()) {
        if name.starts_with("tensorzero::") {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "User-defined function name cannot start with `tensorzero::`: {name}"
                ),
            }));
        }
    }

    Ok(())
}

/// Processes the config input (fresh TOML or snapshot) and returns all the fields
/// needed by load_unwritten_config, avoiding partial moves of UninitializedConfig.
async fn process_config_input(
    input: ConfigInput,
    templates: &mut TemplateConfig<'_>,
) -> Result<ProcessedConfigInput, Error> {
    match input {
        ConfigInput::Fresh(table) => {
            if table.is_empty() {
                tracing::info!(
                    "Config file is empty, so only default functions will be available."
                );
            }

            process_uninitialized_config(UninitializedConfig::try_from(table)?, templates).await
        }
        ConfigInput::Snapshot {
            snapshot,
            runtime_overlay,
        } => {
            let original_snapshot = *snapshot;
            let captured_runtime_overlay = (*runtime_overlay).clone();

            // Destructure overlay to ensure all fields are handled (compile error if field added/removed)
            let RuntimeOverlay {
                gateway: overlay_gateway,
                clickhouse: overlay_clickhouse,
                postgres: overlay_postgres,
                rate_limiting: overlay_rate_limiting,
                object_store_info: overlay_object_store_info,
            } = *runtime_overlay;

            // Destructure uninit_config to overlay specific fields
            let UninitializedConfig {
                gateway: _,        // replaced by overlay
                clickhouse: _,     // replaced by overlay
                postgres: _,       // replaced by overlay
                rate_limiting: _,  // replaced by overlay
                object_storage: _, // replaced by overlay (via object_store_info)
                models,
                embedding_models,
                functions,
                metrics,
                tools,
                evaluations,
                provider_types,
                optimizers,
                autopilot,
            } = original_snapshot
                .config
                .clone()
                .try_into()
                .map_err(|e: &'static str| {
                    Error::new(ErrorDetails::Config {
                        message: e.to_string(),
                    })
                })?;

            // Resolve Options with defaults
            let models = models.unwrap_or_default();
            let embedding_models = embedding_models.unwrap_or_default();
            let functions = functions.unwrap_or_default();
            let metrics = metrics.unwrap_or_default();
            let tools = tools.unwrap_or_default();
            let evaluations = evaluations.unwrap_or_default();
            let provider_types = provider_types.unwrap_or_default();
            let optimizers = optimizers.unwrap_or_default();
            let autopilot = autopilot.unwrap_or_default();

            // Reconstruct with overlaid values for snapshot hash computation
            let overlaid_config = UninitializedConfig {
                gateway: overlay_gateway.clone(),
                clickhouse: overlay_clickhouse.clone(),
                postgres: overlay_postgres.clone(),
                rate_limiting: overlay_rate_limiting.clone(),
                object_storage: overlay_object_store_info
                    .as_ref()
                    .map(|info| info.kind.clone()),
                models: Some(models.clone()),
                embedding_models: Some(embedding_models.clone()),
                functions: Some(functions.clone()),
                metrics: Some(metrics.clone()),
                tools: Some(tools.clone()),
                evaluations: Some(evaluations.clone()),
                provider_types: Some(provider_types.clone()),
                optimizers: Some(optimizers.clone()),
                autopilot: Some(autopilot.clone()),
            };

            let extra_templates = original_snapshot.extra_templates.clone();
            let uninitialized_config = overlaid_config.clone();
            let snapshot = ConfigSnapshot::new(overlaid_config, extra_templates.clone())?;

            // Load all functions from the snapshot (built-in functions are already included)
            let mut loaded_functions = HashMap::new();
            for (name, func_config) in functions {
                let loaded = func_config.load(&name, &metrics)?;
                loaded_functions.insert(name, loaded);
            }

            // Use the overlay object store info directly instead of creating a new one
            let gateway_config = overlay_gateway
                .clone()
                .unwrap_or_default()
                .load(overlay_object_store_info.as_ref())?;
            // Initialize templates from ALL functions (including built-in)
            let all_template_paths = Config::get_templates_from_loaded(&loaded_functions)?;
            // We don't use these since the extra templates come directly from the snapshot
            // We pass in None for the base path to disable searching the file system
            // for snapshotted configs.
            let _unused_extra_templates = templates.initialize(all_template_paths, None).await?;
            templates.add_templates(extra_templates)?;

            Ok(ProcessedConfigInput {
                tools,
                models,
                embedding_models,
                metrics,
                evaluations,
                provider_types,
                optimizers,
                clickhouse: overlay_clickhouse.unwrap_or_default(),
                postgres: overlay_postgres.unwrap_or_default(),
                rate_limiting: overlay_rate_limiting.unwrap_or_default(),
                autopilot,
                // unused
                snapshot,
                uninitialized_config,
                loaded_functions,
                gateway_config,
                object_store_info: overlay_object_store_info,
                runtime_overlay: captured_runtime_overlay,
                loading_errors: vec![],
            })
        }
        ConfigInput::Database {
            config,
            loading_errors,
        } => {
            let mut processed = process_uninitialized_config(*config, templates).await?;
            processed.loading_errors = loading_errors;
            Ok(processed)
        }
    }
}

async fn process_uninitialized_config(
    mut config: UninitializedConfig,
    templates: &mut TemplateConfig<'_>,
) -> Result<ProcessedConfigInput, Error> {
    validate_user_config_names(&config)?;

    // Inject built-in functions into the config (SINGLE INJECTION POINT)
    let built_in_functions = built_in::get_all_built_in_functions()?;
    let mut functions = config.functions.unwrap_or_default();
    functions.extend(built_in_functions);
    config.functions = Some(functions);

    // Destructure the config now that we've added built-in functions
    let UninitializedConfig {
        gateway,
        clickhouse,
        postgres,
        rate_limiting,
        object_storage,
        models,
        embedding_models,
        functions,
        metrics,
        tools,
        evaluations,
        provider_types,
        optimizers,
        autopilot,
    } = config.clone();

    // Resolve Options with defaults
    let gateway = gateway.unwrap_or_default();
    let clickhouse = clickhouse.unwrap_or_default();
    let postgres = postgres.unwrap_or_default();
    let rate_limiting = rate_limiting.unwrap_or_default();
    let models = models.unwrap_or_default();
    let embedding_models = embedding_models.unwrap_or_default();
    let functions = functions.unwrap_or_default();
    let metrics = metrics.unwrap_or_default();
    let tools = tools.unwrap_or_default();
    let evaluations = evaluations.unwrap_or_default();
    let provider_types = provider_types.unwrap_or_default();
    let optimizers = optimizers.unwrap_or_default();
    let autopilot = autopilot.unwrap_or_default();

    // Load ALL functions (user + built-in), including their evaluators
    let mut loaded_functions = HashMap::new();
    for (name, func_config) in functions {
        let loaded = func_config.load(&name, &metrics)?;
        loaded_functions.insert(name, loaded);
    }

    let object_store_info = ObjectStoreInfo::new(object_storage)?;
    let gateway_config = gateway.load(object_store_info.as_ref())?;

    // Initialize templates from ALL functions (including built-in)
    // Build a temporary map of just the function configs for template extraction
    let all_template_paths = Config::get_templates_from_loaded(&loaded_functions)?;
    if gateway_config
        .template_filesystem_access
        .enabled
        .unwrap_or(false)
    {
        deprecation_warning(
            "The `gateway.template_filesystem_access.enabled` flag is deprecated. We now enable filesystem access if and only if `gateway.template_file_system_access.base_path` is set. We will stop allowing this flag in the future.",
        );
    }
    let template_fs_path = gateway_config
        .template_filesystem_access
        .base_path
        .as_ref()
        .map(|x| x.get_real_path());
    let extra_templates = templates
        .initialize(all_template_paths, template_fs_path)
        .await?;

    // Create snapshot from the config (which now includes built-in functions)
    let runtime_overlay =
        RuntimeOverlay::from_uninitialized_config(&config, object_store_info.clone());
    let uninitialized_config = config.clone();
    let snapshot = ConfigSnapshot::new(config, extra_templates.clone())?;

    Ok(ProcessedConfigInput {
        tools,
        models,
        embedding_models,
        metrics,
        evaluations,
        provider_types,
        optimizers,
        clickhouse,
        postgres,
        rate_limiting,
        autopilot,
        snapshot,
        loaded_functions,
        gateway_config,
        object_store_info,
        uninitialized_config,
        runtime_overlay,
        loading_errors: vec![],
    })
}

pub use tensorzero_inference_types::credential_validation::e2e_skip_credential_validation;

impl Config {
    /// Constructs a new `Config`, as if from an empty config file.
    /// This is the only way to construct an empty config file in production code,
    /// as it ensures that things like TensorZero built-in functions will still exist in the config.
    ///
    /// In test code, a `Default` impl is available, but the config it produces might
    /// be completely broken (e.g. no builtin functions will be available).
    pub async fn new_empty() -> Result<UnwrittenConfig, Error> {
        // Use an empty glob, and validate credentials
        Box::pin(
            Self::load_from_path_optional_verify_credentials_allow_empty_glob(
                &ConfigFileGlob::new_empty(),
                true,
                true,
            ),
        )
        .await
    }

    pub async fn load_and_verify_from_path(
        config_glob: &ConfigFileGlob,
    ) -> Result<UnwrittenConfig, Error> {
        Box::pin(Self::load_from_path_optional_verify_credentials(
            config_glob,
            true,
        ))
        .await
    }

    pub async fn load_from_path_optional_verify_credentials(
        config_glob: &ConfigFileGlob,
        validate_credentials: bool,
    ) -> Result<UnwrittenConfig, Error> {
        Self::load_from_path_optional_verify_credentials_allow_empty_glob(
            config_glob,
            validate_credentials,
            false,
        )
        .await
    }

    /// Load config from a ConfigSnapshot (historical config stored in ClickHouse)
    /// with runtime configuration overlaid from the provided RuntimeOverlay.
    ///
    /// The runtime overlay ensures that infrastructure settings (gateway, postgres,
    /// rate limiting, object storage) come from the current runtime environment
    /// rather than the historical snapshot.
    pub async fn load_from_snapshot(
        snapshot: ConfigSnapshot,
        runtime_overlay: RuntimeOverlay,
        validate_credentials: bool,
    ) -> Result<UnwrittenConfig, Error> {
        let unwritten_config = if e2e_skip_credential_validation() || !validate_credentials {
            with_skip_credential_validation(Box::pin(Self::load_unwritten_config(
                ConfigInput::Snapshot {
                    snapshot: Box::new(snapshot),
                    runtime_overlay: Box::new(runtime_overlay),
                },
            )))
            .await?
        } else {
            Box::pin(Self::load_unwritten_config(ConfigInput::Snapshot {
                snapshot: Box::new(snapshot),
                runtime_overlay: Box::new(runtime_overlay),
            }))
            .await?
        };

        if validate_credentials && let Some(object_store) = &unwritten_config.object_store_info {
            object_store.verify().await?;
        }

        Ok(unwritten_config)
    }

    pub async fn load_from_path_optional_verify_credentials_allow_empty_glob(
        config_glob: &ConfigFileGlob,
        validate_credentials: bool,
        allow_empty_glob: bool,
    ) -> Result<UnwrittenConfig, Error> {
        let globbed_config = UninitializedConfig::read_toml_config(config_glob, allow_empty_glob)?;
        let unwritten_config = if e2e_skip_credential_validation() || !validate_credentials {
            with_skip_credential_validation(Box::pin(Self::load_unwritten_config(
                ConfigInput::Fresh(globbed_config.table),
            )))
            .await?
        } else {
            Box::pin(Self::load_unwritten_config(ConfigInput::Fresh(
                globbed_config.table,
            )))
            .await?
        };

        if validate_credentials && let Some(object_store) = &unwritten_config.object_store_info {
            object_store.verify().await?;
        }

        Ok(unwritten_config)
    }

    pub async fn load_from_db(
        pool: &PgPool,
        validate_credentials: bool,
    ) -> Result<UnwrittenConfig, Vec<Error>> {
        let loaded = crate::db::postgres::stored_config_queries::load_config_from_db(pool).await?;
        let input = ConfigInput::Database {
            config: Box::new(loaded.config),
            loading_errors: loaded.loading_errors,
        };
        let unwritten_config = if e2e_skip_credential_validation() || !validate_credentials {
            with_skip_credential_validation(Box::pin(Self::load_unwritten_config(input)))
                .await
                .map_err(|error| vec![error])?
        } else {
            Box::pin(Self::load_unwritten_config(input))
                .await
                .map_err(|error| vec![error])?
        };

        if validate_credentials && let Some(object_store) = &unwritten_config.object_store_info {
            object_store.verify().await.map_err(|error| vec![error])?;
        }

        Ok(unwritten_config)
    }

    pub async fn load_from_uninitialized(
        config: UninitializedConfig,
        validate_credentials: bool,
    ) -> Result<UnwrittenConfig, Error> {
        let input = ConfigInput::Database {
            config: Box::new(config),
            loading_errors: vec![],
        };
        let unwritten_config = if e2e_skip_credential_validation() || !validate_credentials {
            with_skip_credential_validation(Box::pin(Self::load_unwritten_config(input))).await?
        } else {
            Box::pin(Self::load_unwritten_config(input)).await?
        };

        if validate_credentials && let Some(object_store) = &unwritten_config.object_store_info {
            object_store.verify().await?;
        }

        Ok(unwritten_config)
    }

    /// Loads and initializes an unwritten config.
    ///
    /// This is the core config loading function that transforms raw config (TOML table, Config
    /// snapshot, or stoored config in database) into a fully validated and initialized `Config`,
    /// paired with a `ConfigSnapshot` for database storage.
    ///
    /// # Config Loading Flow
    ///
    /// This function performs the following steps:
    ///
    /// 1. **Parse to UninitializedConfig**: Convert the raw config into an `UninitializedConfig`,
    ///    which holds the raw config data before filesystem resources (schemas, templates) are loaded.
    ///
    /// 2. **Initialize Components**: Load and initialize all config components:
    ///    - Object storage (S3, filesystem)
    ///    - Gateway settings (timeouts, OTLP, etc.)
    ///    - HTTP client
    ///    - Built-in functions (tensorzero::*)
    ///    - User-defined functions (with validation against tensorzero:: prefix)
    ///    - Tools
    ///    - Models (with async credential validation)
    ///    - Embedding models
    ///    - Optimizers
    ///    - Templates (load from filesystem, compile with MiniJinja)
    ///
    /// 3. **Create Snapshot**: Create a `ConfigSnapshot` with the sorted TOML and extra templates
    ///    for database storage. The snapshot includes a Blake3 hash for version tracking.
    ///
    /// 4. **Validate**: Run comprehensive validation checks:
    ///    - Function validation (schemas, templates, tools exist)
    ///    - Model validation (timeout settings)
    ///    - Metric name restrictions
    ///    - Name prefix restrictions (tensorzero:: reserved)
    ///
    /// 5. **Load Evaluations**: Add evaluation-specific functions and metrics to the config.
    ///    This happens after validation since evaluations write tensorzero:: prefixed items.
    ///
    /// 6. **Return UnwrittenConfig**: Pair the config and snapshot in an `UnwrittenConfig`.
    ///    The snapshot is written to the database later via `into_config()`.
    ///
    /// # Why UnwrittenConfig?
    ///
    /// This function returns `UnwrittenConfig` (not just `Config`) because:
    /// - Config loading happens **before** database connection setup
    /// - The database connection settings come from the config itself
    /// - We need to write the config snapshot to ClickHouse, but can't do it yet
    /// - `UnwrittenConfig` holds both the ready-to-use config and the snapshot for later DB write
    ///
    /// The caller pattern is:
    /// ```ignore
    /// let unwritten_config = Config::load_unwritten_config(table).await?;
    /// let clickhouse = setup_clickhouse(&unwritten_config).await?;
    /// let config = unwritten_config.into_config(&clickhouse).await?;
    /// ```
    async fn load_unwritten_config(input: ConfigInput) -> Result<UnwrittenConfig, Error> {
        let is_config_snapshot = match &input {
            ConfigInput::Snapshot { .. } => true,
            ConfigInput::Fresh(_) | ConfigInput::Database { .. } => false,
        };
        let mut templates = TemplateConfig::new();
        let ProcessedConfigInput {
            tools,
            models,
            embedding_models,
            metrics,
            evaluations: uninitialized_evaluations,
            provider_types,
            optimizers: uninitialized_optimizers,
            clickhouse,
            postgres,
            rate_limiting,
            autopilot,
            snapshot,
            uninitialized_config,
            loaded_functions,
            gateway_config,
            object_store_info,
            runtime_overlay,
            loading_errors,
        } = process_config_input(input, &mut templates).await?;

        let http_client = TensorzeroHttpClient::new(
            gateway_config.global_outbound_http_timeout,
            gateway_config.global_outbound_http_intra_stream_read_timeout,
        )?;
        let relay_mode = gateway_config.relay.is_some();

        let tools = tools
            .into_iter()
            .map(|(name, config)| config.load(name.clone()).map(|c| (name, Arc::new(c))))
            .collect::<Result<HashMap<String, Arc<StaticToolConfig>>, Error>>()?;
        let provider_type_default_credentials =
            Arc::new(ProviderTypeDefaultCredentials::new(&provider_types));

        let loaded_models = try_join_all(models.into_iter().map(|(name, config)| async {
            config
                .load(
                    &name,
                    &provider_types,
                    &provider_type_default_credentials,
                    relay_mode,
                    is_config_snapshot,
                )
                .await
                .map(|c| (name, c))
        }))
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();

        if relay_mode && !is_config_snapshot {
            let models_without_skip_relay: Vec<&Arc<str>> = loaded_models
                .iter()
                .filter(|(_, config)| !config.skip_relay)
                .map(|(name, _)| name)
                .collect();
            if !models_without_skip_relay.is_empty() {
                let names = models_without_skip_relay
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", ");
                tracing::warn!(
                    "Relay mode is enabled but the following models do not have `skip_relay` set: {names}. \
                     Their configured providers will not be used for inference — requests will be relayed instead. \
                     Set `skip_relay = true` on models that should use their own providers directly."
                );
            }
        }

        let loaded_embedding_models =
            try_join_all(embedding_models.into_iter().map(|(name, config)| async {
                config
                    .load(&provider_types, &provider_type_default_credentials)
                    .await
                    .map(|c| (name, c))
            }))
            .await?
            .into_iter()
            .collect::<HashMap<_, _>>();

        let optimizers = uninitialized_optimizers
            .into_iter()
            .map(|(name, config)| (name, config.load()))
            .collect::<HashMap<_, _>>();
        let models = ModelTable::new(
            loaded_models,
            provider_type_default_credentials.clone(),
            gateway_config.global_outbound_http_timeout,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to load models: {e}"),
            })
        })?;
        let embedding_models = EmbeddingModelTable::new(
            loaded_embedding_models,
            provider_type_default_credentials,
            gateway_config.global_outbound_http_timeout,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to load embedding models: {e}"),
            })
        })?;

        // Split loaded functions into function configs and deferred evaluator artifacts
        let mut functions = HashMap::new();
        let mut deferred_evaluator_artifacts = Vec::new();
        for (name, loaded) in loaded_functions {
            functions.insert(name, Arc::new(loaded.function_config));
            if !loaded.evaluator_functions.is_empty() || !loaded.evaluator_metrics.is_empty() {
                deferred_evaluator_artifacts
                    .push((loaded.evaluator_functions, loaded.evaluator_metrics));
            }
        }

        let mut config = Config {
            gateway: gateway_config,
            clickhouse,
            models: Arc::new(models),
            embedding_models: Arc::new(embedding_models),
            functions,
            metrics,
            tools,
            evaluations: HashMap::new(),
            templates: Arc::new(templates),
            object_store_info,
            provider_types,
            optimizers,
            postgres,
            rate_limiting: rate_limiting.try_into()?,
            http_client,
            autopilot,
            hash: snapshot.hash.clone(),
            loading_errors,
        };

        // Validate the config (before adding tensorzero:: prefixed evaluator artifacts)
        config.validate().await?;

        // Register function-level evaluator functions and metrics after validation
        // (they use tensorzero:: prefix which validation would reject for user-defined items)
        for (evaluator_functions, evaluator_metrics) in deferred_evaluator_artifacts {
            for (fn_name, fn_config) in evaluator_functions {
                let fn_config = Arc::new(fn_config);
                let templates = Arc::get_mut(&mut config.templates).ok_or_else(|| {
                    Error::from(ErrorDetails::Config {
                        message: format!(
                            "Internal error: templates Arc has multiple references. {IMPOSSIBLE_ERROR_MESSAGE}"
                        ),
                    })
                })?;
                for variant in fn_config.variants().values() {
                    for template in variant.get_all_template_paths() {
                        templates.add_template(
                            template.path.get_template_key(),
                            template.contents.clone(),
                        )?;
                    }
                }
                fn_config
                    .validate(
                        &config.tools,
                        &config.models,
                        &config.embedding_models,
                        &config.templates,
                        &fn_name,
                        &config.gateway,
                    )
                    .await?;
                config.functions.insert(fn_name, fn_config);
            }
            config.metrics.extend(evaluator_metrics);
        }

        // We add the evaluations after validation since we will be writing tensorzero:: functions to the functions map
        // and tensorzero:: metrics to the metrics map
        let mut evaluations = HashMap::new();
        for (name, evaluation_config) in uninitialized_evaluations {
            let (evaluation_config, evaluation_function_configs, evaluation_metric_configs) =
                evaluation_config.load(&config.functions, &name)?;
            evaluations.insert(
                name,
                Arc::new(EvaluationConfig::Inference(evaluation_config)),
            );
            for (evaluation_function_name, evaluation_function_config) in
                evaluation_function_configs
            {
                if config.functions.contains_key(&evaluation_function_name) {
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "Duplicate evaluator function name: `{evaluation_function_name}` already exists. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."
                        ),
                    }
                    .into());
                }
                // Get mutable access to templates - this is safe because we just created the Arc
                // and haven't shared it yet
                let templates = Arc::get_mut(&mut config.templates).ok_or_else(|| {
                                    Error::from(ErrorDetails::Config {
                                        message: format!("Internal error: templates Arc has multiple references. {IMPOSSIBLE_ERROR_MESSAGE}"),
                                    })
                                })?;
                for variant in evaluation_function_config.variants().values() {
                    for template in variant.get_all_template_paths() {
                        templates.add_template(
                            template.path.get_template_key(),
                            template.contents.clone(),
                        )?;
                    }
                }
                evaluation_function_config
                    .validate(
                        &config.tools,
                        &config.models,
                        &config.embedding_models,
                        &config.templates,
                        &evaluation_function_name,
                        &config.gateway,
                    )
                    .await?;
                config
                    .functions
                    .insert(evaluation_function_name, evaluation_function_config);
            }
            for (evaluation_metric_name, evaluation_metric_config) in evaluation_metric_configs {
                if config.metrics.contains_key(&evaluation_metric_name) {
                    return Err(ErrorDetails::Config {
                        message: format!("Duplicate evaluator metric name: `{evaluation_metric_name}` already exists. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."),
                    }
                    .into());
                }
                config
                    .metrics
                    .insert(evaluation_metric_name, evaluation_metric_config);
            }
        }
        config.evaluations = evaluations;

        Ok(UnwrittenConfig::new(
            config,
            uninitialized_config,
            snapshot,
            runtime_overlay,
        ))
    }

    /// Validate the config
    #[instrument(skip_all)]
    async fn validate(&mut self) -> Result<(), Error> {
        let batch_writes = self
            .gateway
            .observability
            .batch_writes
            .as_ref()
            .cloned()
            .unwrap_or_default();
        if batch_writes.enabled && self.gateway.observability.async_writes() {
            return Err(ErrorDetails::Config {
                message: "Batch writes and async writes cannot be enabled at the same time. \
                    Async writes are enabled by default; to use batch writes, set \
                    `gateway.observability.async_writes = false`."
                    .to_string(),
            }
            .into());
        }
        if batch_writes.write_queue_capacity == Some(0) {
            return Err(ErrorDetails::Config {
                message: "Batch writes `write_queue_capacity` must be greater than 0 when set"
                    .to_string(),
            }
            .into());
        }
        if batch_writes.flush_interval_ms == Some(0) {
            return Err(ErrorDetails::Config {
                message: "Batch writes flush interval must be greater than 0".to_string(),
            }
            .into());
        }
        if batch_writes.max_rows == Some(0) {
            return Err(ErrorDetails::Config {
                message: "Batch writes max rows must be greater than 0".to_string(),
            }
            .into());
        }
        if let Some(max_rows_postgres) = batch_writes.max_rows_postgres
            && max_rows_postgres == 0
        {
            return Err(ErrorDetails::Config {
                message: "Batch writes Postgres max rows must be greater than 0".to_string(),
            }
            .into());
        }
        // Validate each function
        // Note: We don't check for tensorzero:: prefix here because:
        // 1. Built-in functions are allowed to have this prefix
        // 2. User-defined functions are prevented from using it during loading
        for (function_name, function) in &self.functions {
            function
                .validate(
                    &self.tools,
                    &self.models,
                    &self.embedding_models,
                    &self.templates,
                    function_name,
                    &self.gateway,
                )
                .await?;
        }

        // Ensure that no metrics are named "comment" or "demonstration"
        for metric_name in self.metrics.keys() {
            if metric_name == "comment" || metric_name == "demonstration" {
                return Err(ErrorDetails::Config {
                    message: format!("Metric name '{metric_name}' is reserved and cannot be used"),
                }
                .into());
            }
            if metric_name.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!("Metric name cannot start with 'tensorzero::': {metric_name}"),
                }
                .into());
            }
            // Metric names are interpolated into SQL in some query paths (e.g. LATERAL JOINs),
            // so we validate they contain only safe characters.
            crate::db::postgres::validate_safe_sql_name(metric_name, "Metric name")?;
        }

        // Validate each model
        for (model_name, model) in self.models.iter_static_models() {
            if model_name.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!("Model name cannot start with 'tensorzero::': {model_name}"),
                }
                .into());
            }
            model.validate(model_name, &self.gateway.global_outbound_http_timeout)?;
        }

        namespace::validate_namespaced_model_usage(&self.functions, &self.models)?;
        namespace::validate_namespaced_variant_usage(&self.functions)?;

        for embedding_model_name in self.embedding_models.table.keys() {
            if embedding_model_name.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "Embedding model name cannot start with 'tensorzero::': {embedding_model_name}"
                    ),
                }
                .into());
            }
        }

        // Validate each tool
        for tool_name in self.tools.keys() {
            if tool_name.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!("Tool name cannot start with 'tensorzero::': {tool_name}"),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Get a function by name
    pub fn get_function<'a>(
        &'a self,
        function_name: &str,
    ) -> Result<Cow<'a, Arc<FunctionConfig>>, Error> {
        get_function(&self.functions, function_name)
    }

    /// Get a metric by name, producing an error if it's not found
    pub fn get_metric_or_err<'a>(&'a self, metric_name: &str) -> Result<&'a MetricConfig, Error> {
        self.metrics.get(metric_name).ok_or_else(|| {
            Error::new(ErrorDetails::UnknownMetric {
                name: metric_name.to_string(),
            })
        })
    }

    /// Get a metric by name
    pub fn get_metric<'a>(&'a self, metric_name: &str) -> Option<&'a MetricConfig> {
        self.metrics.get(metric_name)
    }

    /// Get a tool by name
    pub fn get_tool<'a>(&'a self, tool_name: &str) -> Result<&'a Arc<StaticToolConfig>, Error> {
        self.tools.get(tool_name).ok_or_else(|| {
            Error::new(ErrorDetails::UnknownTool {
                name: tool_name.to_string(),
            })
        })
    }

    /// Get a model by name
    pub async fn get_model<'a>(
        &'a self,
        model_name: &Arc<str>,
        relay: Option<&TensorzeroRelay>,
    ) -> Result<CowNoClone<'a, ModelConfig>, Error> {
        self.models.get(model_name, relay).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: model_name.to_string(),
            })
        })
    }

    /// Get all templates from the config
    /// The HashMap returned is a mapping from the path as given in the TOML file
    /// (relative to the directory containing the TOML file) to the file contents.
    /// The former path is used as the name of the template for retrieval by variants later.
    pub fn get_templates(
        functions: &HashMap<String, Arc<FunctionConfig>>,
    ) -> Result<HashMap<String, String>, Error> {
        let mut templates = HashMap::new();

        for function in functions.values() {
            for variant in function.variants().values() {
                let variant_template_paths = variant.get_all_template_paths();
                for path in variant_template_paths {
                    // Duplicates involving real paths are allowed, since we might mention the same filesystem path
                    // in multiple places.
                    // However, 'fake' template names (from judges or agent-generated variants) should always be unique
                    if templates
                        .insert(path.path.get_template_key(), path.contents.clone())
                        .is_some()
                        && !path.path.is_real_path()
                    {
                        return Err(Error::new(ErrorDetails::Config {
                            message: format!(
                                "Duplicate template path: {}. {IMPOSSIBLE_ERROR_MESSAGE}",
                                path.path.get_template_key()
                            ),
                        }));
                    }
                }
            }
        }
        Ok(templates)
    }

    /// Like `get_templates`, but works with `LoadedFunctionConfig` (pre-split).
    fn get_templates_from_loaded(
        loaded_functions: &HashMap<String, LoadedFunctionConfig>,
    ) -> Result<HashMap<String, String>, Error> {
        let mut templates = HashMap::new();
        for loaded in loaded_functions.values() {
            for variant in loaded.function_config.variants().values() {
                let variant_template_paths = variant.get_all_template_paths();
                for path in variant_template_paths {
                    if templates
                        .insert(path.path.get_template_key(), path.contents.clone())
                        .is_some()
                        && !path.path.is_real_path()
                    {
                        return Err(Error::new(ErrorDetails::Config {
                            message: format!(
                                "Duplicate template path: {}. {IMPOSSIBLE_ERROR_MESSAGE}",
                                path.path.get_template_key()
                            ),
                        }));
                    }
                }
            }
        }
        Ok(templates)
    }

    pub fn get_evaluation(&self, evaluation_name: &str) -> Result<Arc<EvaluationConfig>, Error> {
        Ok(self
            .evaluations
            .get(evaluation_name)
            .ok_or_else(|| {
                Error::new(ErrorDetails::UnknownEvaluation {
                    name: evaluation_name.to_string(),
                })
            })?
            .clone())
    }
}

pub enum ConfigInput {
    Fresh(toml::Table),
    Database {
        config: Box<UninitializedConfig>,
        loading_errors: Vec<ConfigLoadingError>,
    },
    Snapshot {
        snapshot: Box<ConfigSnapshot>,
        runtime_overlay: Box<RuntimeOverlay>,
    },
}

#[cfg(feature = "pyo3")]
#[pyclass(name = "Config")]
pub struct ConfigPyClass {
    inner: Arc<Config>,
}

#[cfg(feature = "pyo3")]
impl ConfigPyClass {
    pub fn new(config: Arc<Config>) -> Self {
        Self { inner: config }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ConfigPyClass {
    #[getter]
    fn get_functions(&self) -> FunctionsConfigPyClass {
        FunctionsConfigPyClass {
            inner: self.inner.functions.clone(),
        }
    }
}

#[cfg(feature = "pyo3")]
#[pyclass(mapping, name = "FunctionsConfig")]
pub struct FunctionsConfigPyClass {
    inner: HashMap<String, Arc<FunctionConfig>>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl FunctionsConfigPyClass {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        function_name: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let f = self
            .inner
            .get(function_name)
            .ok_or_else(|| PyKeyError::new_err(function_name.to_string()))?;
        match &**f {
            FunctionConfig::Chat(_) => {
                FunctionConfigChatPyClass { inner: f.clone() }.into_bound_py_any(py)
            }
            FunctionConfig::Json(_) => {
                FunctionConfigJsonPyClass { inner: f.clone() }.into_bound_py_any(py)
            }
        }
    }
}

/// A trait for loading configs
pub trait LoadableConfig<T> {
    fn load(self) -> Result<T, Error>;
}

/// This struct is used to deserialize the TOML config file
/// It does not contain the information that needs to be loaded from the filesystem
/// such as the JSON schemas for the functions and tools.
/// If should be used as part of the `Config::load` method only.
///
/// This allows us to avoid using Option types to represent variables that are initialized after the
/// config is initially parsed.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedConfig {
    pub gateway: Option<UninitializedGatewayConfig>,
    pub clickhouse: Option<ClickHouseConfig>,
    pub postgres: Option<PostgresConfig>,
    pub rate_limiting: Option<UninitializedRateLimitingConfig>,
    pub object_storage: Option<StorageKind>,
    pub models: Option<HashMap<Arc<str>, UninitializedModelConfig>>, // model name => model config
    pub embedding_models: Option<HashMap<Arc<str>, UninitializedEmbeddingModelConfig>>, // embedding model name => embedding model config
    pub functions: Option<HashMap<String, UninitializedFunctionConfig>>, // function name => function config
    pub metrics: Option<HashMap<String, MetricConfig>>, // metric name => metric config
    pub tools: Option<HashMap<String, UninitializedToolConfig>>, // tool name => tool config
    pub evaluations: Option<HashMap<String, UninitializedEvaluationConfig>>, // evaluation name => evaluation
    pub provider_types: Option<ProviderTypesConfig>, // global configuration for all model providers of a particular type
    pub optimizers: Option<HashMap<String, UninitializedOptimizerInfo>>, // optimizer name => optimizer config
    pub autopilot: Option<AutopilotConfig>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedRelayConfig {
    /// If set, all models will be forwarded to this gateway URL
    /// (instead of calling the providers defined in our config)
    pub gateway_url: Option<Url>,
    /// If set, provides a TensorZero API key when invoking `gateway_url`.
    /// If unset, no API key will be sent.
    pub api_key_location: Option<CredentialLocationWithFallback>,
}

/// The result of parsing all of the globbed config files,
/// and merging them into a single `toml::Table`
pub struct UninitializedGlobbedConfig {
    pub table: toml::Table,
}

impl UninitializedConfig {
    /// Validate deprecated config fields, emitting warnings for deprecated locations
    /// and erroring if both old and new locations are set.
    ///
    /// This does NOT migrate values from old fields to new ones. Value migration
    /// is handled by `From<Stored*Config>` impls (for stored snapshots) and by
    /// the consumer (for TOML configs).
    pub(crate) fn warn_on_deprecations(&mut self) -> Result<(), Error> {
        self.resolve_clickhouse_config_deprecation()?;
        self.warn_variant_weight_deprecation();
        self.warn_evaluation_evaluators_deprecation();
        self.warn_gepa_evaluation_name_deprecation()?;
        Ok(())
    }

    #[expect(deprecated)]
    fn resolve_clickhouse_config_deprecation(&mut self) -> Result<(), Error> {
        let old = self
            .gateway
            .as_ref()
            .and_then(|g| g.observability.as_ref())
            .and_then(|o| o.disable_automatic_migrations);
        let new = self
            .clickhouse
            .as_ref()
            .and_then(|c| c.disable_automatic_migrations);

        if old == Some(true) && new == Some(true) {
            return Err(Error::new(ErrorDetails::Config {
                message: "`disable_automatic_migrations` is set in both `[clickhouse]` and `[gateway.observability]`. Remove it from `[gateway.observability]`.".to_string(),
            }));
        }
        if old == Some(true) {
            deprecation_warning(
                "`gateway.observability.disable_automatic_migrations` is deprecated. Use `clickhouse.disable_automatic_migrations` instead.",
            );
        }
        Ok(())
    }

    fn warn_variant_weight_deprecation(&self) {
        let empty = HashMap::new();
        let functions = self.functions.as_ref().unwrap_or(&empty);
        let functions_with_weight: Vec<&str> = functions
            .iter()
            .filter(|(_, func)| {
                let variants = match func {
                    UninitializedFunctionConfig::Chat(c) => &c.variants,
                    UninitializedFunctionConfig::Json(c) => &c.variants,
                };
                variants.values().any(|v| v.inner.weight().is_some())
            })
            .map(|(name, _)| name.as_str())
            .collect();

        if !functions_with_weight.is_empty() {
            // TODO (#4626): Finish deprecation
            deprecation_warning(&format!(
                "The `weight` field on variants is deprecated and will be removed in a future release (2026.6+). \
                 Use the `[functions.<name>.experimentation]` section instead. \
                 Affected functions: {}",
                functions_with_weight.join(", ")
            ));
        }
    }

    fn warn_evaluation_evaluators_deprecation(&self) {
        if self.evaluations.as_ref().is_none_or(|e| e.is_empty()) {
            return;
        }
        deprecation_warning(
            "Top-level evaluations are deprecated — please migrate them to \
             `[functions.function_name.evaluators]` instead.",
        );
    }

    fn warn_gepa_evaluation_name_deprecation(&self) -> Result<(), Error> {
        let mut legacy_gepa_optimizers = Vec::new();

        for (optimizer_name, optimizer) in self.optimizers.iter().flat_map(|m| m.iter()) {
            let UninitializedOptimizerConfig::GEPA(gepa_config) = &optimizer.inner else {
                continue;
            };

            match (
                gepa_config.evaluation_name.as_ref(),
                gepa_config.evaluator_names.as_ref(),
            ) {
                (Some(_), Some(_)) => {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "GEPA optimizer `{optimizer_name}` cannot specify both `evaluation_name` and `evaluator_names`"
                        ),
                    }));
                }
                (None, None) => {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "GEPA optimizer `{optimizer_name}` must specify exactly one of `evaluation_name` or `evaluator_names`"
                        ),
                    }));
                }
                (None, Some(evaluator_names)) if evaluator_names.is_empty() => {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "GEPA optimizer `{optimizer_name}` must specify at least one evaluator name"
                        ),
                    }));
                }
                (Some(_), None) => legacy_gepa_optimizers.push(optimizer_name.as_str()),
                (None, Some(_)) => {}
            }
        }

        if !legacy_gepa_optimizers.is_empty() {
            deprecation_warning(&format!(
                "The `evaluation_name` field on GEPA optimizers is deprecated. Use `evaluator_names` instead. Affected optimizers: {}",
                legacy_gepa_optimizers.join(", ")
            ));
        }

        Ok(())
    }

    /// Read all of the globbed config files from disk, and merge them into a single `UninitializedGlobbedConfig`
    pub fn read_toml_config(
        glob: &ConfigFileGlob,
        allow_empty_glob: bool,
    ) -> Result<UninitializedGlobbedConfig, Error> {
        let table = SpanMap::from_glob(glob, allow_empty_glob)?;
        Ok(UninitializedGlobbedConfig { table })
    }
}

/// TOML-specific version of `UninitializedConfig` that uses TOML shorthand for rate limiting
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TomlUninitializedConfig {
    #[serde(default)]
    gateway: UninitializedGatewayConfig,
    #[serde(default)]
    clickhouse: ClickHouseConfig,
    #[serde(default)]
    postgres: PostgresConfig,
    #[serde(default)]
    rate_limiting: rate_limiting::TomlUninitializedRateLimitingConfig,
    object_storage: Option<StorageKind>,
    #[serde(default)]
    models: HashMap<Arc<str>, UninitializedModelConfig>,
    #[serde(default)]
    embedding_models: HashMap<Arc<str>, UninitializedEmbeddingModelConfig>,
    #[serde(default)]
    functions: HashMap<String, UninitializedFunctionConfig>,
    #[serde(default)]
    metrics: HashMap<String, MetricConfig>,
    #[serde(default)]
    tools: HashMap<String, UninitializedToolConfig>,
    #[serde(default)]
    evaluations: HashMap<String, UninitializedEvaluationConfig>,
    #[serde(default)]
    provider_types: ProviderTypesConfig,
    #[serde(default)]
    optimizers: HashMap<String, UninitializedOptimizerInfo>,
    #[serde(default)]
    autopilot: AutopilotConfig,
}

impl TryFrom<TomlUninitializedConfig> for UninitializedConfig {
    type Error = Error;

    fn try_from(toml_config: TomlUninitializedConfig) -> Result<Self, Self::Error> {
        let rate_limiting = toml_config
            .rate_limiting
            .try_into()
            .map_err(|e: String| Error::new(ErrorDetails::Config { message: e }))?;
        Ok(Self {
            gateway: Some(toml_config.gateway),
            clickhouse: Some(toml_config.clickhouse),
            postgres: Some(toml_config.postgres),
            rate_limiting: Some(rate_limiting),
            object_storage: toml_config.object_storage,
            models: Some(toml_config.models),
            embedding_models: Some(toml_config.embedding_models),
            functions: Some(toml_config.functions),
            metrics: Some(toml_config.metrics),
            tools: Some(toml_config.tools),
            evaluations: Some(toml_config.evaluations),
            provider_types: Some(toml_config.provider_types),
            optimizers: Some(toml_config.optimizers),
            autopilot: Some(toml_config.autopilot),
        })
    }
}

/// Deserialize a TOML table into `UninitializedConfig`
impl TryFrom<toml::Table> for UninitializedConfig {
    type Error = Error;

    fn try_from(table: toml::Table) -> Result<Self, Self::Error> {
        // First deserialize into TOML-specific config, then convert to runtime config
        let toml_config: TomlUninitializedConfig = serde_path_to_error::deserialize(table)
            .map_err(|e| {
                let path = e.path().clone();
                Error::new(ErrorDetails::Config {
                    // Extract the underlying message from the toml error, as
                    // the path-tracking from the toml crate will be incorrect
                    message: format!("{}: {}", path, e.into_inner().message()),
                })
            })?;
        let mut config: UninitializedConfig = toml_config.try_into()?;
        config.warn_on_deprecations()?;
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq, TensorZeroDeserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum UninitializedFunctionConfig {
    Chat(UninitializedFunctionConfigChat),
    Json(UninitializedFunctionConfigJson),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct UninitializedSchema {
    path: ResolvedTomlPathData,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[serde(transparent)]
pub struct UninitializedSchemas {
    inner: HashMap<String, UninitializedSchema>,
}

impl UninitializedSchemas {
    /// Constructs `UninitializedSchemas` from a map of schema names to path data.
    pub fn from_paths(paths: HashMap<String, ResolvedTomlPathData>) -> Self {
        Self {
            inner: paths
                .into_iter()
                .map(|(k, path)| (k, UninitializedSchema { path }))
                .collect(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &ResolvedTomlPathData)> {
        self.inner.iter().map(|(name, schema)| (name, &schema.path))
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedFunctionConfigChat {
    pub variants: HashMap<String, UninitializedVariantInfo>, // variant name => variant config
    pub system_schema: Option<ResolvedTomlPathData>,
    pub user_schema: Option<ResolvedTomlPathData>,
    pub assistant_schema: Option<ResolvedTomlPathData>,
    #[serde(default)]
    pub schemas: UninitializedSchemas,
    #[serde(default)]
    pub tools: Vec<String>, // tool names
    #[serde(default)]
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub description: Option<String>,
    pub experimentation: Option<UninitializedExperimentationConfigWithNamespaces>,
    #[serde(default)]
    pub evaluators: HashMap<String, UninitializedEvaluatorConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedFunctionConfigJson {
    pub variants: HashMap<String, UninitializedVariantInfo>, // variant name => variant config
    pub system_schema: Option<ResolvedTomlPathData>,
    pub user_schema: Option<ResolvedTomlPathData>,
    pub assistant_schema: Option<ResolvedTomlPathData>,
    #[serde(default)]
    pub schemas: UninitializedSchemas,
    pub output_schema: Option<ResolvedTomlPathData>, // schema will default to {} if not specified
    pub description: Option<String>,
    pub experimentation: Option<UninitializedExperimentationConfigWithNamespaces>,
    #[serde(default)]
    pub evaluators: HashMap<String, UninitializedEvaluatorConfig>,
}

/// Holds all of the schemas used by a chat completion function.
/// These are used by variants to construct a `TemplateWithSchema`
#[derive(ts_rs::TS, Debug, Default, Serialize)]
#[ts(export, optional_fields)]
pub struct SchemaData {
    #[serde(flatten)]
    pub inner: HashMap<String, SchemaWithMetadata>,
}

impl SchemaData {
    pub fn get_implicit_system_schema(&self) -> Option<&SchemaWithMetadata> {
        self.inner.get("system")
    }

    pub fn get_implicit_user_schema(&self) -> Option<&SchemaWithMetadata> {
        self.inner.get("user")
    }

    pub fn get_implicit_assistant_schema(&self) -> Option<&SchemaWithMetadata> {
        self.inner.get("assistant")
    }

    pub fn get_named_schema(&self, name: &str) -> Option<&SchemaWithMetadata> {
        self.inner.get(name)
    }

    pub(super) fn load(
        user_schema: Option<JSONSchema>,
        assistant_schema: Option<JSONSchema>,
        system_schema: Option<JSONSchema>,
        schemas: UninitializedSchemas,
        function_name: &str,
    ) -> Result<Self, Error> {
        let mut map = HashMap::new();
        if let Some(user_schema) = user_schema {
            map.insert(
                "user".to_string(),
                SchemaWithMetadata {
                    schema: user_schema,
                    legacy_definition: true,
                },
            );
        }
        if let Some(assistant_schema) = assistant_schema {
            map.insert(
                "assistant".to_string(),
                SchemaWithMetadata {
                    schema: assistant_schema,
                    legacy_definition: true,
                },
            );
        }
        if let Some(system_schema) = system_schema {
            map.insert(
                "system".to_string(),
                SchemaWithMetadata {
                    schema: system_schema,
                    legacy_definition: true,
                },
            );
        }
        for (name, schema) in schemas.inner {
            if map
                .insert(
                    name.clone(),
                    SchemaWithMetadata {
                        schema: JSONSchema::from_path(schema.path)?,
                        legacy_definition: false,
                    },
                )
                .is_some()
            {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "functions.{function_name}: Cannot specify both `schemas.{name}.path` and `{name}_schema`"
                    ),
                }));
            }
        }
        Ok(Self { inner: map })
    }
}

/// Propagates deprecated `timeout_s` from best_of_n/mixture_of_n variants to their candidate variants.
///
/// If a best_of_n or mixture_of_n variant has `timeout_s` set:
/// 1. Emits a deprecation warning
/// 2. Sets `timeouts` on each candidate variant (if not already set)
/// 3. Returns an error if a candidate already has `timeouts` set (conflict)
fn propagate_timeout_s_to_candidates(
    function_name: &str,
    variants: &mut HashMap<String, UninitializedVariantInfo>,
) -> Result<(), Error> {
    use crate::utils::deprecation_warning;

    // Collect timeout_s values from best_of_n/mixture_of_n variants
    let mut timeout_propagations: Vec<(String, f64, Vec<String>)> = Vec::new();

    for (variant_name, variant_info) in variants.iter() {
        match &variant_info.inner {
            UninitializedVariantConfig::BestOfNSampling(config) => {
                if let Some(timeout_s) = config.timeout_s() {
                    timeout_propagations.push((
                        variant_name.clone(),
                        timeout_s,
                        config.candidates.clone(),
                    ));
                }
            }
            UninitializedVariantConfig::MixtureOfN(config) => {
                if let Some(timeout_s) = config.timeout_s() {
                    timeout_propagations.push((
                        variant_name.clone(),
                        timeout_s,
                        config.candidates.clone(),
                    ));
                }
            }
            _ => {}
        }
    }

    // Apply timeout_s to candidate variants
    for (parent_variant_name, timeout_s, candidates) in timeout_propagations {
        deprecation_warning(&format!(
            "Deprecation Warning (#2480 / 2026.2+): `timeout_s` in functions.{function_name}.variants.{parent_variant_name} is deprecated. Please use `[timeouts]` on your candidate variants instead."
        ));

        let timeout_ms = (timeout_s * 1000.0) as u64;
        let timeouts_config = TimeoutsConfig {
            non_streaming: Some(NonStreamingTimeouts {
                total_ms: Some(timeout_ms),
            }),
            streaming: Some(StreamingTimeouts {
                ttft_ms: Some(timeout_ms),
                total_ms: None,
            }),
        };

        for candidate_name in candidates {
            let candidate_variant = variants.get_mut(&candidate_name).ok_or_else(|| {
                Error::new(ErrorDetails::Config {
                    message: format!(
                        "functions.{function_name}.variants.{parent_variant_name}: candidate `{candidate_name}` not found"
                    ),
                })
            })?;

            // Check for conflict
            if candidate_variant.timeouts.is_some() {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "functions.{function_name}.variants.{parent_variant_name}: cannot use `timeout_s` when candidate `{candidate_name}` already has `[timeouts]` configured. Please remove `timeout_s` and configure timeouts directly on the candidate."
                    ),
                }));
            }

            // Set timeouts on candidate
            candidate_variant.timeouts = Some(timeouts_config.clone());
        }
    }

    Ok(())
}

/// Result of loading a function config, including any generated evaluator functions and metrics.
pub struct LoadedFunctionConfig {
    pub function_config: FunctionConfig,
    /// LLM judge functions generated by evaluators on this function
    pub evaluator_functions: HashMap<String, FunctionConfig>,
    /// Metrics generated by evaluators on this function
    pub evaluator_metrics: HashMap<String, MetricConfig>,
}

struct LoadedEvaluators {
    evaluators: HashMap<String, EvaluatorConfig>,
    generated_functions: HashMap<String, FunctionConfig>,
    generated_metrics: HashMap<String, MetricConfig>,
}

/// Load evaluators defined on a function, returning the loaded evaluator configs
/// plus any generated LLM judge functions and metrics.
fn load_function_evaluators(
    function_name: &str,
    uninitialized_evaluators: HashMap<String, UninitializedEvaluatorConfig>,
) -> Result<LoadedEvaluators, Error> {
    let mut evaluators = HashMap::new();
    let mut generated_functions = HashMap::new();
    let mut generated_metrics = HashMap::new();

    for (evaluator_name, evaluator_config) in uninitialized_evaluators {
        let (loaded_evaluator, function_config, metric_config) =
            evaluator_config.load(None, Some(function_name), &evaluator_name)?;

        if let Some(func) = function_config {
            let llm_judge_fn_name =
                get_function_llm_judge_function_name(function_name, &evaluator_name);
            generated_functions.insert(llm_judge_fn_name, func);
        }

        let metric_name = get_function_evaluator_metric_name(function_name, &evaluator_name);
        generated_metrics.insert(metric_name, metric_config);
        evaluators.insert(evaluator_name, loaded_evaluator);
    }

    Ok(LoadedEvaluators {
        evaluators,
        generated_functions,
        generated_metrics,
    })
}

impl UninitializedFunctionConfig {
    pub fn load(
        self,
        function_name: &str,
        metrics: &HashMap<String, MetricConfig>,
    ) -> Result<LoadedFunctionConfig, Error> {
        match self {
            UninitializedFunctionConfig::Chat(mut params) => {
                // Propagate deprecated timeout_s to candidate variants before loading
                propagate_timeout_s_to_candidates(function_name, &mut params.variants)?;

                let schema_data = SchemaData::load(
                    params.user_schema.map(JSONSchema::from_path).transpose()?,
                    params
                        .assistant_schema
                        .map(JSONSchema::from_path)
                        .transpose()?,
                    params
                        .system_schema
                        .map(JSONSchema::from_path)
                        .transpose()?,
                    params.schemas,
                    function_name,
                )?;
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| {
                        variant
                            .load(
                                &schema_data,
                                &ErrorContext {
                                    function_name: function_name.to_string(),
                                    variant_name: name.to_string(),
                                },
                            )
                            .map(|v| (name, Arc::new(v)))
                    })
                    .collect::<Result<HashMap<_, _>, Error>>()?;
                let mut all_template_names = HashSet::new();
                for (name, variant) in &variants {
                    all_template_names.extend(variant.get_all_explicit_template_names());
                    if let VariantConfig::ChatCompletion(chat_config) = &variant.inner
                        && chat_config.json_mode().is_some()
                    {
                        return Err(ErrorDetails::Config {
                            message: format!(
                                "JSON mode is not supported for variant `{name}` (parent function is a chat function)",
                            ),
                        }
                        .into());
                    }
                }
                let experimentation = params
                    .experimentation
                    .map(|config| config.load(&variants, metrics, function_name, true))
                    .transpose()?
                    .unwrap_or_else(|| ExperimentationConfigWithNamespaces {
                        base: ExperimentationConfig::legacy_from_variants_map(&variants),
                        namespaces: std::collections::HashMap::new(),
                    });
                let LoadedEvaluators {
                    evaluators: loaded_evaluators,
                    generated_functions: evaluator_functions,
                    generated_metrics: evaluator_metrics,
                } = load_function_evaluators(function_name, params.evaluators)?;
                Ok(LoadedFunctionConfig {
                    function_config: FunctionConfig::Chat(FunctionConfigChat {
                        variants,
                        schemas: schema_data,
                        tools: params.tools,
                        tool_choice: params.tool_choice,
                        parallel_tool_calls: params.parallel_tool_calls,
                        description: params.description,
                        all_explicit_templates_names: all_template_names,
                        experimentation,
                        evaluators: loaded_evaluators,
                    }),
                    evaluator_functions,
                    evaluator_metrics,
                })
            }
            UninitializedFunctionConfig::Json(mut params) => {
                // Propagate deprecated timeout_s to candidate variants before loading
                propagate_timeout_s_to_candidates(function_name, &mut params.variants)?;

                let schema_data = SchemaData::load(
                    params.user_schema.map(JSONSchema::from_path).transpose()?,
                    params
                        .assistant_schema
                        .map(JSONSchema::from_path)
                        .transpose()?,
                    params
                        .system_schema
                        .map(JSONSchema::from_path)
                        .transpose()?,
                    params.schemas,
                    function_name,
                )?;
                let output_schema = match params.output_schema {
                    Some(path) => JSONSchema::from_path(path)?,
                    None => JSONSchema::default(),
                };
                let json_mode_tool_call_config =
                    create_json_mode_tool_call_config(output_schema.clone());
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| {
                        variant
                            .load(
                                &schema_data,
                                &ErrorContext {
                                    function_name: function_name.to_string(),
                                    variant_name: name.to_string(),
                                },
                            )
                            .map(|v| (name, Arc::new(v)))
                    })
                    .collect::<Result<HashMap<_, _>, Error>>()?;

                let mut all_template_names = HashSet::new();

                for (name, variant) in &variants {
                    let mut variant_missing_mode = None;
                    all_template_names.extend(variant.get_all_explicit_template_names());
                    match &variant.inner {
                        VariantConfig::ChatCompletion(chat_config) => {
                            if chat_config.json_mode().is_none() {
                                variant_missing_mode = Some(name.clone());
                            }
                        }
                        VariantConfig::BestOfNSampling(_best_of_n_config) => {
                            // Evaluator json_mode is optional - it defaults to `strict` at runtime
                        }
                        VariantConfig::MixtureOfN(mixture_of_n_config) => {
                            if mixture_of_n_config.fuser().inner.json_mode().is_none() {
                                variant_missing_mode = Some(format!("{name}.fuser"));
                            }
                        }
                        VariantConfig::Dicl(best_of_n_config) => {
                            if best_of_n_config.json_mode().is_none() {
                                variant_missing_mode = Some(name.clone());
                            }
                        }
                        VariantConfig::ChainOfThought(chain_of_thought_config) => {
                            if chain_of_thought_config.inner.json_mode().is_none() {
                                variant_missing_mode = Some(name.clone());
                            }
                        }
                    }
                    if let Some(variant_name) = variant_missing_mode {
                        return Err(ErrorDetails::Config {
                            message: format!(
                                "`json_mode` must be specified for `[functions.{function_name}.variants.{variant_name}]` (parent function `{function_name}` is a JSON function)"
                            ),
                        }
                        .into());
                    }
                }
                let experimentation = params
                    .experimentation
                    .map(|config| config.load(&variants, metrics, function_name, true))
                    .transpose()?
                    .unwrap_or_else(|| ExperimentationConfigWithNamespaces {
                        base: ExperimentationConfig::legacy_from_variants_map(&variants),
                        namespaces: std::collections::HashMap::new(),
                    });
                let LoadedEvaluators {
                    evaluators: loaded_evaluators,
                    generated_functions: evaluator_functions,
                    generated_metrics: evaluator_metrics,
                } = load_function_evaluators(function_name, params.evaluators)?;
                Ok(LoadedFunctionConfig {
                    function_config: FunctionConfig::Json(FunctionConfigJson {
                        variants,
                        schemas: schema_data,
                        output_schema,
                        json_mode_tool_call_config,
                        description: params.description,
                        all_explicit_template_names: all_template_names,
                        experimentation,
                        evaluators: loaded_evaluators,
                    }),
                    evaluator_functions,
                    evaluator_metrics,
                })
            }
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(ts_rs::TS, Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export, optional_fields)]
#[serde(rename_all = "snake_case")]
// We don't use `#[serde(deny_unknown_fields)]` here - it needs to go on 'UninitializedVariantConfig',
// since we use `#[serde(flatten)]` on the `inner` field.
pub struct UninitializedVariantInfo {
    #[serde(flatten)]
    pub inner: UninitializedVariantConfig,
    pub timeouts: Option<TimeoutsConfig>,
    #[ts(optional)]
    pub namespace: Option<Namespace>,
}

/// NOTE: Contains deprecated variant `ChainOfThought` (#5298 / 2026.2+)
#[derive(ts_rs::TS, Clone, Debug, JsonSchema, PartialEq, TensorZeroDeserialize, Serialize)]
#[ts(export)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum UninitializedVariantConfig {
    ChatCompletion(UninitializedChatCompletionConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(UninitializedBestOfNSamplingConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(UninitializedDiclConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfN(UninitializedMixtureOfNConfig),
    /// DEPRECATED (#5298 / 2026.2+): Use `chat_completion` with reasoning instead.
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(UninitializedChainOfThoughtConfig),
}

impl UninitializedVariantConfig {
    pub fn weight(&self) -> Option<f64> {
        match self {
            Self::ChatCompletion(c) => c.weight,
            Self::BestOfNSampling(c) => c.weight,
            Self::Dicl(c) => c.weight,
            Self::MixtureOfN(c) => c.weight,
            Self::ChainOfThought(c) => c.inner.weight,
        }
    }
}

/// Holds extra information used for enriching error messages
pub struct ErrorContext {
    pub function_name: String,
    pub variant_name: String,
}

impl ErrorContext {
    #[cfg(test)]
    pub fn new_test() -> Self {
        Self {
            function_name: "test".to_string(),
            variant_name: "test".to_string(),
        }
    }
}

impl UninitializedVariantInfo {
    pub fn load(
        self,
        schemas: &SchemaData,
        error_context: &ErrorContext,
    ) -> Result<VariantInfo, Error> {
        let inner = match self.inner {
            UninitializedVariantConfig::ChatCompletion(params) => {
                VariantConfig::ChatCompletion(params.load(schemas, error_context)?)
            }
            UninitializedVariantConfig::BestOfNSampling(params) => {
                VariantConfig::BestOfNSampling(params.load(schemas, error_context)?)
            }
            UninitializedVariantConfig::Dicl(params) => VariantConfig::Dicl(params.load()?),
            UninitializedVariantConfig::MixtureOfN(params) => {
                VariantConfig::MixtureOfN(params.load(schemas, error_context)?)
            }
            UninitializedVariantConfig::ChainOfThought(params) => {
                tracing::warn!(
                    "Deprecation Warning (#5298 / 2026.2+): We are deprecating `experimental_chain_of_thought` now that reasoning models are prevalent. Please use a different variant type (e.g. `chat_completion` with reasoning)."
                );
                VariantConfig::ChainOfThought(params.load(schemas, error_context)?)
            }
        };
        Ok(VariantInfo {
            inner,
            timeouts: self.timeouts.unwrap_or_default(),
            namespace: self.namespace,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedToolConfig {
    pub description: String,
    pub parameters: ResolvedTomlPathData,
    pub name: Option<String>,
    #[serde(default)]
    pub strict: bool,
}

impl UninitializedToolConfig {
    pub(crate) fn convert_for_db(
        &self,
        file_version_ids: &HashMap<String, Uuid>,
    ) -> Result<StoredToolConfig, Error> {
        let file_path = self.parameters.get_template_key();
        let Some(file_version_id) = file_version_ids.get(&file_path) else {
            return Err(Error::new(ErrorDetails::Config {
                message: format!("Missing stored file version ID for file path `{file_path}`."),
            }));
        };
        Ok(StoredToolConfig {
            description: self.description.clone(),
            parameters: StoredFileRef {
                file_version_id: *file_version_id,
                file_path,
            },
            name: self.name.clone(),
            strict: self.strict,
        })
    }

    pub fn load(self, key: String) -> Result<StaticToolConfig, Error> {
        let parameters = JSONSchema::from_path(self.parameters)?;
        Ok(StaticToolConfig {
            name: self.name.unwrap_or_else(|| key.clone()),
            key,
            description: self.description,
            parameters,
            strict: self.strict,
        })
    }
}

#[derive(ts_rs::TS, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct PathWithContents {
    #[ts(type = "string")]
    pub path: ResolvedTomlPathData,
    pub contents: String,
}

impl PathWithContents {
    pub fn from_path(path: ResolvedTomlPathData) -> Result<Self, Error> {
        let contents = path.data().to_string();
        Ok(Self { path, contents })
    }
}

pub(crate) const DEFAULT_POSTGRES_CONNECTION_POOL_SIZE: u32 = 20;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PostgresConfig {
    /// DEPRECATED (2026.3+): Postgres connectivity is now determined by the
    /// `TENSORZERO_POSTGRES_URL` environment variable. This field is accepted
    /// for backward compatibility but will be removed in a future release.
    #[deprecated(
        note = "Postgres connectivity is now determined by the `TENSORZERO_POSTGRES_URL` environment variable. Remove `postgres.enabled` from your config."
    )]
    #[serde(skip_serializing)]
    pub enabled: Option<bool>,
    pub connection_pool_size: Option<u32>,
    /// Retention period in days for inference metadata tables
    /// (chat_inferences, json_inferences, model_inferences — monthly partitions).
    /// If set, old partitions beyond this age will be dropped by pg_cron.
    /// If not set, partitions are retained indefinitely.
    ///
    /// TODO(#5764): Document clearly in user-facing docs:
    /// - WARNING: When first set (or lowered), pg_cron will immediately start dropping
    ///   partitions older than this value on its next daily run. This is irreversible.
    /// - Clients are responsible for managing their own data backups before enabling retention.
    /// - Recommend testing in non-production first and starting with a longer period.
    pub inference_metadata_retention_days: Option<u32>,
    /// Retention period in days for inference data tables
    /// (chat_inference_data, json_inference_data, model_inference_data — daily partitions).
    /// `inference_retention_days` is accepted as a deprecated alias.
    #[serde(alias = "inference_retention_days")]
    pub inference_data_retention_days: Option<u32>,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        #[expect(deprecated)]
        Self {
            enabled: None,
            connection_pool_size: Some(20),
            inference_metadata_retention_days: None,
            inference_data_retention_days: None,
        }
    }
}

impl From<tensorzero_stored_config::StoredPostgresConfig> for PostgresConfig {
    fn from(stored: tensorzero_stored_config::StoredPostgresConfig) -> Self {
        #[expect(deprecated)]
        PostgresConfig {
            enabled: None,
            connection_pool_size: stored.connection_pool_size,
            inference_metadata_retention_days: stored.inference_metadata_retention_days,
            inference_data_retention_days: stored.inference_data_retention_days,
        }
    }
}

impl From<&PostgresConfig> for tensorzero_stored_config::StoredPostgresConfig {
    fn from(config: &PostgresConfig) -> Self {
        tensorzero_stored_config::StoredPostgresConfig {
            connection_pool_size: config.connection_pool_size,
            inference_metadata_retention_days: config.inference_metadata_retention_days,
            inference_data_retention_days: config.inference_data_retention_days,
        }
    }
}

#[cfg(test)]
mod round_trip_tests {
    use std::path::PathBuf;

    use googletest::prelude::*;

    use super::*;
    use crate::config::path::ResolvedTomlPathData;

    // ── TimeoutsConfig ─────────────────────────────────────────────────

    #[gtest]
    fn test_timeouts_config_round_trip_full() {
        let original = TimeoutsConfig {
            non_streaming: Some(NonStreamingTimeouts {
                total_ms: Some(5000),
            }),
            streaming: Some(StreamingTimeouts {
                ttft_ms: Some(1000),
                total_ms: Some(30000),
            }),
        };
        let stored = StoredTimeoutsConfig::from(&original);
        let restored: TimeoutsConfig = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_timeouts_config_round_trip_empty() {
        let original = TimeoutsConfig::default();
        let stored = StoredTimeoutsConfig::from(&original);
        let restored: TimeoutsConfig = stored.into();
        expect_that!(restored, eq(&original));
    }

    // ── MetricConfig ───────────────────────────────────────────────────

    #[gtest]
    fn test_metric_config_type_round_trip() {
        for variant in [MetricConfigType::Boolean, MetricConfigType::Float] {
            let stored: StoredMetricType = variant.into();
            let restored: MetricConfigType = stored.into();
            expect_that!(restored, eq(variant));
        }
    }

    #[gtest]
    fn test_metric_config_optimize_round_trip() {
        for variant in [MetricConfigOptimize::Min, MetricConfigOptimize::Max] {
            let stored: StoredMetricOptimize = variant.into();
            let restored: MetricConfigOptimize = stored.into();
            expect_that!(restored, eq(variant));
        }
    }

    #[gtest]
    fn test_metric_config_level_round_trip() {
        for variant in &[MetricConfigLevel::Inference, MetricConfigLevel::Episode] {
            let stored = StoredMetricLevel::from(variant);
            let restored: MetricConfigLevel = stored.into();
            expect_that!(restored, eq(variant));
        }
    }

    #[gtest]
    fn test_metric_config_round_trip() {
        let original = MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
            description: Some("test metric".to_string()),
        };
        let stored = tensorzero_stored_config::StoredMetricConfig {
            r#type: original.r#type.into(),
            optimize: original.optimize.into(),
            level: (&original.level).into(),
            description: original.description.clone(),
        };
        let restored: MetricConfig = stored.into();
        expect_that!(restored, eq(&original));
    }

    // ── UninitializedToolConfig ────────────────────────────────────────
    //
    // `UninitializedToolConfig` only has a forward conversion to
    // `StoredToolConfig` (the reverse requires looking up a stored file
    // by ID), so this verifies the forward conversion preserves all fields
    // and resolves the stored file version ID correctly.

    #[gtest]
    fn test_uninitialized_tool_config_convert_for_db() {
        let parameters_json = r#"{"type":"object","properties":{}}"#.to_string();
        let parameters = ResolvedTomlPathData::new_for_tests(
            PathBuf::from("tools/my_tool.json"),
            Some(parameters_json),
        );
        let file_path = parameters.get_template_key();

        let original = UninitializedToolConfig {
            description: "Tool description".to_string(),
            parameters,
            name: Some("my_tool".to_string()),
            strict: true,
        };

        let template_id = Uuid::now_v7();
        let mut file_version_ids = HashMap::new();
        file_version_ids.insert(file_path.clone(), template_id);

        let stored = original
            .convert_for_db(&file_version_ids)
            .expect("conversion should succeed when template id is present");

        expect_that!(stored.description, eq(&original.description));
        expect_that!(stored.name.as_deref(), some(eq("my_tool")));
        expect_that!(stored.strict, eq(true));
        expect_that!(stored.parameters.file_path, eq(&file_path));
        expect_that!(stored.parameters.file_version_id, eq(template_id));
    }

    #[gtest]
    fn test_uninitialized_tool_config_convert_for_db_missing_template() {
        let parameters = ResolvedTomlPathData::new_for_tests(
            PathBuf::from("tools/missing.json"),
            Some(r#"{"type":"object"}"#.to_string()),
        );
        let original = UninitializedToolConfig {
            description: "x".to_string(),
            parameters,
            name: None,
            strict: false,
        };

        let result = original.convert_for_db(&HashMap::new());
        expect_that!(result.is_err(), eq(true));
    }
}
