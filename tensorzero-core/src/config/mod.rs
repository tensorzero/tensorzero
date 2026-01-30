use crate::experimentation::{ExperimentationConfig, UninitializedExperimentationConfig};
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
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tensorzero_derive::TensorZeroDeserialize;
use tracing::Span;
use tracing::instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use unwritten::UnwrittenConfig;
use url::Url;

use crate::config::gateway::{GatewayConfig, UninitializedGatewayConfig};
use crate::config::path::{ResolvedTomlPathData, ResolvedTomlPathDirectory};
use crate::config::snapshot::ConfigSnapshot;
use crate::config::span_map::SpanMap;
use crate::db::clickhouse::{ClickHouseConnectionInfo, ExternalDataInfo};
use crate::embeddings::{EmbeddingModelTable, UninitializedEmbeddingModelConfig};
use crate::endpoints::status::TENSORZERO_VERSION;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::evaluations::{EvaluationConfig, UninitializedEvaluationConfig};
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
use crate::optimization::{OptimizerInfo, UninitializedOptimizerInfo};
use crate::tool::{StaticToolConfig, ToolChoice, create_json_mode_tool_call_config};
use crate::variant::best_of_n_sampling::UninitializedBestOfNSamplingConfig;
use crate::variant::chain_of_thought::UninitializedChainOfThoughtConfig;
use crate::variant::chat_completion::UninitializedChatCompletionConfig;
use crate::variant::dicl::UninitializedDiclConfig;
use crate::variant::mixture_of_n::UninitializedMixtureOfNConfig;
use crate::variant::{Variant, VariantConfig, VariantInfo};
use std::error::Error as StdError;

pub mod built_in;
pub mod gateway;
pub mod path;
pub mod provider_types;
pub mod rate_limiting;
pub mod snapshot;
mod span_map;
pub mod stored;
#[cfg(test)]
mod tests;
pub mod unwritten;

tokio::task_local! {
    /// When set, we skip performing credential validation in model providers
    /// This is used when running in e2e test mode, and by the 'evaluations' binary
    /// We need to access this from async code (e.g. when looking up GCP SDK credentials),
    /// so this needs to be a tokio task-local (as a task may be moved between threads)
    ///
    /// Since this needs to be accessed from a `Deserialize` impl, it needs to
    /// be stored in a `static`, since we cannot pass in extra parameters when calling `Deserialize::deserialize`
    static SKIP_CREDENTIAL_VALIDATION: ();
}

pub fn skip_credential_validation() -> bool {
    // tokio::task_local doesn't have an 'is_set' method, so we call 'try_with'
    // (which returns an `Err` if the task-local is not set)
    SKIP_CREDENTIAL_VALIDATION.try_with(|()| ()).is_ok()
}

/// Runs the provider future with credential validation disabled
/// This is safe to repeatedly nest (e.g. `with_skip_credential_validation(async move { with_skip_credential_validation(f).await })`),
/// the original credential validation behavior will be restored after the outermost future completes
pub async fn with_skip_credential_validation<T>(f: impl Future<Output = T>) -> T {
    SKIP_CREDENTIAL_VALIDATION.scope((), f).await
}

// Note - the `Default` impl only exists for convenience in tests
// It might produce a completely broken config - if a test fails,
// use one of the public `Config` constructors instead.
#[derive(Debug)]
#[cfg_attr(any(test, feature = "e2e_tests"), derive(Default))]
pub struct Config {
    pub gateway: GatewayConfig,
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
    pub hash: SnapshotHash,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
pub struct NonStreamingTimeouts {
    #[serde(default)]
    /// The total time allowed for the non-streaming request to complete.
    pub total_ms: Option<u64>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
pub struct StreamingTimeouts {
    #[serde(default)]
    /// The time allowed for the first token to be produced.
    pub ttft_ms: Option<u64>,
}

/// Configures the timeouts for both streaming and non-streaming requests.
/// This can be attached to various other configs (e.g. variants, models, model providers)
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
pub struct TimeoutsConfig {
    #[serde(default)]
    pub non_streaming: NonStreamingTimeouts,
    #[serde(default)]
    pub streaming: StreamingTimeouts,
}

impl TimeoutsConfig {
    pub fn validate(&self, global_outbound_http_timeout: &Duration) -> Result<(), Error> {
        let TimeoutsConfig {
            non_streaming: NonStreamingTimeouts { total_ms },
            streaming: StreamingTimeouts { ttft_ms },
        } = self;

        let global_ms = global_outbound_http_timeout.num_milliseconds();

        if let Some(total_ms) = total_ms
            && Duration::milliseconds(*total_ms as i64) > *global_outbound_http_timeout
        {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "The `timeouts.non_streaming.total_ms` value `{total_ms}` is greater than `gateway.global_outbound_http_timeout_ms`: `{global_ms}`"
                ),
            }));
        }
        if let Some(ttft_ms) = ttft_ms
            && Duration::milliseconds(*ttft_ms as i64) > *global_outbound_http_timeout
        {
            return Err(Error::new(ErrorDetails::Config {
                message: format!(
                    "The `timeouts.streaming.ttft_ms` value `{ttft_ms}` is greater than `gateway.global_outbound_http_timeout_ms`: `{global_ms}`"
                ),
            }));
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TemplateFilesystemAccess {
    /// If `true`, allow minijinja to read from the filesystem (within the tree of the config file) for `{% include %}`
    /// Defaults to `false`
    #[serde(default)]
    enabled: bool,
    base_path: Option<ResolvedTomlPathDirectory>,
}

#[derive(Clone, Debug)]
pub struct ObjectStoreInfo {
    // This will be `None` if we have `StorageKind::Disabled`
    pub object_store: Option<Arc<dyn ObjectStore>>,
    pub kind: StorageKind,
}

impl ObjectStoreInfo {
    pub fn new(config: Option<StorageKind>) -> Result<Option<Self>, Error> {
        let Some(config) = config else {
            return Ok(None);
        };

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

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityConfig {
    pub enabled: Option<bool>,
    #[serde(default)]
    pub async_writes: bool,
    #[serde(default)]
    pub batch_writes: BatchWritesConfig,
    #[serde(default)]
    pub disable_automatic_migrations: bool,
}

fn default_flush_interval_ms() -> u64 {
    100
}

fn default_max_rows() -> usize {
    1000
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchWritesConfig {
    pub enabled: bool,
    // An internal flag to allow us to test batch writes in embedded gateway mode.
    // This can currently cause deadlocks, so we don't want normal embedded clients to use it.
    #[serde(default)]
    pub __force_allow_embedded_batch_writes: bool,
    #[serde(default = "default_flush_interval_ms")]
    pub flush_interval_ms: u64,
    #[serde(default = "default_max_rows")]
    pub max_rows: usize,
}

impl Default for BatchWritesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            __force_allow_embedded_batch_writes: false,
            flush_interval_ms: default_flush_interval_ms(),
            max_rows: default_max_rows(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExportConfig {
    #[serde(default)]
    pub otlp: OtlpConfig,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OtlpConfig {
    #[serde(default)]
    pub traces: OtlpTracesConfig,
}

impl OtlpConfig {
    /// Attaches usage inference to the model provider span (if traces are enabled).
    /// This is used for both streaming and non-streaming requests.
    pub fn apply_usage_to_model_provider_span(&self, span: &Span, usage: &Usage) {
        if self.traces.enabled {
            match self.traces.format {
                OtlpTracesFormat::OpenTelemetry => {
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
                OtlpTracesFormat::OpenInference => {
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
            }
        }
    }

    /// Marks a span as being an OpenInference 'CHAIN' span.
    /// We use this for function/variant/model spans (but not model provider spans).
    /// At the moment, there doesn't seem to be a similar concept in the OpenTelemetry GenAI semantic conventions.
    pub fn mark_openinference_chain_span(&self, span: &Span) {
        if self.traces.enabled {
            match self.traces.format {
                OtlpTracesFormat::OpenInference => {
                    span.set_attribute("openinference.span.kind", "CHAIN");
                }
                OtlpTracesFormat::OpenTelemetry => {}
            }
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OtlpTracesConfig {
    /// Enable OpenTelemetry traces export to the configured OTLP endpoint (configured via OTLP environment variables)
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub format: OtlpTracesFormat,
    /// Extra headers to include in OTLP export requests (can be overridden by dynamic headers at request time)
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "lowercase")]
pub enum OtlpTracesFormat {
    /// Sets 'gen_ai' attributes based on the OpenTelemetry GenAI semantic conventions:
    /// https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai
    #[default]
    OpenTelemetry,
    // Sets attributes based on the OpenInference semantic conventions:
    // https://github.com/Arize-ai/openinference/blob/main/spec/llm_spans.md
    OpenInference,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct MetricConfig {
    pub r#type: MetricConfigType,
    pub optimize: MetricConfigOptimize,
    pub level: MetricConfigLevel,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Copy, Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum MetricConfigType {
    Boolean,
    Float,
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum MetricConfigOptimize {
    Min,
    Max,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

impl MetricConfigLevel {
    pub fn inference_column_name(&self) -> &'static str {
        match self {
            MetricConfigLevel::Inference => "id",
            MetricConfigLevel::Episode => "episode_id",
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
        // Build a matcher from the glob pattern
        let matcher = globset::Glob::new(&glob)
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
pub struct RuntimeOverlay {
    pub gateway: UninitializedGatewayConfig,
    pub postgres: PostgresConfig,
    pub rate_limiting: UninitializedRateLimitingConfig,
    pub object_store_info: Option<ObjectStoreInfo>,
}

impl RuntimeOverlay {
    /// Create a RuntimeOverlay from a live Config.
    /// Uses destructuring to ensure compile-time errors if fields are added/removed.
    pub fn from_config(config: &Config) -> Self {
        // Destructure GatewayConfig to ensure all fields are handled
        let GatewayConfig {
            bind_address,
            observability,
            debug,
            template_filesystem_access,
            export,
            base_path,
            unstable_error_json,
            unstable_disable_feedback_target_validation,
            disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference,
            auth,
            global_outbound_http_timeout,
            relay,
            metrics,
        } = &config.gateway;

        Self {
            gateway: UninitializedGatewayConfig {
                bind_address: *bind_address,
                observability: observability.clone(),
                debug: *debug,
                template_filesystem_access: Some(template_filesystem_access.clone()),
                export: export.clone(),
                base_path: base_path.clone(),
                unstable_disable_feedback_target_validation:
                    *unstable_disable_feedback_target_validation,
                unstable_error_json: *unstable_error_json,
                disable_pseudonymous_usage_analytics: *disable_pseudonymous_usage_analytics,
                fetch_and_encode_input_files_before_inference: Some(
                    *fetch_and_encode_input_files_before_inference,
                ),
                auth: auth.clone(),
                global_outbound_http_timeout_ms: Some(
                    global_outbound_http_timeout.num_milliseconds() as u64,
                ),
                relay: relay.as_ref().map(|relay| relay.original_config.clone()),
                metrics: metrics.clone(),
            },
            postgres: config.postgres.clone(),
            rate_limiting: UninitializedRateLimitingConfig::from(&config.rate_limiting),
            object_store_info: config.object_store_info.clone(),
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
    postgres: PostgresConfig,
    rate_limiting: UninitializedRateLimitingConfig,

    snapshot: ConfigSnapshot,
    /// All functions (user-defined + built-in), loaded and ready to use
    functions: HashMap<String, Arc<FunctionConfig>>,
    gateway_config: GatewayConfig,
    object_store_info: Option<ObjectStoreInfo>,
}

/// Processes the config input (fresh TOML or snapshot) and returns all the fields
/// needed by load_from_toml, avoiding partial moves of UninitializedConfig.
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

            // Deserialize the TOML table into UninitializedConfig
            let mut config = UninitializedConfig::try_from(table)?;

            // Validate that user functions don't use tensorzero:: prefix
            for name in config.functions.keys() {
                if name.starts_with("tensorzero::") {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "User-defined function name cannot start with 'tensorzero::': {name}"
                        ),
                    }));
                }
            }

            // Inject built-in functions into the config (SINGLE INJECTION POINT)
            let built_in_functions = built_in::get_all_built_in_functions()?;
            config.functions.extend(built_in_functions);

            // Destructure the config now that we've added built-in functions
            let UninitializedConfig {
                gateway,
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
            } = config.clone();

            // Load ALL functions (user + built-in)
            let all_functions = functions
                .into_iter()
                .map(|(name, func_config)| {
                    func_config
                        .load(&name, &metrics)
                        .map(|c| (name, Arc::new(c)))
                })
                .collect::<Result<HashMap<String, Arc<FunctionConfig>>, Error>>()?;

            let object_store_info = ObjectStoreInfo::new(object_storage)?;
            let gateway_config = gateway.load(object_store_info.as_ref())?;

            // Initialize templates from ALL functions (including built-in)
            let all_template_paths = Config::get_templates(&all_functions)?;
            if gateway_config.template_filesystem_access.enabled {
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
            let snapshot = ConfigSnapshot::new(config, extra_templates.clone())?;

            Ok(ProcessedConfigInput {
                tools,
                models,
                embedding_models,
                metrics,
                evaluations,
                provider_types,
                optimizers,
                postgres,
                rate_limiting,
                snapshot,
                functions: all_functions,
                gateway_config,
                object_store_info,
            })
        }
        ConfigInput::Snapshot {
            snapshot,
            runtime_overlay,
        } => {
            let original_snapshot = *snapshot;

            // Destructure overlay to ensure all fields are handled (compile error if field added/removed)
            let RuntimeOverlay {
                gateway: overlay_gateway,
                postgres: overlay_postgres,
                rate_limiting: overlay_rate_limiting,
                object_store_info: overlay_object_store_info,
            } = *runtime_overlay;

            // Destructure uninit_config to overlay specific fields
            let UninitializedConfig {
                gateway: _,        // replaced by overlay
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
            } = original_snapshot
                .config
                .clone()
                .try_into()
                .map_err(|e: &'static str| {
                    Error::new(ErrorDetails::Config {
                        message: e.to_string(),
                    })
                })?;

            // Reconstruct with overlaid values for snapshot hash computation
            let overlaid_config = UninitializedConfig {
                gateway: overlay_gateway.clone(),
                postgres: overlay_postgres.clone(),
                rate_limiting: overlay_rate_limiting.clone(),
                object_storage: overlay_object_store_info
                    .as_ref()
                    .map(|info| info.kind.clone()),
                models: models.clone(),
                embedding_models: embedding_models.clone(),
                functions: functions.clone(),
                metrics: metrics.clone(),
                tools: tools.clone(),
                evaluations: evaluations.clone(),
                provider_types: provider_types.clone(),
                optimizers: optimizers.clone(),
            };

            let extra_templates = original_snapshot.extra_templates.clone();
            let snapshot = ConfigSnapshot::new(overlaid_config, extra_templates.clone())?;

            // Load all functions from the snapshot (built-in functions are already included)
            let all_functions = functions
                .into_iter()
                .map(|(name, func_config)| {
                    func_config
                        .load(&name, &metrics)
                        .map(|c| (name, Arc::new(c)))
                })
                .collect::<Result<HashMap<String, Arc<FunctionConfig>>, Error>>()?;

            // Use the overlay object store info directly instead of creating a new one
            let gateway_config = overlay_gateway.load(overlay_object_store_info.as_ref())?;
            // Initialize templates from ALL functions (including built-in)
            let all_template_paths = Config::get_templates(&all_functions)?;
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
                postgres: overlay_postgres,
                rate_limiting: overlay_rate_limiting,
                // unused
                snapshot,
                functions: all_functions,
                gateway_config,
                object_store_info: overlay_object_store_info,
            })
        }
    }
}

/// In e2e test mode, we skip credential validation by default.
/// This can be overridden by setting the `TENSORZERO_E2E_CREDENTIAL_VALIDATION` environment variable to `1`.
/// Outside of e2e test mode, we leave the behavior unchanged (other parts of the codebase might still
/// skip credential validation, e.g. when running in relay mode).
pub fn e2e_skip_credential_validation() -> bool {
    cfg!(any(test, feature = "e2e_tests"))
        && !std::env::var("TENSORZERO_E2E_CREDENTIAL_VALIDATION").is_ok_and(|x| x == "1")
}

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
            with_skip_credential_validation(Box::pin(Self::load_from_toml(ConfigInput::Snapshot {
                snapshot: Box::new(snapshot),
                runtime_overlay: Box::new(runtime_overlay),
            })))
            .await?
        } else {
            Box::pin(Self::load_from_toml(ConfigInput::Snapshot {
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
            with_skip_credential_validation(Box::pin(Self::load_from_toml(ConfigInput::Fresh(
                globbed_config.table,
            ))))
            .await?
        } else {
            Box::pin(Self::load_from_toml(ConfigInput::Fresh(
                globbed_config.table,
            )))
            .await?
        };

        if validate_credentials && let Some(object_store) = &unwritten_config.object_store_info {
            object_store.verify().await?;
        }

        Ok(unwritten_config)
    }

    /// Loads and initializes a config from a parsed TOML table.
    ///
    /// This is the core config loading function that transforms a merged TOML table into
    /// a fully validated and initialized `Config`, paired with a `ConfigSnapshot` for database storage.
    ///
    /// # Config Loading Flow
    ///
    /// This function performs the following steps:
    ///
    /// 1. **Parse to UninitializedConfig**: Deserialize the TOML table into an `UninitializedConfig`,
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
    /// let unwritten_config = Config::load_from_toml(table).await?;
    /// let clickhouse = setup_clickhouse(&unwritten_config).await?;
    /// let config = unwritten_config.into_config(&clickhouse).await?;
    /// ```
    async fn load_from_toml(input: ConfigInput) -> Result<UnwrittenConfig, Error> {
        let mut templates = TemplateConfig::new();
        let ProcessedConfigInput {
            tools,
            models,
            embedding_models,
            metrics,
            evaluations: uninitialized_evaluations,
            provider_types,
            optimizers: uninitialized_optimizers,
            postgres,
            rate_limiting,
            snapshot,
            functions,
            gateway_config,
            object_store_info,
        } = process_config_input(input, &mut templates).await?;

        let http_client = TensorzeroHttpClient::new(gateway_config.global_outbound_http_timeout)?;
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
                )
                .await
                .map(|c| (name, c))
        }))
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();

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

        let mut config = Config {
            gateway: gateway_config,
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
            hash: snapshot.hash.clone(),
        };

        // Validate the config
        config.validate().await?;

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

        Ok(UnwrittenConfig::new(config, snapshot))
    }

    /// Validate the config
    #[instrument(skip_all)]
    async fn validate(&mut self) -> Result<(), Error> {
        if self.gateway.observability.batch_writes.enabled
            && self.gateway.observability.async_writes
        {
            return Err(ErrorDetails::Config {
                message: "Batch writes and async writes cannot be enabled at the same time"
                    .to_string(),
            }
            .into());
        }
        if self.gateway.observability.batch_writes.flush_interval_ms == 0 {
            return Err(ErrorDetails::Config {
                message: "Batch writes flush interval must be greater than 0".to_string(),
            }
            .into());
        }
        if self.gateway.observability.batch_writes.max_rows == 0 {
            return Err(ErrorDetails::Config {
                message: "Batch writes max rows must be greater than 0".to_string(),
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
    Snapshot {
        snapshot: Box<ConfigSnapshot>,
        runtime_overlay: Box<RuntimeOverlay>,
    },
}

/// Writes the config snapshot to the `ConfigSnapshot` table.
/// Takes special care to retain the created_at if there was already a row
/// that had the same hash. Tags are merged with existing tags using mapUpdate.
///
/// This function is gated behind the `TENSORZERO_FF_WRITE_CONFIG_SNAPSHOT=1` feature flag.
/// If the env var is not set to "1", the write is skipped.
pub async fn write_config_snapshot(
    clickhouse: &ClickHouseConnectionInfo,
    snapshot: ConfigSnapshot,
) -> Result<(), Error> {
    // Define the row structure for serialization
    #[derive(Serialize)]
    struct ConfigSnapshotRow<'a> {
        config: &'a str,
        extra_templates: &'a HashMap<String, String>,
        hash: SnapshotHash,
        tensorzero_version: &'static str,
        tags: &'a HashMap<String, String>,
    }

    // Get the pre-computed hash
    let version_hash = snapshot.hash.clone();

    // Serialize StoredConfig to TOML for storage
    let config_string = toml::to_string(&snapshot.config).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize config snapshot: {e}"),
        })
    })?;

    // Create the row
    let row = ConfigSnapshotRow {
        config: &config_string,
        extra_templates: &snapshot.extra_templates,
        hash: version_hash.clone(),
        tensorzero_version: TENSORZERO_VERSION,
        tags: &snapshot.tags,
    };

    // Serialize to JSON
    let json_data = serde_json::to_string(&row).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize config snapshot: {e}"),
        })
    })?;

    // Create the external data info
    let external_data = ExternalDataInfo {
        external_data_name: "new_data".to_string(),
        structure: "config String, extra_templates Map(String, String), hash String, tensorzero_version String, tags Map(String, String)".to_string(),
        format: "JSONEachRow".to_string(),
        data: json_data,
    };

    // Create the query with subquery to preserve created_at and merge tags
    // We use any() aggregate function to handle the case when no row exists (returns default value)
    let query = format!(
        r"INSERT INTO ConfigSnapshot
(config, extra_templates, hash, tensorzero_version, tags, created_at, last_used)
SELECT
    new_data.config,
    new_data.extra_templates,
    toUInt256(new_data.hash) as hash,
    new_data.tensorzero_version,
    mapUpdate(
        (SELECT any(tags) FROM ConfigSnapshot FINAL WHERE hash = toUInt256('{version_hash}')),
        new_data.tags
    ) as tags,
    ifNull((SELECT any(created_at) FROM ConfigSnapshot FINAL WHERE hash = toUInt256('{version_hash}')), now64()) as created_at,
    now64() as last_used
FROM new_data"
    );

    // Execute the query
    clickhouse
        .run_query_with_external_data(external_data, query)
        .await?;

    Ok(())
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
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedConfig {
    #[serde(default)]
    pub gateway: UninitializedGatewayConfig,
    #[serde(default)]
    pub postgres: PostgresConfig,
    #[serde(default)]
    pub rate_limiting: UninitializedRateLimitingConfig,
    pub object_storage: Option<StorageKind>,
    #[serde(default)]
    pub models: HashMap<Arc<str>, UninitializedModelConfig>, // model name => model config
    #[serde(default)]
    pub embedding_models: HashMap<Arc<str>, UninitializedEmbeddingModelConfig>, // embedding model name => embedding model config
    #[serde(default)]
    pub functions: HashMap<String, UninitializedFunctionConfig>, // function name => function config
    #[serde(default)]
    pub metrics: HashMap<String, MetricConfig>, // metric name => metric config
    #[serde(default)]
    pub tools: HashMap<String, UninitializedToolConfig>, // tool name => tool config
    #[serde(default)]
    pub evaluations: HashMap<String, UninitializedEvaluationConfig>, // evaluation name => evaluation
    #[serde(default)]
    pub provider_types: ProviderTypesConfig, // global configuration for all model providers of a particular type
    #[serde(default)]
    pub optimizers: HashMap<String, UninitializedOptimizerInfo>, // optimizer name => optimizer config
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
struct UninitializedGlobbedConfig {
    table: toml::Table,
}

impl UninitializedConfig {
    /// Read all of the globbed config files from disk, and merge them into a single `UninitializedGlobbedConfig`
    fn read_toml_config(
        glob: &ConfigFileGlob,
        allow_empty_glob: bool,
    ) -> Result<UninitializedGlobbedConfig, Error> {
        let table = SpanMap::from_glob(glob, allow_empty_glob)?;
        Ok(UninitializedGlobbedConfig { table })
    }
}

/// Deserialize a TOML table into `UninitializedConfig`
impl TryFrom<toml::Table> for UninitializedConfig {
    type Error = Error;

    fn try_from(table: toml::Table) -> Result<Self, Self::Error> {
        match serde_path_to_error::deserialize(table) {
            Ok(config) => Ok(config),
            Err(e) => {
                let path = e.path().clone();
                Err(Error::new(ErrorDetails::Config {
                    // Extract the underlying message from the toml error, as
                    // the path-tracking from the toml crate will be incorrect
                    message: format!("{}: {}", path, e.into_inner().message()),
                }))
            }
        }
    }
}

#[derive(Clone, Debug, TensorZeroDeserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum UninitializedFunctionConfig {
    Chat(UninitializedFunctionConfigChat),
    Json(UninitializedFunctionConfigJson),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct UninitializedSchema {
    path: ResolvedTomlPathData,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[serde(transparent)]
pub struct UninitializedSchemas {
    inner: HashMap<String, UninitializedSchema>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub description: Option<String>,
    pub experimentation: Option<UninitializedExperimentationConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedFunctionConfigJson {
    pub variants: HashMap<String, UninitializedVariantInfo>, // variant name => variant config
    pub system_schema: Option<ResolvedTomlPathData>,
    pub user_schema: Option<ResolvedTomlPathData>,
    pub assistant_schema: Option<ResolvedTomlPathData>,
    #[serde(default)]
    pub schemas: UninitializedSchemas,
    pub output_schema: Option<ResolvedTomlPathData>, // schema will default to {} if not specified
    #[serde(default)]
    pub description: Option<String>,
    pub experimentation: Option<UninitializedExperimentationConfig>,
}

/// Holds all of the schemas used by a chat completion function.
/// These are used by variants to construct a `TemplateWithSchema`
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
            non_streaming: NonStreamingTimeouts {
                total_ms: Some(timeout_ms),
            },
            streaming: StreamingTimeouts {
                ttft_ms: Some(timeout_ms),
            },
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

impl UninitializedFunctionConfig {
    pub fn load(
        self,
        function_name: &str,
        metrics: &HashMap<String, MetricConfig>,
    ) -> Result<FunctionConfig, Error> {
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
                    .map(|config| config.load(&variants, metrics))
                    .transpose()?
                    .unwrap_or_else(|| ExperimentationConfig::legacy_from_variants_map(&variants));
                Ok(FunctionConfig::Chat(FunctionConfigChat {
                    variants,
                    schemas: schema_data,
                    tools: params.tools,
                    tool_choice: params.tool_choice,
                    parallel_tool_calls: params.parallel_tool_calls,
                    description: params.description,
                    all_explicit_templates_names: all_template_names,
                    experimentation,
                }))
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
                    .map(|config| config.load(&variants, metrics))
                    .transpose()?
                    .unwrap_or_else(|| ExperimentationConfig::legacy_from_variants_map(&variants));
                Ok(FunctionConfig::Json(FunctionConfigJson {
                    variants,
                    schemas: schema_data,
                    output_schema,
                    json_mode_tool_call_config,
                    description: params.description,
                    all_explicit_template_names: all_template_names,
                    experimentation,
                }))
            }
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
// We don't use `#[serde(deny_unknown_fields)]` here - it needs to go on 'UninitializedVariantConfig',
// since we use `#[serde(flatten)]` on the `inner` field.
pub struct UninitializedVariantInfo {
    #[serde(flatten)]
    pub inner: UninitializedVariantConfig,
    #[serde(default)]
    pub timeouts: Option<TimeoutsConfig>,
}

/// NOTE: Contains deprecated variant `ChainOfThought` (#5298 / 2026.2+)
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, JsonSchema, TensorZeroDeserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedToolConfig {
    pub description: String,
    pub parameters: ResolvedTomlPathData,
    pub name: Option<String>,
    #[serde(default)]
    pub strict: bool,
}

impl UninitializedToolConfig {
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct PathWithContents {
    #[cfg_attr(feature = "ts-bindings", ts(type = "string"))]
    pub path: ResolvedTomlPathData,
    pub contents: String,
}

impl PathWithContents {
    pub fn from_path(path: ResolvedTomlPathData) -> Result<Self, Error> {
        let contents = path.data().to_string();
        Ok(Self { path, contents })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct PostgresConfig {
    pub enabled: Option<bool>,
    #[serde(default = "default_connection_pool_size")]
    pub connection_pool_size: u32,
    /// Retention period in days for inference tables (chat_inferences, json_inferences).
    /// If set, old partitions beyond this age will be dropped by pg_cron.
    /// If not set, partitions are retained indefinitely.
    ///
    /// TODO(#5764): Document clearly in user-facing docs:
    /// - WARNING: When first set (or lowered), pg_cron will immediately start dropping
    ///   partitions older than this value on its next daily run. This is irreversible.
    /// - Clients are responsible for managing their own data backups before enabling retention.
    /// - Recommend testing in non-production first and starting with a longer period.
    pub inference_retention_days: Option<u32>,
}

fn default_connection_pool_size() -> u32 {
    20
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            enabled: None,
            connection_pool_size: 20,
            inference_retention_days: None,
        }
    }
}
