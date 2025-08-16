use futures::future::try_join_all;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, PutPayload};
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyKeyError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::IntoPyObjectExt;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tensorzero_derive::TensorZeroDeserialize;
use tracing::instrument;

use crate::config_parser::gateway::{GatewayConfig, UninitializedGatewayConfig};
use crate::config_parser::path::TomlRelativePath;
use crate::config_parser::span_map::SpanMap;
use crate::embeddings::{EmbeddingModelTable, UninitializedEmbeddingModelConfig};
use crate::endpoints::inference::DEFAULT_FUNCTION_NAME;
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::error::{Error, ErrorDetails};
use crate::evaluations::{EvaluationConfig, UninitializedEvaluationConfig};
use crate::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
#[cfg(feature = "pyo3")]
use crate::function::{FunctionConfigChatPyClass, FunctionConfigJsonPyClass};
use crate::inference::types::storage::StorageKind;
use crate::jsonschema_util::StaticJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::{ModelConfig, ModelTable, UninitializedModelConfig};
use crate::model_table::{CowNoClone, ShorthandModelConfig};
use crate::optimization::{OptimizerInfo, UninitializedOptimizerInfo};
use crate::tool::{create_implicit_tool_call_config, StaticToolConfig, ToolChoice};
use crate::variant::best_of_n_sampling::UninitializedBestOfNSamplingConfig;
use crate::variant::chain_of_thought::UninitializedChainOfThoughtConfig;
use crate::variant::chat_completion::UninitializedChatCompletionConfig;
use crate::variant::dicl::UninitializedDiclConfig;
use crate::variant::mixture_of_n::UninitializedMixtureOfNConfig;
use crate::variant::{Variant, VariantConfig, VariantInfo};
use std::error::Error as StdError;

pub mod gateway;
pub mod path;
mod span_map;
#[cfg(test)]
mod tests;

tokio::task_local! {
    /// When set, we skip performing credential validation in model providers
    /// This is used when running in e2e test mode, and by the 'evaluations' binary
    /// We need to access this from async code (e.g. when looking up GCP SDK credentials),
    /// so this needs to be a tokio task-local (as a task may be moved between threads)
    pub(crate) static SKIP_CREDENTIAL_VALIDATION: ();
}

pub fn skip_credential_validation() -> bool {
    // tokio::task_local doesn't have an 'is_set' method, so we call 'try_with'
    // (which returns an `Err` if the task-local is not set)
    SKIP_CREDENTIAL_VALIDATION.try_with(|()| ()).is_ok()
}

#[derive(Debug, Default, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct Config {
    pub gateway: GatewayConfig,
    pub models: ModelTable,                    // model name => model config
    pub embedding_models: EmbeddingModelTable, // embedding model name => embedding model config
    pub functions: HashMap<String, Arc<FunctionConfig>>, // function name => function config
    pub metrics: HashMap<String, MetricConfig>, // metric name => metric config
    pub tools: HashMap<String, Arc<StaticToolConfig>>, // tool name => tool config
    pub evaluations: HashMap<String, Arc<EvaluationConfig>>, // evaluation name => evaluation config
    #[serde(skip)]
    pub templates: TemplateConfig<'static>,
    pub object_store_info: Option<ObjectStoreInfo>,
    pub provider_types: ProviderTypesConfig,
    pub optimizers: HashMap<String, OptimizerInfo>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct NonStreamingTimeouts {
    #[serde(default)]
    /// The total time allowed for the non-streaming request to complete.
    pub total_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct StreamingTimeouts {
    #[serde(default)]
    /// The time allowed for the first token to be produced.
    pub ttft_ms: Option<u64>,
}

/// Configures the timeouts for both streaming and non-streaming requests.
/// This can be attached to various other configs (e.g. variants, models, model providers)
#[derive(Clone, Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct TimeoutsConfig {
    #[serde(default)]
    pub non_streaming: NonStreamingTimeouts,
    #[serde(default)]
    pub streaming: StreamingTimeouts,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct TemplateFilesystemAccess {
    /// If `true`, allow minijinja to read from the filesystem (within the tree of the config file) for '{% include %}'
    /// Defaults to `false`
    #[serde(default)]
    enabled: bool,
    base_path: Option<TomlRelativePath>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GCPProviderTypeConfig {
    #[serde(default)]
    pub batch: Option<GCPBatchConfigType>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "storage_type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
#[serde(deny_unknown_fields)]
pub enum GCPBatchConfigType {
    // In the future, we'll want to allow explicitly setting 'none' at the model provider level,
    // to override the global provider-types batch config.
    None,
    CloudStorage(GCPBatchConfigCloudStorage),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GCPBatchConfigCloudStorage {
    pub input_uri_prefix: String,
    pub output_uri_prefix: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ObjectStoreInfo {
    // This will be `None` if we have `StorageKind::Disabled`
    #[serde(skip)]
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
                                tracing::warn!("Filesystem object store path does not exist: {path}. Treating object store as unconfigured");
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
                        message: "S3_ACCESS_KEY_ID is set but S3_SECRET_ACCESS_KEY is not. Please set either both or none".to_string()
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
                    tracing::warn!("`AWS_ALLOW_HTTP` is set to `true` - this is insecure, and should only be used when running a local S3-compatible object store");
                    if allow_http.is_some() {
                        tracing::info!("Config has `[object_storage.allow_http]` present - this takes precedence over `AWS_ALLOW_HTTP`");
                    }
                }
                if let Some(allow_http) = *allow_http {
                    if allow_http {
                        tracing::warn!("`[object_storage.allow_http]` is set to `true` - this is insecure, and should only be used when running a local S3-compatible object store");
                    }
                    builder = builder.with_allow_http(allow_http);
                }

                if let (Some(bucket_name), Some(endpoint)) = (bucket_name, endpoint) {
                    if endpoint.ends_with(bucket_name) {
                        tracing::warn!("S3-compatible object endpoint `{endpoint}` ends with configured bucket_name `{bucket_name}`. This may be incorrect - if the gateway fails to start, consider setting `bucket_name = null`");
                    }
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
            tracing::info!("Verifying that [object_storage] is configured correctly (writing .tensorzero-validate)");
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

#[derive(Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ObservabilityConfig {
    pub enabled: Option<bool>,
    #[serde(default)]
    pub async_writes: bool,
    #[serde(default)]
    pub batch_writes: BatchWritesConfig,
    /// If `true`, then we skip checking/applying migrations if the `TensorZeroMigration` table
    /// contains exactly the migrations that we expect to have run.
    #[serde(default)]
    pub skip_completed_migrations: bool,
}

fn default_flush_interval_ms() -> u64 {
    100
}

fn default_max_rows() -> usize {
    1000
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct BatchWritesConfig {
    pub enabled: bool,
    #[serde(default = "default_flush_interval_ms")]
    pub flush_interval_ms: u64,
    #[serde(default = "default_max_rows")]
    pub max_rows: usize,
}

impl Default for BatchWritesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            flush_interval_ms: default_flush_interval_ms(),
            max_rows: default_max_rows(),
        }
    }
}

#[derive(Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ExportConfig {
    #[serde(default)]
    pub otlp: OtlpConfig,
}

#[derive(Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct OtlpConfig {
    #[serde(default)]
    pub traces: OtlpTracesConfig,
}

#[derive(Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct OtlpTracesConfig {
    /// Enable OpenTelemetry traces export to the configured OTLP endpoint (configured via OTLP environment variables)
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct MetricConfig {
    pub r#type: MetricConfigType,
    pub optimize: MetricConfigOptimize,
    pub level: MetricConfigLevel,
}

#[derive(Copy, Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum MetricConfigOptimize {
    Min,
    Max,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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
//    deserialized into a `TomlRelativePath`. If we forget to remap any parts of the config
//    in `resolve_toml_relative_paths`, then deserializing the `TomlRelativePath` will fail
//    (rather than succeed with an incorrect path).
//
// At this point, we have a `DeTable` that can be successfully deserialized into an `UninitializedConfig`.
// From this point onward, none of the config-handling code needs to interact with globs, remapped config files,
// or even be aware of whether or not we have multiple config files. All path access goes through `TomlRelativePath`,
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

    pub fn new(glob: String) -> Result<Self, Error> {
        let mut glob_paths = glob::glob(&glob)
            .map_err(|e| {
                Error::new(ErrorDetails::Glob {
                    glob: glob.to_string(),
                    message: e.to_string(),
                })
            })?
            .map(|path_res| {
                path_res.map_err(|e| {
                    Error::new(ErrorDetails::Glob {
                        glob: glob.to_string(),
                        message: format!("Error processing globbed path: `{e}`"),
                    })
                })
            })
            .collect::<Result<Vec<PathBuf>, Error>>()?;
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

impl Config {
    pub async fn load_and_verify_from_path(config_glob: &ConfigFileGlob) -> Result<Config, Error> {
        Self::load_from_path_optional_verify_credentials(config_glob, true).await
    }

    pub async fn load_from_path_optional_verify_credentials(
        config_glob: &ConfigFileGlob,
        validate_credentials: bool,
    ) -> Result<Config, Error> {
        let globbed_config = UninitializedConfig::read_toml_config(config_glob)?;
        let config = if cfg!(feature = "e2e_tests") || !validate_credentials {
            SKIP_CREDENTIAL_VALIDATION
                .scope(
                    (),
                    Self::load_from_toml(globbed_config.table, &globbed_config.span_map),
                )
                .await?
        } else {
            Self::load_from_toml(globbed_config.table, &globbed_config.span_map).await?
        };

        if validate_credentials {
            if let Some(object_store) = &config.object_store_info {
                object_store.verify().await?;
            }
        }

        Ok(config)
    }

    async fn load_from_toml(table: toml::Table, span_map: &SpanMap) -> Result<Config, Error> {
        if table.is_empty() {
            tracing::info!("Config file is empty, so only default functions will be available.");
        }
        let uninitialized_config = UninitializedConfig::try_from(table)?;

        let templates = TemplateConfig::new();

        let functions = uninitialized_config
            .functions
            .into_iter()
            .map(|(name, config)| config.load(&name).map(|c| (name, Arc::new(c))))
            .collect::<Result<HashMap<String, Arc<FunctionConfig>>, Error>>()?;

        let tools = uninitialized_config
            .tools
            .into_iter()
            .map(|(name, config)| config.load(name.clone()).map(|c| (name, Arc::new(c))))
            .collect::<Result<HashMap<String, Arc<StaticToolConfig>>, Error>>()?;

        let models = try_join_all(uninitialized_config.models.into_iter().map(
            |(name, config)| async {
                config
                    .load(&name, &uninitialized_config.provider_types)
                    .await
                    .map(|c| (name, c))
            },
        ))
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();

        let embedding_models = try_join_all(uninitialized_config.embedding_models.into_iter().map(
            |(name, config)| async {
                config
                    .load(&uninitialized_config.provider_types)
                    .await
                    .map(|c| (name, c))
            },
        ))
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();

        let object_store_info = ObjectStoreInfo::new(uninitialized_config.object_storage)?;
        let optimizers = try_join_all(
            uninitialized_config
                .optimizers
                .into_iter()
                .map(|(name, config)| async { config.load().await.map(|c| (name, c)) }),
        )
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();

        let mut config = Config {
            gateway: uninitialized_config.gateway.load()?,
            models: models.try_into().map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to load models: {e}"),
                })
            })?,
            embedding_models: embedding_models.try_into().map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to load embedding models: {e}"),
                })
            })?,
            functions,
            metrics: uninitialized_config.metrics,
            tools,
            evaluations: HashMap::new(),
            templates,
            object_store_info,
            provider_types: uninitialized_config.provider_types,
            optimizers,
        };

        // Initialize the templates
        let template_paths = config.get_templates();
        let template_fs_base_path = if config.gateway.template_filesystem_access.enabled {
            if let Some(base_path) = &config.gateway.template_filesystem_access.base_path {
                Some(base_path.get_real_path().map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to get real path for base path: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"),
                    })
                })?.to_owned())
            } else if let Some(single_file) = span_map.get_single_file() {
                tracing::warn!("Deprecation warning: `[gateway.template_filesystem_access.base_path]` is not set, using config file base path. Please specify `[gateway.template_filesystem_access.base_path]`");
                Some(
                    single_file
                        .parent()
                        .ok_or_else(|| {
                            Error::new(ErrorDetails::Config {
                                message: format!(
                                    "Failed to determine base path for config file `{}`",
                                    single_file.to_string_lossy()
                                ),
                            })
                        })?
                        .to_owned(),
                )
            } else {
                return Err(ErrorDetails::Config {
                    message: "`[gateway.template_filesystem_access]` is enabled, but `[gateway.template_filesystem_access.base_path]` is not set.".to_string()
                }
                .into());
            }
        } else {
            None
        };
        config
            .templates
            .initialize(template_paths, template_fs_base_path.as_deref())?;

        // Validate the config
        config.validate().await?;

        // We add the evaluations after validation since we will be writing tensorzero:: functions to the functions map
        // and tensorzero:: metrics to the metrics map
        let mut evaluations = HashMap::new();
        for (name, evaluation_config) in uninitialized_config.evaluations {
            let (evaluation_config, evaluation_function_configs, evaluation_metric_configs) =
                evaluation_config.load(&config.functions, &name)?;
            evaluations.insert(name, Arc::new(EvaluationConfig::Static(evaluation_config)));
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
                for variant in evaluation_function_config.variants().values() {
                    for template in variant.get_all_template_paths() {
                        config.templates.add_template(
                            template.path.get_template_key(),
                            template.contents.clone(),
                        )?;
                    }
                }
                evaluation_function_config
                    .validate(
                        &config.tools,
                        &mut config.models,
                        &config.embedding_models,
                        &config.templates,
                        &evaluation_function_name,
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

        Ok(config)
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
        for (function_name, function) in &self.functions {
            if function_name.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "Function name cannot start with 'tensorzero::': {function_name}"
                    ),
                }
                .into());
            }
            function
                .validate(
                    &self.tools,
                    &mut self.models, // NOTE: in here there might be some models created using shorthand initialization
                    &self.embedding_models,
                    &self.templates,
                    function_name,
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
            model.validate(model_name)?;
        }

        for embedding_model_name in self.embedding_models.keys() {
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
        if function_name == DEFAULT_FUNCTION_NAME {
            Ok(Cow::Owned(Arc::new(FunctionConfig::Chat(
                FunctionConfigChat {
                    variants: HashMap::new(),
                    schemas: SchemaData::default(),
                    tools: vec![],
                    tool_choice: ToolChoice::None,
                    parallel_tool_calls: None,
                    description: None,
                },
            ))))
        } else {
            Ok(Cow::Borrowed(
                self.functions.get(function_name).ok_or_else(|| {
                    Error::new(ErrorDetails::UnknownFunction {
                        name: function_name.to_string(),
                    })
                })?,
            ))
        }
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
    ) -> Result<CowNoClone<'a, ModelConfig>, Error> {
        self.models.get(model_name).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: model_name.to_string(),
            })
        })
    }

    /// Get all templates from the config
    /// The HashMap returned is a mapping from the path as given in the TOML file
    /// (relative to the directory containing the TOML file) to the file contents.
    /// The former path is used as the name of the template for retrieval by variants later.
    pub fn get_templates(&self) -> HashMap<String, String> {
        let mut templates = HashMap::new();

        for function in self.functions.values() {
            for variant in function.variants().values() {
                let variant_template_paths = variant.get_all_template_paths();
                for path in variant_template_paths {
                    templates.insert(path.path.get_template_key(), path.contents.clone());
                }
            }
        }
        templates
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
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedConfig {
    #[serde(default)]
    pub gateway: UninitializedGatewayConfig,
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
    pub object_storage: Option<StorageKind>,
    #[serde(default)]
    pub optimizers: HashMap<String, UninitializedOptimizerInfo>, // optimizer name => optimizer config
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ProviderTypesConfig {
    #[serde(default)]
    pub gcp_vertex_gemini: Option<GCPProviderTypeConfig>,
}

/// The result of parsing all of the globbed config files,
/// and merging them into a single `toml::Table`
struct UninitializedGlobbedConfig {
    table: toml::Table,
    span_map: SpanMap,
}

impl UninitializedConfig {
    /// Read all of the globbed config files from disk, and merge them into a single `UninitializedGlobbedConfig`
    fn read_toml_config(glob: &ConfigFileGlob) -> Result<UninitializedGlobbedConfig, Error> {
        let (span_map, table) = SpanMap::from_glob(glob)?;
        Ok(UninitializedGlobbedConfig { table, span_map })
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

#[derive(Debug, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
enum UninitializedFunctionConfig {
    Chat(UninitializedFunctionConfigChat),
    Json(UninitializedFunctionConfigJson),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedFunctionConfigChat {
    variants: HashMap<String, UninitializedVariantInfo>, // variant name => variant config
    system_schema: Option<TomlRelativePath>,
    user_schema: Option<TomlRelativePath>,
    assistant_schema: Option<TomlRelativePath>,
    #[serde(default)]
    tools: Vec<String>, // tool names
    #[serde(default)]
    tool_choice: ToolChoice,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedFunctionConfigJson {
    variants: HashMap<String, UninitializedVariantInfo>, // variant name => variant config
    system_schema: Option<TomlRelativePath>,
    user_schema: Option<TomlRelativePath>,
    assistant_schema: Option<TomlRelativePath>,
    output_schema: Option<TomlRelativePath>, // schema will default to {} if not specified
    #[serde(default)]
    description: Option<String>,
}

/// Holds all of the schemas used by a chat completion function.
/// These are used by variants to construct a `TemplateWithSchema`
#[derive(Debug, Default, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct SchemaData {
    pub user: Option<StaticJSONSchema>,
    pub assistant: Option<StaticJSONSchema>,
    pub system: Option<StaticJSONSchema>,
}

impl UninitializedFunctionConfig {
    pub fn load(self, function_name: &str) -> Result<FunctionConfig, Error> {
        match self {
            UninitializedFunctionConfig::Chat(params) => {
                let system_schema = params
                    .system_schema
                    .map(StaticJSONSchema::from_path)
                    .transpose()?;
                let user_schema = params
                    .user_schema
                    .map(StaticJSONSchema::from_path)
                    .transpose()?;
                let assistant_schema = params
                    .assistant_schema
                    .map(StaticJSONSchema::from_path)
                    .transpose()?;
                let schema_data = SchemaData {
                    user: user_schema,
                    assistant: assistant_schema,
                    system: system_schema,
                };
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| variant.load(&schema_data).map(|v| (name, Arc::new(v))))
                    .collect::<Result<HashMap<_, _>, Error>>()?;
                for (name, variant) in &variants {
                    if let VariantConfig::ChatCompletion(chat_config) = &variant.inner {
                        if chat_config.json_mode.is_some() {
                            return Err(ErrorDetails::Config {
                                message: format!(
                                    "JSON mode is not supported for variant `{name}` (parent function is a chat function)",
                                ),
                            }
                            .into());
                        }
                    }
                }
                Ok(FunctionConfig::Chat(FunctionConfigChat {
                    variants,
                    schemas: schema_data,
                    tools: params.tools,
                    tool_choice: params.tool_choice,
                    parallel_tool_calls: params.parallel_tool_calls,
                    description: params.description,
                }))
            }
            UninitializedFunctionConfig::Json(params) => {
                let system_schema = params
                    .system_schema
                    .map(StaticJSONSchema::from_path)
                    .transpose()?;
                let user_schema = params
                    .user_schema
                    .map(StaticJSONSchema::from_path)
                    .transpose()?;
                let assistant_schema = params
                    .assistant_schema
                    .map(StaticJSONSchema::from_path)
                    .transpose()?;
                let schema_data = SchemaData {
                    user: user_schema,
                    assistant: assistant_schema,
                    system: system_schema,
                };
                let output_schema = match params.output_schema {
                    Some(path) => StaticJSONSchema::from_path(path)?,
                    None => StaticJSONSchema::default(),
                };
                let implicit_tool_call_config =
                    create_implicit_tool_call_config(output_schema.clone());
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| variant.load(&schema_data).map(|v| (name, Arc::new(v))))
                    .collect::<Result<HashMap<_, _>, Error>>()?;

                for (name, variant) in &variants {
                    let mut warn_variant = None;
                    match &variant.inner {
                        VariantConfig::ChatCompletion(chat_config) => {
                            if chat_config.json_mode.is_none() {
                                warn_variant = Some(name.clone());
                            }
                        }
                        VariantConfig::BestOfNSampling(best_of_n_config) => {
                            if best_of_n_config.evaluator.inner.json_mode.is_none() {
                                warn_variant = Some(format!("{name}.evaluator"));
                            }
                        }
                        VariantConfig::MixtureOfN(mixture_of_n_config) => {
                            if mixture_of_n_config.fuser.inner.json_mode.is_none() {
                                warn_variant = Some(format!("{name}.fuser"));
                            }
                        }
                        VariantConfig::Dicl(best_of_n_config) => {
                            if best_of_n_config.json_mode.is_none() {
                                warn_variant = Some(name.clone());
                            }
                        }
                        VariantConfig::ChainOfThought(chain_of_thought_config) => {
                            if chain_of_thought_config.inner.json_mode.is_none() {
                                warn_variant = Some(name.clone());
                            }
                        }
                    }
                    if let Some(warn_variant) = warn_variant {
                        tracing::warn!("Deprecation Warning: `json_mode` is not specified for `[functions.{function_name}.variants.{warn_variant}]` (parent function `{function_name}` is a JSON function), defaulting to `strict`. This field will become required in a future release - see https://github.com/tensorzero/tensorzero/issues/1043 on GitHub for details.");
                    }
                }
                Ok(FunctionConfig::Json(FunctionConfigJson {
                    variants,
                    schemas: schema_data,
                    output_schema,
                    implicit_tool_call_config,
                    description: params.description,
                }))
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
// We don't use `#[serde(deny_unknown_fields)]` here - it needs to go on 'UninitializedVariantConfig',
// since we use `#[serde(flatten)]` on the `inner` field.
pub struct UninitializedVariantInfo {
    #[serde(flatten)]
    pub inner: UninitializedVariantConfig,
    #[serde(default)]
    pub timeouts: Option<TimeoutsConfig>,
}

#[derive(Clone, Debug, TensorZeroDeserialize, Serialize, ts_rs::TS)]
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
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(UninitializedChainOfThoughtConfig),
}

impl UninitializedVariantInfo {
    pub fn load(self, schemas: &SchemaData) -> Result<VariantInfo, Error> {
        let inner = match self.inner {
            UninitializedVariantConfig::ChatCompletion(params) => {
                VariantConfig::ChatCompletion(params.load(schemas)?)
            }
            UninitializedVariantConfig::BestOfNSampling(params) => {
                VariantConfig::BestOfNSampling(params.load(schemas)?)
            }
            UninitializedVariantConfig::Dicl(params) => VariantConfig::Dicl(params.load()?),
            UninitializedVariantConfig::MixtureOfN(params) => {
                VariantConfig::MixtureOfN(params.load(schemas)?)
            }
            UninitializedVariantConfig::ChainOfThought(params) => {
                VariantConfig::ChainOfThought(params.load(schemas)?)
            }
        };
        Ok(VariantInfo {
            inner,
            timeouts: self.timeouts.unwrap_or_default(),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedToolConfig {
    pub description: String,
    pub parameters: TomlRelativePath,
    pub name: Option<String>,
    #[serde(default)]
    pub strict: bool,
}

impl UninitializedToolConfig {
    pub fn load(self, name: String) -> Result<StaticToolConfig, Error> {
        let parameters = StaticJSONSchema::from_path(self.parameters)?;
        Ok(StaticToolConfig {
            name: self.name.unwrap_or(name),
            description: self.description,
            parameters,
            strict: self.strict,
        })
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct PathWithContents {
    #[cfg_attr(test, ts(type = "string"))]
    pub path: TomlRelativePath,
    pub contents: String,
}

impl PathWithContents {
    pub fn from_path(path: TomlRelativePath) -> Result<Self, Error> {
        let contents = path.read()?;
        Ok(Self { path, contents })
    }
}
