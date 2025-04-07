use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, PutPayload};
use scoped_tls::scoped_thread_local;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tensorzero_derive::TensorZeroDeserialize;
use tracing::instrument;

use crate::embeddings::EmbeddingModelTable;
use crate::error::{Error, ErrorDetails};
use crate::evaluations::{EvaluationConfig, UninitializedEvaluationConfig};
use crate::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
use crate::inference::types::storage::StorageKind;
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::minijinja_util::TemplateConfig;
use crate::model::{ModelConfig, ModelTable};
use crate::model_table::{CowNoClone, ShorthandModelConfig};
use crate::tool::{create_implicit_tool_call_config, StaticToolConfig, ToolChoice};
use crate::variant::best_of_n_sampling::UninitializedBestOfNSamplingConfig;
use crate::variant::chat_completion::UninitializedChatCompletionConfig;
use crate::variant::dicl::UninitializedDiclConfig;
use crate::variant::mixture_of_n::UninitializedMixtureOfNConfig;
use crate::variant::{Variant, VariantConfig};
use std::error::Error as StdError;

scoped_thread_local!(pub(crate) static SKIP_CREDENTIAL_VALIDATION: ());

#[derive(Debug, Default)]
pub struct Config<'c> {
    pub gateway: GatewayConfig,
    pub models: ModelTable,                    // model name => model config
    pub embedding_models: EmbeddingModelTable, // embedding model name => embedding model config
    pub functions: HashMap<String, Arc<FunctionConfig>>, // function name => function config
    pub metrics: HashMap<String, MetricConfig>, // metric name => metric config
    pub tools: HashMap<String, Arc<StaticToolConfig>>, // tool name => tool config
    pub evaluations: HashMap<String, Arc<EvaluationConfig>>, // evaluation name => evaluation config
    pub templates: TemplateConfig<'c>,
    pub object_store_info: Option<ObjectStoreInfo>,
}

#[derive(Debug, Default)]
pub struct GatewayConfig {
    pub bind_address: Option<std::net::SocketAddr>,
    pub observability: ObservabilityConfig,
    pub debug: bool,
    /// If `true`, allow minijinja to read from the filesystem (within the tree of the config file) for '{% include %}'
    /// Defaults to `false`
    pub enable_template_filesystem_access: bool,
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
            StorageKind::Filesystem { path } => Some(Arc::new(
                LocalFileSystem::new_with_prefix(path).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Failed to create filesystem object store for path: {path}: {e}"
                        ),
                    })
                })?,
            )),
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
                        tracing::warn!("`[object_storage.allow_http]` is set to `true` - this is insecure, and should only be used when running a local S3-compatible object store")
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

/// Note: This struct and the impl below can be removed in favor of a derived impl for Deserialize once we have removed the `disable_observability` flag
/// TODO (#797): Remove this once we have removed the `disable_observability` flag
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedGatewayConfig {
    pub bind_address: Option<std::net::SocketAddr>,
    #[serde(default)]
    pub disable_observability: bool,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    #[serde(default)]
    pub debug: bool,
    #[serde(default)]
    pub enable_template_filesystem_access: bool,
}

impl TryFrom<UninitializedGatewayConfig> for GatewayConfig {
    type Error = Error;
    fn try_from(config: UninitializedGatewayConfig) -> Result<Self, Self::Error> {
        let enabled = match (config.disable_observability, config.observability.enabled) {
            (true, Some(_)) => {
                return Err(Error::new(ErrorDetails::Config {
                    message: "Configuration flag `gateway.disable_observability` and `gateway.observability.enabled` are mutually exclusive. We are deprecating `gateway.disable_observability` in favor of `gateway.observability.enabled`. See https://github.com/tensorzero/tensorzero/issues/797 on GitHub for details.".to_string(),
                }));
            }
            (true, None) => {
                tracing::warn!("Deprecation Warning: The configuration flag `gateway.disable_observability` is deprecated in favor of `gateway.observability.enabled`. See https://github.com/tensorzero/tensorzero/issues/797 on GitHub for details.");
                Some(false)
            }
            (false, Some(enabled)) => Some(enabled),
            (false, None) => None,
        };

        Ok(Self {
            bind_address: config.bind_address,
            observability: ObservabilityConfig {
                enabled,
                async_writes: config.observability.async_writes,
            },
            debug: config.debug,
            enable_template_filesystem_access: config.enable_template_filesystem_access,
        })
    }
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub async_writes: bool,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetricConfig {
    pub r#type: MetricConfigType,
    pub optimize: MetricConfigOptimize,
    pub level: MetricConfigLevel,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MetricConfigType {
    Boolean,
    Float,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MetricConfigOptimize {
    Min,
    Max,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
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

impl<'c> Config<'c> {
    pub async fn load_and_verify_from_path(config_path: &Path) -> Result<Config<'c>, Error> {
        Self::load_from_path_optional_verify_credentials(config_path, true).await
    }

    pub async fn load_from_path_optional_verify_credentials(
        config_path: &Path,
        validate_credentials: bool,
    ) -> Result<Config<'c>, Error> {
        let config_table = match UninitializedConfig::read_toml_config(config_path)? {
            Some(table) => table,
            None => {
                return Err(ErrorDetails::Config {
                    message: format!("Config file not found: {config_path:?}"),
                }
                .into())
            }
        };
        let base_path = match PathBuf::from(&config_path).parent() {
            Some(base_path) => base_path.to_path_buf(),
            None => {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "Failed to get parent directory of config file: {config_path:?}"
                    ),
                }
                .into());
            }
        };
        let config = if cfg!(feature = "e2e_tests") || !validate_credentials {
            SKIP_CREDENTIAL_VALIDATION.set(&(), || Self::load_from_toml(config_table, base_path))?
        } else {
            Self::load_from_toml(config_table, base_path)?
        };

        if validate_credentials {
            if let Some(object_store) = &config.object_store_info {
                object_store.verify().await?;
            }
        }

        Ok(config)
    }

    fn load_from_toml(table: toml::Table, base_path: PathBuf) -> Result<Config<'c>, Error> {
        if table.is_empty() {
            tracing::info!("Config file is empty, so only default functions will be available.")
        }
        let uninitialized_config = UninitializedConfig::try_from(table)?;

        let gateway = uninitialized_config
            .gateway
            .unwrap_or_default()
            .try_into()?;

        let templates = TemplateConfig::new();

        let functions = uninitialized_config
            .functions
            .into_iter()
            .map(|(name, config)| config.load(&name, &base_path).map(|c| (name, Arc::new(c))))
            .collect::<Result<HashMap<String, Arc<FunctionConfig>>, Error>>()?;

        let tools = uninitialized_config
            .tools
            .into_iter()
            .map(|(name, config)| {
                config
                    .load(&base_path, name.clone())
                    .map(|c| (name, Arc::new(c)))
            })
            .collect::<Result<HashMap<String, Arc<StaticToolConfig>>, Error>>()?;

        let object_store_info = ObjectStoreInfo::new(uninitialized_config.object_storage)?;

        let mut config = Config {
            gateway,
            models: uninitialized_config.models,
            embedding_models: uninitialized_config.embedding_models,
            functions,
            metrics: uninitialized_config.metrics,
            tools,
            evaluations: HashMap::new(),
            templates,
            object_store_info,
        };

        // Initialize the templates
        let template_paths = config.get_templates();
        config.templates.initialize(
            template_paths,
            config
                .gateway
                .enable_template_filesystem_access
                .then_some(base_path.clone()),
        )?;

        // Validate the config
        config.validate()?;

        // We add the evaluations after validation since we will be writing tensorzero:: functions to the functions map
        // and tensorzero:: metrics to the metrics map
        let mut evaluations = HashMap::new();
        for (name, evaluation_config) in uninitialized_config.evaluations {
            let (evaluation_config, evaluation_function_configs, evaluation_metric_configs) =
                evaluation_config.load(&config.functions, &base_path, &name)?;
            evaluations.insert(name, Arc::new(EvaluationConfig::Static(evaluation_config)));
            for (evaluation_function_name, evaluation_function_config) in
                evaluation_function_configs
            {
                if config.functions.contains_key(&evaluation_function_name) {
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "Duplicate evaluator function name: `{}` already exists. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.",
                            evaluation_function_name
                        ),
                    }
                    .into());
                }
                for variant in evaluation_function_config.variants().values() {
                    for template in variant.get_all_template_paths() {
                        config.templates.add_template(
                            template.path.to_string_lossy().as_ref(),
                            &template.contents,
                        )?;
                    }
                }
                evaluation_function_config.validate(
                    &config.tools,
                    &mut config.models,
                    &config.embedding_models,
                    &config.templates,
                    &evaluation_function_name,
                )?;
                config
                    .functions
                    .insert(evaluation_function_name, evaluation_function_config);
            }
            for (evaluation_metric_name, evaluation_metric_config) in evaluation_metric_configs {
                if config.metrics.contains_key(&evaluation_metric_name) {
                    return Err(ErrorDetails::Config {
                        message: format!("Duplicate evaluator metric name: `{}` already exists. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.", evaluation_metric_name),
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
    fn validate(&mut self) -> Result<(), Error> {
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
            function.validate(
                &self.tools,
                &mut self.models, // NOTE: in here there might be some models created using shorthand initialization
                &self.embedding_models,
                &self.templates,
                function_name,
            )?;
        }

        // Ensure that no metrics are named "comment" or "demonstration"
        for metric_name in self.metrics.keys() {
            if metric_name == "comment" || metric_name == "demonstration" {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "Metric name '{}' is reserved and cannot be used",
                        metric_name
                    ),
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
    ) -> Result<&'a Arc<FunctionConfig>, Error> {
        self.functions.get(function_name).ok_or_else(|| {
            Error::new(ErrorDetails::UnknownFunction {
                name: function_name.to_string(),
            })
        })
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
    pub fn get_model<'a>(
        &'a self,
        model_name: &Arc<str>,
    ) -> Result<CowNoClone<'a, ModelConfig>, Error> {
        self.models.get(model_name)?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: model_name.to_string(),
            })
        })
    }

    /// Get all templates from the config
    /// The HashMap returned is a mapping from the path as given in the TOML file
    /// (relative to the directory containing the TOML file) to the file contents.
    /// The former path is used as the name of the template for retrievaluation by variants later.
    pub fn get_templates(&self) -> HashMap<String, String> {
        let mut templates = HashMap::new();

        for function in self.functions.values() {
            for variant in function.variants().values() {
                let variant_template_paths = variant.get_all_template_paths();
                for path in variant_template_paths {
                    templates.insert(
                        path.path.to_string_lossy().to_string(),
                        path.contents.clone(),
                    );
                }
            }
        }
        templates
    }
}

/// A trait for loading configs with a base path
pub trait LoadableConfig<T> {
    fn load<P: AsRef<Path>>(self, base_path: P) -> Result<T, Error>;
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
    pub gateway: Option<UninitializedGatewayConfig>,
    #[serde(default)]
    pub models: ModelTable, // model name => model config
    #[serde(default)]
    pub embedding_models: EmbeddingModelTable, // embedding model name => embedding model config
    #[serde(default)]
    pub functions: HashMap<String, UninitializedFunctionConfig>, // function name => function config
    #[serde(default)]
    pub metrics: HashMap<String, MetricConfig>, // metric name => metric config
    #[serde(default)]
    pub tools: HashMap<String, UninitializedToolConfig>, // tool name => tool config
    #[serde(default)]
    pub evaluations: HashMap<String, UninitializedEvaluationConfig>, // evaluation name => evaluation config
    pub object_storage: Option<StorageKind>,
}

impl UninitializedConfig {
    /// Read a file from the file system and parse it as TOML
    fn read_toml_config(path: &Path) -> Result<Option<toml::Table>, Error> {
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(
            std::fs::read_to_string(path)
                .map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to read config file: {}", path.to_string_lossy()),
                    })
                })?
                .parse::<toml::Table>()
                .map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Failed to parse config file `{}` as valid TOML: {}",
                            path.to_string_lossy(),
                            e
                        ),
                    })
                })?,
        ))
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
    variants: HashMap<String, UninitializedVariantConfig>, // variant name => variant config
    system_schema: Option<PathBuf>,
    user_schema: Option<PathBuf>,
    assistant_schema: Option<PathBuf>,
    #[serde(default)]
    tools: Vec<String>, // tool names
    #[serde(default)]
    tool_choice: ToolChoice,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UninitializedFunctionConfigJson {
    variants: HashMap<String, UninitializedVariantConfig>, // variant name => variant config
    system_schema: Option<PathBuf>,
    user_schema: Option<PathBuf>,
    assistant_schema: Option<PathBuf>,
    output_schema: Option<PathBuf>, // schema will default to {} if not specified
}

impl UninitializedFunctionConfig {
    pub fn load<P: AsRef<Path>>(
        self,
        function_name: &str,
        base_path: P,
    ) -> Result<FunctionConfig, Error> {
        match self {
            UninitializedFunctionConfig::Chat(params) => {
                let system_schema = params
                    .system_schema
                    .map(|path| JSONSchemaFromPath::new(path, base_path.as_ref()))
                    .transpose()?;
                let user_schema = params
                    .user_schema
                    .map(|path| JSONSchemaFromPath::new(path, base_path.as_ref()))
                    .transpose()?;
                let assistant_schema = params
                    .assistant_schema
                    .map(|path| JSONSchemaFromPath::new(path, base_path.as_ref()))
                    .transpose()?;
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| variant.load(&base_path).map(|v| (name, v)))
                    .collect::<Result<HashMap<_, _>, Error>>()?;
                for (name, variant) in variants.iter() {
                    if let VariantConfig::ChatCompletion(chat_config) = variant {
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
                    system_schema,
                    user_schema,
                    assistant_schema,
                    tools: params.tools,
                    tool_choice: params.tool_choice,
                    parallel_tool_calls: params.parallel_tool_calls,
                }))
            }
            UninitializedFunctionConfig::Json(params) => {
                let system_schema = params
                    .system_schema
                    .map(|path| JSONSchemaFromPath::new(path, base_path.as_ref()))
                    .transpose()?;
                let user_schema = params
                    .user_schema
                    .map(|path| JSONSchemaFromPath::new(path, base_path.as_ref()))
                    .transpose()?;
                let assistant_schema = params
                    .assistant_schema
                    .map(|path| JSONSchemaFromPath::new(path, base_path.as_ref()))
                    .transpose()?;
                let output_schema = match params.output_schema {
                    Some(path) => JSONSchemaFromPath::new(path, base_path.as_ref())?,
                    None => JSONSchemaFromPath::default(),
                };
                let implicit_tool_call_config =
                    create_implicit_tool_call_config(output_schema.clone());
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| variant.load(&base_path).map(|v| (name, v)))
                    .collect::<Result<HashMap<_, _>, Error>>()?;

                for (name, variant) in variants.iter() {
                    let mut warn_variant = None;
                    match variant {
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
                    }
                    if let Some(warn_variant) = warn_variant {
                        tracing::warn!("Deprecation Warning: `json_mode` is not specified for `[functions.{function_name}.variants.{warn_variant}]` (parent function `{function_name}` is a JSON function), defaulting to `strict`. This field will become required in a future release - see https://github.com/tensorzero/tensorzero/issues/1043 on GitHub for details.");
                    }
                }
                Ok(FunctionConfig::Json(FunctionConfigJson {
                    variants,
                    system_schema,
                    user_schema,
                    assistant_schema,
                    output_schema,
                    implicit_tool_call_config,
                }))
            }
        }
    }
}

#[derive(Debug, TensorZeroDeserialize)]
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
}

impl UninitializedVariantConfig {
    pub fn load<P: AsRef<Path>>(self, base_path: P) -> Result<VariantConfig, Error> {
        match self {
            UninitializedVariantConfig::ChatCompletion(params) => {
                Ok(VariantConfig::ChatCompletion(params.load(base_path)?))
            }
            UninitializedVariantConfig::BestOfNSampling(params) => {
                Ok(VariantConfig::BestOfNSampling(params.load(base_path)?))
            }
            UninitializedVariantConfig::Dicl(params) => {
                Ok(VariantConfig::Dicl(params.load(base_path)?))
            }
            UninitializedVariantConfig::MixtureOfN(params) => {
                Ok(VariantConfig::MixtureOfN(params.load(base_path)?))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct UninitializedToolConfig {
    pub description: String,
    pub parameters: PathBuf,
    #[serde(default)]
    pub strict: bool,
}

impl UninitializedToolConfig {
    pub fn load<P: AsRef<Path>>(
        self,
        base_path: P,
        name: String,
    ) -> Result<StaticToolConfig, Error> {
        let parameters = JSONSchemaFromPath::new(self.parameters, base_path.as_ref())?;
        Ok(StaticToolConfig {
            name,
            description: self.description,
            parameters,
            strict: self.strict,
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct PathWithContents {
    pub path: PathBuf,
    pub contents: String,
}

impl PathWithContents {
    pub fn from_path<P: AsRef<Path>>(path: PathBuf, base_path: Option<P>) -> Result<Self, Error> {
        let full_path = if let Some(base_path) = base_path.as_ref() {
            &base_path.as_ref().join(&path)
        } else {
            &path
        };
        let contents = std::fs::read_to_string(full_path).map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Failed to read file at {}: {}",
                    full_path.to_string_lossy(),
                    e
                ),
            })
        })?;
        Ok(Self { path, contents })
    }
}

#[cfg(test)]
mod tests {

    use std::io::Write;
    use tempfile::NamedTempFile;
    use tracing_test::traced_test;

    use super::*;

    use std::env;

    use crate::{embeddings::EmbeddingProviderConfig, variant::JsonMode};

    /// Ensure that the sample valid config can be parsed without panicking
    #[test]
    fn test_config_from_toml_table_valid() {
        let config = get_sample_valid_config();
        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Config::load_from_toml(config, base_path.clone()).expect("Failed to load config");

        // Ensure that removing the `[metrics]` section still parses the config
        let mut config = get_sample_valid_config();
        config
            .remove("metrics")
            .expect("Failed to remove `[metrics]` section");
        let config = Config::load_from_toml(config, base_path).expect("Failed to load config");

        // Check that the JSON mode is set properly on the JSON variants
        let prompt_a_json_mode = match config
            .functions
            .get("json_with_schemas")
            .unwrap()
            .variants()
            .get("openai_promptA")
            .unwrap()
        {
            VariantConfig::ChatCompletion(chat_config) => &chat_config.json_mode.unwrap(),
            _ => panic!("Expected a chat completion variant"),
        };
        assert_eq!(prompt_a_json_mode, &JsonMode::ImplicitTool);

        let prompt_b_json_mode = match config
            .functions
            .get("json_with_schemas")
            .unwrap()
            .variants()
            .get("openai_promptB")
            .unwrap()
        {
            VariantConfig::ChatCompletion(chat_config) => chat_config.json_mode,
            _ => panic!("Expected a chat completion variant"),
        };
        // The json mode is unset (the default will get filled in when we construct a request,
        // using the variant mode (json/chat)).
        assert_eq!(prompt_b_json_mode, None);
        // Check that the tool choice for get_weather is set to "specific" and the correct tool
        let function = config.functions.get("weather_helper").unwrap();
        match &**function {
            FunctionConfig::Chat(chat_config) => {
                assert_eq!(
                    chat_config.tool_choice,
                    ToolChoice::Specific("get_temperature".to_string())
                );
            }
            _ => panic!("Expected a chat function"),
        }
        // Check that the best of n variant has multiple candidates
        let function = config
            .functions
            .get("templates_with_variables_chat")
            .unwrap();
        match &**function {
            FunctionConfig::Chat(chat_config) => {
                if let Some(variant) = chat_config.variants.get("best_of_n") {
                    match variant {
                        VariantConfig::BestOfNSampling(best_of_n_config) => {
                            assert!(
                                best_of_n_config.candidates.len() > 1,
                                "Best of n variant should have multiple candidates"
                            );
                        }
                        _ => panic!("Expected a best of n variant"),
                    }
                } else {
                    panic!("Expected to find a best of n variant");
                }
            }
            _ => panic!("Expected a chat function"),
        }
        // Check that the async flag is set to false by default
        assert!(!config.gateway.observability.async_writes);

        // To test that variant default weights work correctly,
        // We check `functions.templates_with_variables_json.variants.variant_with_variables.weight`
        // This variant's weight is unspecified, so it should default to 0
        let json_function = config
            .functions
            .get("templates_with_variables_json")
            .unwrap();
        match &**json_function {
            FunctionConfig::Json(json_config) => {
                let variant = json_config.variants.get("variant_with_variables").unwrap();
                match variant {
                    VariantConfig::ChatCompletion(chat_config) => {
                        assert_eq!(chat_config.weight, None); // Default weight should be None
                    }
                    _ => panic!("Expected a chat completion variant"),
                }
            }
            _ => panic!("Expected a JSON function"),
        }

        assert_eq!(config.embedding_models.len(), 1);

        let embedding_model = config
            .embedding_models
            .get("text-embedding-3-small")
            .expect("Error getting embedding model")
            .unwrap();
        assert_eq!(embedding_model.routing, vec!["openai".into()]);
        assert_eq!(embedding_model.providers.len(), 1);
        let provider = embedding_model.providers.get("openai").unwrap();
        assert!(matches!(provider, EmbeddingProviderConfig::OpenAI(_)));

        // Check that the function for the LLM Judge evaluation is added to the functions table
        let function = config
            .functions
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool")
            .unwrap();
        match &**function {
            FunctionConfig::Json(json_config) => {
                assert_eq!(json_config.variants.len(), 1);
                match &json_config.variants["anthropic_promptA"] {
                    VariantConfig::ChatCompletion(chat_config) => {
                        assert_eq!(chat_config.model, "anthropic::claude-3.5-sonnet".into());
                        assert_eq!(chat_config.weight, Some(1.0));
                        assert_eq!(
                            *chat_config.system_template.as_ref().unwrap(),
                            PathWithContents {
                                // We don't use a real path for programmatically generated templates
                                // Instead we use this handle and then the same in minijinja
                                path: PathBuf::from(
                                    "tensorzero::llm_judge::evaluation1::llm_judge_bool::system"
                                ),
                                contents:
                                    "Return True if there is NSFW content in this generation.\n\n"
                                        .to_string(),
                            }
                        );
                        assert_eq!(chat_config.json_mode, Some(JsonMode::ImplicitTool));
                    }
                    _ => panic!("Expected a chat completion variant"),
                }
            }
            _ => panic!("Expected a JSON function"),
        }
        // Check that the metric for the LLM Judge evaluator is added to the metrics table
        let metric = config
            .metrics
            .get("tensorzero::evaluation_name::evaluation1::evaluator_name::llm_judge_bool")
            .unwrap();
        assert_eq!(metric.r#type, MetricConfigType::Boolean);
        assert_eq!(metric.optimize, MetricConfigOptimize::Min);
        assert_eq!(metric.level, MetricConfigLevel::Inference);

        // Check that the metric for the exact match evaluation is added to the metrics table
        let metric = config
            .metrics
            .get("tensorzero::evaluation_name::evaluation1::evaluator_name::em_evaluator")
            .unwrap();
        assert_eq!(metric.r#type, MetricConfigType::Boolean);
        assert_eq!(metric.optimize, MetricConfigOptimize::Max);
        assert_eq!(metric.level, MetricConfigLevel::Inference);

        // Check that the metric for the LLM Judge float evaluation is added to the metrics table
        let metric = config
            .metrics
            .get("tensorzero::evaluation_name::evaluation1::evaluator_name::llm_judge_float")
            .unwrap();
        assert_eq!(metric.r#type, MetricConfigType::Float);
        assert_eq!(metric.optimize, MetricConfigOptimize::Min);
        assert_eq!(metric.level, MetricConfigLevel::Inference);
    }

    /// Ensure that the config parsing correctly handles the `gateway.bind_address` field
    #[test]
    fn test_config_gateway_bind_address() {
        let mut config = get_sample_valid_config();
        let base_path = PathBuf::new();

        // Test with a valid bind address
        let parsed_config = Config::load_from_toml(config.clone(), base_path.clone()).unwrap();
        assert_eq!(
            parsed_config.gateway.bind_address.unwrap().to_string(),
            "0.0.0.0:3000"
        );

        // Test with missing gateway section
        config.remove("gateway");
        let parsed_config = Config::load_from_toml(config.clone(), base_path.clone()).unwrap();
        assert!(parsed_config.gateway.bind_address.is_none());

        // Test with missing bind_address
        config.insert(
            "gateway".to_string(),
            toml::Value::Table(toml::Table::new()),
        );
        let parsed_config = Config::load_from_toml(config.clone(), base_path.clone()).unwrap();
        assert!(parsed_config.gateway.bind_address.is_none());

        // Test with invalid bind address
        config["gateway"].as_table_mut().unwrap().insert(
            "bind_address".to_string(),
            toml::Value::String("invalid_address".to_string()),
        );
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::Config {
                message: "gateway.bind_address: invalid socket address syntax".to_string()
            })
        );
    }

    /// Ensure that the config parsing fails when the `[models]` section is missing
    #[test]
    fn test_config_from_toml_table_missing_models() {
        let mut config = get_sample_valid_config();
        let base_path = PathBuf::new();
        config
            .remove("models")
            .expect("Failed to remove `[models]` section");

        // Remove all functions except generate_draft so we are sure what error will be thrown
        config["functions"]
            .as_table_mut()
            .unwrap()
            .retain(|k, _| k == "generate_draft");

        assert_eq!(
            Config::load_from_toml(config, base_path).unwrap_err(),
            Error::new(ErrorDetails::Config {
                message: "Model name 'gpt-3.5-turbo' not found in model table".to_string()
            })
        );
    }

    /// Ensure that the config parsing fails when the `[providers]` section is missing
    #[test]
    fn test_config_from_toml_table_missing_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307` section")
            .remove("providers")
            .expect("Failed to remove `[providers]` section");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::Config {
                message: "models.claude-3-haiku-20240307: missing field `providers`".to_string()
            })
        );
    }

    /// Ensure that the config parsing fails when the model credentials are missing
    #[test]
    fn test_config_from_toml_table_missing_credentials() {
        let mut config = get_sample_valid_config();
        let base_path = PathBuf::new();

        // Add a new variant called generate_draft_dummy to the generate_draft function
        let generate_draft = config["functions"]["generate_draft"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft` section");

        let variants = generate_draft["variants"]
            .as_table_mut()
            .expect("Failed to get `variants` section");

        variants.insert(
            "generate_draft_dummy".into(),
            toml::Value::Table({
                let mut table = toml::Table::new();
                table.insert("type".into(), "chat_completion".into());
                table.insert("weight".into(), 1.0.into());
                table.insert("model".into(), "dummy".into());
                table.insert(
                    "system_template".into(),
                    "fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
                        .into(),
                );
                table
            }),
        );

        // Add a new model "dummy" with a provider of type "dummy" with name "bad_credentials"
        let models = config["models"].as_table_mut().unwrap();
        models.insert(
            "dummy".into(),
            toml::Value::Table({
                let mut dummy_model = toml::Table::new();
                dummy_model.insert(
                    "providers".into(),
                    toml::Value::Table({
                        let mut providers = toml::Table::new();
                        providers.insert(
                            "bad_credentials".into(),
                            toml::Value::Table({
                                let mut provider = toml::Table::new();
                                provider.insert("type".into(), "dummy".into());
                                provider.insert("model_name".into(), "bad_credentials".into());
                                provider
                                    .insert("api_key_location".into(), "env::not_a_place".into());
                                provider
                            }),
                        );
                        providers
                    }),
                );
                dummy_model.insert(
                    "routing".into(),
                    toml::Value::Array(vec![toml::Value::String("bad_credentials".into())]),
                );
                dummy_model
            }),
        );

        let error = Config::load_from_toml(config.clone(), base_path.clone()).unwrap_err();
        assert_eq!(
            error,
            Error::new(ErrorDetails::Config {
                message: "models.dummy.providers.bad_credentials: Invalid api_key_location for Dummy provider"
                    .to_string()
            })
        );
    }

    /// Ensure that the config parsing fails when referencing a nonexistent function
    #[test]
    fn test_config_from_toml_table_nonexistent_function() {
        let mut config = get_sample_valid_config();
        config
            .remove("functions")
            .expect("Failed to remove `[functions]` section");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message:
                    "Function `generate_draft` not found (referenced in `[evaluations.evaluation1]`)"
                        .to_string()
            }
            .into()
        );
    }

    /// Ensure that the config parsing fails when the `[variants]` section is missing
    #[test]
    fn test_config_from_toml_table_missing_variants() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft` section")
            .remove("variants")
            .expect("Failed to remove `[variants]` section");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "functions.generate_draft: missing field `variants`".to_string()
            }
            .into()
        );
    }

    /// Ensure that the config parsing fails when there are extra variables at the root level
    #[test]
    fn test_config_from_toml_table_extra_variables_root() {
        let mut config = get_sample_valid_config();
        config.insert("enable_agi".into(), true.into());
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown field `enable_agi`, expected one of"));
    }

    /// Ensure that the config parsing fails when there are extra variables for models
    #[test]
    fn test_config_from_toml_table_extra_variables_models() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307` section")
            .insert("enable_agi".into(), true.into());
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown field `enable_agi`, expected"));
    }

    /// Ensure that the config parsing fails when there models with blacklisted names
    #[test]
    fn test_config_from_toml_table_blacklisted_models() {
        let mut config = get_sample_valid_config();

        let claude_config = config["models"]
            .as_table_mut()
            .expect("Failed to get `models` section")
            .remove("claude-3-haiku-20240307")
            .expect("Failed to remove claude config");
        config["models"]
            .as_table_mut()
            .expect("Failed to get `models` section")
            .insert("anthropic::claude-3-haiku-20240307".into(), claude_config);

        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        let error = result.unwrap_err().to_string();
        assert!(error
            .contains("models: Model name 'anthropic::claude-3-haiku-20240307' contains a reserved prefix"),
        "Unexpected error: {error}");
    }

    /// Ensure that the config parsing fails when there are extra variables for providers
    #[test]
    fn test_config_from_toml_table_extra_variables_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]["providers"]["anthropic"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307.providers.anthropic` section")
            .insert("enable_agi".into(), true.into());
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown field `enable_agi`, expected"));
    }

    /// Ensure that the config parsing fails when there are extra variables for functions
    #[test]
    fn test_config_from_toml_table_extra_variables_functions() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft` section")
            .insert("enable_agi".into(), true.into());
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown field `enable_agi`, expected"));
    }

    /// Ensure that the config parsing defaults properly for JSON functions with no output schema
    #[test]
    fn test_config_from_toml_table_json_function_no_output_schema() {
        let mut config = get_sample_valid_config();
        config["functions"]["json_with_schemas"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft` section")
            .remove("output_schema");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        let config = result.unwrap();
        // Check that the output schema is set to {}
        let output_schema = match &**config.functions.get("json_with_schemas").unwrap() {
            FunctionConfig::Json(json_config) => &json_config.output_schema,
            _ => panic!("Expected a JSON function"),
        };
        assert_eq!(output_schema, &JSONSchemaFromPath::default());
        assert_eq!(output_schema.value, &serde_json::json!({}));
    }

    /// Ensure that the config parsing fails when there are extra variables for variants
    #[test]
    fn test_config_from_toml_table_extra_variables_variants() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]["variants"]["openai_promptA"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft.variants.openai_promptA` section")
            .insert("enable_agi".into(), true.into());
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown field `enable_agi`, expected"));
    }

    /// Ensure that the config parsing fails when there are extra variables for metrics
    #[test]
    fn test_config_from_toml_table_extra_variables_metrics() {
        let mut config = get_sample_valid_config();
        config["metrics"]["task_success"]
            .as_table_mut()
            .expect("Failed to get `metrics.task_success` section")
            .insert("enable_agi".into(), true.into());
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown field `enable_agi`, expected"));
    }

    /// Ensure that the config validation fails when a model has no providers in `routing`
    #[test]
    fn test_config_validate_model_empty_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["gpt-3.5-turbo"]["routing"] = toml::Value::Array(vec![]);
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        let error = result.unwrap_err();
        assert!(error
            .to_string()
            .contains("`models.gpt-3.5-turbo`: `routing` must not be empty"));
    }

    /// Ensure that the config validation fails when there are duplicate routing entries
    #[test]
    fn test_config_validate_model_duplicate_routing_entry() {
        let mut config = get_sample_valid_config();
        config["models"]["gpt-3.5-turbo"]["routing"] =
            toml::Value::Array(vec!["openai".into(), "openai".into()]);
        let result = Config::load_from_toml(config, PathBuf::new());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("`models.gpt-3.5-turbo.routing`: duplicate entry `openai`"));
    }

    /// Ensure that the config validation fails when a routing entry does not exist in providers
    #[test]
    fn test_config_validate_model_routing_entry_not_in_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["gpt-3.5-turbo"]["routing"] = toml::Value::Array(vec!["closedai".into()]);
        let result = Config::load_from_toml(config, PathBuf::new());
        assert!(result.unwrap_err().to_string().contains("`models.gpt-3.5-turbo`: `routing` contains entry `closedai` that does not exist in `providers`"));
    }

    /// Ensure that the config loading fails when the system schema does not exist
    #[test]
    fn test_config_system_schema_does_not_exist() {
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_chat"]["system_schema"] =
            "non_existent_file.json".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::JsonSchema {
                message: "Failed to read JSON Schema `non_existent_file.json`: No such file or directory (os error 2)".to_string()
            }.into()
        );
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_json"]["system_schema"] =
            "non_existent_file.json".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::JsonSchema {
                message: "Failed to read JSON Schema `non_existent_file.json`: No such file or directory (os error 2)".to_string()
            }.into()
        );
    }

    /// Ensure that the config loading fails when the user schema does not exist
    #[test]
    fn test_config_user_schema_does_not_exist() {
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_chat"]["user_schema"] =
            "non_existent_file.json".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::JsonSchema {
                message: "Failed to read JSON Schema `non_existent_file.json`: No such file or directory (os error 2)".to_string()
            }.into()
        );
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_json"]["user_schema"] =
            "non_existent_file.json".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::JsonSchema {
                message: "Failed to read JSON Schema `non_existent_file.json`: No such file or directory (os error 2)".to_string()
            }.into()
        );
    }

    /// Ensure that the config loading fails when the assistant schema does not exist
    #[test]
    fn test_config_assistant_schema_does_not_exist() {
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_chat"]["assistant_schema"] =
            "non_existent_file.json".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::JsonSchema {
                message: "Failed to read JSON Schema `non_existent_file.json`: No such file or directory (os error 2)".to_string()
            }.into()
        );
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_json"]["assistant_schema"] =
            "non_existent_file.json".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::JsonSchema {
                message: "Failed to read JSON Schema `non_existent_file.json`: No such file or directory (os error 2)".to_string()
            }.into()
        );
    }

    /// Ensure that the config loading fails when the system schema is missing but is needed
    #[test]
    fn test_config_system_schema_is_needed() {
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_chat"]
            .as_table_mut()
            .unwrap()
            .remove("system_schema");

        sample_config["functions"]["templates_with_variables_chat"]["variants"]
            .as_table_mut()
            .unwrap()
            .remove("best_of_n");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_chat.variants.variant_with_variables.system_template`: schema is required when template is specified and needs variables".to_string()
            }.into()
        );
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_json"]
            .as_table_mut()
            .unwrap()
            .remove("system_schema");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_json.variants.variant_with_variables.system_template`: schema is required when template is specified and needs variables".to_string()
            }.into()
        );
    }

    /// Ensure that the config loading fails when the user schema is missing but is needed
    #[test]
    fn test_config_user_schema_is_needed() {
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_chat"]
            .as_table_mut()
            .unwrap()
            .remove("user_schema");
        sample_config["functions"]["templates_with_variables_chat"]["variants"]
            .as_table_mut()
            .unwrap()
            .remove("best_of_n");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_chat.variants.variant_with_variables.user_template`: schema is required when template is specified and needs variables".to_string()
            }.into()
        );

        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_json"]
            .as_table_mut()
            .unwrap()
            .remove("user_schema");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_json.variants.variant_with_variables.user_template`: schema is required when template is specified and needs variables".to_string()
            }.into()
        );
    }

    /// Ensure that the config loading fails when the assistant schema is missing but is needed
    #[test]
    fn test_config_assistant_schema_is_needed() {
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_chat"]
            .as_table_mut()
            .unwrap()
            .remove("assistant_schema");

        sample_config["functions"]["templates_with_variables_chat"]["variants"]
            .as_table_mut()
            .unwrap()
            .remove("best_of_n");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_chat.variants.variant_with_variables.assistant_template`: schema is required when template is specified and needs variables".to_string()
            }.into()
        );
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_json"]
            .as_table_mut()
            .unwrap()
            .remove("assistant_schema");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_json.variants.variant_with_variables.assistant_template`: schema is required when template is specified and needs variables".to_string()
            }.into()
        );
    }

    /// Ensure that config loading fails when a nonexistent candidate is specified in a variant
    #[test]
    fn test_config_best_of_n_candidate_not_found() {
        let mut sample_config = get_sample_valid_config();
        sample_config["functions"]["templates_with_variables_chat"]["variants"]
            .as_table_mut()
            .unwrap()
            .get_mut("best_of_n")
            .unwrap()
            .as_table_mut()
            .unwrap()
            .insert(
                "candidates".into(),
                toml::Value::Array(vec!["non_existent_candidate".into()]),
            );
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(sample_config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::UnknownCandidate {
                name: "non_existent_candidate".to_string()
            }
            .into()
        );
    }

    /// Ensure that the config validation fails when a function variant has a negative weight
    #[test]
    fn test_config_validate_function_variant_negative_weight() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]["variants"]["openai_promptA"]["weight"] =
            toml::Value::Float(-1.0);
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.generate_draft.variants.openai_promptA`: `weight` must be non-negative".to_string()
            }.into()
        );
    }

    /// Ensure that the config validation fails when a variant has a model that does not exist in the models section
    #[test]
    fn test_config_validate_variant_model_not_in_models() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]["variants"]["openai_promptA"]["model"] =
            "non_existent_model".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);

        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "Model name 'non_existent_model' not found in model table".to_string()
            }
            .into()
        );
    }

    /// Ensure that the config validation fails when a variant has a template that does not exist
    #[test]
    fn test_config_validate_variant_template_nonexistent() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]["variants"]["openai_promptA"]["system_template"] =
            "nonexistent_template".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);

        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "Failed to read file at nonexistent_template: No such file or directory (os error 2)".to_string()
            }
            .into()
        );
    }

    /// Ensure that the config validation fails when an evaluation points at a nonexistent function
    #[test]
    fn test_config_validate_evaluation_function_nonexistent() {
        let mut config = get_sample_valid_config();
        config["evaluations"]["evaluation1"]["function_name"] = "nonexistent_function".into();
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);

        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message:
                    "Function `nonexistent_function` not found (referenced in `[evaluations.evaluation1]`)"
                        .to_string()
            }
            .into()
        );
    }

    /// Ensure that the config validation fails when an evaluation name contains `::`
    #[test]
    fn test_config_validate_evaluation_name_contains_double_colon() {
        let mut config = get_sample_valid_config();
        let evaluation1 = config["evaluations"]["evaluation1"].clone();
        config
            .get_mut("evaluations")
            .unwrap()
            .as_table_mut()
            .unwrap()
            .insert("bad::evaluation".to_string(), evaluation1);
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);

        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message:
                    "evaluation names cannot contain \"::\" (referenced in `[evaluations.bad::evaluation]`)"
                        .to_string()
            }
            .into()
        );
    }

    /// Ensure that the config validation fails when a function has a tool that does not exist in the tools section
    #[test]
    fn test_config_validate_function_nonexistent_tool() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]
            .as_table_mut()
            .unwrap()
            .insert("tools".to_string(), toml::Value::Array(vec![]));
        config["functions"]["generate_draft"]["tools"] =
            toml::Value::Array(vec!["non_existent_tool".into()]);
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);

        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.generate_draft.tools`: tool `non_existent_tool` is not present in the config".to_string()
            }.into()
        );
    }

    /// Ensure that the config validation fails when a function name starts with `tensorzero::`
    #[test]
    fn test_config_validate_function_name_tensorzero_prefix() {
        let mut config = get_sample_valid_config();

        // Rename an existing function to start with `tensorzero::`
        let old_function_entry = config["functions"]
            .as_table_mut()
            .unwrap()
            .remove("generate_draft")
            .expect("Did not find function `generate_draft`");
        config["functions"]
            .as_table_mut()
            .unwrap()
            .insert("tensorzero::bad_function".to_string(), old_function_entry);

        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::Config {
                message: "Function name cannot start with 'tensorzero::': tensorzero::bad_function"
                    .to_string()
            })
        );
    }

    /// Ensure that the config validation fails when a metric name starts with `tensorzero::`
    #[test]
    fn test_config_validate_metric_name_tensorzero_prefix() {
        let mut config = get_sample_valid_config();

        // Rename an existing metric to start with `tensorzero::`
        let old_metric_entry = config["metrics"]
            .as_table_mut()
            .unwrap()
            .remove("task_success")
            .expect("Did not find metric `task_success`");
        config["metrics"]
            .as_table_mut()
            .unwrap()
            .insert("tensorzero::bad_metric".to_string(), old_metric_entry);

        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::Config {
                message: "Metric name cannot start with 'tensorzero::': tensorzero::bad_metric"
                    .to_string()
            })
        );
    }

    /// Ensure that the config validation fails when a model name starts with `tensorzero::`
    #[test]
    fn test_config_validate_model_name_tensorzero_prefix() {
        let mut config = get_sample_valid_config();

        // Rename an existing model to start with `tensorzero::`
        let old_model_entry = config["models"]
            .as_table_mut()
            .unwrap()
            .remove("gpt-3.5-turbo")
            .expect("Did not find model `gpt-3.5-turbo`");
        config["models"]
            .as_table_mut()
            .unwrap()
            .insert("tensorzero::bad_model".to_string(), old_model_entry);

        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::Config {
                message: "models: Model name 'tensorzero::bad_model' contains a reserved prefix"
                    .to_string()
            })
        );
    }

    /// Ensure that the config validation fails when an embedding model name starts with `tensorzero::`
    #[test]
    fn test_config_validate_embedding_model_name_tensorzero_prefix() {
        let mut config = get_sample_valid_config();

        // Rename an existing embedding model to start with `tensorzero::`
        let old_embedding_model_entry = config["embedding_models"]
            .as_table_mut()
            .unwrap()
            .remove("text-embedding-3-small")
            .expect("Did not find embedding model `text-embedding-3-small`");
        config["embedding_models"].as_table_mut().unwrap().insert(
            "tensorzero::bad_embedding_model".to_string(),
            old_embedding_model_entry,
        );

        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
                result.unwrap_err(),
                Error::new(ErrorDetails::Config {
                    message:
                        "embedding_models: Embedding model name 'tensorzero::bad_embedding_model' contains a reserved prefix"
                            .to_string()
                })
            );
    }

    /// Ensure that the config validation fails when a tool name starts with `tensorzero::`
    #[test]
    fn test_config_validate_tool_name_tensorzero_prefix() {
        let mut config = get_sample_valid_config();

        // Clone an existing tool and add a new one with tensorzero:: prefix
        let old_tool_entry = config["tools"]
            .as_table()
            .unwrap()
            .get("get_temperature")
            .expect("Did not find tool `get_temperature`")
            .clone();
        config["tools"]
            .as_table_mut()
            .unwrap()
            .insert("tensorzero::bad_tool".to_string(), old_tool_entry);

        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::Config {
                message: "Tool name cannot start with 'tensorzero::': tensorzero::bad_tool"
                    .to_string()
            })
        );
    }

    #[test]
    fn test_config_validate_chat_function_json_mode() {
        let mut config = get_sample_valid_config();

        // Insert `json_mode = "on"` into a variant config for a chat function.
        config["functions"]["generate_draft"]["variants"]["openai_promptA"]
            .as_table_mut()
            .unwrap()
            .insert("json_mode".to_string(), "on".into());

        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);

        // Check that the config is rejected, since `generate_draft` is not a json function
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("JSON mode is not supported for variant `openai_promptA` (parent function is a chat function)"),
            "Unexpected error message: {err_msg}"
        );
    }

    /// If you also want to confirm a variant name starting with `tensorzero::` fails
    /// (only do this if your `function.validate` logic checks variant names):
    #[test]
    fn test_config_validate_variant_name_tensorzero_prefix() {
        let mut config = get_sample_valid_config();

        // For demonstration, rename an existing variant inside `generate_draft`:
        let old_variant_entry = config["functions"]["generate_draft"]["variants"]
            .as_table_mut()
            .unwrap()
            .remove("openai_promptA")
            .expect("Did not find variant `openai_promptA`");
        config["functions"]["generate_draft"]["variants"]
            .as_table_mut()
            .unwrap()
            .insert("tensorzero::bad_variant".to_string(), old_variant_entry);

        // This test will only pass if your code actually rejects variant names with that prefix
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);

        // Adjust the expected message if your code gives a different error shape for variants
        // Or remove this test if variant names are *not* validated in that manner
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("tensorzero::bad_variant"));
    }

    /// Ensure that the config validation fails when a model provider's name starts with `tensorzero::`
    #[test]
    fn test_config_validate_model_provider_name_tensorzero_prefix() {
        let mut config = get_sample_valid_config();

        // Rename an existing provider to start with `tensorzero::`
        let old_openai_provider = config["models"]["gpt-3.5-turbo"]["providers"]
            .as_table_mut()
            .unwrap()
            .remove("openai")
            .expect("Did not find provider `openai` under `gpt-3.5-turbo`");
        config["models"]["gpt-3.5-turbo"]["providers"]
            .as_table_mut()
            .unwrap()
            .insert("tensorzero::openai".to_string(), old_openai_provider);

        // Update the routing entry to match the new provider name
        let routing = config["models"]["gpt-3.5-turbo"]["routing"]
            .as_array_mut()
            .expect("Expected routing to be an array");
        for entry in routing.iter_mut() {
            if entry.as_str() == Some("openai") {
                *entry = toml::Value::String("tensorzero::openai".to_string());
            }
        }

        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);

        assert!(result.unwrap_err().to_string().contains("`models.gpt-3.5-turbo.routing`: Provider name cannot start with 'tensorzero::': tensorzero::openai"));
    }

    /// Ensure that get_templates returns the correct templates
    #[test]
    fn test_get_all_templates() {
        let config_table = get_sample_valid_config();
        let config =
            Config::load_from_toml(config_table, PathBuf::new()).expect("Failed to load config");

        // Get all templates
        let templates = config.get_templates();

        // Check if all expected templates are present
        assert_eq!(
            *templates
                .get("fixtures/config/functions/generate_draft/promptA/system_template.minijinja")
                .unwrap(),
            include_str!(
                "../fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
            )
            .to_string()
        );
        assert_eq!(
            *templates
                .get("fixtures/config/functions/generate_draft/promptA/system_template.minijinja")
                .unwrap(),
            include_str!(
                "../fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
            )
            .to_string()
        );
        assert_eq!(
            *templates
                .get(
                    "fixtures/config/functions/json_with_schemas/promptA/system_template.minijinja"
                )
                .unwrap(),
            include_str!(
                "../fixtures/config/functions/json_with_schemas/promptA/system_template.minijinja"
            )
            .to_string()
        );
        assert_eq!(
            *templates
                .get(
                    "fixtures/config/functions/json_with_schemas/promptB/system_template.minijinja"
                )
                .unwrap(),
            include_str!(
                "../fixtures/config/functions/json_with_schemas/promptB/system_template.minijinja"
            )
            .to_string()
        );
        assert_eq!(
            *templates.get("fixtures/config/functions/templates_without_variables/variant_without_templates/system_template.minijinja")
            .unwrap(),
            include_str!(
                "../fixtures/config/functions/templates_without_variables/variant_without_templates/system_template.minijinja"
            ).to_string()
        );
        assert_eq!(
            *templates.get("fixtures/config/functions/templates_without_variables/variant_without_templates/user_template.minijinja")
            .unwrap(),
            include_str!(
                "../fixtures/config/functions/templates_without_variables/variant_without_templates/user_template.minijinja"
            ).to_string()
        );
        assert_eq!(
            *templates.get("fixtures/config/functions/templates_without_variables/variant_without_templates/assistant_template.minijinja")
            .unwrap(),
            include_str!(
                "../fixtures/config/functions/templates_without_variables/variant_without_templates/assistant_template.minijinja"
            ).to_string()
        );
        assert_eq!(
            *templates.get("fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja")
            .unwrap(),
            include_str!(
                "../fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja"
            ).to_string()
        );
        assert_eq!(
            *templates.get("fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja")
            .unwrap(),
            include_str!(
                "../fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja"
            ).to_string()
        );
        assert_eq!(
                    *templates.get("fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja")
                    .unwrap(),
                    include_str!(
                        "../fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja"
                    ).to_string()
                );

        assert_eq!(
            *templates
                .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::system")
                .unwrap(),
            "Return True if there is NSFW content in this generation.\n\n".to_string(),
        );

        assert_eq!(
            *templates
                .get("tensorzero::llm_judge::evaluation1::llm_judge_float::system")
                .unwrap(),
            "Return a number between 0 and 1 where 1 is very NSFW and 0 is the least NSFW content.\n\n"
                .to_string(),
        );
        assert_eq!(
            *templates
                .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::user")
                .unwrap(),
            include_str!("evaluations/llm_judge_user_template.minijinja").to_string()
        );
        assert_eq!(
            *templates
                .get("tensorzero::llm_judge::evaluation1::llm_judge_float::user")
                .unwrap(),
            include_str!("evaluations/llm_judge_user_template.minijinja").to_string()
        );

        // Check the total number of templates
        assert_eq!(templates.len(), 14);
    }

    #[test]
    fn test_load_bad_config_error_path() {
        let config_str = r#"
[functions.bash_assistant]
type = "chat"

[functions.bash_assistant.variants.anthropic_claude_3_7_sonnet_20250219]
type = "chat_completion"
model = "anthropic::claude-3-7-sonnet-20250219"
max_tokens = 2048

[functions.bash_assistant.variants.anthropic_claude_3_7_sonnet_20250219.extra_body]
tools = [{ type = "bash_20250124", name = "bash" }]
thinking = { type = "enabled", budget_tokens = 1024 }
        "#;
        let config = toml::from_str(config_str).expect("Failed to parse sample config");
        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let err = Config::load_from_toml(config, base_path.clone())
            .expect_err("Config loading should fail")
            .to_string();
        assert_eq!(err, "functions.bash_assistant: variants.anthropic_claude_3_7_sonnet_20250219: extra_body: invalid type: map, expected a sequence");
    }

    #[test]
    fn test_config_load_shorthand_models_only() {
        let config_str = r#"
        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                  GENERAL                                   │
        # └────────────────────────────────────────────────────────────────────────────┘

        [gateway]
        bind_address = "0.0.0.0:3000"


        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                 FUNCTIONS                                  │
        # └────────────────────────────────────────────────────────────────────────────┘

        [functions.generate_draft]
        type = "chat"
        system_schema = "fixtures/config/functions/generate_draft/system_schema.json"

        [functions.generate_draft.variants.openai_promptA]
        type = "chat_completion"
        weight = 0.9
        model = "openai::gpt-3.5-turbo"
        system_template = "fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
        "#;
        env::set_var("OPENAI_API_KEY", "sk-something");
        env::set_var("ANTHROPIC_API_KEY", "sk-something");
        env::set_var("AZURE_OPENAI_API_KEY", "sk-something");

        let config = toml::from_str(config_str).expect("Failed to parse sample config");
        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Config::load_from_toml(config, base_path.clone()).expect("Failed to load config");
    }

    #[tokio::test]
    #[traced_test]
    async fn test_empty_config() {
        let tempfile = NamedTempFile::new().unwrap();
        write!(&tempfile, "").unwrap();
        Config::load_and_verify_from_path(tempfile.path())
            .await
            .unwrap();
        assert!(logs_contain(
            "Config file is empty, so only default functions will be available."
        ))
    }

    #[tokio::test]
    async fn test_invalid_toml() {
        let config_str = r#"
        [models.my-model]
        routing = ["dummy"]

        [models.my-model]
        routing = ["other"]
        "#;

        let tmpfile = NamedTempFile::new().unwrap();
        std::fs::write(tmpfile.path(), config_str).unwrap();

        let err = Config::load_and_verify_from_path(tmpfile.path())
            .await
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("duplicate key `my-model` in table `models`"),
            "Unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_model_provider_unknown_field() {
        let config_str = r#"
        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                  GENERAL                                   │
        # └────────────────────────────────────────────────────────────────────────────┘

        [gateway]
        bind_address = "0.0.0.0:3000"

        [functions]

        [models.my-model]
        routing = ["dummy"]

        [models.my-model.providers.dummy]
        type = "dummy"
        my_bad_key = "foo"
        "#;

        let config = toml::from_str(config_str).expect("Failed to parse sample config");
        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let err = Config::load_from_toml(config, base_path.clone())
            .expect_err("Config should fail to load");
        assert!(
            err.to_string().contains("unknown field `my_bad_key`"),
            "Unexpected error: {err:?}"
        );
    }

    /// Get a sample valid config for testing
    fn get_sample_valid_config() -> toml::Table {
        let config_str = include_str!("../fixtures/config/tensorzero.toml");
        env::set_var("OPENAI_API_KEY", "sk-something");
        env::set_var("ANTHROPIC_API_KEY", "sk-something");
        env::set_var("AZURE_OPENAI_API_KEY", "sk-something");

        toml::from_str(config_str).expect("Failed to parse sample config")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bedrock_err_no_auto_detect_region() {
        let config_str = r#"
        [gateway]
        bind_address = "0.0.0.0:3000"


        [models."my-model"]
        routing = ["aws-bedrock"]

        [models.my-model.providers.aws-bedrock]
        type = "aws_bedrock"
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        "#;
        let config = toml::from_str(config_str).expect("Failed to parse sample config");

        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let err =
            Config::load_from_toml(config, base_path.clone()).expect_err("Failed to load bedrock");
        let err_msg = err.to_string();
        assert!(
            err_msg
                .contains("requires a region to be provided, or `allow_auto_detect_region = true`"),
            "Unexpected error message: {err_msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bedrock_err_auto_detect_region_no_aws_credentials() {
        // We want auto-detection to fail, so we clear this environment variable.
        // We use 'nextest' as our runner, so each test runs in its own process
        std::env::remove_var("AWS_REGION");
        std::env::remove_var("AWS_DEFAULT_REGION");

        let config_str = r#"
        [gateway]
        bind_address = "0.0.0.0:3000"

        [models."my-model"]
        routing = ["aws-bedrock"]

        [models.my-model.providers.aws-bedrock]
        type = "aws_bedrock"
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        allow_auto_detect_region = true
        "#;
        let config = toml::from_str(config_str).expect("Failed to parse sample config");

        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let err =
            Config::load_from_toml(config, base_path.clone()).expect_err("Failed to load bedrock");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Failed to determine AWS region."),
            "Unexpected error message: {err_msg}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bedrock_region_and_allow_auto() {
        let config_str = r#"
        [gateway]
        bind_address = "0.0.0.0:3000"

        [functions.basic_test]
        type = "chat"

        [functions.basic_test.variants.test]
        type = "chat_completion"
        weight = 1
        model = "my-model"

        [models."my-model"]
        routing = ["aws-bedrock"]

        [models.my-model.providers.aws-bedrock]
        type = "aws_bedrock"
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        allow_auto_detect_region = true
        region = "us-east-2"
        "#;
        let config = toml::from_str(config_str).expect("Failed to parse sample config");

        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Config::load_from_toml(config, base_path.clone())
            .expect("Failed to construct config with valid AWS bedrock provider");
    }

    #[traced_test]
    #[tokio::test]
    async fn test_config_load_no_config_file() {
        let err = Config::load_and_verify_from_path(Path::new("nonexistent.toml"))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Config file not found"),
            "Unexpected error message: {err}"
        );
    }

    #[traced_test]
    #[tokio::test]
    async fn test_config_load_invalid_s3_creds() {
        // Set invalid credentials (tests are isolated per-process)
        // to make sure that the write fails quickly.
        std::env::set_var("AWS_ACCESS_KEY_ID", "invalid");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "invalid");
        let tempfile = NamedTempFile::new().unwrap();
        write!(
            &tempfile,
            r#"
            [object_storage]
            type = "s3_compatible"
            bucket_name = "tensorzero-fake-bucket"
            region = "us-east-1"

            [functions]"#
        )
        .unwrap();
        let err = Config::load_and_verify_from_path(tempfile.path())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Failed to write `.tensorzero-validate` to object store."),
            "Unexpected error message: {err}"
        );
    }

    #[traced_test]
    #[tokio::test]
    async fn test_config_blocked_s3_http_endpoint_default() {
        // Set invalid credentials (tests are isolated per-process)
        // to make sure that the write fails quickly.
        std::env::set_var("AWS_ACCESS_KEY_ID", "invalid");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "invalid");
        let tempfile = NamedTempFile::new().unwrap();
        write!(
            &tempfile,
            r#"
            [object_storage]
            type = "s3_compatible"
            bucket_name = "tensorzero-fake-bucket"
            region = "us-east-1"
            endpoint = "http://tensorzero.invalid"
            [functions]"#
        )
        .unwrap();
        let err = Config::load_and_verify_from_path(tempfile.path())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Failed to write `.tensorzero-validate` to object store."),
            "Unexpected error message: {err}"
        );
        assert!(
            err.contains("BadScheme"),
            "Missing `BadScheme` in error: {err}"
        );
        assert!(logs_contain("Consider setting `[object_storage.allow_http]` to `true` if you are using a non-HTTPs endpoint"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_config_blocked_s3_http_endpoint_override() {
        // Set invalid credentials (tests are isolated per-process)
        // to make sure that the write fails quickly.
        std::env::set_var("AWS_ACCESS_KEY_ID", "invalid");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "invalid");
        std::env::set_var("AWS_ALLOW_HTTP", "true");
        let tempfile = NamedTempFile::new().unwrap();
        write!(
            &tempfile,
            r#"
            [object_storage]
            type = "s3_compatible"
            bucket_name = "tensorzero-fake-bucket"
            region = "us-east-1"
            endpoint = "http://tensorzero.invalid"
            allow_http = false
            [functions]"#
        )
        .unwrap();
        let err = Config::load_and_verify_from_path(tempfile.path())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Failed to write `.tensorzero-validate` to object store."),
            "Unexpected error message: {err}"
        );
        assert!(
            err.contains("BadScheme"),
            "Missing `BadScheme` in error: {err}"
        );
        assert!(logs_contain("Consider setting `[object_storage.allow_http]` to `true` if you are using a non-HTTPs endpoint"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_config_s3_allow_http_config() {
        // Set invalid credentials (tests are isolated per-process)
        // to make sure that the write fails quickly.
        std::env::set_var("AWS_ACCESS_KEY_ID", "invalid");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "invalid");
        // Make `object_store` fail immediately (with the expected dns resolution error)
        // to speed up this test.
        std::env::set_var("TENSORZERO_E2E_DISABLE_S3_RETRY", "true");
        let tempfile = NamedTempFile::new().unwrap();
        write!(
            &tempfile,
            r#"
            [object_storage]
            type = "s3_compatible"
            bucket_name = "tensorzero-fake-bucket"
            region = "us-east-1"
            endpoint = "http://tensorzero.invalid"
            allow_http = true
            [functions]"#
        )
        .unwrap();
        let err = Config::load_and_verify_from_path(tempfile.path())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Failed to write `.tensorzero-validate` to object store."),
            "Unexpected error message: {err}"
        );
        assert!(
            err.contains("failed to lookup address information"),
            "Missing dns error in error: {err}"
        );
        assert!(logs_contain(
            "[object_storage.allow_http]` is set to `true` - this is insecure"
        ));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_config_s3_allow_http_env_var() {
        // Set invalid credentials (tests are isolated per-process)
        // to make sure that the write fails quickly.
        std::env::set_var("AWS_ACCESS_KEY_ID", "invalid");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "invalid");
        // Make `object_store` fail immediately (with the expected dns resolution error)
        // to speed up this test.
        std::env::set_var("TENSORZERO_E2E_DISABLE_S3_RETRY", "true");
        std::env::set_var("AWS_ALLOW_HTTP", "true");
        let tempfile = NamedTempFile::new().unwrap();
        write!(
            &tempfile,
            r#"
            [object_storage]
            type = "s3_compatible"
            bucket_name = "tensorzero-fake-bucket"
            region = "us-east-1"
            endpoint = "http://tensorzero.invalid"
            [functions]"#
        )
        .unwrap();
        let err = Config::load_and_verify_from_path(tempfile.path())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Failed to write `.tensorzero-validate` to object store."),
            "Unexpected error message: {err}"
        );
        assert!(
            err.contains("failed to lookup address information"),
            "Missing dns error in error: {err}"
        );
        assert!(!logs_contain("HTTPS"));
    }

    #[traced_test]
    #[test]
    fn test_deprecated_missing_json_mode() {
        let config_str = r#"
        [gateway]
        bind_address = "0.0.0.0:3000"

        [functions.basic_test]
        type = "json"

        [functions.basic_test.variants.good_variant]
        type = "chat_completion"
        model = "my-model"
        json_mode = "off"

        [functions.basic_test.variants.test]
        type = "chat_completion"
        model = "my-model"

        [functions.basic_test.variants.dicl]
        type = "experimental_dynamic_in_context_learning"
        model = "my-model"
        embedding_model = "openai::text-embedding-3-small"
        k = 3
        max_tokens = 100

        [functions.basic_test.variants.mixture_of_n_variant]
        type = "experimental_mixture_of_n"
        candidates = ["test"]

        [functions.basic_test.variants.mixture_of_n_variant.fuser]
        model = "my-model"

        [functions.basic_test.variants.best_of_n_variant]
        type = "experimental_best_of_n_sampling"
        candidates = ["test"]

        [functions.basic_test.variants.best_of_n_variant.evaluator]
        model = "my-model"

        [models."my-model"]
        routing = ["openai"]

        [models.my-model.providers.openai]
        type = "openai"
        model_name = "gpt-4o-mini-2024-07-18"
        "#;
        let config = toml::from_str(config_str).expect("Failed to parse sample config");

        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        SKIP_CREDENTIAL_VALIDATION
            .set(&(), || Config::load_from_toml(config, base_path))
            .unwrap();

        assert!(!logs_contain("good_variant"));
        assert!(logs_contain("Deprecation Warning: `json_mode` is not specified for `[functions.basic_test.variants.test]`"));
        assert!(logs_contain("Deprecation Warning: `json_mode` is not specified for `[functions.basic_test.variants.dicl]`"));
        assert!(logs_contain("Deprecation Warning: `json_mode` is not specified for `[functions.basic_test.variants.mixture_of_n_variant.fuser]`"));
        assert!(logs_contain("Deprecation Warning: `json_mode` is not specified for `[functions.basic_test.variants.best_of_n_variant.evaluator]`"));
    }

    #[tokio::test]
    async fn test_config_load_optional_credentials_validation() {
        let config_str = r#"
        [models."my-model"]
        routing = ["openai"]

        [models.my-model.providers.openai]
        type = "openai"
        model_name = "gpt-4o-mini-2024-07-18"
        api_key_location = "path::/not/a/path"
        "#;

        let tmpfile = NamedTempFile::new().unwrap();
        std::fs::write(tmpfile.path(), config_str).unwrap();

        let res = Config::load_from_path_optional_verify_credentials(tmpfile.path(), true).await;
        if cfg!(feature = "e2e_tests") {
            assert!(res.is_ok());
        } else {
            assert_eq!(res.unwrap_err().to_string(), "models.my-model.providers.openai: API key missing for provider: openai: Failed to read credentials file - No such file or directory (os error 2)");
        }

        // Should not fail since validation is disabled
        Config::load_from_path_optional_verify_credentials(tmpfile.path(), false)
            .await
            .expect("Failed to load config");
    }
}
