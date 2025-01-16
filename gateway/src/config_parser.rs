use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::instrument;

use crate::embeddings::EmbeddingModelConfig;
use crate::error::{Error, ErrorDetails};
use crate::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::minijinja_util::TemplateConfig;
use crate::model::{ModelConfig, ModelTable};
use crate::tool::{
    ImplicitToolConfig, StaticToolConfig, ToolCallConfig, ToolChoice, ToolConfig,
    IMPLICIT_TOOL_NAME,
};
use crate::variant::best_of_n_sampling::BestOfNSamplingConfig;
use crate::variant::chat_completion::ChatCompletionConfig;
use crate::variant::dicl::UninitializedDiclConfig;
use crate::variant::mixture_of_n::MixtureOfNConfig;
use crate::variant::{Variant, VariantConfig};

#[derive(Debug, Default)]
pub struct Config<'c> {
    pub gateway: GatewayConfig,
    pub models: ModelTable, // model name => model config
    pub embedding_models: HashMap<Arc<str>, EmbeddingModelConfig>, // embedding model name => embedding model config
    pub functions: HashMap<String, FunctionConfig>, // function name => function config
    pub metrics: HashMap<String, MetricConfig>,     // metric name => metric config
    pub tools: HashMap<String, StaticToolConfig>,   // tool name => tool config
    pub templates: TemplateConfig<'c>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayConfig {
    pub bind_address: Option<std::net::SocketAddr>,
    #[serde(default)]
    pub disable_observability: bool,
    #[serde(default)]
    pub debug: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricConfig {
    pub r#type: MetricConfigType,
    pub optimize: MetricConfigOptimize,
    pub level: MetricConfigLevel,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricConfigType {
    Boolean,
    Float,
}

#[derive(Debug, Deserialize)]
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
    #[instrument]
    pub fn load() -> Result<Config<'c>, Error> {
        let config_path = UninitializedConfig::get_config_path();
        let config_table = UninitializedConfig::read_toml_config(&config_path)?;
        let base_path = match PathBuf::from(&config_path).parent() {
            Some(base_path) => base_path.to_path_buf(),
            None => {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "Failed to get parent directory of config file: {config_path}"
                    ),
                }
                .into())
            }
        };
        let config = Self::load_from_toml(config_table, base_path)?;
        Ok(config)
    }

    fn load_from_toml(table: toml::Table, base_path: PathBuf) -> Result<Config<'c>, Error> {
        let config = UninitializedConfig::try_from(table)?;

        let gateway = config.gateway.unwrap_or_default();

        let templates = TemplateConfig::new();

        let functions = config
            .functions
            .into_iter()
            .map(|(name, config)| config.load(&base_path).map(|c| (name, c)))
            .collect::<Result<HashMap<String, FunctionConfig>, Error>>()?;

        let tools = config
            .tools
            .into_iter()
            .map(|(name, config)| config.load(&base_path, name.clone()).map(|c| (name, c)))
            .collect::<Result<HashMap<String, StaticToolConfig>, Error>>()?;

        let mut config = Config {
            gateway,
            models: config.models,
            embedding_models: config.embedding_models,
            functions,
            metrics: config.metrics,
            tools,
            templates,
        };

        // Initialize the templates
        let template_paths = config.get_templates(&base_path);
        config.templates.initialize(template_paths)?;

        // Validate the config
        config.validate()?;

        Ok(config)
    }

    /// Validate the config
    #[instrument(skip_all)]
    fn validate(&mut self) -> Result<(), Error> {
        // Validate each function
        for (function_name, function) in &self.functions {
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
        }

        // Validate each model
        for (model_name, model) in self.models.iter() {
            // Ensure that the model has at least one provider
            if model.routing.is_empty() {
                return Err(ErrorDetails::Config {
                    message: format!("`models.{model_name}`: `routing` must not be empty"),
                }
                .into());
            }

            // Ensure that routing entries are unique and exist as keys in providers
            let mut seen_providers = std::collections::HashSet::new();
            for provider in &model.routing {
                if !seen_providers.insert(provider) {
                    return Err(ErrorDetails::Config {
                        message: format!(
                            "`models.{model_name}.routing`: duplicate entry `{provider}`"
                        ),
                    }
                    .into());
                }

                if !model.providers.contains_key(provider) {
                    return Err(ErrorDetails::Config {
                message: format!(
                    "`models.{model_name}`: `routing` contains entry `{provider}` that does not exist in `providers`"
                ),
            }
            .into());
                }
            }

            // Validate each provider
            for provider_name in model.providers.keys() {
                if !seen_providers.contains(provider_name) {
                    return Err(ErrorDetails::Config {
                        message: format!(
                    "`models.{model_name}`: Provider `{provider_name}` is not listed in `routing`"
                ),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    /// Get a function by name
    pub fn get_function<'a>(&'a self, function_name: &str) -> Result<&'a FunctionConfig, Error> {
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
    pub fn get_tool<'a>(&'a self, tool_name: &str) -> Result<&'a StaticToolConfig, Error> {
        self.tools.get(tool_name).ok_or_else(|| {
            Error::new(ErrorDetails::UnknownTool {
                name: tool_name.to_string(),
            })
        })
    }

    /// Get a model by name
    pub fn get_model<'a>(&'a self, model_name: &Arc<str>) -> Result<&'a ModelConfig, Error> {
        self.models.get(model_name).ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: model_name.to_string(),
            })
        })
    }

    /// Get all templates from the config
    /// The HashMap returned is a mapping from the path as given in the TOML file
    /// (relative to the directory containing the TOML file) to the path on the filesystem.
    /// The former path is used as the name of the template for retrieval by variants later.
    pub fn get_templates<P: AsRef<Path>>(&self, base_path: P) -> HashMap<String, PathBuf> {
        let mut templates = HashMap::new();

        for function in self.functions.values() {
            for variant in function.variants().values() {
                let variant_template_paths = variant.get_all_template_paths();
                for path in variant_template_paths {
                    templates.insert(
                        path.to_string_lossy().to_string(),
                        base_path.as_ref().join(path),
                    );
                }
            }
        }
        templates
    }
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
    pub gateway: Option<GatewayConfig>,
    #[serde(default)]
    pub models: ModelTable, // model name => model config
    #[serde(default)]
    pub embedding_models: HashMap<Arc<str>, EmbeddingModelConfig>, // embedding model name => embedding model config
    pub functions: HashMap<String, UninitializedFunctionConfig>, // function name => function config
    #[serde(default)]
    pub metrics: HashMap<String, MetricConfig>, // metric name => metric config
    #[serde(default)]
    pub tools: HashMap<String, UninitializedToolConfig>, // tool name => tool config
}

impl UninitializedConfig {
    /// Load and validate the TensorZero config file
    /// Use a path provided as a CLI argument (`./gateway path/to/tensorzero.toml`), or default to
    /// `tensorzero.toml` in the current directory if no path is provided.
    fn get_config_path() -> String {
        match std::env::args().nth(1) {
            Some(path) => path,
            None => "config/tensorzero.toml".to_string(),
        }
    }

    /// Read a file from the file system and parse it as TOML
    fn read_toml_config(path: &str) -> Result<toml::Table, Error> {
        std::fs::read_to_string(path)
            .map_err(|_| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to read config file: {path}"),
                })
            })?
            .parse::<toml::Table>()
            .map_err(|_| {
                Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse config file as valid TOML: {path}"),
                })
            })
    }
}

/// Deserialize a TOML table into `UninitializedConfig`
impl TryFrom<toml::Table> for UninitializedConfig {
    type Error = Error;

    fn try_from(table: toml::Table) -> Result<Self, Self::Error> {
        // NOTE: We'd like to use `serde_path_to_error` here but it has a bug with enums:
        //       https://github.com/dtolnay/path-to-error/issues/1
        match table.try_into() {
            Ok(config) => Ok(config),
            Err(e) => Err(Error::new(ErrorDetails::Config {
                message: format!("{e}"),
            })),
        }
    }
}

#[derive(Debug, Deserialize)]
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
    parallel_tool_calls: bool,
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
    pub fn load<P: AsRef<Path>>(self, base_path: P) -> Result<FunctionConfig, Error> {
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
                let implicit_tool = ToolConfig::Implicit(ImplicitToolConfig {
                    parameters: output_schema.clone(),
                });
                let implicit_tool_call_config = ToolCallConfig {
                    tools_available: vec![implicit_tool],
                    tool_choice: ToolChoice::Specific(IMPLICIT_TOOL_NAME.to_string()),
                    parallel_tool_calls: false,
                };
                let variants = params
                    .variants
                    .into_iter()
                    .map(|(name, variant)| variant.load(&base_path).map(|v| (name, v)))
                    .collect::<Result<HashMap<_, _>, Error>>()?;
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

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum UninitializedVariantConfig {
    ChatCompletion(ChatCompletionConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(BestOfNSamplingConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(UninitializedDiclConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfN(MixtureOfNConfig),
}

impl UninitializedVariantConfig {
    pub fn load<P: AsRef<Path>>(self, base_path: P) -> Result<VariantConfig, Error> {
        match self {
            UninitializedVariantConfig::ChatCompletion(params) => {
                Ok(VariantConfig::ChatCompletion(params))
            }
            UninitializedVariantConfig::BestOfNSampling(params) => {
                Ok(VariantConfig::BestOfNSampling(params))
            }
            UninitializedVariantConfig::Dicl(params) => {
                Ok(VariantConfig::Dicl(params.load(base_path)?))
            }
            UninitializedVariantConfig::MixtureOfN(params) => Ok(VariantConfig::MixtureOfN(params)),
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

#[cfg(test)]
mod tests {

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
            VariantConfig::ChatCompletion(chat_config) => &chat_config.json_mode,
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
            VariantConfig::ChatCompletion(chat_config) => &chat_config.json_mode,
            _ => panic!("Expected a chat completion variant"),
        };
        assert_eq!(prompt_b_json_mode, &JsonMode::On);
        // Check that the tool choice for get_weather is set to "specific" and the correct tool
        let function = config.functions.get("weather_helper").unwrap();
        match function {
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
        match function {
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

        // To test that variant default weights work correctly,
        // We check `functions.templates_with_variables_json.variants.variant_with_variables.weight`
        // This variant's weight is unspecified, so it should default to 0
        let json_function = config
            .functions
            .get("templates_with_variables_json")
            .unwrap();
        match json_function {
            FunctionConfig::Json(json_config) => {
                let variant = json_config.variants.get("variant_with_variables").unwrap();
                match variant {
                    VariantConfig::ChatCompletion(chat_config) => {
                        assert_eq!(chat_config.weight, 0.0); // Default weight should be 0
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
            .unwrap();
        assert_eq!(embedding_model.routing, vec!["openai".into()]);
        assert_eq!(embedding_model.providers.len(), 1);
        let provider = embedding_model.providers.get("openai").unwrap();
        assert!(matches!(provider, EmbeddingProviderConfig::OpenAI(_)));
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
                message: "invalid socket address syntax\nin `gateway.bind_address`\n".to_string()
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
                message: "missing field `providers`\nin `models.claude-3-haiku-20240307`\n"
                    .to_string()
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
                message: "Invalid api_key_location for Dummy provider\nin `models.dummy.providers.bad_credentials`\n"
                    .to_string()
            })
        );
    }

    /// Ensure that the config parsing fails when the `[functions]` section is missing
    #[test]
    fn test_config_from_toml_table_missing_functions() {
        let mut config = get_sample_valid_config();
        config
            .remove("functions")
            .expect("Failed to remove `[functions]` section");
        let base_path = PathBuf::new();
        let result = Config::load_from_toml(config, base_path);
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "missing field `functions`\n".to_string()
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
                message: "missing field `variants`\nin `functions.generate_draft`\n".to_string()
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
        let error = result.unwrap_err();
        assert!(error
            .to_string()
            .contains("Model name 'anthropic::claude-3-haiku-20240307' contains a reserved prefix\nin `models`"));
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
        let output_schema = match config.functions.get("json_with_schemas").unwrap() {
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
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`models.gpt-3.5-turbo`: `routing` must not be empty".to_string()
            }
            .into()
        );
    }

    /// Ensure that the config validation fails when there are duplicate routing entries
    #[test]
    fn test_config_validate_model_duplicate_routing_entry() {
        let mut config = get_sample_valid_config();
        config["models"]["gpt-3.5-turbo"]["routing"] =
            toml::Value::Array(vec!["openai".into(), "openai".into()]);
        let result = Config::load_from_toml(config, PathBuf::new());
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`models.gpt-3.5-turbo.routing`: duplicate entry `openai`".to_string()
            }
            .into()
        );
    }

    /// Ensure that the config validation fails when a routing entry does not exist in providers
    #[test]
    fn test_config_validate_model_routing_entry_not_in_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["gpt-3.5-turbo"]["routing"] = toml::Value::Array(vec!["closedai".into()]);
        let result = Config::load_from_toml(config, PathBuf::new());
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`models.gpt-3.5-turbo`: `routing` contains entry `closedai` that does not exist in `providers`".to_string()
            }
            .into()
        );
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

    /// Ensure that get_templates returns the correct templates
    #[test]
    fn test_get_all_templates() {
        let config_table = get_sample_valid_config();
        let config =
            Config::load_from_toml(config_table, PathBuf::new()).expect("Failed to load config");

        // Get all templates
        let templates = config.get_templates(PathBuf::from("/base/path"));

        // Check if all expected templates are present
        assert_eq!(
            templates.get("fixtures/config/functions/generate_draft/promptA/system_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/generate_draft/promptA/system_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/json_with_schemas/promptA/system_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/json_with_schemas/promptA/system_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/json_with_schemas/promptB/system_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/json_with_schemas/promptB/system_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/templates_without_variables/variant_without_templates/system_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/templates_without_variables/variant_without_templates/system_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/templates_without_variables/variant_without_templates/user_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/templates_without_variables/variant_without_templates/user_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/templates_without_variables/variant_without_templates/assistant_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/templates_without_variables/variant_without_templates/assistant_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja"
            ))
        );
        assert_eq!(
            templates.get("fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja"),
            Some(&PathBuf::from(
                "/base/path/fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja"
            ))
        );

        // Check the total number of templates
        assert_eq!(templates.len(), 10);
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

    /// Get a sample valid config for testing
    fn get_sample_valid_config() -> toml::Table {
        let config_str = r#"
        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                  GENERAL                                   │
        # └────────────────────────────────────────────────────────────────────────────┘

        [gateway]
        bind_address = "0.0.0.0:3000"

        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                   MODELS                                   │
        # └────────────────────────────────────────────────────────────────────────────┘

        [models."gpt-3.5-turbo"]
        routing = ["openai", "azure"]

        [models."gpt-3.5-turbo".providers.openai]
        type = "openai"
        model_name = "gpt-3.5-turbo"

        [models."gpt-3.5-turbo".providers.azure]
        type = "azure"
        deployment_id = "gpt-35-turbo"
        endpoint = "https://your-endpoint.openai.azure.com"

        [models.claude-3-haiku-20240307]
        routing = ["anthropic"]

        [models.claude-3-haiku-20240307.providers.anthropic]
        type = "anthropic"
        model_name = "claude-3-haiku-20240307"

        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                              EMBEDDING MODELS                              │
        # └────────────────────────────────────────────────────────────────────────────┘

        [embedding_models.text-embedding-3-small]
        routing = ["openai"]

        [embedding_models.text-embedding-3-small.providers.openai]
        type = "openai"
        model_name = "text-embedding-3-small"

        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                 FUNCTIONS                                  │
        # └────────────────────────────────────────────────────────────────────────────┘

        [functions.generate_draft]
        type = "chat"
        system_schema = "fixtures/config/functions/generate_draft/system_schema.json"

        [functions.generate_draft.variants.openai_promptA]
        type = "chat_completion"
        weight = 0.9
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/generate_draft/promptA/system_template.minijinja"

        [functions.generate_draft.variants.openai_promptB]
        type = "chat_completion"
        weight = 0.1
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/generate_draft/promptB/system_template.minijinja"

        [functions.json_with_schemas]
        type = "json"
        system_schema = "fixtures/config/functions/json_with_schemas/system_schema.json"
        output_schema = "fixtures/config/functions/json_with_schemas/output_schema.json"

        [functions.json_with_schemas.variants.openai_promptA]
        type = "chat_completion"
        weight = 0.9
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/json_with_schemas/promptA/system_template.minijinja"
        json_mode = "implicit_tool"

        [functions.json_with_schemas.variants.openai_promptB]
        type = "chat_completion"
        weight = 0.1
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/json_with_schemas/promptB/system_template.minijinja"

        [functions.weather_helper]
        type = "chat"
        tools = ["get_temperature"]
        tool_choice = {specific = "get_temperature"}

        [functions.weather_helper.variants.openai_promptA]
        type = "chat_completion"
        weight = 1.0
        model = "gpt-3.5-turbo"

        [functions.templates_without_variables_chat]
        type = "chat"

        [functions.templates_without_variables_chat.variants.variant_without_templates]
        type = "chat_completion"
        weight = 1.0
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/templates_without_variables/variant_without_templates/system_template.minijinja"
        user_template = "fixtures/config/functions/templates_without_variables/variant_without_templates/user_template.minijinja"
        assistant_template = "fixtures/config/functions/templates_without_variables/variant_without_templates/assistant_template.minijinja"

        [functions.templates_with_variables_chat]
        type = "chat"
        system_schema = "fixtures/config/functions/templates_with_variables/system_schema.json"
        user_schema = "fixtures/config/functions/templates_with_variables/user_schema.json"
        assistant_schema = "fixtures/config/functions/templates_with_variables/assistant_schema.json"

        [functions.templates_with_variables_chat.variants.variant_with_variables]
        type = "chat_completion"
        weight = 1.0
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja"
        user_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja"
        assistant_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja"

        [functions.templates_with_variables_chat.variants.best_of_n]
        type = "experimental_best_of_n_sampling"
        weight = 1.0
        candidates = ["variant_with_variables", "variant_with_variables"]

        [functions.templates_with_variables_chat.variants.best_of_n.evaluator]
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja"
        user_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja"
        assistant_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja"

        [functions.templates_without_variables_json]
        type = "json"
        output_schema = "fixtures/config/functions/json_with_schemas/output_schema.json"

        [functions.templates_without_variables_json.variants.variant_without_templates]
        type = "chat_completion"
        weight = 1.0
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/templates_without_variables/variant_without_templates/system_template.minijinja"
        user_template = "fixtures/config/functions/templates_without_variables/variant_without_templates/user_template.minijinja"
        assistant_template = "fixtures/config/functions/templates_without_variables/variant_without_templates/assistant_template.minijinja"

        [functions.templates_with_variables_json]
        type = "json"
        system_schema = "fixtures/config/functions/templates_with_variables/system_schema.json"
        user_schema = "fixtures/config/functions/templates_with_variables/user_schema.json"
        assistant_schema = "fixtures/config/functions/templates_with_variables/assistant_schema.json"
        output_schema = "fixtures/config/functions/json_with_schemas/output_schema.json"

        [functions.templates_with_variables_json.variants.variant_with_variables]
        type = "chat_completion"
        model = "gpt-3.5-turbo"
        system_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja"
        user_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja"
        assistant_template = "fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja"

        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                  METRICS                                   │
        # └────────────────────────────────────────────────────────────────────────────┘

        [metrics.task_success]
        type = "boolean"
        optimize = "max"
        level = "inference"

        [metrics.user_rating]
        type = "float"
        optimize = "max"
        level = "episode"

        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                   TOOLS                                    │
        # └────────────────────────────────────────────────────────────────────────────┘
        [tools.get_temperature]
        description = "Get the weather for a given location"
        parameters = "fixtures/config/tools/get_temperature.json"
        "#;
        env::set_var("OPENAI_API_KEY", "sk-something");
        env::set_var("ANTHROPIC_API_KEY", "sk-something");
        env::set_var("AZURE_OPENAI_API_KEY", "sk-something");

        toml::from_str(config_str).expect("Failed to parse sample config")
    }

    #[test]
    fn test_tensorzero_example_file() {
        env::set_var("OPENAI_API_KEY", "sk-something");
        env::set_var("ANTHROPIC_API_KEY", "sk-something");
        env::set_var("AZURE_OPENAI_API_KEY", "sk-something");
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let config_path = format!("{}/fixtures/config/tensorzero.toml", manifest_dir);
        let config_pathbuf = PathBuf::from(&config_path);
        let base_path = config_pathbuf
            .parent()
            .expect("Failed to get parent directory of config file");
        let config_table = UninitializedConfig::read_toml_config(&config_path)
            .expect("Failed to read tensorzero.example.toml");

        Config::load_from_toml(config_table, base_path.to_path_buf())
            .expect("Failed to load config");
    }
}
