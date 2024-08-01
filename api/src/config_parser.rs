use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::Error;
use crate::function::FunctionConfig;
use crate::minijinja_util::initialize_templates;
use crate::model::ModelConfig;
use crate::variant::VariantConfig;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub models: HashMap<String, ModelConfig>, // model name => model config
    pub functions: HashMap<String, FunctionConfig>, // function name => function config
    pub metrics: Option<HashMap<String, MetricConfig>>, // metric name => metric config
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricConfig {
    pub r#type: MetricConfigType,
    pub optimize: MetricConfigOptimize,
    pub level: MetricConfigLevel,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricConfigType {
    Boolean,
    Float,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricConfigOptimize {
    Min,
    Max,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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

/// Deserialize a TOML table into `Config`
impl From<toml::Table> for Config {
    fn from(table: toml::Table) -> Self {
        // TODO: We'd like to use `serde_path_to_error` here but it has a bug with enums:
        //       https://github.com/dtolnay/path-to-error/issues/1
        table.try_into().unwrap_or_else(|e| {
            panic!("Failed to parse config:\n{e}");
        })
    }
}

impl Config {
    /// Load and validate the TensorZero config file
    pub fn load() -> Config {
        let config_path = Config::get_config_path();
        let config_table = Config::read_toml_config(&config_path);
        let config = Config::from(config_table);
        config.validate();
        initialize_templates(&config.get_templates());
        config
    }

    /// Get the path for the TensorZero config file
    ///
    /// Use a path provided as a CLI argument (`./api path/to/tensorzero.toml`), or default to
    /// `tensorzero.toml` in the current directory if no path is provided.
    fn get_config_path() -> String {
        match std::env::args().nth(1) {
            Some(path) => path,
            None => "tensorzero.toml".to_string(),
        }
    }

    /// Read a file from the file system and parse it as TOML
    fn read_toml_config(path: &str) -> toml::Table {
        std::fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("Failed to read config file: {path}"))
            .parse::<toml::Table>()
            .expect("Failed to parse config file as valid TOML")
    }

    /// Validate the config
    fn validate(&self) {
        // Validate each model
        for (model_name, model) in &self.models {
            // Ensure that the model has at least one provider
            assert!(
                !model.routing.is_empty(),
                "Invalid Config: `models.{model_name}`: `routing` must not be empty",
            );

            // Ensure that routing entries are unique and exist as keys in providers
            let mut seen_providers = std::collections::HashSet::new();
            for provider in &model.routing {
                assert!(
                    seen_providers.insert(provider),
                    "Invalid Config: `models.{model_name}.routing`: duplicate entry `{provider}`",
                );

                assert!(
                    model.providers.contains_key(provider),
                    "Invalid Config: `models.{model_name}`: `routing` contains entry `{provider}` that does not exist in `providers`",
                );
            }

            // Validate each provider
            for provider_name in model.providers.keys() {
                assert!(
                    seen_providers.contains(provider_name),
                    "Invalid Config: `models.{model_name}`: Provider `{provider_name}` is not listed in `routing`",
                );
            }
        }

        // Validate each function
        for (function_name, function) in &self.functions {
            // Validate each variant
            for (variant_name, variant) in function.variants() {
                // Ensure that the weight is non-negative
                assert!(
                    variant.weight() >= 0.0,
                    "Invalid Config: `functions.{function_name}.variants.{variant_name}.weight`: must be non-negative",
                );

                // Ensure that the variant type is correct
                match function {
                    FunctionConfig::Chat(function) => {
                        // Check that the variant type matches the function type
                        assert!(
                            matches!(variant, VariantConfig::ChatCompletion(_)),
                            "Invalid Config: `functions.{function_name}.variants.{variant_name}`: variant type must be `chat_completion`",
                        );

                        // Check that system schema <=> system template
                        match (&function.system_schema, &variant.system_template()) {
                            (Some(_), None) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `system_template` is required when `system_schema` is specified");
                            }
                            (None, Some(_)) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `system_template` is required when `system_schema` is specified");
                            }
                            _ => {}
                        }

                        // Check that user schema <=> user template
                        match (&function.user_schema, &variant.user_template()) {
                            (Some(_), None) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `user_template` is required when `user_schema` is specified");
                            }
                            (None, Some(_)) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `user_template` is required when `user_schema` is specified");
                            }
                            _ => {}
                        }

                        // Check that assistant schema <=> assistant template
                        match (&function.assistant_schema, &variant.assistant_template()) {
                            (Some(_), None) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `assistant_template` is required when `assistant_schema` is specified");
                            }
                            (None, Some(_)) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `assistant_template` is required when `assistant_schema` is specified");
                            }
                            _ => {}
                        }
                    }
                    FunctionConfig::Tool(function) => {
                        // Check that the variant type matches the function type
                        assert!(
                            matches!(variant, VariantConfig::ChatCompletion(_)),
                            "Invalid Config: `functions.{function_name}.variants.{variant_name}`: variant type must be `chat_completion`",
                        );

                        // Check that system schema <=> system template
                        match (&function.system_schema, &variant.system_template()) {
                            (Some(_), None) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `system_template` is required when `system_schema` is specified");
                            }
                            (None, Some(_)) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `system_template` is required when `system_schema` is specified");
                            }
                            _ => {}
                        }

                        // Check that user schema <=> user template
                        match (&function.user_schema, &variant.user_template()) {
                            (Some(_), None) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `user_template` is required when `user_schema` is specified");
                            }
                            (None, Some(_)) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `user_template` is required when `user_schema` is specified");
                            }
                            _ => {}
                        }

                        // Check that assistant schema <=> assistant template
                        match (&function.assistant_schema, &variant.assistant_template()) {
                            (Some(_), None) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `assistant_template` is required when `assistant_schema` is specified");
                            }
                            (None, Some(_)) => {
                                panic!("Invalid Config: `functions.{function_name}.variants.{variant_name}`: `assistant_template` is required when `assistant_schema` is specified");
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Ensure that no metrics are named "comment" or "demonstration"
        if let Some(metrics) = &self.metrics {
            for metric_name in metrics.keys() {
                assert!(
                    metric_name != "comment" && metric_name != "demonstration",
                    "Invalid Config: Metric name '{}' is reserved and cannot be used",
                    metric_name
                );
            }
        }
    }

    /// Get a function by name
    pub fn get_function<'a>(&'a self, function_name: &str) -> Result<&'a FunctionConfig, Error> {
        self.functions
            .get(function_name)
            .ok_or_else(|| Error::UnknownFunction {
                name: function_name.to_string(),
            })
    }

    /// Get a metric by name
    pub fn get_metric<'a>(&'a self, metric_name: &str) -> Result<&'a MetricConfig, Error> {
        self.metrics
            .as_ref()
            .and_then(|metrics| metrics.get(metric_name))
            .ok_or_else(|| Error::UnknownMetric {
                name: metric_name.to_string(),
            })
    }

    /// Get all templates from the config
    pub fn get_templates(&self) -> Vec<&PathBuf> {
        let mut templates = Vec::new();
        for function in self.functions.values() {
            for variant in function.variants().values() {
                match variant {
                    VariantConfig::ChatCompletion(chat_config) => {
                        if let Some(ref path) = chat_config.system_template {
                            templates.push(path);
                        }
                        if let Some(ref path) = chat_config.user_template {
                            templates.push(path);
                        }
                        if let Some(ref path) = chat_config.assistant_template {
                            templates.push(path);
                        }
                    }
                }
            }
        }
        templates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensure that the sample valid config can be parsed without panicking
    #[test]
    fn test_config_from_toml_table_valid() {
        let config = get_sample_valid_config();
        let _ = Config::from(config);

        // Ensure that removing the `[metrics]` section still parses the config
        let mut config = get_sample_valid_config();
        config
            .remove("metrics")
            .expect("Failed to remove `[metrics]` section");
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when the `[models]` section is missing
    #[test]
    #[should_panic(expected = "Failed to parse config:\nmissing field `models`\n")]
    fn test_config_from_toml_table_missing_models() {
        let mut config = get_sample_valid_config();
        config
            .remove("models")
            .expect("Failed to remove `[models]` section");
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when the `[providers]` section is missing
    #[test]
    #[should_panic(expected = "Failed to parse config:\nmissing field `providers`")]
    fn test_config_from_toml_table_missing_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307` section")
            .remove("providers")
            .expect("Failed to remove `[providers]` section");
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when the `[functions]` section is missing
    #[test]
    #[should_panic(expected = "Failed to parse config:\nmissing field `functions`\n")]
    fn test_config_from_toml_table_missing_functions() {
        let mut config = get_sample_valid_config();
        config
            .remove("functions")
            .expect("Failed to remove `[functions]` section");
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when the `[variants]` section is missing
    #[test]
    #[should_panic(expected = "Failed to parse config:\nmissing field `variants`")]
    fn test_config_from_toml_table_missing_variants() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft` section")
            .remove("variants")
            .expect("Failed to remove `[variants]` section");
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when there are extra variables at the root level
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nunknown field `enable_agi`, expected one of"
    )]
    fn test_config_from_toml_table_extra_variables_root() {
        let mut config = get_sample_valid_config();
        config.insert("enable_agi".into(), true.into());
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when there are extra variables for models
    #[test]
    #[should_panic(expected = "Failed to parse config:\nunknown field `enable_agi`, expected")]
    fn test_config_from_toml_table_extra_variables_models() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307` section")
            .insert("enable_agi".into(), true.into());
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when there are extra variables for providers
    #[test]
    #[should_panic(expected = "Failed to parse config:\nunknown field `enable_agi`, expected")]
    fn test_config_from_toml_table_extra_variables_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]["providers"]["anthropic"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307.providers.anthropic` section")
            .insert("enable_agi".into(), true.into());
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when there are extra variables for functions
    #[test]
    #[should_panic(expected = "Failed to parse config:\nunknown field `enable_agi`, expected")]
    fn test_config_from_toml_table_extra_variables_functions() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft` section")
            .insert("enable_agi".into(), true.into());
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when there are extra variables for variants
    #[test]
    #[should_panic(expected = "Failed to parse config:\nunknown field `enable_agi`, expected")]
    fn test_config_from_toml_table_extra_variables_variants() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]["variants"]["openai_promptA"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft.variants.openai_promptA` section")
            .insert("enable_agi".into(), true.into());
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when there are extra variables for metrics
    #[test]
    #[should_panic(expected = "Failed to parse config:\nunknown field `enable_agi`, expected")]
    fn test_config_from_toml_table_extra_variables_metrics() {
        let mut config = get_sample_valid_config();
        config["metrics"]["task_success"]
            .as_table_mut()
            .expect("Failed to get `metrics.task_success` section")
            .insert("enable_agi".into(), true.into());
        let _ = Config::from(config);
    }

    /// Ensure that the config validation panics when a model has no providers in `routing`
    #[test]
    #[should_panic(
        expected = "Invalid Config: `models.gpt-3.5-turbo`: `routing` must not be empty"
    )]
    fn test_config_validate_model_empty_providers() {
        let mut config = Config::from(get_sample_valid_config());
        config
            .models
            .get_mut("gpt-3.5-turbo")
            .expect("Failed to get `models.gpt-3.5-turbo`")
            .routing
            .clear();
        config.validate();
    }

    /// Ensure that the config validation panics when there are duplicate routing entries
    #[test]
    #[should_panic(
        expected = "Invalid Config: `models.gpt-3.5-turbo.routing`: duplicate entry `openai`"
    )]
    fn test_config_validate_model_duplicate_routing_entry() {
        let mut config = Config::from(get_sample_valid_config());
        config
            .models
            .get_mut("gpt-3.5-turbo")
            .expect("Failed to get `models.gpt-3.5-turbo`")
            .routing
            .push("openai".into());
        config.validate();
    }

    /// Ensure that the config validation panics when a routing entry does not exist in providers
    #[test]
    #[should_panic(
        expected = "Invalid Config: `models.gpt-3.5-turbo`: `routing` contains entry `closedai` that does not exist in `providers`"
    )]
    fn test_config_validate_model_routing_entry_not_in_providers() {
        let mut config = Config::from(get_sample_valid_config());
        config
            .models
            .get_mut("gpt-3.5-turbo")
            .expect("Failed to get `models.gpt-3.5-turbo`")
            .routing
            .push("closedai".into());
        config.validate();
    }

    /// Ensure that the config validation panics when a function variant has a negative weight
    #[test]
    #[should_panic(
        expected = "Invalid Config: `functions.generate_draft.variants.openai_promptA.weight`: must be non-negative"
    )]
    fn test_config_validate_function_variant_negative_weight() {
        let mut config = Config::from(get_sample_valid_config());
        match config
            .functions
            .get_mut("generate_draft")
            .expect("Failed to get `functions.generate_draft`")
        {
            FunctionConfig::Chat(params) => {
                match params.variants.get_mut("openai_promptA").unwrap() {
                    VariantConfig::ChatCompletion(params) => {
                        params.weight = -1.0;
                    }
                }
            }
            _ => panic!(),
        }
        config.validate();
    }

    /// Get a sample valid config for testing
    fn get_sample_valid_config() -> toml::Table {
        let config_str = r#"
        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                  GENERAL                                   │
        # └────────────────────────────────────────────────────────────────────────────┘

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
        model_name = "gpt-35-turbo"
        deployment_id = "gpt-35-turbo"
        api_base = "https://your-endpoint.openai.azure.com/"

        [models.claude-3-haiku-20240307]
        routing = ["anthropic"]

        [models.claude-3-haiku-20240307.providers.anthropic]
        type = "anthropic"
        model_name = "claude-3-haiku-20240307"

        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                 FUNCTIONS                                  │
        # └────────────────────────────────────────────────────────────────────────────┘

        [functions.generate_draft]
        type = "chat"
        system_schema = "../functions/generate_draft/system_schema.json"
        output_schema = "../functions/generate_draft/output_schema.json"

        [functions.generate_draft.variants.openai_promptA]
        type = "chat_completion"
        weight = 0.9
        model = "gpt-3.5-turbo"
        system_template = "../functions/generate_draft/promptA/system.jinja"

        [functions.generate_draft.variants.openai_promptB]
        type = "chat_completion"
        weight = 0.1
        model = "gpt-3.5-turbo"
        system_template = "../functions/generate_draft/promptB/system.jinja"

        [functions.extract_data]
        type = "tool"
        system_schema = "../functions/extract_data/system_schema.json"
        output_schema = "../functions/extract_data/output_schema.json"

        [functions.extract_data.variants.openai_promptA]
        type = "chat_completion"
        weight = 0.9
        model = "gpt-3.5-turbo"
        system_template = "../functions/extract_data/promptA/system.jinja"

        [functions.extract_data.variants.openai_promptB]
        type = "chat_completion"
        weight = 0.1
        model = "gpt-3.5-turbo"
        system_template = "../functions/extract_data/promptB/system.jinja"

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
        "#;

        toml::from_str(config_str).expect("Failed to parse sample config")
    }
}
