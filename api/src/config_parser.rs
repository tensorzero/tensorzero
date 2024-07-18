use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[allow(dead_code)] // TODO: temporary
    pub models: HashMap<String, ModelConfig>, // model name => model config
    #[allow(dead_code)] // TODO: temporary
    pub functions: HashMap<String, FunctionConfig>, // function name => function config
    #[allow(dead_code)] // TODO: temporary
    pub metrics: Option<HashMap<String, MetricConfig>>, // metric name => metric config
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    #[allow(dead_code)] // TODO: temporary
    pub routing: Vec<String>, // [provider name A, provider name B, ...]
    #[allow(dead_code)] // TODO: temporary
    pub providers: HashMap<String, ProviderConfig>, // provider name => provider config
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    #[allow(dead_code)] // TODO: temporary
    pub r#type: ProviderConfigType,
    // TODO: consider moving name and api_base to a provider-specific child object (based on `type`)
    #[allow(dead_code)] // TODO: temporary
    pub name: String,
    #[allow(dead_code)] // TODO: temporary
    pub api_base: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderConfigType {
    #[serde(rename = "openai")]
    OpenAI,
    Anthropic,
    Azure,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConfig {
    #[allow(dead_code)] // TODO: temporary
    pub r#type: FunctionConfigType,
    // TODO: consider moving {user|assistant|system}_schema to a "chat" child object
    #[allow(dead_code)] // TODO: temporary
    pub system_schema: Option<String>,
    #[allow(dead_code)] // TODO: temporary
    pub user_schema: Option<String>,
    #[allow(dead_code)] // TODO: temporary
    pub assistant_schema: Option<String>,
    #[allow(dead_code)] // TODO: temporary
    pub output_schema: Option<String>,
    #[allow(dead_code)] // TODO: temporary
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionConfigType {
    Chat,
    Tool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariantConfig {
    #[allow(dead_code)] // TODO: temporary
    pub weight: f64,
    #[allow(dead_code)] // TODO: temporary
    pub generation: Option<GenerationConfig>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    #[allow(dead_code)] // TODO: temporary
    pub model: String,
    #[allow(dead_code)] // TODO: temporary
    pub system_template: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricConfig {
    #[allow(dead_code)] // TODO: temporary
    pub r#type: MetricConfigType,
    #[allow(dead_code)] // TODO: temporary
    pub optimize: MetricConfigOptimize,
    #[allow(dead_code)] // TODO: temporary
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

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricConfigLevel {
    Inference,
    Episode,
}

/// Deserialize a TOML table into `Config`
impl From<toml::Table> for Config {
    fn from(table: toml::Table) -> Self {
        serde_path_to_error::deserialize(table).unwrap_or_else(|e| {
            panic!("Failed to parse config:\n{e}");
        })
    }
}

impl Config {
    /// Load and validate the TensorZero config file
    pub fn load() -> Config {
        let config_path = Config::get_config_path();
        let config_table = Config::read_toml_config(&config_path);
        #[allow(clippy::let_and_return)] // TODO: temporary
        let config = Config::from(config_table);
        config.validate();
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
                "Invalid Config: `models.{model_name}`: `providers` must not be empty",
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
            for (provider_name, provider) in model.providers.iter() {
                // Ensure that the provider has the necessary fields
                #[allow(clippy::single_match)] // TODO: temporary
                match provider.r#type {
                    ProviderConfigType::Azure => {
                        assert!(
                            provider.api_base.is_some(),
                            "Invalid Config: `models.{model_name}.providers.{provider_name}`: Azure provider requires `api_base`",
                        );
                    }
                    _ => {}
                }
            }
        }

        // Validate each function
        for (function_name, function) in &self.functions {
            for (variant_name, variant) in &function.variants {
                assert!(
                    variant.weight >= 0.0,
                    "Invalid Config: `functions.{function_name}.variants.{variant_name}.weight`: must be non-negative",
                );

                match function.r#type {
                    FunctionConfigType::Chat | FunctionConfigType::Tool => {
                        assert!(
                            variant.generation.is_some(),
                            "Invalid Config: `functions.{function_name}.variants.{variant_name}`: `generation` is required",
                        );
                    }
                }
            }
        }

        // NOTE: There is nothing to validate in metrics for now
        // if let Some(metrics) = &self.metrics {
        //     for (metric_name, metric) in metrics {
        //         // ...
        //     }
        // }
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
    #[should_panic(
        expected = "Failed to parse config:\nmodels.claude-3-haiku-20240307: missing field `providers`\n"
    )]
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
    #[should_panic(
        expected = "Failed to parse config:\nfunctions.generate_draft: missing field `variants`\n"
    )]
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
        expected = "Failed to parse config:\nenable_agi: unknown field `enable_agi`, expected"
    )]
    fn test_config_from_toml_table_extra_variables_root() {
        let mut config = get_sample_valid_config();
        config.insert("enable_agi".into(), true.into());
        let _ = Config::from(config);
    }

    /// Ensure that the config parsing panics when there are extra variables for models
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nmodels.claude-3-haiku-20240307.enable_agi: unknown field `enable_agi`, expected"
    )]
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
    #[should_panic(
        expected = "Failed to parse config:\nmodels.claude-3-haiku-20240307.providers.anthropic.enable_agi: unknown field `enable_agi`, expected"
    )]
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
    #[should_panic(
        expected = "Failed to parse config:\nfunctions.generate_draft.enable_agi: unknown field `enable_agi`, expected"
    )]
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
    #[should_panic(
        expected = "Failed to parse config:\nfunctions.generate_draft.variants.openai_promptA.enable_agi: unknown field `enable_agi`, expected"
    )]
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
    #[should_panic(
        expected = "Failed to parse config:\nmetrics.task_success.enable_agi: unknown field `enable_agi`, expected"
    )]
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
        expected = "Invalid Config: `models.gpt-3.5-turbo`: `providers` must not be empty"
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

    /// Ensure that the config validation panics when a model has an Azure provider without `api_base`
    #[test]
    #[should_panic(
        expected = "Invalid Config: `models.gpt-3.5-turbo.providers.azure`: Azure provider requires `api_base`"
    )]
    fn test_config_validate_model_azure_provider_missing_api_base() {
        let mut config = Config::from(get_sample_valid_config());
        config
            .models
            .get_mut("gpt-3.5-turbo")
            .expect("Failed to get `models.gpt-3.5-turbo`")
            .providers
            .get_mut("azure")
            .expect("Failed to get `models.gpt-3.5-turbo.providers.azure`")
            .api_base = None;
        config.validate();
    }

    /// Ensure that the config validation panics when a function variant has a negative weight
    #[test]
    #[should_panic(
        expected = "Invalid Config: `functions.generate_draft.variants.openai_promptA.weight`: must be non-negative"
    )]
    fn test_config_validate_function_variant_negative_weight() {
        let mut config = Config::from(get_sample_valid_config());
        config
            .functions
            .get_mut("generate_draft")
            .expect("Failed to get `functions.generate_draft`")
            .variants
            .get_mut("openai_promptA")
            .expect("Failed to get `functions.generate_draft.variants.openai_promptA`")
            .weight = -1.0;
        config.validate();
    }

    /// Ensure that the config validation panics when a `chat` or `tool` function variant has no `generation`
    #[test]
    #[should_panic(
        expected = "Invalid Config: `functions.generate_draft.variants.openai_promptA`: `generation` is required"
    )]
    fn test_config_validate_function_variant_missing_generation() {
        let mut config = Config::from(get_sample_valid_config());
        config
            .functions
            .get_mut("generate_draft")
            .expect("Failed to get `functions.generate_draft`")
            .variants
            .get_mut("openai_promptA")
            .expect("Failed to get `functions.generate_draft.variants.openai_promptA`")
            .generation = None;
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
        name = "gpt-3.5-turbo"

        [models."gpt-3.5-turbo".providers.azure]
        type = "azure"
        name = "gpt-35-turbo"
        api_base = "https://your-endpoint.openai.azure.com/"

        [models.claude-3-haiku-20240307]
        routing = ["anthropic"]

        [models.claude-3-haiku-20240307.providers.anthropic]
        type = "anthropic"
        name = "claude-3-haiku-20240307"

        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                 FUNCTIONS                                  │
        # └────────────────────────────────────────────────────────────────────────────┘

        [functions.generate_draft]
        type = "chat"  # "chat", "tool"
        system_schema = "to/do.json"
        output_schema = "to/do.json"

        [functions.generate_draft.variants.openai_promptA]
        weight = 0.9
        generation.model = "gpt-3.5-turbo"
        generation.system_template = "to/do/promptA/system.jinja"

        [functions.generate_draft.variants.openai_promptB]
        weight = 0.1
        generation.model = "gpt-3.5-turbo"
        generation.system_template = "to/do/promptB/system.jinja"

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
