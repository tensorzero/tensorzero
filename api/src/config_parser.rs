use serde::Deserialize;
use std::collections::HashMap;

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │                                    TYPES                                     │
// └──────────────────────────────────────────────────────────────────────────────┘

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
    #[allow(dead_code)] // TODO: temporary
    pub system_schema: String,
    #[allow(dead_code)] // TODO: temporary
    pub output_schema: String,
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

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │                                  FUNCTIONS                                   │
// └──────────────────────────────────────────────────────────────────────────────┘
/// Load and validate the TensorZero config file
pub fn get_config() -> Config {
    let config_path = get_config_path();
    let config_table = read_toml_config(&config_path);
    #[allow(clippy::let_and_return)] // TODO: temporary
    let config = parse_toml_config(config_table);

    // TODO: sanity check config (e.g. routing has corresponding models, weights are non-negative)

    config
}

/// Get the path for `tensorzero.toml`
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

/// Deserialize a TOML table into a `Config`
fn parse_toml_config(config_table: toml::Table) -> Config {
    serde_path_to_error::deserialize(config_table).unwrap_or_else(|e| {
        panic!("Failed to parse config:\n{e}");
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensure that the sample valid config can be parsed without panicking
    #[test]
    fn test_parse_toml_config_valid() {
        let config = get_sample_valid_config();
        let _ = parse_toml_config(config);

        // Ensure that removing the `[metrics]` section still parses the config
        let mut config = get_sample_valid_config();
        config
            .remove("metrics")
            .expect("Failed to remove `[metrics]` section");
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when the `[models]` section is missing
    #[test]
    #[should_panic(expected = "Failed to parse config:\nmissing field `models`\n")]
    fn test_parse_toml_config_missing_models() {
        let mut config = get_sample_valid_config();
        config
            .remove("models")
            .expect("Failed to remove `[models]` section");
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when the `[providers]` section is missing
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nmodels.claude-3-haiku-20240307: missing field `providers`\n"
    )]
    fn test_parse_toml_config_missing_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307` section")
            .remove("providers")
            .expect("Failed to remove `[providers]` section");
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when the `[functions]` section is missing
    #[test]
    #[should_panic(expected = "Failed to parse config:\nmissing field `functions`\n")]
    fn test_parse_toml_config_missing_functions() {
        let mut config = get_sample_valid_config();
        config
            .remove("functions")
            .expect("Failed to remove `[functions]` section");
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when the `[variants]` section is missing
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nfunctions.generate_draft: missing field `variants`\n"
    )]
    fn test_parse_toml_config_missing_variants() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft` section")
            .remove("variants")
            .expect("Failed to remove `[variants]` section");
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when there are extra variables at the root level
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nenable_agi: unknown field `enable_agi`, expected"
    )]
    fn test_parse_toml_config_extra_variables_root() {
        let mut config = get_sample_valid_config();
        config.insert("enable_agi".into(), true.into());
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when there are extra variables for models
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nmodels.claude-3-haiku-20240307.enable_agi: unknown field `enable_agi`, expected"
    )]
    fn test_parse_toml_config_extra_variables_models() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307` section")
            .insert("enable_agi".into(), true.into());
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when there are extra variables for providers
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nmodels.claude-3-haiku-20240307.providers.anthropic.enable_agi: unknown field `enable_agi`, expected"
    )]
    fn test_parse_toml_config_extra_variables_providers() {
        let mut config = get_sample_valid_config();
        config["models"]["claude-3-haiku-20240307"]["providers"]["anthropic"]
            .as_table_mut()
            .expect("Failed to get `models.claude-3-haiku-20240307.providers.anthropic` section")
            .insert("enable_agi".into(), true.into());
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when there are extra variables for functions
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nfunctions.generate_draft.enable_agi: unknown field `enable_agi`, expected"
    )]
    fn test_parse_toml_config_extra_variables_functions() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft` section")
            .insert("enable_agi".into(), true.into());
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when there are extra variables for variants
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nfunctions.generate_draft.variants.openai_promptA.enable_agi: unknown field `enable_agi`, expected"
    )]
    fn test_parse_toml_config_extra_variables_variants() {
        let mut config = get_sample_valid_config();
        config["functions"]["generate_draft"]["variants"]["openai_promptA"]
            .as_table_mut()
            .expect("Failed to get `functions.generate_draft.variants.openai_promptA` section")
            .insert("enable_agi".into(), true.into());
        let _ = parse_toml_config(config);
    }

    /// Ensure that the config panics when there are extra variables for metrics
    #[test]
    #[should_panic(
        expected = "Failed to parse config:\nmetrics.task_success.enable_agi: unknown field `enable_agi`, expected"
    )]
    fn test_parse_toml_config_extra_variables_metrics() {
        let mut config = get_sample_valid_config();
        config["metrics"]["task_success"]
            .as_table_mut()
            .expect("Failed to get `metrics.task_success` section")
            .insert("enable_agi".into(), true.into());
        let _ = parse_toml_config(config);
    }

    /// Get a sample valid config for testing
    fn get_sample_valid_config() -> toml::Table {
        let config_str = r#"
        # ┌──────────────────────────────────────────────────────────────────────────────┐
        # │                                   GENERAL                                    │
        # └──────────────────────────────────────────────────────────────────────────────┘

        # ┌────────────────────────────────────────────────────────────────────────────┐
        # │                                   MODELS                                   │
        # └────────────────────────────────────────────────────────────────────────────┘

        [models."gpt-3.5-turbo"]
        routing = ["openai", "azure"]

        [models."gpt-3.5-turbo".providers.openai]
        type = "openai"
        name = "gpt-3.5-turbo"

        [models."gpt-3.5-turbo".providers.azure]
        type = "openai"
        name = "gpt-35-turbo"
        api_base = "https://your-endpoint.openai.azure.com/"

        [models.claude-3-haiku-20240307]
        routing = ["anthropic"]

        [models.claude-3-haiku-20240307.providers.anthropic]
        type = "anthropic"
        name = "claude-3-haiku-20240307"

        # ┌──────────────────────────────────────────────────────────────────────────────┐
        # │                                  FUNCTIONS                                   │
        # └──────────────────────────────────────────────────────────────────────────────┘

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

        # ┌──────────────────────────────────────────────────────────────────────────────┐
        # │                                   METRICS                                    │
        # └──────────────────────────────────────────────────────────────────────────────┘

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
