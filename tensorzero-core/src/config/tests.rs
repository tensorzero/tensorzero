use super::*;
use std::{io::Write, path::PathBuf};
use tempfile::NamedTempFile;
use toml::de::DeTable;

use std::env;

use crate::{embeddings::EmbeddingProviderConfig, inference::types::Role, variant::JsonMode};

/// Ensure that the sample valid config can be parsed without panicking
#[tokio::test]
async fn test_config_from_toml_table_valid() {
    let config = get_sample_valid_config();

    Config::load_from_toml(config)
        .await
        .expect("Failed to load config");

    // Ensure that removing the `[metrics]` section still parses the config
    let mut config = get_sample_valid_config();
    config
        .remove("metrics")
        .expect("Failed to remove `[metrics]` section");
    let ConfigLoadInfo { config, .. } = Config::load_from_toml(config)
        .await
        .expect("Failed to load config");

    // Check that the JSON mode is set properly on the JSON variants
    let prompt_a_json_mode = match &config
        .functions
        .get("json_with_schemas")
        .unwrap()
        .variants()
        .get("openai_promptA")
        .unwrap()
        .inner
    {
        VariantConfig::ChatCompletion(chat_config) => chat_config.json_mode().unwrap(),
        _ => panic!("Expected a chat completion variant"),
    };
    assert_eq!(prompt_a_json_mode, &JsonMode::Tool);

    let prompt_b_json_mode = match &config
        .functions
        .get("json_with_schemas")
        .unwrap()
        .variants()
        .get("openai_promptB")
        .unwrap()
        .inner
    {
        VariantConfig::ChatCompletion(chat_config) => chat_config.json_mode(),
        _ => panic!("Expected a chat completion variant"),
    };
    assert_eq!(prompt_b_json_mode, Some(&JsonMode::Strict));
    // Check that the tool choice for get_weather is set to "specific" and the correct tool
    let function = config.functions.get("weather_helper").unwrap();
    match &**function {
        FunctionConfig::Chat(chat_config) => {
            assert_eq!(
                chat_config.tool_choice,
                ToolChoice::Specific("get_temperature".to_string())
            );
        }
        FunctionConfig::Json(_) => panic!("Expected a chat function"),
    }
    // Check that the best of n variant has multiple candidates
    let function = config
        .functions
        .get("templates_with_variables_chat")
        .unwrap();
    match &**function {
        FunctionConfig::Chat(chat_config) => {
            if let Some(variant) = chat_config.variants.get("best_of_n") {
                match &variant.inner {
                    VariantConfig::BestOfNSampling(best_of_n_config) => {
                        assert!(
                            best_of_n_config.candidates().len() > 1,
                            "Best of n variant should have multiple candidates"
                        );
                    }
                    _ => panic!("Expected a best of n variant"),
                }
            } else {
                panic!("Expected to find a best of n variant");
            }
        }
        FunctionConfig::Json(_) => panic!("Expected a chat function"),
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
            match &variant.inner {
                VariantConfig::ChatCompletion(chat_config) => {
                    assert_eq!(chat_config.weight(), None); // Default weight should be None
                }
                _ => panic!("Expected a chat completion variant"),
            }
        }
        FunctionConfig::Chat(_) => panic!("Expected a JSON function"),
    }

    assert_eq!(config.embedding_models.table.len(), 1);

    let embedding_model = config
        .embedding_models
        .get("text-embedding-3-small")
        .await
        .expect("Error getting embedding model")
        .unwrap();
    assert_eq!(embedding_model.routing, vec!["openai".into()]);
    assert_eq!(embedding_model.providers.len(), 1);
    let provider = embedding_model.providers.get("openai").unwrap();
    assert!(matches!(provider.inner, EmbeddingProviderConfig::OpenAI(_)));

    // Check that the function for the LLM Judge evaluation is added to the functions table
    let function = config
        .functions
        .get("tensorzero::llm_judge::evaluation1::llm_judge_bool")
        .unwrap();
    match &**function {
        FunctionConfig::Json(json_config) => {
            assert_eq!(json_config.variants.len(), 7);
            match &json_config.variants["anthropic_promptA"].inner {
                VariantConfig::ChatCompletion(chat_config) => {
                    assert_eq!(chat_config.model(), &"anthropic::claude-3.5-sonnet".into());
                    assert_eq!(chat_config.weight(), Some(1.0));
                    assert_eq!(
                            chat_config.templates().get_implicit_system_template().unwrap().template,
                            PathWithContents {
                                // We don't use a real path for programmatically generated templates
                                // Instead we use this handle and then the same in minijinja
                                path: ResolvedTomlPathData::new_for_tests(
                                    PathBuf::from(
                                        "tensorzero::llm_judge::evaluation1::llm_judge_bool::anthropic_promptA::system"
                                    ),
                                    Some("Return True if there is NSFW content in this generation.\n\n".to_string())
                                ),
                                contents:
                                    "Return True if there is NSFW content in this generation.\n\n"
                                        .to_string(),
                            }
                        );
                    assert_eq!(chat_config.json_mode(), Some(&JsonMode::Tool));
                }
                _ => panic!("Expected a chat completion variant"),
            }
            match &json_config.variants["best_of_3"].inner {
                VariantConfig::BestOfNSampling(best_of_n_config) => {
                    assert_eq!(best_of_n_config.candidates().len(), 3);
                    assert_eq!(
                        best_of_n_config.evaluator().inner.model().as_ref(),
                        "openai::gpt-4o-mini"
                    );
                    assert_eq!(
                        best_of_n_config.evaluator().inner.json_mode(),
                        Some(&JsonMode::Strict)
                    );
                    assert_eq!(best_of_n_config.evaluator().inner.temperature(), Some(0.3));
                }
                _ => panic!("Expected a best of n sampling variant"),
            }
            match &json_config.variants["mixture_of_3"].inner {
                VariantConfig::MixtureOfN(mixture_of_n_config) => {
                    assert_eq!(mixture_of_n_config.candidates().len(), 3);
                    assert_eq!(
                        mixture_of_n_config.fuser().inner.model().as_ref(),
                        "openai::gpt-4o-mini"
                    );
                    assert_eq!(
                        mixture_of_n_config.fuser().inner.json_mode(),
                        Some(&JsonMode::Strict)
                    );
                    assert_eq!(mixture_of_n_config.fuser().inner.temperature(), Some(0.3));
                }
                _ => panic!("Expected a mixture of n sampling variant"),
            }
            match &json_config.variants["dicl"].inner {
                VariantConfig::Dicl(dicl_config) => {
                    assert_eq!(
                        dicl_config.system_instructions(),
                        crate::variant::dicl::default_system_instructions()
                    );
                    assert_eq!(
                        dicl_config.embedding_model().as_ref(),
                        "text-embedding-3-small"
                    );
                    assert_eq!(dicl_config.k(), 3);
                    assert_eq!(dicl_config.model().as_ref(), "openai::gpt-4o-mini");
                }
                _ => panic!("Expected a Dicl variant"),
            }
            match &json_config.variants["dicl_custom_system"].inner {
                VariantConfig::Dicl(dicl_config) => {
                    assert_eq!(
                        dicl_config.system_instructions(),
                        "Return True if there is NSFW content in this generation.\n\n"
                    );
                    assert_eq!(
                        dicl_config.embedding_model().as_ref(),
                        "text-embedding-3-small"
                    );
                    assert_eq!(dicl_config.k(), 3);
                    assert_eq!(dicl_config.model().as_ref(), "openai::gpt-4o-mini");
                }
                _ => panic!("Expected a Dicl variant"),
            }
        }
        FunctionConfig::Chat(_) => panic!("Expected a JSON function"),
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

    // Check that there are 2 tools and both have name "get_temperature"
    assert_eq!(config.tools.len(), 2);
    assert_eq!(
        config.tools.get("get_temperature").unwrap().name,
        "get_temperature"
    );
    assert_eq!(
        config.tools.get("get_temperature_with_name").unwrap().name,
        "get_temperature"
    );

    assert_eq!(config.postgres.connection_pool_size, 10);
}

/// Ensure that the config parsing correctly handles the `gateway.bind_address` field
#[tokio::test]
async fn test_config_gateway_bind_address() {
    let mut config = get_sample_valid_config();

    // Test with a valid bind address
    let ConfigLoadInfo {
        config: parsed_config,
        ..
    } = Config::load_from_toml(config.clone()).await.unwrap();
    assert_eq!(
        parsed_config.gateway.bind_address.unwrap().to_string(),
        "0.0.0.0:3000"
    );

    // Test with missing gateway section
    config.remove("gateway");
    let ConfigLoadInfo {
        config: parsed_config,
        ..
    } = Config::load_from_toml(config.clone()).await.unwrap();
    assert!(parsed_config.gateway.bind_address.is_none());

    // Test with missing bind_address
    config.insert(
        "gateway".to_string(),
        toml::Value::Table(toml::Table::new()),
    );
    let ConfigLoadInfo {
        config: parsed_config,
        ..
    } = Config::load_from_toml(config.clone()).await.unwrap();
    assert!(parsed_config.gateway.bind_address.is_none());

    // Test with invalid bind address
    config["gateway"].as_table_mut().unwrap().insert(
        "bind_address".to_string(),
        toml::Value::String("invalid_address".to_string()),
    );
    let result = Config::load_from_toml(config).await;
    assert_eq!(
        result.unwrap_err(),
        Error::new(ErrorDetails::Config {
            message: "gateway.bind_address: invalid socket address syntax".to_string()
        })
    );
}

/// Ensure that the config parsing fails when the `[models]` section is missing
#[tokio::test]
async fn test_config_from_toml_table_missing_models() {
    let mut config = get_sample_valid_config();

    config
        .remove("models")
        .expect("Failed to remove `[models]` section");

    // Remove all functions except generate_draft so we are sure what error will be thrown
    config["functions"]
        .as_table_mut()
        .unwrap()
        .retain(|k, _| k == "generate_draft");

    assert_eq!(
        Config::load_from_toml(config).await.unwrap_err(),
        Error::new(ErrorDetails::Config {
            message: "Model name 'gpt-4.1-mini' not found in model table".to_string()
        })
    );
}

/// Ensure that the config parsing fails when the `[providers]` section is missing
#[tokio::test]
async fn test_config_from_toml_table_missing_providers() {
    let mut config = get_sample_valid_config();
    config["models"]["claude-3-haiku-20240307"]
        .as_table_mut()
        .expect("Failed to get `models.claude-3-haiku-20240307` section")
        .remove("providers")
        .expect("Failed to remove `[providers]` section");

    let result = Config::load_from_toml(config).await;
    assert_eq!(
        result.unwrap_err(),
        Error::new(ErrorDetails::Config {
            message: "models.claude-3-haiku-20240307: missing field `providers`".to_string()
        })
    );
}

/// Ensure that the config parsing fails when the model credentials are missing
#[tokio::test]
async fn test_config_from_toml_table_missing_credentials() {
    let mut config = get_sample_valid_config();

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
                [
                    (
                        "__tensorzero_remapped_path".into(),
                        "tensorzero-core/fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
                            .into(),
                    ),
                    (
                        "__data".into(),
                        std::fs::read_to_string("tensorzero-core/fixtures/config/functions/generate_draft/promptA/system_template.minijinja")
                            .unwrap_or_else(|_| "You are a helpful assistant.".to_string())
                            .into(),
                    ),
                ]
                .into_iter()
                .collect::<toml::Table>()
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
                            provider.insert("api_key_location".into(), "env::not_a_place".into());
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

    let error = Config::load_from_toml(config.clone()).await.unwrap_err();
    assert_eq!(
            error,
            Error::new(ErrorDetails::Config {
                message: "models.dummy.providers.bad_credentials: Invalid api_key_location for Dummy provider"
                    .to_string()
            })
        );
}

/// Ensure that the config parsing fails when referencing a nonexistent function
#[tokio::test]
async fn test_config_from_toml_table_nonexistent_function() {
    let mut config = get_sample_valid_config();
    config
        .remove("functions")
        .expect("Failed to remove `[functions]` section");

    let result = Config::load_from_toml(config).await;
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
#[tokio::test]
async fn test_config_from_toml_table_missing_variants() {
    let mut config = get_sample_valid_config();
    config["functions"]["generate_draft"]
        .as_table_mut()
        .expect("Failed to get `functions.generate_draft` section")
        .remove("variants")
        .expect("Failed to remove `[variants]` section");

    let result = Config::load_from_toml(config).await;
    assert_eq!(
        result.unwrap_err(),
        ErrorDetails::Config {
            message: "functions.generate_draft: missing field `variants`".to_string()
        }
        .into()
    );
}

/// Ensure that the config parsing fails when there are extra variables at the root level
#[tokio::test]
async fn test_config_from_toml_table_extra_variables_root() {
    let mut config = get_sample_valid_config();
    config.insert("enable_agi".into(), true.into());

    let result = Config::load_from_toml(config).await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown field `enable_agi`, expected one of"));
}

/// Ensure that the config parsing fails when there are extra variables for models
#[tokio::test]
async fn test_config_from_toml_table_extra_variables_models() {
    let mut config = get_sample_valid_config();
    config["models"]["claude-3-haiku-20240307"]
        .as_table_mut()
        .expect("Failed to get `models.claude-3-haiku-20240307` section")
        .insert("enable_agi".into(), true.into());

    let result = Config::load_from_toml(config).await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown field `enable_agi`, expected"));
}

/// Ensure that the config parsing fails when there models with blacklisted names
#[tokio::test]
async fn test_config_from_toml_table_blacklisted_models() {
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

    let result = Config::load_from_toml(config).await;
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains(
            "models: Model name 'anthropic::claude-3-haiku-20240307' contains a reserved prefix"
        ),
        "Unexpected error: {error}"
    );
}

/// Ensure that the config parsing fails when there are extra variables for providers
#[tokio::test]
async fn test_config_from_toml_table_extra_variables_providers() {
    let mut config = get_sample_valid_config();
    config["models"]["claude-3-haiku-20240307"]["providers"]["anthropic"]
        .as_table_mut()
        .expect("Failed to get `models.claude-3-haiku-20240307.providers.anthropic` section")
        .insert("enable_agi".into(), true.into());

    let result = Config::load_from_toml(config).await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown field `enable_agi`, expected"));
}

/// Ensure that the config parsing fails when there are extra variables for functions
#[tokio::test]
async fn test_config_from_toml_table_extra_variables_functions() {
    let mut config = get_sample_valid_config();
    config["functions"]["generate_draft"]
        .as_table_mut()
        .expect("Failed to get `functions.generate_draft` section")
        .insert("enable_agi".into(), true.into());

    let result = Config::load_from_toml(config).await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown field `enable_agi`, expected"));
}

/// Ensure that the config parsing defaults properly for JSON functions with no output schema
#[tokio::test]
async fn test_config_from_toml_table_json_function_no_output_schema() {
    let mut config = get_sample_valid_config();
    config["functions"]["json_with_schemas"]
        .as_table_mut()
        .expect("Failed to get `functions.generate_draft` section")
        .remove("output_schema");

    let result = Config::load_from_toml(config).await;
    let ConfigLoadInfo { config, .. } = result.unwrap();
    // Check that the output schema is set to {}
    let output_schema = match &**config.functions.get("json_with_schemas").unwrap() {
        FunctionConfig::Json(json_config) => &json_config.output_schema,
        FunctionConfig::Chat(_) => panic!("Expected a JSON function"),
    };
    assert_eq!(output_schema, &StaticJSONSchema::default());
    assert_eq!(output_schema.value, serde_json::json!({}));
}

/// Ensure that the config parsing fails when there are extra variables for variants
#[tokio::test]
async fn test_config_from_toml_table_extra_variables_variants() {
    let mut config = get_sample_valid_config();
    config["functions"]["generate_draft"]["variants"]["openai_promptA"]
        .as_table_mut()
        .expect("Failed to get `functions.generate_draft.variants.openai_promptA` section")
        .insert("enable_agi".into(), true.into());

    let result = Config::load_from_toml(config).await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown field `enable_agi`, expected"));
}

/// Ensure that the config parsing fails when there are extra variables for metrics
#[tokio::test]
async fn test_config_from_toml_table_extra_variables_metrics() {
    let mut config = get_sample_valid_config();
    config["metrics"]["task_success"]
        .as_table_mut()
        .expect("Failed to get `metrics.task_success` section")
        .insert("enable_agi".into(), true.into());

    let result = Config::load_from_toml(config).await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown field `enable_agi`, expected"));
}

/// Ensure that the config validation fails when a model has no providers in `routing`
#[tokio::test]
async fn test_config_validate_model_empty_providers() {
    let mut config = get_sample_valid_config();
    config["models"]["gpt-4.1-mini"]["routing"] = toml::Value::Array(vec![]);

    let result = Config::load_from_toml(config).await;
    let error = result.unwrap_err();
    assert!(error
        .to_string()
        .contains("`models.gpt-4.1-mini`: `routing` must not be empty"));
}

/// Ensure that the config validation fails when there are duplicate routing entries
#[tokio::test]
async fn test_config_validate_model_duplicate_routing_entry() {
    let mut config = get_sample_valid_config();
    config["models"]["gpt-4.1-mini"]["routing"] =
        toml::Value::Array(vec!["openai".into(), "openai".into()]);
    let result = Config::load_from_toml(config).await;
    let error = result.unwrap_err().to_string();
    assert!(error.contains("`models.gpt-4.1-mini.routing`: duplicate entry `openai`"));
}

/// Ensure that the config validation fails when a routing entry does not exist in providers
#[tokio::test]
async fn test_config_validate_model_routing_entry_not_in_providers() {
    let mut config = get_sample_valid_config();
    config["models"]["gpt-4.1-mini"]["routing"] = toml::Value::Array(vec!["closedai".into()]);
    let result = Config::load_from_toml(config).await;
    assert!(result.unwrap_err().to_string().contains("`models.gpt-4.1-mini`: `routing` contains entry `closedai` that does not exist in `providers`"));
}

/// Ensure that the config loading fails when the system schema does not exist
#[tokio::test]
async fn test_config_system_schema_does_not_exist() {
    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_chat"]["system_schema"] = [
        (
            "__tensorzero_remapped_path".into(),
            "non_existent_file.json".into(),
        ),
        ("__data".into(), "invalid json content".into()),
    ]
    .into_iter()
    .collect::<toml::Table>()
    .into();

    let result = Config::load_from_toml(sample_config).await;
    let error = result.unwrap_err();
    if let ErrorDetails::JsonSchema { message } = error.get_details() {
        assert!(message.contains("expected value") || message.contains("invalid type"));
    } else {
        panic!("Expected JsonSchema error, got: {error:?}");
    }

    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_json"]["system_schema"] = [
        (
            "__tensorzero_remapped_path".into(),
            "non_existent_file.json".into(),
        ),
        ("__data".into(), "invalid json content".into()),
    ]
    .into_iter()
    .collect::<toml::Table>()
    .into();

    let result = Config::load_from_toml(sample_config).await;
    let error = result.unwrap_err();
    if let ErrorDetails::JsonSchema { message } = error.get_details() {
        assert!(message.contains("expected value") || message.contains("invalid type"));
    } else {
        panic!("Expected JsonSchema error, got: {error:?}");
    }
}

/// Ensure that the config loading fails when the user schema does not exist
#[tokio::test]
async fn test_config_user_schema_does_not_exist() {
    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_chat"]["user_schema"] = [
        (
            "__tensorzero_remapped_path".into(),
            "non_existent_file.json".into(),
        ),
        ("__data".into(), "invalid json content".into()),
    ]
    .into_iter()
    .collect::<toml::Table>()
    .into();

    let result = Config::load_from_toml(sample_config).await;
    let error = result.unwrap_err();
    if let ErrorDetails::JsonSchema { message } = error.get_details() {
        assert!(message.contains("expected value") || message.contains("invalid type"));
    } else {
        panic!("Expected JsonSchema error, got: {error:?}");
    }

    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_json"]["user_schema"] = [
        (
            "__tensorzero_remapped_path".into(),
            "non_existent_file.json".into(),
        ),
        ("__data".into(), "invalid json content".into()),
    ]
    .into_iter()
    .collect::<toml::Table>()
    .into();

    let result = Config::load_from_toml(sample_config).await;
    let error = result.unwrap_err();
    if let ErrorDetails::JsonSchema { message } = error.get_details() {
        assert!(message.contains("expected value") || message.contains("invalid type"));
    } else {
        panic!("Expected JsonSchema error, got: {error:?}");
    }
}

/// Ensure that the config loading fails when the assistant schema does not exist
#[tokio::test]
async fn test_config_assistant_schema_does_not_exist() {
    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_chat"]["assistant_schema"] = [
        (
            "__tensorzero_remapped_path".into(),
            "non_existent_file.json".into(),
        ),
        ("__data".into(), "invalid json content".into()),
    ]
    .into_iter()
    .collect::<toml::Table>()
    .into();

    let result = Config::load_from_toml(sample_config).await;
    let error = result.unwrap_err();
    if let ErrorDetails::JsonSchema { message } = error.get_details() {
        assert!(message.contains("expected value") || message.contains("invalid type"));
    } else {
        panic!("Expected JsonSchema error, got: {error:?}");
    }

    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_json"]["assistant_schema"] = [
        (
            "__tensorzero_remapped_path".into(),
            "non_existent_file.json".into(),
        ),
        ("__data".into(), "invalid json content".into()),
    ]
    .into_iter()
    .collect::<toml::Table>()
    .into();

    let result = Config::load_from_toml(sample_config).await;
    let error = result.unwrap_err();
    if let ErrorDetails::JsonSchema { message } = error.get_details() {
        assert!(message.contains("expected value") || message.contains("invalid type"));
    } else {
        panic!("Expected JsonSchema error, got: {error:?}");
    }
}

/// Ensure that the config loading fails when the system schema is missing but is needed
#[tokio::test]
async fn test_config_system_schema_is_needed() {
    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_chat"]
        .as_table_mut()
        .unwrap()
        .remove("system_schema");

    sample_config["functions"]["templates_with_variables_chat"]["variants"]
        .as_table_mut()
        .unwrap()
        .remove("best_of_n");

    let result = Config::load_from_toml(sample_config).await;
    assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_chat.variants.variant_with_variables.system_template`: template needs variables: [message] but only `system_text` is allowed when template has no schema".to_string()
            }.into()
        );
    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_json"]
        .as_table_mut()
        .unwrap()
        .remove("system_schema");

    let result = Config::load_from_toml(sample_config).await;
    assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_json.variants.variant_with_variables.system_template`: template needs variables: [message] but only `system_text` is allowed when template has no schema".to_string()
            }.into()
        );
}

/// Ensure that the config loading fails when the user schema is missing but is needed
#[tokio::test]
async fn test_config_user_schema_is_needed() {
    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_chat"]
        .as_table_mut()
        .unwrap()
        .remove("user_schema");
    sample_config["functions"]["templates_with_variables_chat"]["variants"]
        .as_table_mut()
        .unwrap()
        .remove("best_of_n");

    let result = Config::load_from_toml(sample_config).await;
    assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_chat.variants.variant_with_variables.user_template`: template needs variables: [message] but only `user_text` is allowed when template has no schema".to_string()
            }.into()
        );

    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_json"]
        .as_table_mut()
        .unwrap()
        .remove("user_schema");

    let result = Config::load_from_toml(sample_config).await;
    assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_json.variants.variant_with_variables.user_template`: template needs variables: [message] but only `user_text` is allowed when template has no schema".to_string()
            }.into()
        );
}

/// Ensure that the config loading fails when the assistant schema is missing but is needed
#[tokio::test]
async fn test_config_assistant_schema_is_needed() {
    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_chat"]
        .as_table_mut()
        .unwrap()
        .remove("assistant_schema");

    sample_config["functions"]["templates_with_variables_chat"]["variants"]
        .as_table_mut()
        .unwrap()
        .remove("best_of_n");

    let result = Config::load_from_toml(sample_config).await;
    assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_chat.variants.variant_with_variables.assistant_template`: template needs variables: [message] but only `assistant_text` is allowed when template has no schema".to_string()
            }.into()
        );
    let mut sample_config = get_sample_valid_config();
    sample_config["functions"]["templates_with_variables_json"]
        .as_table_mut()
        .unwrap()
        .remove("assistant_schema");

    let result = Config::load_from_toml(sample_config).await;
    assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.templates_with_variables_json.variants.variant_with_variables.assistant_template`: template needs variables: [message] but only `assistant_text` is allowed when template has no schema".to_string()
            }.into()
        );
}

/// Ensure that config loading fails when a nonexistent candidate is specified in a variant
#[tokio::test]
async fn test_config_best_of_n_candidate_not_found() {
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

    let result = Config::load_from_toml(sample_config).await;
    assert_eq!(
        result.unwrap_err(),
        ErrorDetails::UnknownCandidate {
            name: "non_existent_candidate".to_string()
        }
        .into()
    );
}

/// Ensure that the config validation fails when a function variant has a negative weight
#[tokio::test]
async fn test_config_validate_function_variant_negative_weight() {
    let mut config = get_sample_valid_config();
    config["functions"]["generate_draft"]["variants"]["openai_promptA"]["weight"] =
        toml::Value::Float(-1.0);

    let result = Config::load_from_toml(config).await;
    assert_eq!(
        result.unwrap_err(),
        ErrorDetails::Config {
            message:
                "`functions.generate_draft.variants.openai_promptA`: `weight` must be non-negative"
                    .to_string()
        }
        .into()
    );
}

/// Ensure that the config validation fails when a variant has a model that does not exist in the models section
#[tokio::test]
async fn test_config_validate_variant_model_not_in_models() {
    let mut config = get_sample_valid_config();
    config["functions"]["generate_draft"]["variants"]["openai_promptA"]["model"] =
        "non_existent_model".into();

    let result = Config::load_from_toml(config).await;

    assert_eq!(
        result.unwrap_err(),
        ErrorDetails::Config {
            message: "Model name 'non_existent_model' not found in model table".to_string()
        }
        .into()
    );
}

/// Ensure that the config validation fails when a variant has a template that does not exist
#[tokio::test]
async fn test_config_validate_variant_template_nonexistent() {
    let mut config = get_sample_valid_config();
    config["functions"]["generate_draft"]["variants"]["openai_promptA"]["system_template"] = [
        (
            "__tensorzero_remapped_path".into(),
            "nonexistent_template".into(),
        ),
        (
            "__data".into(),
            "invalid template content with {{ unclosed".into(),
        ),
    ]
    .into_iter()
    .collect::<toml::Table>()
    .into();

    let result = Config::load_from_toml(config).await;

    // With eager loading, this should now fail during template parsing
    let error = result.unwrap_err();
    if let ErrorDetails::MiniJinjaTemplate { message, .. } = error.get_details() {
        assert!(
            message.contains("expected")
                || message.contains("unclosed")
                || message.contains("invalid")
        );
    } else {
        panic!("Expected MiniJinjaTemplate error, got: {error:?}");
    }
}

/// Ensure that the config validation fails when an evaluation points at a nonexistent function
#[tokio::test]
async fn test_config_validate_evaluation_function_nonexistent() {
    let mut config = get_sample_valid_config();
    config["evaluations"]["evaluation1"]["function_name"] = "nonexistent_function".into();

    let result = Config::load_from_toml(config).await;

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
#[tokio::test]
async fn test_config_validate_evaluation_name_contains_double_colon() {
    let mut config = get_sample_valid_config();
    let evaluation1 = config["evaluations"]["evaluation1"].clone();
    config
        .get_mut("evaluations")
        .unwrap()
        .as_table_mut()
        .unwrap()
        .insert("bad::evaluation".to_string(), evaluation1);

    let result = Config::load_from_toml(config).await;

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
#[tokio::test]
async fn test_config_validate_function_nonexistent_tool() {
    let mut config = get_sample_valid_config();
    config["functions"]["generate_draft"]
        .as_table_mut()
        .unwrap()
        .insert("tools".to_string(), toml::Value::Array(vec![]));
    config["functions"]["generate_draft"]["tools"] =
        toml::Value::Array(vec!["non_existent_tool".into()]);

    let result = Config::load_from_toml(config).await;

    assert_eq!(
            result.unwrap_err(),
            ErrorDetails::Config {
                message: "`functions.generate_draft.tools`: tool `non_existent_tool` is not present in the config".to_string()
            }.into()
        );
}

/// Ensure that the config validation fails when a function name starts with `tensorzero::`
#[tokio::test]
async fn test_config_validate_function_name_tensorzero_prefix() {
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

    let result = Config::load_from_toml(config).await;
    assert_eq!(
        result.unwrap_err(),
        Error::new(ErrorDetails::Config {
            message: "User-defined function name cannot start with 'tensorzero::': tensorzero::bad_function"
                .to_string()
        })
    );
}

/// Ensure that the config validation fails when a metric name starts with `tensorzero::`
#[tokio::test]
async fn test_config_validate_metric_name_tensorzero_prefix() {
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

    let result = Config::load_from_toml(config).await;
    assert_eq!(
        result.unwrap_err(),
        Error::new(ErrorDetails::Config {
            message: "Metric name cannot start with 'tensorzero::': tensorzero::bad_metric"
                .to_string()
        })
    );
}

/// Ensure that the config validation fails when a model name starts with `tensorzero::`
#[tokio::test]
async fn test_config_validate_model_name_tensorzero_prefix() {
    let mut config = get_sample_valid_config();

    // Rename an existing model to start with `tensorzero::`
    let old_model_entry = config["models"]
        .as_table_mut()
        .unwrap()
        .remove("gpt-4.1-mini")
        .expect("Did not find model `gpt-4.1-mini`");
    config["models"]
        .as_table_mut()
        .unwrap()
        .insert("tensorzero::bad_model".to_string(), old_model_entry);

    let result = Config::load_from_toml(config).await;
    assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::Config {
                message: "Failed to load models: Model name 'tensorzero::bad_model' contains a reserved prefix"
                    .to_string()
            })
        );
}

/// Ensure that the config validation fails when an embedding model name starts with `tensorzero::`
#[tokio::test]
async fn test_config_validate_embedding_model_name_tensorzero_prefix() {
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

    let result = Config::load_from_toml(config).await;
    assert_eq!(
                result.unwrap_err(),
                Error::new(ErrorDetails::Config {
                    message:
                        "Failed to load embedding models: Embedding model name 'tensorzero::bad_embedding_model' contains a reserved prefix"
                            .to_string()
                })
            );
}

/// Ensure that the config validation fails when a tool name starts with `tensorzero::`
#[tokio::test]
async fn test_config_validate_tool_name_tensorzero_prefix() {
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

    let result = Config::load_from_toml(config).await;
    assert_eq!(
        result.unwrap_err(),
        Error::new(ErrorDetails::Config {
            message: "Tool name cannot start with 'tensorzero::': tensorzero::bad_tool".to_string()
        })
    );
}

#[tokio::test]
async fn test_config_validate_chat_function_json_mode() {
    let mut config = get_sample_valid_config();

    // Insert `json_mode = "on"` into a variant config for a chat function.
    config["functions"]["generate_draft"]["variants"]["openai_promptA"]
        .as_table_mut()
        .unwrap()
        .insert("json_mode".to_string(), "on".into());

    let result = Config::load_from_toml(config).await;

    // Check that the config is rejected, since `generate_draft` is not a json function
    let err_msg = result.unwrap_err().to_string();
    assert!(
            err_msg.contains("JSON mode is not supported for variant `openai_promptA` (parent function is a chat function)"),
            "Unexpected error message: {err_msg}"
        );
}

/// If you also want to confirm a variant name starting with `tensorzero::` fails
/// (only do this if your `function.validate` logic checks variant names):
#[tokio::test]
async fn test_config_validate_variant_name_tensorzero_prefix() {
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

    let result = Config::load_from_toml(config).await;

    // Adjust the expected message if your code gives a different error shape for variants
    // Or remove this test if variant names are *not* validated in that manner
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("tensorzero::bad_variant"));
}

/// Ensure that the config validation fails when a model provider's name starts with `tensorzero::`
#[tokio::test]
async fn test_config_validate_model_provider_name_tensorzero_prefix() {
    let mut config = get_sample_valid_config();

    // Rename an existing provider to start with `tensorzero::`
    let old_openai_provider = config["models"]["gpt-4.1-mini"]["providers"]
        .as_table_mut()
        .unwrap()
        .remove("openai")
        .expect("Did not find provider `openai` under `gpt-4.1-mini`");
    config["models"]["gpt-4.1-mini"]["providers"]
        .as_table_mut()
        .unwrap()
        .insert("tensorzero::openai".to_string(), old_openai_provider);

    // Update the routing entry to match the new provider name
    let routing = config["models"]["gpt-4.1-mini"]["routing"]
        .as_array_mut()
        .expect("Expected routing to be an array");
    for entry in routing.iter_mut() {
        if entry.as_str() == Some("openai") {
            *entry = toml::Value::String("tensorzero::openai".to_string());
        }
    }

    let result = Config::load_from_toml(config).await;

    assert!(result.unwrap_err().to_string().contains("`models.gpt-4.1-mini.routing`: Provider name cannot start with 'tensorzero::': tensorzero::openai"));
}

/// Ensure that get_templates returns the correct templates
#[tokio::test]
async fn test_get_all_templates() {
    let config_table = get_sample_valid_config();
    let ConfigLoadInfo { config, .. } = Config::load_from_toml(config_table)
        .await
        .expect("Failed to load config");

    // Get all templates
    let templates = config.get_templates();

    // Check if all expected templates are present
    assert_eq!(
        *templates
            .get(&format!(
                "{}/fixtures/config/functions/generate_draft/promptA/system_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        include_str!(
            "../../fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
        )
        .to_string()
    );
    assert_eq!(
        *templates
            .get(&format!(
                "{}/fixtures/config/functions/generate_draft/promptA/system_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        include_str!(
            "../../fixtures/config/functions/generate_draft/promptA/system_template.minijinja"
        )
        .to_string()
    );
    assert_eq!(
        *templates
            .get(&format!(
                "{}/fixtures/config/functions/json_with_schemas/promptA/system_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        include_str!(
            "../../fixtures/config/functions/json_with_schemas/promptA/system_template.minijinja"
        )
        .to_string()
    );
    assert_eq!(
        *templates
            .get(&format!(
                "{}/fixtures/config/functions/json_with_schemas/promptB/system_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        include_str!(
            "../../fixtures/config/functions/json_with_schemas/promptB/system_template.minijinja"
        )
        .to_string()
    );
    assert_eq!(
            *templates.get(&format!(
                "{}/fixtures/config/functions/templates_without_variables/variant_without_templates/system_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
            include_str!(
                "../../fixtures/config/functions/templates_without_variables/variant_without_templates/system_template.minijinja"
            ).to_string()
        );
    assert_eq!(
            *templates.get(&format!(
                "{}/fixtures/config/functions/templates_without_variables/variant_without_templates/user_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
            include_str!(
                "../../fixtures/config/functions/templates_without_variables/variant_without_templates/user_template.minijinja"
            ).to_string()
        );
    assert_eq!(
            *templates.get(&format!(
                "{}/fixtures/config/functions/templates_without_variables/variant_without_templates/assistant_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
            include_str!(
                "../../fixtures/config/functions/templates_without_variables/variant_without_templates/assistant_template.minijinja"
            ).to_string()
        );
    assert_eq!(
            *templates.get(&format!(
                "{}/fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
            include_str!(
                "../../fixtures/config/functions/templates_with_variables/variant_with_variables/assistant_template.minijinja"
            ).to_string()
        );
    assert_eq!(
            *templates.get(&format!(
                "{}/fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
            include_str!(
                "../../fixtures/config/functions/templates_with_variables/variant_with_variables/user_template.minijinja"
            ).to_string()
        );
    assert_eq!(
                    *templates.get(&format!(
                        "{}/fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja",
                        env!("CARGO_MANIFEST_DIR")
                    ))
                    .unwrap(),
                    include_str!(
                        "../../fixtures/config/functions/templates_with_variables/variant_with_variables/system_template.minijinja"
                    ).to_string()
                );

    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::anthropic_promptA::system")
            .unwrap(),
        "Return True if there is NSFW content in this generation.\n\n".to_string(),
    );

    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_float::anthropic_promptA::system")
            .unwrap(),
        "Return a number between 0 and 1 where 1 is very NSFW and 0 is the least NSFW content.\n\n"
            .to_string(),
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::anthropic_promptA::user")
            .unwrap(),
        include_str!("../evaluations/llm_judge_user_template.minijinja").to_string()
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_float::anthropic_promptA::user")
            .unwrap(),
        include_str!("../evaluations/llm_judge_user_template.minijinja").to_string()
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::llama_promptA::system")
            .unwrap(),
        "Return True if there is NSFW content in this generation.\n\n".to_string(),
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::llama_promptA::user")
            .unwrap(),
        include_str!("../evaluations/llm_judge_user_template.minijinja").to_string()
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::openai_promptA::system")
            .unwrap(),
        "Return True if there is NSFW content in this generation.\n\n".to_string()
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::openai_promptA::user")
            .unwrap(),
        include_str!("../evaluations/llm_judge_user_template.minijinja").to_string()
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::best_of_3::system")
            .unwrap(),
        "Return True if there is NSFW content in this generation.\n\n".to_string()
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::best_of_3::user")
            .unwrap(),
        include_str!("../evaluations/llm_judge_user_template.minijinja").to_string()
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::mixture_of_3::system")
            .unwrap(),
        "Return True if there is NSFW content in this generation.\n\n".to_string()
    );
    assert_eq!(
        *templates
            .get("tensorzero::llm_judge::evaluation1::llm_judge_bool::mixture_of_3::user")
            .unwrap(),
        include_str!("../evaluations/llm_judge_user_template.minijinja").to_string()
    );

    // Check the total number of templates
    assert_eq!(templates.len(), 22);
}

#[tokio::test]
async fn test_load_bad_extra_body_delete() {
    let config_str = r#"
        [functions.bash_assistant]
        type = "chat"

        [functions.bash_assistant.variants.anthropic_claude_3_7_sonnet_20250219]
        type = "chat_completion"
        model = "anthropic::claude-3-7-sonnet-20250219"
        max_tokens = 2048
        extra_body = [{ pointer = "/invalid-field-should-be-deleted", delete = false }]
        "#;
    let config = toml::from_str(config_str).expect("Failed to parse sample config");

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config loading should fail")
        .to_string();
    assert_eq!(err, "functions.bash_assistant: variants.anthropic_claude_3_7_sonnet_20250219: extra_body.[0]: Error deserializing replacement config: `delete` must be `true`, or not set");
}

#[tokio::test]
async fn test_load_bad_config_error_path() {
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

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config loading should fail")
        .to_string();
    assert_eq!(err, "functions.bash_assistant: variants.anthropic_claude_3_7_sonnet_20250219: extra_body: invalid type: map, expected a sequence");
}

#[tokio::test]
async fn test_config_load_shorthand_models_only() {
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file
        .write_all(
            r#"
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

        [functions.generate_draft.variants.openai_promptA]
        type = "chat_completion"
        weight = 0.9
        model = "openai::gpt-4.1-mini"
        "#
            .as_bytes(),
        )
        .unwrap();

    let config = UninitializedConfig::read_toml_config(
        &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
        false,
    )
    .unwrap();
    env::set_var("OPENAI_API_KEY", "sk-something");
    env::set_var("ANTHROPIC_API_KEY", "sk-something");
    env::set_var("AZURE_OPENAI_API_KEY", "sk-something");

    Config::load_from_toml(config.table)
        .await
        .expect("Failed to load config");
}

#[tokio::test]
async fn test_empty_config() {
    let logs_contain = crate::utils::testing::capture_logs();
    let tempfile = NamedTempFile::new().unwrap();
    write!(&tempfile, "").unwrap();
    Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(tempfile.path()).unwrap())
        .await
        .unwrap();
    assert!(logs_contain(
        "Config file is empty, so only default functions will be available."
    ));
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

    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(tmpfile.path()).unwrap())
            .await
            .unwrap_err()
            .to_string();

    assert!(
        err.contains("duplicate key"),
        "Message is missing 'duplicate key': {err}"
    );
    assert!(
        err.contains("models.my-model"),
        "Message is missing 'models.my-model': {err}"
    );
}

#[tokio::test]
async fn test_model_provider_unknown_field() {
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

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");
    assert!(
        err.to_string().contains("unknown field `my_bad_key`"),
        "Unexpected error: {err:?}"
    );
}

/// Get a sample valid config for testing
fn get_sample_valid_config() -> toml::Table {
    let config_str = include_str!("../../fixtures/config/tensorzero.toml");
    env::set_var("OPENAI_API_KEY", "sk-something");
    env::set_var("ANTHROPIC_API_KEY", "sk-something");
    env::set_var("AZURE_OPENAI_API_KEY", "sk-something");

    let table = DeTable::parse(config_str).expect("Failed to parse sample config");

    // Note - we deliberately use an unusual base_path here (not the immediate parent of the config file)

    let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fake_path = base_path.join("fake_path.toml");

    path::resolve_toml_relative_paths(table.into_inner(), &SpanMap::new_single_file(fake_path))
        .expect("Failed to resolve paths")
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

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Failed to load bedrock");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("requires a region to be provided, or `allow_auto_detect_region = true`"),
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

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Failed to load bedrock");
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

    Config::load_from_toml(config)
        .await
        .expect("Failed to construct config with valid AWS bedrock provider");
}
#[tokio::test]
async fn test_config_load_no_config_file() {
    let err = &ConfigFileGlob::new_from_path(Path::new("nonexistent.toml"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Error using glob: `nonexistent.toml`: No files matched the glob pattern. Ensure that the path exists, and contains at least one file."),
        "Unexpected error message: {err}"
    );
}

#[tokio::test]
#[cfg_attr(feature = "e2e_tests", ignore)]
async fn test_config_missing_filesystem_object_store() {
    let tempfile = NamedTempFile::new().unwrap();
    write!(
        &tempfile,
        r#"
            [object_storage]
            type = "filesystem"
            path = "/fake-tensorzero-path/other-path"

            [functions]"#
    )
    .unwrap();
    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(tempfile.path()).unwrap())
            .await
            .unwrap_err()
            .to_string();
    assert!(
            err.contains("Failed to create filesystem object store: path does not exist: /fake-tensorzero-path/other-path"),
            "Unexpected error message: {err}"
        );
}

#[tokio::test]
async fn test_config_no_verify_creds_missing_filesystem_object_store() {
    let logs_contain = crate::utils::testing::capture_logs();
    let tempfile = NamedTempFile::new().unwrap();
    write!(
        &tempfile,
        r#"
            [object_storage]
            type = "filesystem"
            path = "/fake-tensorzero-path/other-path"

            [functions]"#
    )
    .unwrap();
    let ConfigLoadInfo { config, .. } = Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new_from_path(tempfile.path()).unwrap(),
        false,
    )
    .await
    .unwrap();
    assert!(config.object_store_info.is_none());
    assert!(logs_contain("Filesystem object store path does not exist: /fake-tensorzero-path/other-path. Treating object store as unconfigured"));
}
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
    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(tempfile.path()).unwrap())
            .await
            .unwrap_err()
            .to_string();
    assert!(
        err.contains("Failed to write `.tensorzero-validate` to object store."),
        "Unexpected error message: {err}"
    );
}
#[tokio::test]
async fn test_config_blocked_s3_http_endpoint_default() {
    let logs_contain = crate::utils::testing::capture_logs();
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
    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(tempfile.path()).unwrap())
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
#[tokio::test]
async fn test_config_blocked_s3_http_endpoint_override() {
    let logs_contain = crate::utils::testing::capture_logs();
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
    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(tempfile.path()).unwrap())
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
#[tokio::test]
async fn test_config_s3_allow_http_config() {
    let logs_contain = crate::utils::testing::capture_logs();
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
    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(tempfile.path()).unwrap())
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
#[tokio::test]
async fn test_config_s3_allow_http_env_var() {
    let logs_contain = crate::utils::testing::capture_logs();
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
    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(tempfile.path()).unwrap())
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

#[tokio::test]
async fn test_missing_json_mode_chat() {
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

        [models."my-model"]
        routing = ["openai"]

        [models.my-model.providers.openai]
        type = "openai"
        model_name = "gpt-4o-mini-2024-07-18"
        "#;
    let config = toml::from_str(config_str).expect("Failed to parse sample config");

    let err = SKIP_CREDENTIAL_VALIDATION
        .scope((), Config::load_from_toml(config))
        .await
        .unwrap_err();

    assert_eq!(err.to_string(), "`json_mode` must be specified for `[functions.basic_test.variants.test]` (parent function `basic_test` is a JSON function)");
}

#[tokio::test]
async fn test_missing_json_mode_dicl() {
    let config_str = r#"
        [gateway]
        bind_address = "0.0.0.0:3000"

        [functions.basic_test]
        type = "json"

        [functions.basic_test.variants.good_variant]
        type = "chat_completion"
        model = "my-model"
        json_mode = "off"

        [functions.basic_test.variants.dicl]
        type = "experimental_dynamic_in_context_learning"
        model = "my-model"
        embedding_model = "openai::text-embedding-3-small"
        k = 3
        max_tokens = 100

        [models."my-model"]
        routing = ["openai"]

        [models.my-model.providers.openai]
        type = "openai"
        model_name = "gpt-4o-mini-2024-07-18"
        "#;
    let config = toml::from_str(config_str).expect("Failed to parse sample config");

    let err = SKIP_CREDENTIAL_VALIDATION
        .scope((), Config::load_from_toml(config))
        .await
        .unwrap_err();

    assert_eq!(err.to_string(), "`json_mode` must be specified for `[functions.basic_test.variants.dicl]` (parent function `basic_test` is a JSON function)");
}

#[tokio::test]
async fn test_missing_json_mode_mixture_of_n() {
    let config_str = r#"
        [gateway]
        bind_address = "0.0.0.0:3000"

        [functions.basic_test]
        type = "json"

        [functions.basic_test.variants.good_variant]
        type = "chat_completion"
        model = "my-model"
        json_mode = "off"


        [functions.basic_test.variants.mixture_of_n_variant]
        type = "experimental_mixture_of_n"
        candidates = ["test"]

        [functions.basic_test.variants.mixture_of_n_variant.fuser]
        model = "my-model"

        [models."my-model"]
        routing = ["openai"]

        [models.my-model.providers.openai]
        type = "openai"
        model_name = "gpt-4o-mini-2024-07-18"
        "#;
    let config = toml::from_str(config_str).expect("Failed to parse sample config");

    let err = SKIP_CREDENTIAL_VALIDATION
        .scope((), Config::load_from_toml(config))
        .await
        .unwrap_err();

    assert_eq!(err.to_string(), "`json_mode` must be specified for `[functions.basic_test.variants.mixture_of_n_variant.fuser]` (parent function `basic_test` is a JSON function)");
}

#[tokio::test]
async fn test_missing_json_mode_best_of_n() {
    // Test that evaluator json_mode is optional (it defaults to `strict` at runtime)
    let config_str = r#"
        [gateway]
        bind_address = "0.0.0.0:3000"

        [functions.basic_test]
        type = "json"

        [functions.basic_test.variants.good_variant]
        type = "chat_completion"
        model = "my-model"
        json_mode = "off"

        [functions.basic_test.variants.best_of_n_variant]
        type = "experimental_best_of_n_sampling"
        candidates = ["good_variant"]

        [functions.basic_test.variants.best_of_n_variant.evaluator]
        model = "my-model"

        [models."my-model"]
        routing = ["openai"]

        [models.my-model.providers.openai]
        type = "openai"
        model_name = "gpt-4o-mini-2024-07-18"
        "#;

    let config = toml::from_str(config_str).expect("Failed to parse sample config");

    // This should succeed (evaluator's `json_mode` is optional)
    SKIP_CREDENTIAL_VALIDATION
        .scope((), Config::load_from_toml(config))
        .await
        .expect("Config should load successfully with missing evaluator json_mode");
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

    let res = Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new_from_path(tmpfile.path()).unwrap(),
        true,
    )
    .await;
    if cfg!(feature = "e2e_tests") {
        assert!(res.is_ok());
    } else {
        assert_eq!(res.unwrap_err().to_string(), "models.my-model.providers.openai: API key missing for provider openai: Failed to read credentials file - No such file or directory (os error 2)");
    }

    // Should not fail since validation is disabled
    Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new_from_path(tmpfile.path()).unwrap(),
        false,
    )
    .await
    .expect("Failed to load config");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_gcp_no_endpoint_and_model() {
    let config_str = r#"
        [gateway]
        bind_address = "0.0.0.0:3000"


        [models."my-model"]
        routing = ["gcp-vertex-gemini"]

        [models.my-model.providers.gcp-vertex-gemini]
        type = "gcp_vertex_gemini"
        location = "us-central1"
        project_id = "test-project"
        model_id = "gemini-2.0-flash-001"
        endpoint_id = "4094940393049"
        "#;
    let config = toml::from_str(config_str).expect("Failed to parse sample config");

    let err = SKIP_CREDENTIAL_VALIDATION
        .scope((), Config::load_from_toml(config))
        .await
        .unwrap_err();

    let err_msg = err.to_string();
    assert!(
            err_msg
                .contains("models.my-model.providers.gcp-vertex-gemini: Exactly one of model_id or endpoint_id must be provided"),
            "Unexpected error message: {err_msg}"
        );
}

#[tokio::test]
async fn test_config_duplicate_user_schema() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .display()
        .to_string();
    // We write an absolute path into the config file, since we're writing the config file in a temp dir.
    temp_file
        .write_all(
            format!(
                r#"
        [functions.bad_user_schema]
        type = "chat"
        user_schema = "{base_path}/fixtures/config/functions/json_success/user_schema.json"
        schemas.user.path = "{base_path}/fixtures/config/functions/json_success/user_schema.json"

        [functions.bad_user_schema.variants.good]
        type = "chat_completion"
        model = "dummy::good"
        "#
            )
            .as_bytes(),
        )
        .unwrap();

    let config = UninitializedConfig::read_toml_config(
        &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
        false,
    )
    .unwrap();
    let err = Config::load_from_toml(config.table)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(
        err.to_string(),
        "functions.bad_user_schema: Cannot specify both `schemas.user.path` and `user_schema`"
    );
}

#[tokio::test]
async fn test_config_named_schema_no_template() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .display()
        .to_string();
    // We write an absolute path into the config file, since we're writing the config file in a temp dir.
    temp_file
        .write_all(
            format!(
                r#"
        [functions.bad_custom_schema]
        type = "chat"
        schemas.my_custom_schema.path = "{base_path}/fixtures/config/functions/json_success/user_schema.json"

        [functions.bad_custom_schema.variants.good]
        type = "chat_completion"
        model = "dummy::good"
        "#
            )
            .as_bytes(),
        )
        .unwrap();

    let config = UninitializedConfig::read_toml_config(
        &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
        false,
    )
    .unwrap();
    let err = Config::load_from_toml(config.table)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(
        err.to_string(),
        "`functions.bad_custom_schema.variants.good.templates.my_custom_schema` is required when `functions.bad_custom_schema.schemas.my_custom_schema` is specified"
    );
}

#[tokio::test]
async fn test_config_duplicate_user_template() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .display()
        .to_string();
    // We write an absolute path into the config file, since we're writing the config file in a temp dir.
    temp_file
        .write_all(
            format!(r#"
        [functions.test]
        type = "chat"

        [functions.test.variants.bad_user_template]
        type = "chat_completion"
        model = "dummy::echo_request_messages"
        user_template = "{base_path}/fixtures/config/functions/json_success/prompt/user_template.minijinja"
        templates.user.path = "{base_path}/fixtures/config/functions/json_success/prompt/user_template.minijinja"
        "#).as_bytes(),
        )
        .unwrap();

    let config = UninitializedConfig::read_toml_config(
        &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
        false,
    )
    .unwrap();
    let err = Config::load_from_toml(config.table)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(err.to_string(), "functions.test.variants.bad_user_template: Cannot specify both `templates.user.path` and `user_template`");
}

#[tokio::test]
async fn test_config_invalid_template_no_schema() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .display()
        .to_string();
    // We write an absolute path into the config file, since we're writing the config file in a temp dir.
    temp_file
        .write_all(
            format!(r#"
        [functions.no_schema]
        type = "chat"

        [functions.no_schema.variants.invalid_system_template]
        type = "chat_completion"
        model = "dummy::echo_request_messages"
        system_template = "{base_path}/fixtures/config/functions/basic_test/prompt/system_template.minijinja"
        "#).as_bytes(),
        )
        .unwrap();

    let config = UninitializedConfig::read_toml_config(
        &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
        false,
    )
    .unwrap();
    let err = Config::load_from_toml(config.table)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(err.to_string(), "`functions.no_schema.variants.invalid_system_template.system_template`: template needs variables: [assistant_name] but only `system_text` is allowed when template has no schema");
}

#[tokio::test]
async fn deny_timeout_with_default_global_timeout() {
    let config = r#"
    [models.slow_with_timeout]
    routing = ["slow"]

    [models.slow_with_timeout.providers.slow]
    type = "dummy"
    model_name = "good"
    timeouts = { non_streaming.total_ms = 99999999 }
    "#;
    let config = toml::from_str(config).unwrap();

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(
            err.to_string(),
            "The `timeouts.non_streaming.total_ms` value `99999999` is greater than `gateway.global_outbound_http_timeout_ms`: `300000`"
        );
}

#[tokio::test]
async fn deny_timeout_with_non_default_global_timeout() {
    let config = r#"
    gateway.global_outbound_http_timeout_ms = 200
    [models.slow_with_timeout]
    routing = ["slow"]

    [models.slow_with_timeout.providers.slow]
    type = "dummy"
    model_name = "good"
    timeouts = { non_streaming.total_ms = 500 }
    "#;
    let config = toml::from_str(config).unwrap();

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(
            err.to_string(),
            "The `timeouts.non_streaming.total_ms` value `500` is greater than `gateway.global_outbound_http_timeout_ms`: `200`"
        );
}

#[tokio::test]
async fn deny_bad_timeout_fields() {
    let config = r#"
    [models.slow_with_timeout]
    routing = ["slow"]

    [models.slow_with_timeout.providers.slow]
    type = "dummy"
    model = "good"
    timeouts = { bad_field = 1 }
    "#;
    let config = toml::from_str(config).unwrap();

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(
            err.to_string(),
            "models.slow_with_timeout.providers.slow.timeouts.bad_field: unknown field `bad_field`, expected `non_streaming` or `streaming`"
        );
}

#[tokio::test]
async fn deny_bad_timeouts_non_streaming_field() {
    let config = r#"
        [models.slow_with_timeout]
        routing = ["slow"]

        [models.slow_with_timeout.providers.slow]
        type = "dummy"
        model = "good"
        timeouts = { non_streaming = { unknown_field = 1000 } }
        "#;
    let config = toml::from_str(config).unwrap();

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(
            err.to_string(),
            "models.slow_with_timeout.providers.slow.timeouts.non_streaming.unknown_field: unknown field `unknown_field`, expected `total_ms`"
        );
}

#[tokio::test]
async fn deny_user_schema_and_input_wrapper() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_str = r#"
        [functions.basic_test]
        type = "chat"
        user_schema = "user_schema.json"

        [functions.basic_test.variants.good_variant]
        type = "chat_completion"
        model = "my-model"
        input_wrappers.user = "user_wrapper.minijinja"
        "#;
    let config_path = temp_dir.path().join("config.toml");
    let user_schema_path = temp_dir.path().join("user_schema.json");
    let user_wrapper_path = temp_dir.path().join("user_wrapper.minijinja");
    std::fs::write(&config_path, config_str).unwrap();
    std::fs::write(&user_schema_path, "{}").unwrap();
    std::fs::write(&user_wrapper_path, "Plain user wrapper").unwrap();

    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(&config_path).unwrap())
            .await
            .unwrap_err()
            .to_string();

    assert_eq!(
            err.to_string(),
            "functions.basic_test.variants.good_variant: Cannot provide both `input_wrappers.user` and `user_schema`"
        );
}

#[tokio::test]
async fn deny_user_template_and_input_wrapper() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_str = r#"
        [functions.basic_test]
        type = "chat"

        [functions.basic_test.variants.good_variant]
        type = "chat_completion"
        model = "my-model"
        user_template = "user_template.minijinja"
        input_wrappers.user = "user_wrapper.minijinja"
        "#;
    let config_path = temp_dir.path().join("config.toml");
    let user_template_path = temp_dir.path().join("user_template.minijinja");
    let user_wrapper_path = temp_dir.path().join("user_wrapper.minijinja");
    std::fs::write(&config_path, config_str).unwrap();
    std::fs::write(&user_template_path, "Plain user template").unwrap();
    std::fs::write(&user_wrapper_path, "Plain user wrapper").unwrap();

    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(&config_path).unwrap())
            .await
            .unwrap_err()
            .to_string();

    assert_eq!(
            err.to_string(),
            "functions.basic_test.variants.good_variant: Cannot provide both `input_wrappers.user` and `user` template"
        );
}

#[tokio::test]
async fn deny_fuser_user_template_and_input_wrapper() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_str = r#"
        [functions.basic_test]
        type = "chat"

        [functions.basic_test.variants.good_variant]
        type = "experimental_mixture_of_n"
        candidates = []

        [functions.basic_test.variants.good_variant.fuser]
        model = "dummy::echo_request_messages"
        user_template = "user_template.minijinja"
        input_wrappers.user = "user_wrapper.minijinja"
        "#;
    let config_path = temp_dir.path().join("config.toml");
    let user_template_path = temp_dir.path().join("user_template.minijinja");
    let user_wrapper_path = temp_dir.path().join("user_wrapper.minijinja");
    std::fs::write(&config_path, config_str).unwrap();
    std::fs::write(&user_template_path, "Plain user template").unwrap();
    std::fs::write(&user_wrapper_path, "Plain user wrapper").unwrap();

    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(&config_path).unwrap())
            .await
            .unwrap_err()
            .to_string();

    assert_eq!(
            err.to_string(),
            "functions.basic_test.variants.good_variant.fuser: Cannot provide both `input_wrappers.user` and `user` template"
        );
}

#[tokio::test]
async fn deny_evaluator_user_template_and_input_wrapper() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_str = r#"
        [functions.basic_test]
        type = "chat"

        [functions.basic_test.variants.good_variant]
        type = "experimental_best_of_n_sampling"
        candidates = []

        [functions.basic_test.variants.good_variant.evaluator]
        model = "dummy::echo_request_messages"
        user_template = "user_template.minijinja"
        input_wrappers.user = "user_wrapper.minijinja"
        "#;
    let config_path = temp_dir.path().join("config.toml");
    let user_template_path = temp_dir.path().join("user_template.minijinja");
    let user_wrapper_path = temp_dir.path().join("user_wrapper.minijinja");
    std::fs::write(&config_path, config_str).unwrap();
    std::fs::write(&user_template_path, "Plain user template").unwrap();
    std::fs::write(&user_wrapper_path, "Plain user wrapper").unwrap();

    let err =
        Config::load_and_verify_from_path(&ConfigFileGlob::new_from_path(&config_path).unwrap())
            .await
            .unwrap_err()
            .to_string();

    assert_eq!(
            err.to_string(),
            "functions.basic_test.variants.good_variant.evaluator: Cannot provide both `input_wrappers.user` and `user` template"
        );
}

#[tokio::test]
async fn deny_bad_timeouts_streaming_field() {
    let config = r#"
        [models.slow_with_timeout]
        routing = ["slow"]

        [models.slow_with_timeout.providers.slow]
        type = "dummy"
        model = "good"
        timeouts = { streaming = { unknown_field = 1000 } }
        "#;
    let config = toml::from_str(config).unwrap();

    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(
            err.to_string(),
            "models.slow_with_timeout.providers.slow.timeouts.streaming.unknown_field: unknown field `unknown_field`, expected `ttft_ms`"
        );
}

#[tokio::test]
async fn test_glob_relative_path() {
    let temp_dir = tempfile::tempdir().unwrap();
    let base_config = r#"
        [functions.no_schema]
        type = "chat"

        [functions.no_schema.variants.my_variant]
        type = "chat_completion"
        model = "dummy::echo_request_messages"
        system_template = "./first/first_template.minijinja"
        "#;

    std::fs::write(temp_dir.path().join("base_config.toml"), base_config).unwrap();

    let first_dir = temp_dir.path().join("first");
    std::fs::create_dir(&first_dir).unwrap();
    std::fs::write(first_dir.join("first_template.minijinja"), "Hello, world!").unwrap();

    let second_dir = temp_dir.path().join("second");
    std::fs::create_dir(&second_dir).unwrap();
    std::fs::write(
        second_dir.join("second_template.minijinja"),
        "Hello, world!",
    )
    .unwrap();

    std::fs::write(
        second_dir.join("second_config.toml"),
        r#"functions.no_schema.variants.my_variant.user_template = "second_template.minijinja""#,
    )
    .unwrap();

    std::fs::write(
        second_dir.join("second_template.minijinja"),
        "My second template",
    )
    .unwrap();

    let ConfigLoadInfo { config, .. } = Config::load_and_verify_from_path(
        &ConfigFileGlob::new_from_path(temp_dir.path().join("**/*.toml").as_path()).unwrap(),
    )
    .await
    .unwrap();

    let function = config.get_function("no_schema").unwrap();
    let FunctionConfig::Chat(function) = &**function else {
        panic!("Function should be a chat function");
    };
    let VariantConfig::ChatCompletion(variant) = &function.variants["my_variant"].inner else {
        panic!("Variant should be a chat completion variant");
    };
    assert_eq!(
        variant
            .templates()
            .get_implicit_template(Role::User)
            .unwrap()
            .template
            .path
            .get_template_key(),
        format!(
            "{}/second/second_template.minijinja",
            temp_dir.path().display()
        )
    );
    assert_eq!(
        variant
            .templates()
            .get_implicit_system_template()
            .unwrap()
            .template
            .contents,
        "Hello, world!"
    );

    assert_eq!(
        variant
            .templates()
            .get_implicit_system_template()
            .unwrap()
            .template
            .path
            .get_template_key(),
        format!(
            "{}/./first/first_template.minijinja",
            temp_dir.path().display()
        )
    );

    assert_eq!(
        variant
            .templates()
            .get_implicit_template(Role::User)
            .unwrap()
            .template
            .contents,
        "My second template"
    );
}

#[tokio::test]
async fn test_invalid_glob() {
    let err = ConfigFileGlob::new("/fake/tensorzero-config-test/path/to/**/fake.toml".to_string())
        .unwrap_err();

    assert_eq!(
        err.to_string(),
        "Error using glob: `/fake/tensorzero-config-test/path/to/**/fake.toml`: No files matched the glob pattern. Ensure that the path exists, and contains at least one file."
    );
}

#[tokio::test]
async fn test_glob_duplicate_key() {
    let config_a = r#"
        [functions.first]
        type = "chat"

        [functions.first.variants.first_variant]
        type = "chat_completion"
        model = "dummy::echo_request_messages"
        "#;

    let config_b = r"
        functions.first.variants.first_variant.max_tokens = 200
        ";

    let config_c = r#"
        functions.first.variants.first_variant.model = "other_model"
        "#;

    let temp_dir = tempfile::tempdir().unwrap();
    let config_a_path = temp_dir.path().join("config_a.toml");
    let config_b_path = temp_dir.path().join("config_b.toml");
    let config_c_path = temp_dir.path().join("config_c.toml");
    std::fs::write(&config_a_path, config_a).unwrap();
    std::fs::write(&config_b_path, config_b).unwrap();
    std::fs::write(&config_c_path, config_c).unwrap();

    let err = Config::load_and_verify_from_path(
        &ConfigFileGlob::new_from_path(temp_dir.path().join("*.toml").as_path()).unwrap(),
    )
    .await
    .expect_err("Config should fail to load");

    assert_eq!(
        err.to_string(),
        format!(
            "`functions.first.variants.first_variant.model`: Found duplicate values in globbed TOML config files `{}` and `{}`",
            config_a_path.display(),
            config_c_path.display()
        )
    );
}

#[tokio::test]
async fn test_glob_merge_non_map() {
    let config_a = r#"
        [functions.first]
        type = "chat"

        [functions.first.variants.first_variant]
        type = "chat_completion"
        model = "dummy::echo_request_messages"
        "#;

    let config_b = r"
        functions.first = 123
        ";

    let temp_dir = tempfile::tempdir().unwrap();
    let config_a_path = temp_dir.path().join("config_a.toml");
    let config_b_path = temp_dir.path().join("config_b.toml");
    std::fs::write(&config_a_path, config_a).unwrap();
    std::fs::write(&config_b_path, config_b).unwrap();

    let err = Config::load_and_verify_from_path(
        &ConfigFileGlob::new_from_path(temp_dir.path().join("*.toml").as_path()).unwrap(),
    )
    .await
    .expect_err("Config should fail to load");

    assert_eq!(
        err.to_string(),
        format!(
            "`functions.first`: Cannot merge `integer` from file `{}` into a table from file `{}`",
            config_b_path.display(),
            config_a_path.display()
        )
    );
}

#[tokio::test]
async fn test_config_schema_missing_template() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .display()
        .to_string();
    // We write an absolute path into the config file, since we're writing the config file in a temp dir.
    temp_file
        .write_all(
            format!(r#"
        [functions.test]
        type = "chat"
        schemas.my_custom_schema.path = "{base_path}/fixtures/config/functions/json_success/user_schema.json"

        [functions.test.variants.missing_template]
        type = "chat_completion"
        model = "dummy::echo_request_messages"
        "#).as_bytes(),
        )
        .unwrap();

    let config = UninitializedConfig::read_toml_config(
        &ConfigFileGlob::new_from_path(temp_file.path()).unwrap(),
        false,
    )
    .unwrap();
    let err = Config::load_from_toml(config.table)
        .await
        .expect_err("Config should fail to load");

    assert_eq!(err.to_string(), "`functions.test.variants.missing_template.templates.my_custom_schema` is required when `functions.test.schemas.my_custom_schema` is specified");
}

#[tokio::test]
async fn test_experimentation_with_variant_weights_error_uniform() {
    let config_str = r#"
        [models.test]
        routing = ["test"]

        [models.test.providers.test]
        type = "dummy"
        model_name = "test"

        [functions.test_function]
        type = "chat"

        [functions.test_function.variants.variant_a]
        type = "chat_completion"
        model = "test"
        weight = 0.5

        [functions.test_function.variants.variant_b]
        type = "chat_completion"
        model = "test"
        weight = 0.5

        [functions.test_function.experimentation]
        type = "uniform"
        "#;

    let config = toml::from_str(config_str).expect("Failed to parse config");
    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");

    let err_msg = err.to_string();
    assert!(
        err_msg.contains(
            "Cannot mix `experimentation` configuration with individual variant `weight` values"
        ),
        "Unexpected error message: {err_msg}"
    );
    assert!(
        err_msg.contains("variant_a") && err_msg.contains("variant_b"),
        "Error should list both variants with weights: {err_msg}"
    );
}

#[tokio::test]
async fn test_experimentation_with_variant_weights_error_static_weights() {
    let config_str = r#"
        [models.test]
        routing = ["test"]

        [models.test.providers.test]
        type = "dummy"
        model_name = "test"

        [functions.test_function]
        type = "chat"

        [functions.test_function.variants.variant_a]
        type = "chat_completion"
        model = "test"
        weight = 0.7

        [functions.test_function.variants.variant_b]
        type = "chat_completion"
        model = "test"

        [functions.test_function.experimentation]
        type = "static_weights"
        candidate_variants = {"variant_a" = 0.3, "variant_b" = 0.7}
        fallback_variants = ["variant_a", "variant_b"]
        "#;

    let config = toml::from_str(config_str).expect("Failed to parse config");
    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");

    let err_msg = err.to_string();
    assert!(
        err_msg.contains(
            "Cannot mix `experimentation` configuration with individual variant `weight` values"
        ),
        "Unexpected error message: {err_msg}"
    );
    assert!(
        err_msg.contains("variant_a"),
        "Error should list the variant with weight: {err_msg}"
    );
}

#[tokio::test]
async fn test_experimentation_with_variant_weights_error_track_and_stop() {
    let config_str = r#"
        [models.test]
        routing = ["test"]

        [models.test.providers.test]
        type = "dummy"
        model_name = "test"

        [metrics.test_metric]
        type = "boolean"
        optimize = "max"
        level = "inference"

        [functions.test_function]
        type = "chat"

        [functions.test_function.variants.variant_a]
        type = "chat_completion"
        model = "test"
        weight = 0.6

        [functions.test_function.variants.variant_b]
        type = "chat_completion"
        model = "test"

        [functions.test_function.experimentation]
        type = "track_and_stop"
        metric = "test_metric"
        candidate_variants = ["variant_a", "variant_b"]
        fallback_variants = ["variant_a"]
        min_samples_per_variant = 100
        delta = 0.05
        epsilon = 0.1
        update_period_s = 60
        "#;

    let config = toml::from_str(config_str).expect("Failed to parse config");
    let err = Config::load_from_toml(config)
        .await
        .expect_err("Config should fail to load");

    let err_msg = err.to_string();
    assert!(
        err_msg.contains(
            "Cannot mix `experimentation` configuration with individual variant `weight` values"
        ),
        "Unexpected error message: {err_msg}"
    );
    assert!(
        err_msg.contains("variant_a"),
        "Error should list the variant with weight: {err_msg}"
    );
}

// Unit tests for glob pattern matching functionality

#[test]
fn test_extract_base_path_from_glob_with_recursive_pattern() {
    // Pattern with ** should extract up to that component
    let base = extract_base_path_from_glob("/tmp/config/**/*.toml");
    assert_eq!(base, PathBuf::from("/tmp/config"));

    let base = extract_base_path_from_glob("config/**/*.toml");
    assert_eq!(base, PathBuf::from("config"));
}

#[test]
fn test_extract_base_path_from_glob_with_wildcard_in_filename() {
    // Pattern with * in filename should extract directory
    let base = extract_base_path_from_glob("/tmp/config/*.toml");
    assert_eq!(base, PathBuf::from("/tmp/config"));

    let base = extract_base_path_from_glob("config/*.toml");
    assert_eq!(base, PathBuf::from("config"));
}

#[test]
fn test_extract_base_path_from_glob_wildcard_only() {
    // Pattern starting with wildcard should use current directory
    let base = extract_base_path_from_glob("*.toml");
    assert_eq!(base, PathBuf::from("."));

    let base = extract_base_path_from_glob("**/*.toml");
    assert_eq!(base, PathBuf::from("."));
}

#[test]
fn test_extract_base_path_from_glob_no_pattern() {
    let base = extract_base_path_from_glob("/tmp/config/file.toml");
    assert_eq!(base, PathBuf::from("/tmp/config/file.toml"));

    let base = extract_base_path_from_glob("/tmp/config");
    assert_eq!(base, PathBuf::from("/tmp/config"));

    let base = extract_base_path_from_glob("config/file.toml");
    assert_eq!(base, PathBuf::from("config/file.toml"));

    let base = extract_base_path_from_glob("config");
    assert_eq!(base, PathBuf::from("config"));

    let base = extract_base_path_from_glob("file.toml");
    assert_eq!(base, PathBuf::from("file.toml"));
}

#[test]
fn test_extract_base_path_from_glob_without_parent() {
    let base = extract_base_path_from_glob("file?.toml");
    assert_eq!(base, PathBuf::from("."));

    let base = extract_base_path_from_glob("*");
    assert_eq!(base, PathBuf::from("."));
}

#[test]
fn test_extract_base_path_from_glob_question_mark() {
    // Question mark is also a glob metacharacter
    let base = extract_base_path_from_glob("/tmp/config/file?.toml");
    assert_eq!(base, PathBuf::from("/tmp/config"));
}

#[test]
fn test_extract_base_path_from_glob_brackets() {
    // Brackets are glob metacharacters
    let base = extract_base_path_from_glob("/tmp/config/file[0-9].toml");
    assert_eq!(base, PathBuf::from("/tmp/config"));
}

#[test]
fn test_extract_base_path_from_glob_braces() {
    // Braces are glob metacharacters
    let base = extract_base_path_from_glob("/tmp/config/{a,b}.toml");
    assert_eq!(base, PathBuf::from("/tmp/config"));
}

#[tokio::test]
async fn test_config_file_glob_integration() {
    // Integration test: create temp files and verify glob matching works
    let temp_dir = tempfile::tempdir().unwrap();

    // Create directory structure
    let config_dir = temp_dir.path().join("config");
    std::fs::create_dir(&config_dir).unwrap();

    // Create some test files
    std::fs::write(config_dir.join("base.toml"), "[test]\nkey = \"base\"").unwrap();
    std::fs::write(config_dir.join("dev.toml"), "[test]\nkey = \"dev\"").unwrap();
    std::fs::write(config_dir.join("README.md"), "# README").unwrap();

    // Test glob pattern matching
    let glob_pattern = format!("{}/*.toml", config_dir.display());
    let config_glob = ConfigFileGlob::new(glob_pattern).unwrap();

    // Should match 2 .toml files, not the .md file
    assert_eq!(config_glob.paths.len(), 2);
    assert!(config_glob
        .paths
        .iter()
        .all(|p| p.extension().unwrap() == "toml"));
}

#[tokio::test]
async fn test_config_file_glob_recursive() {
    // Test recursive glob pattern matching
    let temp_dir = tempfile::tempdir().unwrap();

    // Create nested directory structure
    let base_dir = temp_dir.path().join("config");
    let sub_dir = base_dir.join("subdir");
    std::fs::create_dir_all(&sub_dir).unwrap();

    // Create files at different levels
    std::fs::write(base_dir.join("base.toml"), "[test]\nkey = \"base\"").unwrap();
    std::fs::write(sub_dir.join("nested.toml"), "[test]\nkey = \"nested\"").unwrap();

    // Test recursive glob pattern
    let glob_pattern = format!("{}/**/*.toml", base_dir.display());
    let config_glob = ConfigFileGlob::new(glob_pattern).unwrap();

    // Should match both files
    assert_eq!(config_glob.paths.len(), 2);
}

/// Test that built-in functions are automatically loaded
#[tokio::test]
async fn test_built_in_functions_loaded() {
    // Load a minimal config (empty table)
    let config = toml::Table::new();
    let ConfigLoadInfo { config, .. } = Config::load_from_toml(config)
        .await
        .expect("Failed to load config");

    // Check that both built-in functions are available
    assert!(config.functions.contains_key("tensorzero::hello_chat"));
    assert!(config.functions.contains_key("tensorzero::hello_json"));

    // Verify hello_chat is a Chat function with no variants
    let hello_chat = config.functions.get("tensorzero::hello_chat").unwrap();
    match &**hello_chat {
        FunctionConfig::Chat(chat_config) => {
            assert!(chat_config.variants.is_empty());
            assert!(chat_config.tools.is_empty());
            assert!(chat_config.description.is_some());
        }
        FunctionConfig::Json(_) => panic!("Expected tensorzero::hello_chat to be a Chat function"),
    }

    // Verify hello_json is a JSON function with no variants
    let hello_json = config.functions.get("tensorzero::hello_json").unwrap();
    match &**hello_json {
        FunctionConfig::Json(json_config) => {
            assert!(json_config.variants.is_empty());
            assert!(json_config.description.is_some());
        }
        FunctionConfig::Chat(_) => panic!("Expected tensorzero::hello_json to be a JSON function"),
    }
}

/// Test that built-in functions can be retrieved via get_function
#[tokio::test]
async fn test_get_built_in_function() {
    let config = toml::Table::new();
    let ConfigLoadInfo { config, .. } = Config::load_from_toml(config)
        .await
        .expect("Failed to load config");

    // Should be able to get both built-in functions
    assert!(config.get_function("tensorzero::hello_chat").is_ok());
    assert!(config.get_function("tensorzero::hello_json").is_ok());
}

/// Test that built-in functions work alongside user-defined functions
#[tokio::test]
async fn test_built_in_and_user_functions_coexist() {
    let config = get_sample_valid_config();

    let ConfigLoadInfo { config, .. } = Config::load_from_toml(config)
        .await
        .expect("Failed to load config");

    // Check that both built-in and user functions exist
    assert!(config.functions.contains_key("tensorzero::hello_chat"));
    assert!(config.functions.contains_key("tensorzero::hello_json"));
    assert!(config.functions.contains_key("generate_draft"));
}

/// Test that the deprecated `gateway.template_filesystem_access.enabled` option
/// is still accepted and emits a deprecation warning
#[tokio::test]
async fn test_deprecated_template_filesystem_access_enabled() {
    let logs_contain = crate::utils::testing::capture_logs();
    let tempfile = NamedTempFile::new().unwrap();
    write!(
        &tempfile,
        r"
            [gateway.template_filesystem_access]
            enabled = true

            [functions]"
    )
    .unwrap();

    // Should successfully load config despite deprecated field
    let _config = Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new_from_path(tempfile.path()).unwrap(),
        false,
    )
    .await
    .unwrap();

    // Should emit deprecation warning
    assert!(logs_contain(
        "The `gateway.template_filesystem_access.enabled` flag is deprecated"
    ));
}
