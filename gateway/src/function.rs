use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::types::{InputMessage, InputMessageRole};
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::variant::VariantConfig;

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum FunctionConfig {
    Chat(FunctionConfigChat),
    Tool(FunctionConfigTool),
}

impl FunctionConfig {
    pub fn variants(&self) -> &HashMap<String, VariantConfig> {
        match self {
            FunctionConfig::Chat(params) => &params.variants,
            FunctionConfig::Tool(params) => &params.variants,
        }
    }

    pub fn output_schema(&self) -> Option<&JSONSchemaFromPath> {
        match self {
            FunctionConfig::Chat(params) => params.output_schema.as_ref(),
            FunctionConfig::Tool(params) => params.output_schema.as_ref(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConfigChat {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: Option<JSONSchemaFromPath>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionConfigTool {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: Option<JSONSchemaFromPath>,
}

impl FunctionConfig {
    /// Validate the input against the function's input schemas.
    /// The validation is done based on the function's type:
    /// - For a chat function, the input is validated against the system, user, and assistant schemas.
    /// - For a tool function, the input is validated against the system, user, and assistant schemas.
    pub fn validate_input(&self, input: &[InputMessage]) -> Result<(), Error> {
        match &self {
            FunctionConfig::Chat(params) => {
                FunctionConfig::validate_chat_like_input(
                    &params.system_schema,
                    &params.user_schema,
                    &params.assistant_schema,
                    input,
                )?;
            }
            FunctionConfig::Tool(params) => {
                FunctionConfig::validate_chat_like_input(
                    &params.system_schema,
                    &params.user_schema,
                    &params.assistant_schema,
                    input,
                )?;
            }
        }

        Ok(())
    }

    /// Validate an input that is a chat-like function (i.e. chat or tool).
    /// The validation is done based on the input's role and the function's schemas.
    fn validate_chat_like_input(
        system_schema: &Option<JSONSchemaFromPath>,
        user_schema: &Option<JSONSchemaFromPath>,
        assistant_schema: &Option<JSONSchemaFromPath>,
        input: &[InputMessage],
    ) -> Result<(), Error> {
        for (index, message) in input.iter().enumerate() {
            match (
                &message.role,
                &system_schema,
                &user_schema,
                &assistant_schema,
            ) {
                (InputMessageRole::System, Some(ref system_schema), _, _) => {
                    system_schema.validate(&message.content)
                }
                (InputMessageRole::User, _, Some(ref user_schema), _) => {
                    user_schema.validate(&message.content)
                }
                (InputMessageRole::Assistant, _, _, Some(ref assistant_schema)) => {
                    assistant_schema.validate(&message.content)
                }
                _ => {
                    if !message.content.is_string() {
                        return Err(Error::InvalidMessage {
                            message: format!("Message at index {} has non-string content but there is no schema given for role {}.", index, message.role),
                        });
                    } else {
                        Ok(())
                    }
                }
            }?;
        }

        Ok(())
    }

    pub fn load<P: AsRef<Path>>(&mut self, base_path: P) -> Result<(), Error> {
        match self {
            FunctionConfig::Chat(params) => {
                params
                    .system_schema
                    .as_mut()
                    .map(|schema| schema.load(base_path.as_ref()));
                params
                    .user_schema
                    .as_mut()
                    .map(|schema| schema.load(base_path.as_ref()));
                params
                    .assistant_schema
                    .as_mut()
                    .map(|schema| schema.load(base_path.as_ref()));
                params
                    .output_schema
                    .as_mut()
                    .map(|schema| schema.load(base_path.as_ref()));
                Ok(())
            }
            FunctionConfig::Tool(params) => {
                params
                    .system_schema
                    .as_mut()
                    .map(|schema| schema.load(base_path.as_ref()));
                params
                    .user_schema
                    .as_mut()
                    .map(|schema| schema.load(base_path.as_ref()));
                params
                    .assistant_schema
                    .as_mut()
                    .map(|schema| schema.load(base_path.as_ref()));
                params
                    .output_schema
                    .as_mut()
                    .map(|schema| schema.load(base_path.as_ref()));
                Ok(())
            }
        }
    }
}

/// Sample a variant from the function based on variant weights (uniform random selection)
pub fn sample_variant(
    variants: &mut HashMap<String, VariantConfig>,
    function_name: &str,
    episode_id: &Uuid,
) -> Result<(String, VariantConfig), Error> {
    // Compute the total weight of all variants
    let total_weight = variants
        .values()
        .map(|variant| variant.weight())
        .sum::<f64>();

    // If the total weight is non-positive, return an error
    // NOTE: We enforce non-negative weights at the config parsing stage, but it's good to be extra
    //       safe here to ensure that we catch any regressions we might introduce in the future.
    if total_weight <= 0. {
        return Err(Error::InvalidFunctionVariants {
            message: format!("Function `{function_name}` variants have non-positive total weight"),
        });
    }

    // Sample a random threshold between 0 and the total weight
    let random_threshold = get_uniform_value(function_name, episode_id) * total_weight;

    // Iterate over the variants to find the one that corresponds to the sampled threshold
    let mut cumulative_weight = 0.;
    let mut sampled_variant_name = String::new();
    for (variant_name, variant) in variants.iter() {
        cumulative_weight += variant.weight();
        if cumulative_weight > random_threshold {
            sampled_variant_name.clone_from(variant_name);
            break;
        }
    }

    // If we didn't find a variant (which should only happen due to rare numerical precision issues),
    // use the last variant as a fallback
    if sampled_variant_name.is_empty() {
        sampled_variant_name.clone_from(variants.keys().last().ok_or_else(|| {
            Error::InvalidFunctionVariants {
                message: format!("Function `{function_name}` has no variants"),
            }
        })?);
    }

    // Remove and return the sampled variant
    variants
        .remove(&sampled_variant_name)
        .map(|variant| (sampled_variant_name, variant))
        .ok_or_else(|| Error::InvalidFunctionVariants {
            message: format!("Failed to remove sampled variant from function `{function_name}`"),
        })
}

/// Implements a uniform distribution over the interval [0, 1) using a hash function.
/// This function is deterministic but should have good statistical properties.
fn get_uniform_value(function_name: &str, episode_id: &Uuid) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(function_name.as_bytes());
    hasher.update(episode_id.as_bytes());
    let hash_value = hasher.finalize();
    let truncated_hash =
        u32::from_be_bytes([hash_value[0], hash_value[1], hash_value[2], hash_value[3]]);
    truncated_hash as f64 / u32::MAX as f64
}

#[cfg(test)]
mod tests {
    use crate::variant::ChatCompletionConfig;

    use super::*;
    use serde_json::json;
    use std::{io::Write, path::PathBuf};
    use tempfile::NamedTempFile;

    fn create_test_schema() -> JSONSchemaFromPath {
        let schema = r#"
        {
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"],
            "additionalProperties": false
        }
        "#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", schema).expect("Failed to write schema to temporary file");

        let mut schema = JSONSchemaFromPath::new(temp_file.path().to_owned());
        schema
            .load::<&std::path::Path>(&PathBuf::from(""))
            .expect("Failed to load schema");
        schema
    }

    #[test]
    fn test_validate_input_chat_no_schema() {
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidMessage {
                message: "Message at index 0 has non-string content but there is no schema given for role \"system\".".to_string()
            }
        );
    }

    #[test]
    fn test_validate_input_chat_system_schema() {
        let system_schema = create_test_schema();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: Some(system_schema.clone()),
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: json!("system content"),
                schema: system_schema.value().unwrap().clone(),
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_user_schema() {
        let user_schema = create_test_schema();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(user_schema.clone()),
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()],
                data: json!("user content"),
                schema: user_schema.value().unwrap().clone(),
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_assistant_schema() {
        let assistant_schema = create_test_schema();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: Some(assistant_schema.clone()),
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()],
                data: json!("assistant content"),
                schema: assistant_schema.value().unwrap().clone(),
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_all_schemas() {
        let system_schema = create_test_schema();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: Some(system_schema.clone()),
            user_schema: Some(user_schema),
            assistant_schema: Some(assistant_schema),
            output_schema: None,
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: json!("system content"),
                schema: system_schema.value().unwrap().clone(),
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_tool_no_schema() {
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system_content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidMessage {
                message: "Message at index 1 has non-string content but there is no schema given for role \"user\".".to_string()
            }
        );
    }

    #[test]
    fn test_validate_input_tool_system_schema() {
        let system_schema = create_test_schema();
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: Some(system_schema.clone()),
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: json!("system content"),
                schema: system_schema.value().unwrap().clone(),
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_tool_user_schema() {
        let user_schema = create_test_schema();
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(user_schema.clone()),
            assistant_schema: None,
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()],
                data: json!("user content"),
                schema: user_schema.value().unwrap().clone(),
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_tool_assistant_schema() {
        let assistant_schema = create_test_schema();
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: Some(assistant_schema.clone()),
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()],
                data: json!("assistant content"),
                schema: assistant_schema.value().unwrap().clone(),
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_tool_all_schemas() {
        let system_schema = create_test_schema();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
        let tool_config = FunctionConfigTool {
            variants: HashMap::new(),
            system_schema: Some(system_schema.clone()),
            user_schema: Some(user_schema),
            assistant_schema: Some(assistant_schema),
            output_schema: None,
        };
        let function_config = FunctionConfig::Tool(tool_config);

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!("system content"),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!("user content"),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!("assistant content"),
            },
        ];

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: json!("system content"),
                schema: system_schema.value().unwrap().clone(),
            }
        );

        let input = vec![
            InputMessage {
                role: InputMessageRole::System,
                content: json!({ "name": "system name" }),
            },
            InputMessage {
                role: InputMessageRole::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: InputMessageRole::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        assert!(function_config.validate_input(&input).is_ok());
    }

    /// Tests the `sample_variant` function with a variety of test cases through Monte Carlo simulations.
    ///
    /// NOTE: If this test fails, it might be due to sampling. Please run it again to check if the
    ///       issue persists.
    #[test]
    fn test_sample_variant() {
        // Helper function to create a HashMap of variant names to their weights
        fn create_variants(variant_weights: &[(&str, f64)]) -> HashMap<String, VariantConfig> {
            variant_weights
                .iter()
                .map(|&(name, weight)| {
                    (
                        name.to_string(),
                        VariantConfig::ChatCompletion(ChatCompletionConfig {
                            weight,
                            model: "model-name".to_string(),
                            system_template: None,
                            user_template: None,
                            assistant_template: None,
                        }),
                    )
                })
                .collect()
        }

        // Helper function to test the distribution of variant weights by sampling them many times
        // and checking if the observed distribution is close to the expected distribution
        fn test_variant_distribution(
            variants: &HashMap<String, VariantConfig>,
            sample_size: usize,
            tolerance: f64,
        ) {
            let total_weight: f64 = variants.values().map(|v| v.weight()).sum();
            let mut counts: HashMap<String, usize> = HashMap::new();

            for _ in 0..sample_size {
                let (variant_name, _) =
                    sample_variant(&mut variants.clone(), "test_function", &Uuid::now_v7())
                        .unwrap();
                *counts.entry(variant_name.clone()).or_insert(0) += 1;
            }

            for (variant_name, variant) in variants {
                let expected_prob = variant.weight() / total_weight;
                let actual_prob =
                    *counts.get(variant_name).unwrap_or(&0) as f64 / sample_size as f64;
                let diff = (expected_prob - actual_prob).abs();

                assert!(
                    diff <= tolerance,
                    "Probability for variant {} is outside the acceptable range",
                    variant_name
                );
            }
        }

        // Test case 1: Equal weights
        let variants = create_variants(&[("A", 1.0), ("B", 1.0), ("C", 1.0)]);
        test_variant_distribution(&variants, 10_000, 0.02);

        // Test case 2: Unequal weights
        let variants = create_variants(&[("X", 1.0), ("Y", 2.0), ("Z", 3.0)]);
        test_variant_distribution(&variants, 10_000, 0.02);

        // Test case 3: Extreme weights
        let variants = create_variants(&[("Rare", 0.01), ("Common", 0.99)]);
        test_variant_distribution(&variants, 10_000, 0.005);

        // Test case 4: Single weights
        let variants = create_variants(&[("Solo", 1.0)]);
        test_variant_distribution(&variants, 10_000, 0.0);
    }

    #[test]
    fn test_get_uniform_value() {
        // Test with function name and episode ID
        let episode_id = Uuid::now_v7();
        let value1 = get_uniform_value("test_function", &episode_id);
        let value2 = get_uniform_value("test_function", &episode_id);

        // Values should be the same due to deterministic input
        assert_eq!(value1, value2);
        assert!((0.0..1.0).contains(&value1));
        assert!((0.0..1.0).contains(&value2));

        // Test with different function names
        let value3 = get_uniform_value("another_function", &episode_id);
        assert_ne!(value1, value3);
        assert!((0.0..1.0).contains(&value3));

        // Test with different episode IDs
        let value4 = get_uniform_value("test_function", &Uuid::now_v7());
        assert_ne!(value1, value4);
        assert_ne!(value3, value4);
        assert!((0.0..1.0).contains(&value4));
    }
}
