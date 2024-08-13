use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::types::{Input, Role};
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::variant::VariantConfig;

#[derive(Debug)]
pub enum FunctionConfig {
    Chat(FunctionConfigChat),
    Json(FunctionConfigJson),
}

impl FunctionConfig {
    pub fn variants(&self) -> &HashMap<String, VariantConfig> {
        match self {
            FunctionConfig::Chat(params) => &params.variants,
            FunctionConfig::Json(params) => &params.variants,
        }
    }

    // TODO (viraj): rip this in the next PR
    pub fn output_schema(&self) -> Option<&JSONSchemaFromPath> {
        match self {
            FunctionConfig::Chat(_) => None,
            FunctionConfig::Json(params) => Some(&params.output_schema),
        }
    }
}

#[derive(Debug)]
pub struct FunctionConfigChat {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub tools: Vec<String>, // tool names
}

#[derive(Debug)]
pub struct FunctionConfigJson {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: JSONSchemaFromPath, // schema is mandatory for JSON functions
}

impl FunctionConfig {
    /// Validate the input against the function's input schemas.
    /// The validation is done based on the function's type:
    /// - For a chat function, the input is validated against the system, user, and assistant schemas.
    /// - For a JSON function, the input is validated against the system, user, and assistant schemas.
    pub fn validate_input(&self, input: &Input) -> Result<(), Error> {
        match &self {
            FunctionConfig::Chat(params) => {
                FunctionConfig::validate_chat_like_input(
                    &params.system_schema,
                    &params.user_schema,
                    &params.assistant_schema,
                    input,
                )?;
            }
            FunctionConfig::Json(params) => {
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
        input: &Input,
    ) -> Result<(), Error> {
        match (input.system.as_ref(), system_schema) {
            (Some(system), Some(ref system_schema)) => system_schema.validate(system),
            (None, None) => Ok(()),
            (Some(system), None) => {
                if system.is_string() {
                    Ok(())
                } else {
                    Err(Error::InvalidMessage { message: "`input.system` has non-string content but there is no schema given for `system`.".to_string() })
                }
            }
            (None, Some(_)) => Err(Error::InvalidMessage {
                message: "`input.system` is empty but a system template is present.".to_string(),
            }),
        }?;
        for (index, message) in input.messages.iter().enumerate() {
            match (&message.role, &user_schema, &assistant_schema) {
                (Role::User, Some(ref user_schema), _) => user_schema.validate(&message.content),
                (Role::Assistant, _, Some(ref assistant_schema)) => {
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

    // If the total weight is non-positive, perform uniform sampling
    // NOTE: We enforce non-negative weights at the config parsing stage,
    //       but there's a chance we pin a weight-zero variant in the config.
    //       This check also ensures that we catch any regressions we might introduce in the future.
    if total_weight <= 0. {
        // Perform uniform sampling if total weight is non-positive
        let random_index =
            (get_uniform_value(function_name, episode_id) * variants.len() as f64).floor() as usize;
        let sampled_variant_name = variants
            .keys()
            .nth(random_index)
            .ok_or_else(|| Error::InvalidFunctionVariants {
                message: format!("Function `{function_name}` has no variants"),
            })?
            .clone();
        return variants
            .remove(&sampled_variant_name)
            .map(|variant| (sampled_variant_name, variant))
            .ok_or_else(|| Error::InvalidFunctionVariants {
                message: format!(
                    "Failed to remove sampled variant from function `{function_name}`"
                ),
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
    use crate::{inference::types::InputMessage, variant::ChatCompletionConfig};

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

        JSONSchemaFromPath::new(temp_file.path().to_owned(), PathBuf::new())
            .expect("Failed to create schema")
    }

    #[test]
    fn test_validate_input_chat_no_schema() {
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];

        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];
        let input = Input {
            system: Some(json!("system name")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidMessage {
                message: "Message at index 0 has non-string content but there is no schema given for role user.".to_string()
            }
        );
    }

    #[test]
    fn test_validate_input_chat_system_schema() {
        let system_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: Some(system_schema),
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: json!("system content"),
                schema: system_value,
            }
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];
        let input = Input {
            system: Some(json!({ "name": "system name" })),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_user_schema() {
        let user_schema = create_test_schema();
        let user_value = user_schema.value.clone();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(user_schema),
            assistant_schema: None,
            tools: vec![],
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };
        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()],
                data: json!("user content"),
                schema: user_value,
            }
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_assistant_schema() {
        let assistant_schema = create_test_schema();
        let assistant_value = assistant_schema.value.clone();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: Some(assistant_schema),
            tools: vec![],
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };
        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()],
                data: json!("assistant content"),
                schema: assistant_value,
            }
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_all_schemas() {
        let system_schema = create_test_schema();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: Some(system_schema),
            user_schema: Some(user_schema),
            assistant_schema: Some(assistant_schema),
            tools: vec![],
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];

        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: json!("system content"),
                schema: system_value,
            }
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        let input = Input {
            system: Some(json!({ "name": "system name" })),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_json_no_schema() {
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&json!({})),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];

        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::InvalidMessage {
                message: "Message at index 0 has non-string content but there is no schema given for role user.".to_string()
            }
        );
    }

    #[test]
    fn test_validate_input_json_system_schema() {
        let system_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: Some(system_schema),
            user_schema: None,
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&json!({})),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];

        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: json!("system content"),
                schema: system_value,
            }
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];

        let input = Input {
            system: Some(json!({ "name": "system name" })),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_json_user_schema() {
        let user_schema = create_test_schema();
        let user_value = user_schema.value.clone();
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(user_schema),
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&json!({})),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];

        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()],
                data: json!("user content"),
                schema: user_value,
            }
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_json_assistant_schema() {
        let assistant_schema = create_test_schema();
        let assistant_value = assistant_schema.value.clone();
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: Some(assistant_schema),
            output_schema: JSONSchemaFromPath::from_value(&json!({})),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()],
                data: json!("assistant content"),
                schema: assistant_value,
            }
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_json_all_schemas() {
        let system_schema = create_test_schema();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: Some(system_schema),
            user_schema: Some(user_schema),
            assistant_schema: Some(assistant_schema),
            output_schema: JSONSchemaFromPath::from_value(&json!({})),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!("user content"),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!("assistant content"),
            },
        ];
        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: json!("system content"),
                schema: system_value,
            }
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: json!({ "name": "user name" }),
            },
            InputMessage {
                role: Role::Assistant,
                content: json!({ "name": "assistant name" }),
            },
        ];

        let input = Input {
            system: Some(json!({ "name": "system name" })),
            messages,
        };

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

        // Test case 5: All zero weights
        let variants = create_variants(&[("A", 0.0), ("B", 0.0), ("C", 0.0)]);
        let sample_size = 10_000;
        let mut counts: HashMap<String, usize> = HashMap::new();

        for _ in 0..sample_size {
            let (variant_name, _) =
                sample_variant(&mut variants.clone(), "test_function", &Uuid::now_v7()).unwrap();
            *counts.entry(variant_name).or_insert(0) += 1;
        }

        // Check if all variants are sampled approximately equally
        let expected_count = sample_size / variants.len();
        let tolerance = (expected_count as f64 * 0.1) as usize; // 10% tolerance

        for (variant_name, count) in counts {
            assert!(
                (count as i32 - expected_count as i32).abs() <= tolerance as i32,
                "Variant {} was not sampled uniformly. Expected {} +/- {}, got {}",
                variant_name,
                expected_count,
                tolerance,
                count
            );
        }
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
