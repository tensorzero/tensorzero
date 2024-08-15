use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::{Error, ResultExt};
use crate::inference::types::{
    ChatInferenceResult, ContentBlock, InferenceResult, Input, InputMessageContent,
    JsonInferenceResult, ModelInferenceResponse, Role, Usage,
};
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::tool::{ToolCallConfig, ToolChoice, ToolConfig};
use crate::variant::VariantConfig;

#[derive(Debug)]
pub enum FunctionConfig {
    Chat(FunctionConfigChat),
    Json(FunctionConfigJson),
}

#[derive(Debug, Default)]
pub struct FunctionConfigChat {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub tools: Vec<String>, // tool names
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JsonEnforcement {
    #[default]
    Default,
    Strict,
    ImplicitTool,
    Off,
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
    pub fn variants(&self) -> &HashMap<String, VariantConfig> {
        match self {
            FunctionConfig::Chat(params) => &params.variants,
            FunctionConfig::Json(params) => &params.variants,
        }
    }
}

impl FunctionConfig {
    /// Validate the input against the function's input schemas.
    /// The validation is done based on the function's type:
    /// - For a chat function, the input is validated against the system, user, and assistant schemas.
    /// - For a JSON function, the input is validated against the system, user, and assistant schemas.
    ///
    /// We do not validate ContentBlocks that are not text (tool calls and tool responses).
    ///  TODO (#30):
    ///   We should also enforce that no dynamic tool calling information should be passed to JSON functions
    ///   as they can't call tools.
    pub fn validate_input(&self, input: &Input) -> Result<(), Error> {
        match &self {
            FunctionConfig::Chat(params) => {
                validate_all_text_input(
                    params.system_schema.as_ref(),
                    params.user_schema.as_ref(),
                    params.assistant_schema.as_ref(),
                    input,
                )?;
            }
            FunctionConfig::Json(params) => {
                validate_all_text_input(
                    params.system_schema.as_ref(),
                    params.user_schema.as_ref(),
                    params.assistant_schema.as_ref(),
                    input,
                )?;
            }
        }
        Ok(())
    }

    pub fn prepare_tool_config(
        &'static self,
        // TODO (#126): implement dynamic tool calling
        // dynamic_tool_config: &'a DynamicToolConfig,
        static_tools: &'static HashMap<String, ToolConfig>,
    ) -> Result<Option<ToolCallConfig>, Error> {
        match self {
            FunctionConfig::Chat(params) => Ok(Some(ToolCallConfig::new(
                &params.tools,
                &params.tool_choice,
                params.parallel_tool_calls,
                static_tools,
                // TODO (#126): implement dynamic tool calling
                // dynamic_tool_config,
            )?)),
            FunctionConfig::Json(_) => {
                // if dynamic_tool_config.allowed_tools.is_some() {
                //     return Err(Error::InvalidRequest {
                //         message: "Cannot pass `allowed_tools` to a JSON function.".to_string(),
                //     });
                // }
                // if dynamic_tool_config.additional_tools.is_some() {
                //     return Err(Error::InvalidRequest {
                //         message: "Cannot pass `additional_tools` to a JSON function.".to_string(),
                //     });
                // }
                // if dynamic_tool_config.tool_choice.is_some() {
                //     return Err(Error::InvalidRequest {
                //         message: "Cannot pass `tool_choice` to a JSON function".to_string(),
                //     });
                // }
                Ok(None)
            }
        }
    }

    pub fn prepare_response(
        &self,
        inference_id: Uuid,
        content_blocks: Vec<ContentBlock>,
        usage: Usage,
        model_inference_responses: Vec<ModelInferenceResponse>,
        tool_config: Option<&ToolCallConfig>,
    ) -> Result<InferenceResult, Error> {
        match self {
            FunctionConfig::Chat(..) => Ok(InferenceResult::Chat(ChatInferenceResult::new(
                inference_id,
                content_blocks,
                usage,
                model_inference_responses,
                tool_config,
            ))),
            FunctionConfig::Json(params) => {
                // Parse the content blocks into a JSON object
                // We assume here that the last content block that's text or a tool call is the JSON object.
                // (this is because we could have used an implicit tool call and there is no other reason for a tool call in a JSON function).
                let raw = content_blocks
                    .into_iter()
                    .rev()
                    .find_map(|content_block| match content_block {
                        ContentBlock::Text(text) => Some(text.text),
                        ContentBlock::ToolCall(tool_call) => Some(tool_call.arguments),
                        _ => None,
                    })
                    .ok_or(Error::Inference {
                        message: "No valid content blocks found in JSON function response"
                            .to_string(),
                    })?;
                let parsed_output = serde_json::from_str::<Value>(&raw)
                    .map_err(|e| Error::OutputParsing {
                        message: format!(
                            "Failed to parse output from JSON function response {}",
                            e
                        ),
                        raw_output: raw.clone(),
                    })
                    .ok_or_log();

                // If the parsed output fails validation, we log the error and set `parsed_output` to None
                let parsed_output = parsed_output.and_then(|parsed_output| {
                    match params.output_schema.validate(&parsed_output) {
                        Ok(_) => Some(parsed_output),
                        Err(e) => {
                            e.log();
                            None
                        }
                    }
                });
                Ok(InferenceResult::Json(JsonInferenceResult::new(
                    inference_id,
                    raw,
                    parsed_output,
                    usage,
                    model_inference_responses,
                )))
            }
        }
    }
}

/// Validate all input messages that contain text.
/// The validation is done based on the input's role and the function's schemas.
/// We first validate the system message (if it exists)
/// Next we validate all messages containing text blocks.
/// If there are multiple text blocks in a message we reject.
fn validate_all_text_input(
    system_schema: Option<&JSONSchemaFromPath>,
    user_schema: Option<&JSONSchemaFromPath>,
    assistant_schema: Option<&JSONSchemaFromPath>,
    input: &Input,
) -> Result<(), Error> {
    match (input.system.as_ref(), system_schema) {
        // If there is any system message passed we validate it
        (Some(system), _) => validate_single_message(system, system_schema, None),
        // If there is no system message and no schema we accept
        (None, None) => Ok(()),
        // If no system message is passed and we have a schema we fail
        (None, Some(_)) => Err(Error::InvalidMessage {
            message: "`input.system` is empty but a system template is present.".to_string(),
        }),
    }?;
    for (index, message) in input.messages.iter().enumerate() {
        let mut content: Option<&Value> = None;
        for block in message.content.iter() {
            if let InputMessageContent::Text { value } = block {
                // Throw an error if we have multiple text blocks in a message
                if content.is_some() {
                    return Err(Error::InvalidMessage {
                        message: format!(
                            "Message at index {index} has multiple text content blocks"
                        ),
                    });
                }
                content = Some(value);
            }
        }
        if let Some(content) = content {
            match &message.role {
                Role::Assistant => validate_single_message(
                    content,
                    assistant_schema,
                    Some((index, &message.role)),
                )?,
                Role::User => {
                    validate_single_message(content, user_schema, Some((index, &message.role)))?
                }
            }
        }
    }
    Ok(())
}

/// Validates a single message according to the following rules:
/// If there is no schema, the message `content` must be a string
/// Otherwise, the message must contain JSON content that matches the schema
fn validate_single_message(
    content: &Value,
    schema: Option<&JSONSchemaFromPath>,
    index_role: Option<(usize, &Role)>,
) -> Result<(), Error> {
    match schema {
        Some(schema) => schema.validate(content),
        None => {
            if content.is_string() {
                Ok(())
            } else {
                Err(match index_role {
                    Some(index_role) => Error::InvalidMessage {
                        message: format!("Message at index {} has non-string content but there is no schema given for role {}.", index_role.0, index_role.1),
                    },
                    None => Error::InvalidMessage {message:"Message has non-string content but there is no schema given for role system.".to_string()},
                })
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
    use crate::inference::types::Latency;
    use crate::tool::ToolCall;
    use crate::variant::ChatCompletionConfig;
    use crate::{inference::types::InputMessage, variant::JsonEnforcement};

    use super::*;
    use serde_json::json;
    use std::time::Duration;
    use std::{io::Write, path::PathBuf};
    use tempfile::NamedTempFile;
    use tracing_test::traced_test;

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
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!({ "name": "assistant name" }).into()],
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
                message: "Message at index 1 has non-string content but there is no schema given for role assistant.".to_string()
            }
        );

        // Test case for multiple text content blocks in one message
        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![
                    "first user content".to_string().into(),
                    "second user content".to_string().into(),
                ],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                message: "Message at index 0 has multiple text content blocks".to_string()
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
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                content: vec![json!({ "name": "user name" }).into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!({ "name": "assistant name" }).into()],
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
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                content: vec![json!({ "name": "user name" }).into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!({ "name": "assistant name" }).into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                content: vec![json!({ "name": "user name" }).into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!({ "name": "assistant name" }).into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!("assistant content").to_string().into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!("assistant content").to_string().into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!("assistant content").to_string().into()],
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
                content: vec![json!({ "name": "user name" }).into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!({ "name": "assistant name" }).into()],
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
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!("assistant content").to_string().into()],
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
                content: vec![json!({ "name": "user name" }).into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!({ "name": "assistant name" }).into()],
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
                            json_mode: JsonEnforcement::Default,
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

    #[test]
    #[traced_test]
    fn test_prepare_response_json() {
        // The Chat stuff is tested in types::test_create_chat_inference_response
        // Here we focus on the JSON stuff
        let output_schema = JSONSchemaFromPath::from_value(&json!({
          "$schema": "http://json-schema.org/draft-07/schema#",
          "type": "object",
          "properties": {
            "name": {
              "type": "string"
            },
            "age": {
              "type": "integer",
              "minimum": 0
            }
          },
          "required": ["name", "age"],
          "additionalProperties": false
        }));
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema,
        });

        // Test with a non-JSON content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec!["Hello, world!".to_string().into()];
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponse::new(
            content_blocks.clone(),
            "content".to_string(),
            usage.clone(),
            latency,
        );
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                None,
            )
            .unwrap();
        assert!(logs_contain(
            "Failed to parse output from JSON function response"
        ));
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, "Hello, world!");
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_responses, vec![model_response]);
            }
            _ => unreachable!(),
        }

        // Test with a correct content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"name": "Jerry", "age": 30}"#.to_string().into()];
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponse::new(
            content_blocks.clone(),
            "content".to_string(),
            usage.clone(),
            latency,
        );
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                None,
            )
            .unwrap();
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(
                    result.output.parsed.unwrap(),
                    json!({"name": "Jerry", "age": 30}),
                );
                assert_eq!(result.output.raw, r#"{"name": "Jerry", "age": 30}"#);
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_responses, vec![model_response]);
            }
            _ => unreachable!(),
        }

        // Test with an incorrect JSON content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"name": "Jerry", "age": "thirty"}"#.to_string().into()];
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponse::new(
            content_blocks.clone(),
            "content".to_string(),
            usage.clone(),
            latency,
        );
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                None,
            )
            .unwrap();
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, r#"{"name": "Jerry", "age": "thirty"}"#);
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_responses, vec![model_response]);
            }
            _ => unreachable!(),
        }

        // Test with a tool content block with bad output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: "tool_call_arguments".to_string(),
        };
        let content_blocks = vec![ContentBlock::ToolCall(tool_call)];
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 10,
        };
        let model_response = ModelInferenceResponse::new(
            content_blocks.clone(),
            "content".to_string(),
            usage.clone(),
            Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
        );
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                None,
            )
            .unwrap();
        assert!(logs_contain("JSON Schema validation failed"));
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, "tool_call_arguments");
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_responses, vec![model_response]);
            }
            _ => unreachable!(),
        }

        // Test with a tool content block with good output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: r#"{"name": "Jerry", "age": 30}"#.to_string(),
        };
        let content_blocks = vec![ContentBlock::ToolCall(tool_call)];
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 10,
        };
        let model_response = ModelInferenceResponse::new(
            content_blocks.clone(),
            "content".to_string(),
            usage.clone(),
            Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
        );
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                None,
            )
            .unwrap();
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(
                    result.output.parsed.unwrap(),
                    json!({"name": "Jerry", "age": 30}),
                );
                assert_eq!(result.output.raw, r#"{"name": "Jerry", "age": 30}"#);
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_responses, vec![model_response]);
            }
            _ => unreachable!(),
        }

        // Test with no content blocks
        let inference_id = Uuid::now_v7();
        let content_blocks = Vec::new();
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 10,
        };
        let model_response = ModelInferenceResponse::new(
            content_blocks.clone(),
            "content".to_string(),
            usage.clone(),
            Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
        );
        let error = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                None,
            )
            .unwrap_err();
        assert_eq!(
            error,
            Error::Inference {
                message: "No valid content blocks found in JSON function response".to_string()
            }
        );
    }
}
