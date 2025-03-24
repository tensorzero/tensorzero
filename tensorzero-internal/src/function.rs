use serde_json::Value;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::InferenceParams;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    ChatInferenceResult, ContentBlockOutput, InferenceResult, Input, InputMessageContent,
    JsonInferenceResult, ModelInferenceResponseWithMetadata, Role, TextKind, Usage,
};
use crate::jsonschema_util::{JSONSchemaFromPath, JsonSchemaRef};
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::tool::{DynamicToolParams, StaticToolConfig, ToolCallConfig, ToolChoice};
use crate::variant::{InferenceConfig, JsonMode, Variant, VariantConfig};

#[derive(Debug)]
pub enum FunctionConfig {
    Chat(FunctionConfigChat),
    Json(FunctionConfigJson),
}

#[derive(Copy, Clone, Debug)]
pub enum FunctionConfigType {
    Chat,
    Json,
}

impl FunctionConfig {
    pub fn config_type(&self) -> FunctionConfigType {
        match self {
            FunctionConfig::Chat(_) => FunctionConfigType::Chat,
            FunctionConfig::Json(_) => FunctionConfigType::Json,
        }
    }
}

#[derive(Debug, Default)]
pub struct FunctionConfigChat {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub tools: Vec<String>, // tool names
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Default)]
pub struct FunctionConfigJson {
    pub variants: HashMap<String, VariantConfig>, // variant name => variant config
    pub system_schema: Option<JSONSchemaFromPath>,
    pub user_schema: Option<JSONSchemaFromPath>,
    pub assistant_schema: Option<JSONSchemaFromPath>,
    pub output_schema: JSONSchemaFromPath, // schema is mandatory for JSON functions
    pub implicit_tool_call_config: ToolCallConfig,
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
    pub fn validate_inference_params(
        &self,
        params: &crate::endpoints::inference::Params,
    ) -> Result<(), Error> {
        if let FunctionConfig::Chat(_) = self {
            if let Some(JsonMode::ImplicitTool) = &params.params.chat_completion.json_mode {
                return Err(ErrorDetails::InvalidRequest {
                    message: "JSON mode `implicit_tool` is not supported for chat functions"
                        .to_string(),
                }
                .into());
            }
        }
        self.validate_input(&params.input)
    }
    /// Validate the input against the function's input schemas.
    /// The validation is done based on the function's type:
    /// - For a chat function, the input is validated against the system, user, and assistant schemas.
    /// - For a JSON function, the input is validated against the system, user, and assistant schemas.
    ///
    /// We do not validate ContentBlocks that are not text (tool calls and tool responses).
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

    /// Prepare the tool config for the function.
    /// For a Chat function, this will incorporate the tool information configured in the function as
    /// well as the dynamic tool calling information passed in `dynamic_tool_params`.
    /// JSON functions do not get tool_configs even if they end up using tools under the hood.
    pub fn prepare_tool_config(
        &self,
        dynamic_tool_params: DynamicToolParams,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<Option<ToolCallConfig>, Error> {
        match self {
            FunctionConfig::Chat(params) => Ok(ToolCallConfig::new(
                &params.tools,
                &params.tool_choice,
                params.parallel_tool_calls,
                static_tools,
                dynamic_tool_params,
            )?),
            FunctionConfig::Json(_) => {
                if dynamic_tool_params.allowed_tools.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `allowed_tools` to a JSON function.".to_string(),
                    }
                    .into());
                }
                if dynamic_tool_params.additional_tools.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `additional_tools` to a JSON function.".to_string(),
                    }
                    .into());
                }
                if dynamic_tool_params.tool_choice.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `tool_choice` to a JSON function".to_string(),
                    }
                    .into());
                }
                if dynamic_tool_params.parallel_tool_calls.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `parallel_tool_calls` to a JSON function".to_string(),
                    }
                    .into());
                }
                Ok(None)
            }
        }
    }

    #[instrument(skip_all, fields(inference_id))]
    #[allow(clippy::too_many_arguments)]
    pub async fn prepare_response<'a, 'request>(
        &self,
        inference_id: Uuid,
        content_blocks: Vec<ContentBlockOutput>,
        usage: Usage,
        model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
        inference_config: &'request InferenceConfig<'a, 'request>,
        inference_params: InferenceParams,
        original_response: Option<String>,
    ) -> Result<InferenceResult, Error> {
        match self {
            FunctionConfig::Chat(..) => Ok(InferenceResult::Chat(
                ChatInferenceResult::new(
                    inference_id,
                    content_blocks,
                    usage,
                    model_inference_results,
                    inference_config.tool_config,
                    inference_params,
                    original_response,
                )
                .await,
            )),
            FunctionConfig::Json(params) => {
                // Parse the content blocks into a JSON object
                // We assume here that the last content block that's text or a tool call is the JSON object.
                // (this is because we could have used an implicit tool call and there is no other reason for a tool call in a JSON function).
                let raw = content_blocks
                    .iter()
                    .rev()
                    .find_map(|content_block| match content_block {
                        ContentBlockOutput::Text(text) => Some(&text.text),
                        ContentBlockOutput::ToolCall(tool_call) => Some(&tool_call.arguments),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::Inference {
                            message: "No valid content blocks found in JSON function response"
                                .to_string(),
                        })
                    })?;
                let parsed_output = serde_json::from_str::<Value>(raw)
                    .map_err(|e| {
                        Error::new(ErrorDetails::OutputParsing {
                            message: format!(
                                "Failed to parse output from JSON function response {}",
                                e
                            ),
                            raw_output: raw.to_string(),
                        })
                    })
                    .ok();
                let output_schema = match &inference_config.dynamic_output_schema {
                    Some(schema) => JsonSchemaRef::Dynamic(schema),
                    None => JsonSchemaRef::Static(&params.output_schema),
                };

                // If the parsed output fails validation, we log the error and set `parsed_output` to None
                let parsed_output = match parsed_output {
                    Some(parsed_output) => match output_schema.validate(&parsed_output).await {
                        Ok(_) => Some(parsed_output),
                        Err(_) => None,
                    },
                    None => None,
                };
                Ok(InferenceResult::Json(JsonInferenceResult::new(
                    inference_id,
                    raw.to_string(),
                    parsed_output,
                    usage,
                    model_inference_results,
                    output_schema.value().clone(),
                    inference_params,
                    original_response,
                )))
            }
        }
    }

    pub fn system_schema(&self) -> Option<&JSONSchemaFromPath> {
        match self {
            FunctionConfig::Chat(params) => params.system_schema.as_ref(),
            FunctionConfig::Json(params) => params.system_schema.as_ref(),
        }
    }

    pub fn user_schema(&self) -> Option<&JSONSchemaFromPath> {
        match self {
            FunctionConfig::Chat(params) => params.user_schema.as_ref(),
            FunctionConfig::Json(params) => params.user_schema.as_ref(),
        }
    }

    pub fn assistant_schema(&self) -> Option<&JSONSchemaFromPath> {
        match self {
            FunctionConfig::Chat(params) => params.assistant_schema.as_ref(),
            FunctionConfig::Json(params) => params.assistant_schema.as_ref(),
        }
    }

    #[instrument(skip_all, fields(function_name = %function_name))]
    pub fn validate(
        &self,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
        models: &mut ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig,
        function_name: &str,
    ) -> Result<(), Error> {
        // Validate each variant
        for (variant_name, variant) in self.variants() {
            if variant_name.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "Variant name cannot start with 'tensorzero::': {variant_name}"
                    ),
                }
                .into());
            }
            variant.validate(
                self,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            )?;
        }
        match self {
            FunctionConfig::Chat(params) => {
                for tool in params.tools.iter() {
                    static_tools.get(tool).ok_or_else(|| Error::new(ErrorDetails::Config {
                        message: format!("`functions.{function_name}.tools`: tool `{tool}` is not present in the config"),
                    }))?;
                }
                Ok(())
            }
            FunctionConfig::Json(_) => Ok(()),
        }
    }
}

/// Validate all input messages that contain text (not raw_text).
/// The validation is done based on the input's role and the function's schemas.
/// We first validate the system message (if it exists)
/// Next we validate all messages containing text blocks.
/// If there are multiple text or raw text blocks in a message we reject.
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
        (None, Some(_)) => Err(Error::new(ErrorDetails::InvalidMessage {
            message: "`input.system` is empty but a system template is present.".to_string(),
        })),
    }?;
    for (index, message) in input.messages.iter().enumerate() {
        // Only for Text blocks, not RawText blocks since we don't validate those
        let mut content: Option<Cow<'_, Value>> = None;
        let mut text_seen = false;
        for block in message.content.iter() {
            match block {
                InputMessageContent::Text(kind) => {
                    // Throw an error if we have multiple text blocks in a message
                    if text_seen {
                        return Err(Error::new(ErrorDetails::InvalidMessage {
                            message: format!(
                                "Message at index {index} has multiple text content blocks"
                            ),
                        }));
                    }
                    content = Some(match kind {
                        TextKind::Arguments { arguments } => {
                            Cow::Owned(Value::Object(arguments.clone()))
                        }
                        TextKind::Text { text } => Cow::Owned(Value::String(text.clone())),
                        TextKind::LegacyValue { value } => Cow::Borrowed(value),
                    });
                    text_seen = true;
                }
                InputMessageContent::RawText { .. } => {
                    // Throw an error if we have multiple raw text blocks in a message
                    if text_seen {
                        return Err(Error::new(ErrorDetails::InvalidMessage {
                            message: format!(
                                "Message at index {index} has multiple text content blocks"
                            ),
                        }));
                    }
                    text_seen = true;
                }
                _ => {}
            }
        }
        if let Some(content) = content {
            match &message.role {
                Role::Assistant => validate_single_message(
                    &content,
                    assistant_schema,
                    Some((index, &message.role)),
                )?,
                Role::User => {
                    validate_single_message(&content, user_schema, Some((index, &message.role)))?
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
                    Some(index_role) => Error::new(ErrorDetails::InvalidMessage {
                        message: format!("Message at index {} has non-string content but there is no schema given for role {}.", index_role.0, index_role.1),
                    }),
                    None => Error::new(ErrorDetails::InvalidMessage {
                        message: "Message has non-string content but there is no schema given for role system.".to_string(),
                    }),
                })
            }
        }
    }
}

/// Sample a variant from the function based on variant weights (uniform random selection)
pub fn sample_variant<'a>(
    candidate_variant_names: &mut Vec<&'a str>,
    variants: &'a HashMap<String, VariantConfig>,
    function_name: &str,
    episode_id: &Uuid,
) -> Result<(&'a str, &'a VariantConfig), Error> {
    // Compute the total weight of variants present in variant_names
    let total_weight = candidate_variant_names
        .iter()
        .filter_map(|name| variants.get(*name))
        .map(|variant| variant.weight().unwrap_or(0.0))
        .sum::<f64>();

    // If the total weight is non-positive, perform uniform sampling
    // NOTE: We enforce non-negative weights at the config parsing stage,
    //       but there's a chance we pin a weight-zero variant in the config.
    //       This check also ensures that we catch any regressions we might introduce in the future.
    if total_weight <= 0. {
        if candidate_variant_names.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidFunctionVariants {
                message: format!("Function `{function_name}` has no variants"),
            }));
        }
        // Perform uniform sampling if total weight is non-positive
        let random_index = (get_uniform_value(function_name, episode_id)
            * candidate_variant_names.len() as f64)
            .floor() as usize;
        // Reorders this list (in place) by swapping the element at index with the last element.
        // This should not matter and is more efficient than `remove`
        let sampled_variant_name = if random_index < candidate_variant_names.len() {
            // could panic if random_index is out of bounds
            candidate_variant_names.swap_remove(random_index)
        } else {
            return Err(Error::new(ErrorDetails::InvalidFunctionVariants {
                message: format!(
                    "Invalid index {} for function `{}` with {} variants",
                    random_index,
                    function_name,
                    candidate_variant_names.len()
                ),
            }));
        };
        let variant = variants.get(sampled_variant_name).ok_or_else(|| {
            Error::new(ErrorDetails::InvalidFunctionVariants {
                message: format!(
                    "Function `{function_name}` has no variant `{sampled_variant_name}`"
                ),
            })
        })?;
        return Ok((sampled_variant_name, variant));
    }

    // Sample a random threshold between 0 and the total weight
    let random_threshold = get_uniform_value(function_name, episode_id) * total_weight;

    // Iterate over the variants to find the one that corresponds to the sampled threshold
    let mut cumulative_weight = 0.;
    let mut sampled_variant_name = "";
    for (i, variant_name) in candidate_variant_names.iter().enumerate() {
        let variant = variants.get(*variant_name).ok_or_else(|| {
            Error::new(ErrorDetails::InvalidFunctionVariants {
                message: format!("Function `{function_name}` has no variant `{variant_name}`"),
            })
        })?;
        cumulative_weight += variant.weight().unwrap_or(0.0);
        if cumulative_weight > random_threshold {
            sampled_variant_name = candidate_variant_names.swap_remove(i);
            break;
        }
    }

    // If we didn't find a variant (which should only happen due to rare numerical precision issues),
    // use the last variant as a fallback
    if sampled_variant_name.is_empty() {
        sampled_variant_name = candidate_variant_names.swap_remove(variants.len() - 1);
    }

    let variant = variants.get(sampled_variant_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{function_name}` has no variant `{sampled_variant_name}`"),
        })
    })?;
    Ok((sampled_variant_name, variant))
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
    use crate::endpoints::inference::InferenceIds;
    use crate::inference::types::FinishReason;
    use crate::inference::types::InputMessage;
    use crate::inference::types::Latency;
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::minijinja_util::TemplateConfig;
    use crate::tool::ToolCall;
    use crate::variant::chat_completion::ChatCompletionConfig;

    use super::*;
    use serde_json::json;
    use std::time::Duration;
    use std::time::Instant;
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
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "assistant name" })
                        .as_object()
                        .unwrap()
                        .clone(),
                })],
            },
        ];
        let input = Input {
            system: Some(json!("system name")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::new(ErrorDetails::InvalidMessage {
                message: "Message at index 1 has non-string content but there is no schema given for role assistant.".to_string(),
            })
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
            Error::new(ErrorDetails::InvalidMessage {
                message: "Message at index 0 has multiple text content blocks".to_string(),
            })
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
            Error::new(ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("system content")),
                schema: Box::new(system_value),
            })
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
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("user content")),
                schema: Box::new(user_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "user name" }).as_object().unwrap().clone(),
                })],
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
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("assistant content")),
                schema: Box::new(assistant_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "assistant name" })
                        .as_object()
                        .unwrap()
                        .clone(),
                })],
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
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText {
                    value: "raw text".to_string(),
                }],
            },
        ];

        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("system content")),
                schema: Box::new(system_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "user name" }).as_object().unwrap().clone(),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "assistant name" })
                        .as_object()
                        .unwrap()
                        .clone(),
                })],
            },
        ];

        let input = Input {
            system: Some(json!({ "name": "system name" })),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_raw_bypass_schemas() {
        let system_schema = create_test_schema();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
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
                content: vec![InputMessageContent::RawText {
                    value: "user content".to_string(),
                }],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::RawText {
                    value: "assistant content".to_string(),
                }],
            },
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText {
                    value: "raw text".to_string(),
                }],
            },
        ];

        let input = Input {
            system: Some(json!({ "name": "system name" })),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert!(validation_result.is_ok());
    }

    #[test]
    fn test_validate_input_chat_multiple_text_blocks() {
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
                content: vec![
                    "user content".to_string().into(),
                    "extra content".to_string().into(),
                ],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText {
                    value: "raw text".to_string(),
                }],
            },
        ];

        let input = Input {
            system: Some(Value::String("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::InvalidMessage {
                message: "Message at index 0 has multiple text content blocks".to_string(),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![
                    InputMessageContent::Text(TextKind::Arguments {
                        arguments: json!({ "name": "user name" }).as_object().unwrap().clone(),
                    }),
                    InputMessageContent::RawText {
                        value: "raw text".to_string(),
                    },
                ],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "assistant name" })
                        .as_object()
                        .unwrap()
                        .clone(),
                })],
            },
        ];

        let input = Input {
            system: Some(Value::String("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input).unwrap_err();
        assert_eq!(
            validation_result,
            ErrorDetails::InvalidMessage {
                message: "Message at index 0 has multiple text content blocks".to_string(),
            }
            .into()
        );
    }

    #[test]
    fn test_validate_input_json_no_schema() {
        let output_schema = json!({});
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&json!({})).unwrap(),
            implicit_tool_call_config,
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
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText {
                    value: "raw text".to_string(),
                }],
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
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "user name" }).as_object().unwrap().clone(),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "assistant name" })
                        .as_object()
                        .unwrap()
                        .clone(),
                })],
            },
        ];

        let input = Input {
            system: Some(json!("system content")),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::InvalidMessage {
                message: "Message at index 0 has non-string content but there is no schema given for role user.".to_string()
            }.into()
        );
    }

    #[test]
    fn test_validate_input_json_system_schema() {
        let system_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let output_schema = json!({});
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: Some(system_schema),
            user_schema: None,
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&output_schema).unwrap(),
            implicit_tool_call_config,
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
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("system content")),
                schema: Box::new(system_value),
            }
            .into()
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
        let output_schema = json!({});
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: Some(user_schema),
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&output_schema).unwrap(),
            implicit_tool_call_config,
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
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("user content")),
                schema: Box::new(user_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "user name" }).as_object().unwrap().clone(),
                })],
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
        let output_schema = json!({});
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: Some(assistant_schema),
            output_schema: JSONSchemaFromPath::from_value(&output_schema).unwrap(),
            implicit_tool_call_config,
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
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("assistant content")),
                schema: Box::new(assistant_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "assistant name" })
                        .as_object()
                        .unwrap()
                        .clone(),
                })],
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
        let output_schema = json!({});
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: Some(system_schema),
            user_schema: Some(user_schema),
            assistant_schema: Some(assistant_schema),
            output_schema: JSONSchemaFromPath::from_value(&output_schema).unwrap(),
            implicit_tool_call_config,
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
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("system content")),
                schema: Box::new(system_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "user name" }).as_object().unwrap().clone(),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({ "name": "assistant name" })
                        .as_object()
                        .unwrap()
                        .clone(),
                })],
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
                            weight: Some(weight),
                            model: "model-name".into(),
                            ..Default::default()
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
            let total_weight: f64 = variants.values().map(|v| v.weight().unwrap_or(0.0)).sum();
            let mut counts: HashMap<String, usize> = HashMap::new();

            for _ in 0..sample_size {
                let mut variant_names = variants.keys().map(AsRef::as_ref).collect();
                let (variant_name, _) = sample_variant(
                    &mut variant_names,
                    variants,
                    "test_function",
                    &Uuid::now_v7(),
                )
                .unwrap();
                *counts.entry(variant_name.to_string()).or_insert(0) += 1;
            }

            for (variant_name, variant) in variants {
                let expected_prob = variant.weight().unwrap_or(0.0) / total_weight;
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
            let mut variant_names = variants.keys().map(AsRef::as_ref).collect();
            let (variant_name, _) = sample_variant(
                &mut variant_names,
                &variants,
                "test_function",
                &Uuid::now_v7(),
            )
            .unwrap();
            *counts.entry(variant_name.to_string()).or_insert(0) += 1;
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

    #[tokio::test]
    #[traced_test]
    async fn test_prepare_response_json() {
        // The Chat stuff is tested in types::test_create_chat_inference_response
        // Here we focus on the JSON stuff
        let output_schema = json!({
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
        });
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = JSONSchemaFromPath::from_value(&output_schema).unwrap();
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema,
            implicit_tool_call_config,
        });
        let raw_request = "raw_request".to_string();

        // Test with a non-JSON content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec!["Hello, world!".to_string().into()];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::Stop),
            latency,
            cached: false,
        };
        let templates = TemplateConfig::default();
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            tool_config: None,
            function_name: "",
            variant_name: Some(""),
            templates: &templates,
            dynamic_output_schema: None,
            extra_body: Default::default(),
            extra_cache_key: None,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
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
                assert_eq!(result.finish_reason, Some(FinishReason::Stop));
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with a correct content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"name": "Jerry", "age": 30}"#.to_string().into()];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ToolCall),
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
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
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with an incorrect JSON content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"name": "Jerry", "age": "thirty"}"#.to_string().into()];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ToolCall),
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, r#"{"name": "Jerry", "age": "thirty"}"#);
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_results, vec![model_response]);
                assert_eq!(result.finish_reason, Some(FinishReason::ToolCall));
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with a tool content block with bad output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: "tool_call_arguments".to_string(),
        };
        let content_blocks = vec![ContentBlockOutput::ToolCall(tool_call)];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ToolCall),
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert!(logs_contain("JSON Schema validation failed"));
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, "tool_call_arguments");
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_results, vec![model_response]);
                assert_eq!(result.finish_reason, Some(FinishReason::ToolCall));
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with a tool content block with good output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: r#"{"name": "Jerry", "age": 30}"#.to_string(),
        };
        let content_blocks = vec![ContentBlockOutput::ToolCall(tool_call)];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ContentFilter),
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
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
                assert_eq!(result.model_inference_results, vec![model_response]);
                assert_eq!(result.finish_reason, Some(FinishReason::ContentFilter));
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with no content blocks
        let inference_id = Uuid::now_v7();
        let content_blocks = Vec::new();
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: None,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let error = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(
            error,
            ErrorDetails::Inference {
                message: "No valid content blocks found in JSON function response".to_string()
            }
            .into()
        );

        let dynamic_output_schema = DynamicJSONSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                }
            },
            "required": ["answer"]
        }));
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            tool_config: None,
            function_name: "",
            variant_name: Some(""),
            templates: &templates,
            dynamic_output_schema: Some(&dynamic_output_schema),
            extra_body: Default::default(),
            extra_cache_key: None,
        };
        // Test with a correct content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"answer": "42"}"#.to_string().into()];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::Stop),
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(result.output.parsed.unwrap(), json!({"answer": "42"}),);
                assert_eq!(result.output.raw, r#"{"answer": "42"}"#);
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with an incorrect JSON content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"response": "forty-two"}"#.to_string().into()];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: None,
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, r#"{"response": "forty-two"}"#);
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with a tool content block with bad output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: "tool_call_arguments".to_string(),
        };
        let content_blocks = vec![ContentBlockOutput::ToolCall(tool_call)];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ToolCall),
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert!(logs_contain("JSON Schema validation failed"));
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, "tool_call_arguments");
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with a tool content block with good output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: r#"{"answer": "42"}"#.to_string(),
        };
        let content_blocks = vec![ContentBlockOutput::ToolCall(tool_call)];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: None,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(result.output.parsed.unwrap(), json!({"answer": "42"}),);
                assert_eq!(result.output.raw, r#"{"answer": "42"}"#);
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            _ => panic!("Expected a JSON inference result"),
        }

        // Test with an empty output schema
        let output_schema = json!({});
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = JSONSchemaFromPath::from_value(&output_schema).unwrap();
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema,
            implicit_tool_call_config,
        });
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"answer": "42"}"#.to_string().into()];
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 10,
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: vec![],
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage: usage.clone(),
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::Stop),
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                usage.clone(),
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(result.output.parsed.unwrap(), json!({"answer": "42"}),);
                assert_eq!(result.output.raw, r#"{"answer": "42"}"#);
                assert_eq!(result.usage, usage);
                assert_eq!(result.model_inference_results, vec![model_response]);
                assert_eq!(result.finish_reason, Some(FinishReason::Stop));
            }
            _ => panic!("Expected a JSON inference result"),
        }
    }
}
