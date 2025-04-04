use std::collections::HashMap;

use anyhow::{bail, Result};
use serde_json::{json, Value};
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    DynamicToolParams, InferenceOutput, InferenceParams, InferenceResponse, Role,
};
use tensorzero_internal::cache::CacheEnabledMode;
use tensorzero_internal::endpoints::datasets::Datapoint;
use tensorzero_internal::evaluations::{
    get_llm_judge_function_name, LLMJudgeConfig, LLMJudgeInputFormat, LLMJudgeOutputType,
};
use tensorzero_internal::inference::types::{
    ContentBlockChatOutput, JsonInferenceOutput, TextKind,
};
use uuid::Uuid;

use crate::ThrottledTensorZeroClient;

pub struct LLMJudgeEvaluationResult {
    pub inference_id: Uuid,
    pub value: Value,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_llm_judge_evaluator(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
    tensorzero_client: &ThrottledTensorZeroClient,
    llm_judge_config: &LLMJudgeConfig,
    evaluation_name: &str,
    evaluator_name: &str,
    evaluation_run_id: Uuid,
    input: &ClientInput,
) -> Result<Option<LLMJudgeEvaluationResult>> {
    let judge_input =
        match prepare_llm_judge_input(llm_judge_config, input, inference_response, datapoint)? {
            Some(input) => input,
            None => return Ok(None),
        };

    let params = ClientInferenceParams {
        function_name: Some(get_llm_judge_function_name(evaluation_name, evaluator_name)),
        model_name: None,
        episode_id: None,
        input: judge_input,
        stream: Some(false),
        include_original_response: false,
        params: InferenceParams::default(),
        variant_name: None,
        dryrun: Some(false),
        internal: true,
        tags: HashMap::from([
            (
                "tensorzero::evaluation_run_id".to_string(),
                evaluation_run_id.to_string(),
            ),
            (
                "tensorzero::evaluation_name".to_string(),
                evaluation_name.to_string(),
            ),
        ]),
        dynamic_tool_params: DynamicToolParams::default(),
        output_schema: None,
        credentials: HashMap::new(),
        cache_options: tensorzero::CacheParamsOptions {
            max_age_s: None,
            enabled: CacheEnabledMode::On,
        },
        extra_body: Default::default(),
    };
    let result = tensorzero_client.inference(params).await?;
    let response = match result {
        InferenceOutput::NonStreaming(response) => response,
        InferenceOutput::Streaming(..) => {
            bail!("Streaming not supported for LLM judge evaluations. This is a bug, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.")
        }
    };
    let inference_id = response.inference_id();
    let output = match response {
        InferenceResponse::Chat(..) => {
            bail!("Chat output not supported for LLM judge evaluations. This is a bug, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.")
        }
        InferenceResponse::Json(json_response) => json_response
            .output
            .parsed
            .ok_or_else(|| anyhow::anyhow!("JSON output does not contain a `parsed` field"))?,
    };
    let value = match llm_judge_config.output_type {
        LLMJudgeOutputType::Float | LLMJudgeOutputType::Boolean => Some(
            output
                .get("score")
                .ok_or_else(|| anyhow::anyhow!("JSON output does not contain a `score` field"))?
                .clone(),
        ),
    };
    match value {
        Some(value) => Ok(Some(LLMJudgeEvaluationResult {
            inference_id,
            value,
        })),
        None => Ok(None),
    }
}

fn prepare_llm_judge_input(
    llm_judge_config: &LLMJudgeConfig,
    input: &ClientInput,
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
) -> Result<Option<ClientInput>> {
    let generated_output = match &inference_response {
        InferenceResponse::Chat(chat_response) => {
            prepare_serialized_chat_output(&chat_response.content)?
        }
        InferenceResponse::Json(json_response) => {
            prepare_serialized_json_output(&json_response.output)?
        }
    };
    let reference_output = match handle_reference_output(llm_judge_config, datapoint) {
        Ok(reference_output) => reference_output,
        // Reference output is optional so if it's needed but not present, we can just return None
        Err(_e) => return Ok(None),
    };
    match &llm_judge_config.input_format {
        LLMJudgeInputFormat::Serialized => {
            let serialized_input = prepare_serialized_input(input)?;
            Ok(Some(ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Arguments{
                        arguments: json!({"input": serialized_input, "generated_output": generated_output, "reference_output": reference_output})
                            .as_object()
                            .expect("Arguments should be an object")
                            .clone()
                    })],
                }],
            }))
        }
        LLMJudgeInputFormat::Messages => {
            let mut messages = prepare_messages_input(input)?;
            let final_message = prepare_final_message_messages_input(
                llm_judge_config,
                &generated_output,
                reference_output.as_deref(),
            );
            match final_message {
                Some(final_message) => {
                    messages.push(ClientInputMessage {
                        role: Role::User,
                        content: vec![ClientInputMessageContent::Text(TextKind::Text {
                            text: final_message,
                        })],
                    });
                }
                None => return Ok(None),
            }
            Ok(Some(ClientInput {
                system: None,
                messages,
            }))
        }
    }
}

fn prepare_final_message_messages_input(
    llm_judge_config: &LLMJudgeConfig,
    generated_output: &str,
    reference_output: Option<&str>,
) -> Option<String> {
    if llm_judge_config.include.reference_output {
        let reference_output = reference_output?;
        // Includes hardcoded placeholders for generated_output and reference_output
        Some(format!(
            include_str!("message_output_template_with_reference.txt"),
            generated_output, reference_output
        ))
    } else {
        Some(format!(
            include_str!("message_output_template_without_reference.txt"),
            generated_output
        ))
    }
}

fn prepare_serialized_input(input: &ClientInput) -> Result<String> {
    for message in &input.messages {
        for content in &message.content {
            match content {
                ClientInputMessageContent::Image(..) => {
                    bail!("Image content not supported for LLM judge evaluations")
                }
                ClientInputMessageContent::Unknown { .. } => {
                    bail!("Unknown content not supported for LLM judge evaluations")
                }
                ClientInputMessageContent::Text { .. }
                | ClientInputMessageContent::ToolCall { .. }
                | ClientInputMessageContent::ToolResult { .. }
                | ClientInputMessageContent::RawText { .. }
                | ClientInputMessageContent::Thought(_) => {}
            }
        }
    }
    Ok(serde_json::to_string(input)?)
}

fn prepare_messages_input(input: &ClientInput) -> Result<Vec<ClientInputMessage>> {
    let mut messages = Vec::new();
    if let Some(system) = &input.system {
        match system {
            Value::String(system) => {
                messages.push(ClientInputMessage {
                    role: Role::User,
                    // TODO (Viraj): decide how we should serialize system messages
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: system.clone(),
                    })],
                });
            }
            Value::Object(system) => {
                let system_message = serde_json::to_string(system)?;
                messages.push(ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: system_message,
                    })],
                })
            }
            _ => {
                bail!("System message is not a string or object");
            }
        }
    }
    for message in &input.messages {
        let content = serialize_content_for_messages_input(&message.content)?;
        messages.push(ClientInputMessage {
            role: message.role,
            content,
        });
    }
    Ok(messages)
}

fn serialize_content_for_messages_input(
    content: &Vec<ClientInputMessageContent>,
) -> Result<Vec<ClientInputMessageContent>> {
    let mut serialized_content = Vec::new();
    for content_block in content {
        match content_block {
            ClientInputMessageContent::Image(image) => {
                serialized_content.push(ClientInputMessageContent::Image(image.clone()));
            }
            ClientInputMessageContent::Unknown { .. } => {
                bail!("Unknown content not supported for LLM judge evaluations")
            }
            ClientInputMessageContent::ToolCall { .. }
            | ClientInputMessageContent::ToolResult { .. }
            | ClientInputMessageContent::RawText { .. }
            | ClientInputMessageContent::Thought(_) => {
                serialized_content.push(content_block.clone());
            }
            ClientInputMessageContent::Text(text) => match text {
                TextKind::Text { text } => {
                    serialized_content.push(ClientInputMessageContent::Text(TextKind::Text {
                        text: text.clone(),
                    }));
                }
                TextKind::Arguments { arguments } => {
                    let arguments_string = serde_json::to_string(arguments)?;
                    serialized_content.push(ClientInputMessageContent::Text(TextKind::Text {
                        text: arguments_string,
                    }));
                }
                TextKind::LegacyValue { value } => match value {
                    Value::String(string) => {
                        serialized_content.push(ClientInputMessageContent::Text(TextKind::Text {
                            text: string.clone(),
                        }));
                    }
                    Value::Object(object) => {
                        let object_string = serde_json::to_string(object)?;
                        serialized_content.push(ClientInputMessageContent::Text(TextKind::Text {
                            text: object_string,
                        }));
                    }
                    _ => bail!("Legacy value is not a string"),
                },
            },
        }
    }
    Ok(serialized_content)
}

/// We prepare the serialized output by converting the content blocks to a string.
/// The only reason this doesn't directly use serde_json::to_string is because we want to
/// strip out the Unknown content blocks, which we don't want to include in the output.
fn prepare_serialized_chat_output(content: &Vec<ContentBlockChatOutput>) -> Result<String> {
    let mut blocks_to_serialized = Vec::new();
    for block in content {
        if let ContentBlockChatOutput::Unknown { .. } = block {
            continue;
        }
        blocks_to_serialized.push(block);
    }
    if blocks_to_serialized.is_empty() {
        bail!("No valid content blocks to serialize");
    }
    Ok(serde_json::to_string(&blocks_to_serialized)?)
}

fn prepare_serialized_json_output(output: &JsonInferenceOutput) -> Result<String> {
    if output.parsed.is_none() {
        bail!("JSON output does not contain a `parsed` field");
    }
    Ok(output.raw.clone())
}

/// Handles the reference output for the LLM judge evaluator.
/// If the reference output is not needed, we return None.
/// If the reference output is needed, we return the serialized output of the datapoint.
/// If the reference output is needed but not present, we throw an error. (this could be mapped to None above this call)
fn handle_reference_output(
    llm_judge_config: &LLMJudgeConfig,
    datapoint: &Datapoint,
) -> Result<Option<String>> {
    if !llm_judge_config.include.reference_output {
        return Ok(None);
    }
    match datapoint {
        Datapoint::ChatInference(chat_datapoint) => match &chat_datapoint.output {
            Some(output) => prepare_serialized_chat_output(output).map(Some),
            None => bail!("Datapoint does not contain an output when this is expected"),
        },
        Datapoint::JsonInference(json_datapoint) => match &json_datapoint.output {
            Some(output) => prepare_serialized_json_output(output).map(Some),
            None => bail!("Datapoint does not contain an output when this is expected"),
        },
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use serde_json::json;
    use tensorzero::Image;
    use tensorzero::Role;
    use tensorzero_internal::{
        inference::types::{ContentBlockChatOutput, Text},
        tool::{ToolCallOutput, ToolResult},
    };

    use url::Url;

    #[test]
    fn test_prepare_serialized_input() {
        // No system, just a text user message
        let input = ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        };
        let serialized_input = prepare_serialized_input(&input).unwrap();
        assert_eq!(
            serialized_input,
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"Hello, world!"}]}]}"#
        );

        // System message, user message with a text and tool block
        let input = ClientInput {
            system: Some(json!("You are a helpful assistant")),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![
                    ClientInputMessageContent::Text(TextKind::Text {
                        text: "Hello, world!".to_string(),
                    }),
                    ClientInputMessageContent::ToolResult(ToolResult {
                        name: "tool".to_string(),
                        result: "it's 24 degrees and cloudy in SF".to_string(),
                        id: "foooo".to_string(),
                    }),
                ],
            }],
        };
        let serialized_input = prepare_serialized_input(&input).unwrap();
        assert_eq!(
            serialized_input,
            r#"{"system":"You are a helpful assistant","messages":[{"role":"user","content":[{"type":"text","text":"Hello, world!"},{"type":"tool_result","name":"tool","result":"it's 24 degrees and cloudy in SF","id":"foooo"}]}]}"#
        );
        // Input contains an image
        let input = ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Image(Image::Url {
                    url: Url::parse("https://example.com/image.png").unwrap(),
                })],
            }],
        };
        let error = prepare_serialized_input(&input).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Image content not supported for LLM judge evaluations"
        );
    }

    #[test]
    fn test_prepare_serialized_chat_output() {
        let content = vec![ContentBlockChatOutput::Text(Text {
            text: "Hello, world!".to_string(),
        })];
        let serialized_output = prepare_serialized_chat_output(&content).unwrap();
        assert_eq!(
            serialized_output,
            r#"[{"type":"text","text":"Hello, world!"}]"#
        );
        // Text and Unknown content blocks
        let content = vec![
            ContentBlockChatOutput::Text(Text {
                text: "Hello, world!".to_string(),
            }),
            ContentBlockChatOutput::Unknown {
                data: json!({"foo": "bar"}),
                model_provider_name: Some("foo".to_string()),
            },
        ];
        let serialized_output = prepare_serialized_chat_output(&content).unwrap();
        assert_eq!(
            serialized_output,
            r#"[{"type":"text","text":"Hello, world!"}]"#
        );
        // Tool call and text content blocks
        let content = vec![
            ContentBlockChatOutput::ToolCall(ToolCallOutput {
                name: Some("tool".to_string()),
                arguments: Some(json!({"foo": "bar"})),
                id: "foooo".to_string(),
                raw_name: "tool".to_string(),
                raw_arguments: r#"{"foo": "bar"}"#.to_string(),
            }),
            ContentBlockChatOutput::Text(Text {
                text: "Hello, world!".to_string(),
            }),
        ];
        let serialized_output = prepare_serialized_chat_output(&content).unwrap();
        assert_eq!(
            serialized_output,
            r#"[{"type":"tool_call","arguments":{"foo":"bar"},"id":"foooo","name":"tool","raw_arguments":"{\"foo\": \"bar\"}","raw_name":"tool"},{"type":"text","text":"Hello, world!"}]"#
        );
    }
}
