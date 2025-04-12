use std::collections::HashMap;

use anyhow::{bail, Result};
use serde_json::{json, Value};
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    DynamicToolParams, Image, InferenceOutput, InferenceParams, InferenceResponse, Role,
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

use crate::helpers::get_cache_options;
use crate::ThrottledTensorZeroClient;

pub struct LLMJudgeEvaluationResult {
    pub inference_id: Uuid,
    pub value: Value,
}

pub struct RunLLMJudgeEvaluatorParams<'a> {
    pub inference_response: &'a InferenceResponse,
    pub datapoint: &'a Datapoint,
    pub tensorzero_client: &'a ThrottledTensorZeroClient,
    pub llm_judge_config: &'a LLMJudgeConfig,
    pub evaluation_name: &'a str,
    pub evaluator_name: &'a str,
    pub evaluation_run_id: Uuid,
    pub input: &'a ClientInput,
    pub inference_cache: CacheEnabledMode,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_llm_judge_evaluator(
    params: RunLLMJudgeEvaluatorParams<'_>,
) -> Result<Option<LLMJudgeEvaluationResult>> {
    let RunLLMJudgeEvaluatorParams {
        inference_response,
        datapoint,
        tensorzero_client,
        llm_judge_config,
        evaluation_name,
        evaluator_name,
        evaluation_run_id,
        input,
        inference_cache,
    } = params;
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
        cache_options: get_cache_options(inference_cache),
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

/// We prepare the input for the LLM judge evaluator.
/// This is heavily informed by the config for the evaluator.
/// There are two flags that matter here:
///  - include.reference_output: Whether we include the reference output in the input.
///  - input_format: Serialized or Messages.
///
/// If we are using the serialized format, we serialize the input and include it the generated and reference outputs in a
/// TextKind::Arguments block that should get templated into the first user message by the LLM Judge.
///
/// If we are using the messages format, we first convert the original system message to a user message by serializing it,
/// append all the other messages as is, and finally append the generated and reference outputs in a TextKind::Text block that we format here.
/// Since we don't want to use a template on the gateway side, we must format this here.
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
        // Here, we serialize the input and include it in the first user message as a TextKind::Arguments block
        // alongside the generated output and optionally the reference output.
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
        // Here, we convert the input to a list of messages and append the generated output and optionally the reference output
        // to the last user message.
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
                    bail!("Image content not supported for LLM judge evaluations with `serialized` input format. If you want image evaluations, try the `messages` input format.")
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
                // The image was already converted from a ResolvedImage to a Base64Image before this.
                if let Image::Url { .. } = image {
                    bail!("URL images not supported for LLM judge evaluations. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.")
                }
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
                // Since the LLM Judge does not have the template of the original function,
                // we instead serialize the arguments and send them as a TextKind::Text block.
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
                    // Same behavior as Arguments above.
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
    match (&output.raw, &output.parsed) {
        (_, None) => bail!("JSON output does not contain a `parsed` field"),
        (None, _) => bail!("JSON output does not contain a `raw` field"),
        (Some(raw), Some(_parsed)) => Ok(raw.clone()),
    }
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
    use tensorzero_internal::endpoints::datasets::ChatInferenceDatapoint;
    use tensorzero_internal::endpoints::datasets::JsonInferenceDatapoint;
    use tensorzero_internal::endpoints::inference::ChatInferenceResponse;
    use tensorzero_internal::endpoints::inference::JsonInferenceResponse;
    use tensorzero_internal::evaluations::LLMJudgeIncludeConfig;
    use tensorzero_internal::evaluations::LLMJudgeOptimize;
    use tensorzero_internal::inference::types::ResolvedInput;
    use tensorzero_internal::inference::types::Usage;
    use tensorzero_internal::tool::ToolCallInput;
    use tensorzero_internal::{
        inference::types::{ContentBlockChatOutput, Text, Thought},
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
            "Image content not supported for LLM judge evaluations with `serialized` input format. If you want image evaluations, try the `messages` input format."
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

    #[test]
    fn test_prepare_llm_judge_input() {
        let llm_judge_config = LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Float,
            cutoff: None,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig::default(),
        };
        let input = ClientInput {
            system: Some(json!("You are a helpful assistant")),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "bar".to_string(),
                })],
            }],
        };
        // Reference output disabled, serialized input format
        let input = prepare_llm_judge_input(
            &llm_judge_config,
            &input,
            &InferenceResponse::Chat(ChatInferenceResponse {
                content: vec![ContentBlockChatOutput::Text(Text {
                    text: "Hi world!".to_string(),
                })],
                inference_id: Uuid::now_v7(),
                variant_name: "foo".to_string(),
                usage: Usage::default(),
                original_response: None,
                finish_reason: None,
                episode_id: Uuid::now_v7(),
            }),
            &Datapoint::ChatInference(ChatInferenceDatapoint {
                dataset_name: "foo".to_string(),
                function_name: "foo".to_string(),
                id: Uuid::now_v7(),
                episode_id: Some(Uuid::now_v7()),
                input: ResolvedInput {
                    // This shouldn't get used
                    system: None,
                    messages: Vec::new(),
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "Hello, world!".to_string(),
                })]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                is_deleted: false,
                source_inference_id: None,
            }),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            input,
            ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Arguments {
                        arguments: json!({
                            "input": "{\"system\":\"You are a helpful assistant\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"bar\"}]}]}",
                            "generated_output": "[{\"type\":\"text\",\"text\":\"Hi world!\"}]",
                            "reference_output": null
                        })
                            .as_object()
                            .unwrap()
                            .clone(),
                    })],
                }],
            }
        );

        // Reference output enabled, serialized input format, no includes
        let llm_judge_config = LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Float,
            cutoff: None,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig {
                reference_output: true,
            },
        };
        let input = prepare_llm_judge_input(
            &llm_judge_config,
            &input,
            &InferenceResponse::Chat(ChatInferenceResponse {
                content: vec![ContentBlockChatOutput::Text(Text {
                    text: "Hi, world!".to_string(),
                })],
                inference_id: Uuid::now_v7(),
                variant_name: "foo".to_string(),
                usage: Usage::default(),
                original_response: None,
                finish_reason: None,
                episode_id: Uuid::now_v7(),
            }),
            &Datapoint::ChatInference(ChatInferenceDatapoint {
                dataset_name: "foo".to_string(),
                function_name: "foo".to_string(),
                id: Uuid::now_v7(),
                episode_id: Some(Uuid::now_v7()),
                input: ResolvedInput {
                    // This shouldn't get used
                    system: None,
                    messages: Vec::new(),
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "Hello, world!".to_string(),
                })]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                is_deleted: false,
                source_inference_id: None,
            }),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            input,
            ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Arguments {
                        arguments: json!({
                            "input": "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"arguments\":{\"input\":\"{\\\"system\\\":\\\"You are a helpful assistant\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"bar\\\"}]}]}\",\"generated_output\":\"[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Hi world!\\\"}]\",\"reference_output\":null}}]}]}",
                            "generated_output": "[{\"type\":\"text\",\"text\":\"Hi, world!\"}]",
                            "reference_output": "[{\"type\":\"text\",\"text\":\"Hello, world!\"}]"
                        })
                            .as_object()
                            .unwrap()
                            .clone(),
                    })],
                }],
            }
        );
    }

    #[test]
    fn test_prepare_messages_input() {
        // Test with simple string system message
        let input = ClientInput {
            system: Some(json!("You are a helpful assistant")),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Hello!".to_string(),
                })],
            }],
        };
        let messages = prepare_messages_input(&input).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].content.len(), 1);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) = &messages[0].content[0] {
            assert_eq!(text, "You are a helpful assistant");
        } else {
            panic!("Expected TextKind::Text");
        }
        assert_eq!(messages[1].role, Role::User);

        // Test with object system message
        let input = ClientInput {
            system: Some(json!({
                "instructions": "Be helpful",
                "persona": "assistant"
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Hello!".to_string(),
                })],
            }],
        };
        let messages = prepare_messages_input(&input).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) = &messages[0].content[0] {
            assert_eq!(
                text,
                r#"{"instructions":"Be helpful","persona":"assistant"}"#
            );
        } else {
            panic!("Expected TextKind::Text");
        }

        // Test with invalid system message
        let input = ClientInput {
            system: Some(json!([1, 2, 3])),
            messages: vec![],
        };
        let err = prepare_messages_input(&input).unwrap_err();
        assert_eq!(err.to_string(), "System message is not a string or object");
    }

    #[test]
    fn test_serialize_content_for_messages_input() {
        // Test with TextKind::Text
        let content = vec![ClientInputMessageContent::Text(TextKind::Text {
            text: "Hello, world!".to_string(),
        })];
        let serialized = serialize_content_for_messages_input(&content).unwrap();
        assert_eq!(serialized.len(), 1);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) = &serialized[0] {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected TextKind::Text");
        }

        // Test with TextKind::Arguments
        let content = vec![ClientInputMessageContent::Text(TextKind::Arguments {
            arguments: json!({"key": "value"}).as_object().unwrap().clone(),
        })];
        let serialized = serialize_content_for_messages_input(&content).unwrap();
        assert_eq!(serialized.len(), 1);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) = &serialized[0] {
            assert_eq!(text, r#"{"key":"value"}"#);
        } else {
            panic!("Expected TextKind::Text");
        }

        // Test with TextKind::LegacyValue (string)
        let content = vec![ClientInputMessageContent::Text(TextKind::LegacyValue {
            value: json!("legacy text"),
        })];
        let serialized = serialize_content_for_messages_input(&content).unwrap();
        assert_eq!(serialized.len(), 1);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) = &serialized[0] {
            assert_eq!(text, "legacy text");
        } else {
            panic!("Expected TextKind::Text");
        }

        // Test with TextKind::LegacyValue (object)
        let content = vec![ClientInputMessageContent::Text(TextKind::LegacyValue {
            value: json!({"legacy": "object"}),
        })];
        let serialized = serialize_content_for_messages_input(&content).unwrap();
        assert_eq!(serialized.len(), 1);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) = &serialized[0] {
            assert_eq!(text, r#"{"legacy":"object"}"#);
        } else {
            panic!("Expected TextKind::Text");
        }

        // Test with TextKind::LegacyValue (non-string, non-object)
        let content = vec![ClientInputMessageContent::Text(TextKind::LegacyValue {
            value: json!([1, 2, 3]),
        })];
        let err = serialize_content_for_messages_input(&content).unwrap_err();
        assert_eq!(err.to_string(), "Legacy value is not a string");

        // Test with ToolCall, ToolResult, etc. (should pass through)
        let content = vec![
            ClientInputMessageContent::ToolCall(ToolCallInput {
                name: Some("tool".to_string()),
                arguments: Some(json!({"arg": "value"})),
                id: "toolid".to_string(),
                raw_name: None,
                raw_arguments: None,
            }),
            ClientInputMessageContent::ToolResult(ToolResult {
                name: "tool".to_string(),
                result: "result".to_string(),
                id: "toolid".to_string(),
            }),
            ClientInputMessageContent::RawText {
                value: "raw text".to_string(),
            },
            ClientInputMessageContent::Thought(Thought {
                text: "thought".to_string(),
                signature: None,
            }),
        ];
        let serialized = serialize_content_for_messages_input(&content).unwrap();
        assert_eq!(serialized.len(), 4);

        // Test with Unknown content (should error)
        let content = vec![ClientInputMessageContent::Unknown {
            data: json!({"unknown": "data"}),
            model_provider_name: Some("provider".to_string()),
        }];
        let err = serialize_content_for_messages_input(&content).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unknown content not supported for LLM judge evaluations"
        );
    }

    #[test]
    fn test_prepare_serialized_json_output() {
        // Test with parsed field
        let output = JsonInferenceOutput {
            raw: Some(r#"{"key":"value"}"#.to_string()),
            parsed: Some(json!({"key":"value"})),
        };
        let serialized = prepare_serialized_json_output(&output).unwrap();
        assert_eq!(serialized, r#"{"key":"value"}"#);

        // Test without parsed field
        let output = JsonInferenceOutput {
            raw: Some(r#"{"key":"value"}"#.to_string()),
            parsed: None,
        };
        let err = prepare_serialized_json_output(&output).unwrap_err();
        assert_eq!(
            err.to_string(),
            "JSON output does not contain a `parsed` field"
        );
    }

    #[test]
    fn test_handle_reference_output() {
        // Test with reference output disabled
        let config = LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Float,
            cutoff: None,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig {
                reference_output: false,
            },
        };
        let datapoint = Datapoint::ChatInference(ChatInferenceDatapoint {
            dataset_name: "dataset".to_string(),
            function_name: "function".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: ResolvedInput {
                system: None,
                messages: Vec::new(),
            },
            output: None,
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            is_deleted: false,
            source_inference_id: None,
        });
        let result = handle_reference_output(&config, &datapoint).unwrap();
        assert_eq!(result, None);

        // Test with reference output enabled but missing in datapoint
        let config = LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Float,
            cutoff: None,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig {
                reference_output: true,
            },
        };
        let datapoint = Datapoint::ChatInference(ChatInferenceDatapoint {
            dataset_name: "dataset".to_string(),
            function_name: "function".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: ResolvedInput {
                system: None,
                messages: Vec::new(),
            },
            output: None,
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            is_deleted: false,
            source_inference_id: None,
        });
        let err = handle_reference_output(&config, &datapoint).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Datapoint does not contain an output when this is expected"
        );

        // Test with reference output enabled and present (chat)
        let datapoint = Datapoint::ChatInference(ChatInferenceDatapoint {
            dataset_name: "dataset".to_string(),
            function_name: "function".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: ResolvedInput {
                system: None,
                messages: Vec::new(),
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Reference text".to_string(),
            })]),
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            is_deleted: false,
            source_inference_id: None,
        });
        let result = handle_reference_output(&config, &datapoint)
            .unwrap()
            .unwrap();
        assert_eq!(result, r#"[{"type":"text","text":"Reference text"}]"#);

        // Test with reference output enabled and present (json)
        let datapoint = Datapoint::JsonInference(JsonInferenceDatapoint {
            dataset_name: "dataset".to_string(),
            function_name: "function".to_string(),
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: ResolvedInput {
                system: None,
                messages: Vec::new(),
            },
            output: Some(JsonInferenceOutput {
                raw: Some(r#"{"result":"json reference"}"#.to_string()),
                parsed: Some(json!({"result":"json reference"})),
            }),
            output_schema: json!({}),
            tags: None,
            auxiliary: String::new(),
            is_deleted: false,
            source_inference_id: None,
        });
        let result = handle_reference_output(&config, &datapoint)
            .unwrap()
            .unwrap();
        assert_eq!(result, r#"{"result":"json reference"}"#);
    }

    #[test]
    fn test_prepare_final_message_messages_input() {
        // Test with reference output required but not provided
        let config = LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Messages,
            output_type: LLMJudgeOutputType::Float,
            cutoff: None,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig {
                reference_output: true,
            },
        };
        let result = prepare_final_message_messages_input(&config, "Generated", None);
        assert_eq!(result, None);

        // Test with reference output required and provided
        let message =
            prepare_final_message_messages_input(&config, "Generated", Some("Reference")).unwrap();
        let expected = format!(
            include_str!("message_output_template_with_reference.txt"),
            "Generated", "Reference"
        );
        assert_eq!(message, expected);

        // Test without reference output
        let config = LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Messages,
            output_type: LLMJudgeOutputType::Float,
            cutoff: None,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig {
                reference_output: false,
            },
        };
        let message = prepare_final_message_messages_input(&config, "Generated", None).unwrap();
        let expected = format!(
            include_str!("message_output_template_without_reference.txt"),
            "Generated"
        );
        assert_eq!(message, expected);
    }

    #[test]
    fn test_prepare_llm_judge_input_messages_format() {
        // Test with Messages format, no reference output
        let llm_judge_config = LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Messages,
            output_type: LLMJudgeOutputType::Float,
            cutoff: None,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig {
                reference_output: false,
            },
        };
        let input = ClientInput {
            system: Some(json!("System instruction")),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "User message".to_string(),
                })],
            }],
        };
        let prepared_input = prepare_llm_judge_input(
            &llm_judge_config,
            &input,
            &InferenceResponse::Chat(ChatInferenceResponse {
                content: vec![ContentBlockChatOutput::Text(Text {
                    text: "Generated output".to_string(),
                })],
                inference_id: Uuid::now_v7(),
                variant_name: "model".to_string(),
                usage: Usage::default(),
                original_response: None,
                finish_reason: None,
                episode_id: Uuid::now_v7(),
            }),
            &Datapoint::ChatInference(ChatInferenceDatapoint {
                dataset_name: "dataset".to_string(),
                function_name: "function".to_string(),
                id: Uuid::now_v7(),
                episode_id: Some(Uuid::now_v7()),
                input: ResolvedInput {
                    system: None,
                    messages: Vec::new(),
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "Reference output".to_string(),
                })]),
                tool_params: None,
                tags: None,
                auxiliary: String::new(),
                is_deleted: false,
                source_inference_id: None,
            }),
        )
        .unwrap()
        .unwrap();

        // Check structure of prepared input
        assert_eq!(prepared_input.system, None);
        assert_eq!(prepared_input.messages.len(), 3); // System converted to user + original user + final message

        // First message should be system converted to user
        assert_eq!(prepared_input.messages[0].role, Role::User);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) =
            &prepared_input.messages[0].content[0]
        {
            assert_eq!(text, "System instruction");
        } else {
            panic!("Expected TextKind::Text");
        }

        // Second message should be original user message
        assert_eq!(prepared_input.messages[1].role, Role::User);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) =
            &prepared_input.messages[1].content[0]
        {
            assert_eq!(text, "User message");
        } else {
            panic!("Expected TextKind::Text");
        }

        // Third message should contain the generated output
        assert_eq!(prepared_input.messages[2].role, Role::User);
        if let ClientInputMessageContent::Text(TextKind::Text { text }) =
            &prepared_input.messages[2].content[0]
        {
            let expected = format!(
                include_str!("message_output_template_without_reference.txt"),
                "[{\"type\":\"text\",\"text\":\"Generated output\"}]"
            );
            assert_eq!(text, &expected);
        } else {
            panic!("Expected TextKind::Text");
        }
    }

    #[test]
    fn test_prepare_serialized_chat_output_error_cases() {
        // Test with only Unknown blocks
        let content = vec![ContentBlockChatOutput::Unknown {
            data: json!({"foo": "bar"}),
            model_provider_name: Some("provider".to_string()),
        }];
        let err = prepare_serialized_chat_output(&content).unwrap_err();
        assert_eq!(err.to_string(), "No valid content blocks to serialize");

        // Test with empty content
        let content = Vec::new();
        let err = prepare_serialized_chat_output(&content).unwrap_err();
        assert_eq!(err.to_string(), "No valid content blocks to serialize");
    }

    #[test]
    fn test_prepare_llm_judge_input_with_json_response() {
        // Test with JSON response
        let llm_judge_config = LLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Serialized,
            output_type: LLMJudgeOutputType::Float,
            cutoff: None,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig::default(),
        };
        let input = ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Query".to_string(),
                })],
            }],
        };
        let prepared_input = prepare_llm_judge_input(
            &llm_judge_config,
            &input,
            &InferenceResponse::Json(JsonInferenceResponse {
                output: JsonInferenceOutput {
                    raw: Some(r#"{"result":"json output"}"#.to_string()),
                    parsed: Some(json!({"result":"json output"})),
                },
                inference_id: Uuid::now_v7(),
                variant_name: "model".to_string(),
                usage: Usage::default(),
                original_response: None,
                finish_reason: None,
                episode_id: Uuid::now_v7(),
            }),
            &Datapoint::JsonInference(JsonInferenceDatapoint {
                dataset_name: "dataset".to_string(),
                function_name: "function".to_string(),
                id: Uuid::now_v7(),
                episode_id: Some(Uuid::now_v7()),
                input: ResolvedInput {
                    system: None,
                    messages: Vec::new(),
                },
                output: Some(JsonInferenceOutput {
                    raw: Some(r#"{"result":"reference output"}"#.to_string()),
                    parsed: Some(json!({"result":"reference output"})),
                }),
                output_schema: json!({}),
                tags: None,
                auxiliary: String::new(),
                is_deleted: false,
                source_inference_id: None,
            }),
        )
        .unwrap()
        .unwrap();

        // Check the prepared input
        assert_eq!(prepared_input.system, None);
        assert_eq!(prepared_input.messages.len(), 1);
        assert_eq!(prepared_input.messages[0].role, Role::User);
        if let ClientInputMessageContent::Text(TextKind::Arguments { arguments }) =
            &prepared_input.messages[0].content[0]
        {
            assert_eq!(
                arguments.get("input").and_then(|v| v.as_str()).unwrap(),
                r#"{"messages":[{"role":"user","content":[{"type":"text","text":"Query"}]}]}"#
            );
            assert_eq!(
                arguments
                    .get("generated_output")
                    .and_then(|v| v.as_str())
                    .unwrap(),
                r#"{"result":"json output"}"#
            );
            assert!(arguments.get("reference_output").unwrap().is_null());
        } else {
            panic!("Expected TextKind::Arguments");
        }
    }
}
