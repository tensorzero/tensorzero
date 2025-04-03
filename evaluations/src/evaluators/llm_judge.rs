use std::collections::HashMap;

use anyhow::{bail, Result};
use serde_json::{json, Value};
use tensorzero::{
    ClientInferenceParams, DynamicToolParams, InferenceOutput, InferenceParams, InferenceResponse,
    Input, InputMessage, InputMessageContent, Role,
};
use tensorzero_internal::cache::CacheEnabledMode;
use tensorzero_internal::endpoints::datasets::Datapoint;
use tensorzero_internal::evaluations::{
    get_llm_judge_function_name, LLMJudgeConfig, LLMJudgeOutputType,
};
use tensorzero_internal::inference::types::{
    ContentBlockChatOutput, JsonInferenceOutput, ResolvedInput, ResolvedInputMessageContent,
    TextKind,
};
use uuid::Uuid;

use crate::ThrottledTensorZeroClient;

pub struct LLMJudgeEvaluationResult {
    pub inference_id: Uuid,
    pub value: Value,
}

pub async fn run_llm_judge_evaluator(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
    tensorzero_client: &ThrottledTensorZeroClient,
    llm_judge_config: &LLMJudgeConfig,
    evaluation_name: &str,
    evaluator_name: &str,
    evaluation_run_id: Uuid,
) -> Result<Option<LLMJudgeEvaluationResult>> {
    let resolved_input = datapoint.input();
    let serialized_datapoint_input = prepare_serialized_input(resolved_input)?;
    // TODO (Viraj): add a flag to LLM judge config that is `input_format = "serialized" | "messages"`
    // optional, default is "serialized"
    // we only support images in "messages" format and error telling people that.
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
    let input = Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(TextKind::Arguments{
                arguments: json!({"input": serialized_datapoint_input, "generated_output": generated_output, "reference_output": reference_output})
                    .as_object()
                    .expect("Arguments should be an object")
                    .clone()
            })],
        }],
    };

    let params = ClientInferenceParams {
        function_name: Some(get_llm_judge_function_name(evaluation_name, evaluator_name)),
        model_name: None,
        episode_id: None,
        input,
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

pub fn prepare_serialized_input(resolved_input: &ResolvedInput) -> Result<String> {
    for message in &resolved_input.messages {
        for content in &message.content {
            match content {
                ResolvedInputMessageContent::Image(..) => {
                    bail!("Image content not supported for LLM judge evaluations")
                }
                ResolvedInputMessageContent::Unknown { .. } => {
                    bail!("Unknown content not supported for LLM judge evaluations")
                }
                ResolvedInputMessageContent::Text { .. }
                | ResolvedInputMessageContent::ToolCall { .. }
                | ResolvedInputMessageContent::ToolResult { .. }
                | ResolvedInputMessageContent::RawText { .. }
                | ResolvedInputMessageContent::Thought(_) => {}
            }
        }
    }
    Ok(serde_json::to_string(resolved_input)?)
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
    use tensorzero::Role;
    use tensorzero_internal::{
        inference::types::{
            resolved_input::ImageWithPath,
            storage::{StorageKind, StoragePath},
            Base64Image, ContentBlockChatOutput, ImageKind, ResolvedInput, ResolvedInputMessage,
            ResolvedInputMessageContent, Text,
        },
        tool::{ToolCallOutput, ToolResult},
    };

    use url::Url;

    #[test]
    fn test_prepare_serialized_input() {
        // No system, just a text user message
        let resolved_input = ResolvedInput {
            system: None,
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!("Hello, world!"),
                }],
            }],
        };
        let serialized_input = prepare_serialized_input(&resolved_input).unwrap();
        assert_eq!(
            serialized_input,
            r#"{"messages":[{"role":"user","content":[{"type":"text","value":"Hello, world!"}]}]}"#
        );

        // System message, user message with a text and tool block
        let resolved_input = ResolvedInput {
            system: Some(json!("You are a helpful assistant")),
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![
                    ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    },
                    ResolvedInputMessageContent::ToolResult(ToolResult {
                        name: "tool".to_string(),
                        result: "it's 24 degrees and cloudy in SF".to_string(),
                        id: "foooo".to_string(),
                    }),
                ],
            }],
        };
        let serialized_input = prepare_serialized_input(&resolved_input).unwrap();
        assert_eq!(
            serialized_input,
            r#"{"system":"You are a helpful assistant","messages":[{"role":"user","content":[{"type":"text","value":"Hello, world!"},{"type":"tool_result","name":"tool","result":"it's 24 degrees and cloudy in SF","id":"foooo"}]}]}"#
        );
        // Input contains an image
        let resolved_input = ResolvedInput {
            system: None,
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Image(ImageWithPath {
                    image: Base64Image {
                        url: Some(Url::parse("https://example.com/image.png").unwrap()),
                        data: None,
                        mime_type: ImageKind::Png,
                    },
                    storage_path: StoragePath {
                        kind: StorageKind::Filesystem {
                            path: "/tmp/image.png".to_string(),
                        },
                        path: "foo".to_string().into(),
                    },
                })],
            }],
        };
        let error = prepare_serialized_input(&resolved_input).unwrap_err();
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
