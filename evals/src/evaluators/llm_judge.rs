use std::collections::HashMap;

use anyhow::{bail, Result};
use serde_json::{json, Value};
use tensorzero::{
    ClientInferenceParams, DynamicToolParams, InferenceOutput, InferenceParams, InferenceResponse,
    Input, InputMessage, InputMessageContent, Role,
};
use tensorzero_internal::cache::CacheEnabledMode;
use tensorzero_internal::endpoints::datasets::Datapoint;
use tensorzero_internal::evals::{get_llm_judge_function_name, LLMJudgeConfig, LLMJudgeOutputType};
use tensorzero_internal::inference::types::{
    ContentBlockChatOutput, JsonInferenceOutput, ResolvedInput, ResolvedInputMessageContent,
    TextKind,
};
use uuid::Uuid;

use crate::ThrottledTensorZeroClient;

pub(super) async fn run_llm_judge_evaluator(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
    tensorzero_client: &ThrottledTensorZeroClient,
    llm_judge_config: &LLMJudgeConfig,
    eval_name: &str,
    evaluator_name: &str,
    eval_run_id: Uuid,
) -> Result<Option<Value>> {
    let resolved_input = datapoint.input();
    let serialized_datapoint_input = prepare_serialized_input(resolved_input)?;
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
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert LLM judge arguments to Map. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."))?
                    .clone()
            })],
        }],
    };

    let params = ClientInferenceParams {
        function_name: Some(get_llm_judge_function_name(eval_name, evaluator_name)),
        model_name: None,
        episode_id: None,
        input,
        stream: Some(false),
        include_original_response: false,
        params: InferenceParams::default(),
        variant_name: None,
        dryrun: Some(false),
        internal: true,
        tags: HashMap::from([(
            "tensorzero::eval_run_id".to_string(),
            eval_run_id.to_string(),
        )]),
        dynamic_tool_params: DynamicToolParams::default(),
        output_schema: None,
        credentials: HashMap::new(),
        cache_options: tensorzero::CacheParamsOptions {
            max_age_s: None,
            enabled: CacheEnabledMode::On,
        },
    };
    let result = tensorzero_client.inference(params).await?;
    let response = match result {
        InferenceOutput::NonStreaming(response) => response,
        InferenceOutput::Streaming(..) => {
            bail!("Streaming not supported for LLM judge evals. This is a bug, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.")
        }
    };
    let output = match response {
        InferenceResponse::Chat(..) => {
            bail!("Chat output not supported for LLM judge evals. This is a bug, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.")
        }
        InferenceResponse::Json(json_response) => json_response
            .output
            .parsed
            .ok_or_else(|| anyhow::anyhow!("JSON output does not contain a `parsed` field"))?,
    };
    match llm_judge_config.output_type {
        LLMJudgeOutputType::Float => Ok(Some(
            output
                .get("score")
                .ok_or_else(|| anyhow::anyhow!("JSON output does not contain a `score` field"))?
                .clone(),
        )),
        LLMJudgeOutputType::Boolean => Ok(Some(
            output
                .get("score")
                .ok_or_else(|| anyhow::anyhow!("JSON output does not contain a `score` field"))?
                .clone(),
        )),
    }
}

pub fn prepare_serialized_input(resolved_input: &ResolvedInput) -> Result<String> {
    for message in &resolved_input.messages {
        for content in &message.content {
            match content {
                ResolvedInputMessageContent::Image(..) => {
                    bail!("Image content not supported for LLM judge evals")
                }
                ResolvedInputMessageContent::Unknown { .. } => {
                    bail!("Unknown content not supported for LLM judge evals")
                }
                _ => {}
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
    use std::{path::PathBuf, sync::Arc};

    use super::*;

    use serde_json::json;
    use tensorzero::{ClientBuilder, ClientBuilderMode, Role};
    use tensorzero_internal::{
        endpoints::{
            datasets::{ChatInferenceDatapoint, JsonInferenceDatapoint},
            inference::{ChatInferenceResponse, JsonInferenceResponse},
        },
        evals::{LLMJudgeIncludeConfig, LLMJudgeOptimize},
        inference::types::{
            resolved_input::ImageWithPath,
            storage::{StorageKind, StoragePath},
            Base64Image, ContentBlockChatOutput, ImageKind, ResolvedInput, ResolvedInputMessage,
            ResolvedInputMessageContent, Text, Usage,
        },
        tool::{ToolCallOutput, ToolResult},
    };
    use tokio::sync::Semaphore;
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
            "Image content not supported for LLM judge evals"
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

    #[tokio::test(flavor = "multi_thread")]
    async fn test_run_llm_judge_evaluator_chat() {
        let tensorzero_client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(PathBuf::from(&format!(
                "{}/../tensorzero-internal/tests/e2e/tensorzero.toml",
                std::env::var("CARGO_MANIFEST_DIR").unwrap()
            ))),
            clickhouse_url: None,
            timeout: None,
        })
        .build()
        .await
        .unwrap();
        let tensorzero_client = Arc::new(ThrottledTensorZeroClient::new(
            tensorzero_client,
            Semaphore::new(1),
        ));
        let inference_response = InferenceResponse::Chat(ChatInferenceResponse {
            content: vec![ContentBlockChatOutput::Text(Text {
                text: "Hello, world!".to_string(),
            })],
            original_response: None,
            finish_reason: None,
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
            variant_name: "test_variant".to_string(),
        });
        let datapoint = Datapoint::ChatInference(ChatInferenceDatapoint {
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            auxiliary: "".to_string(),
            dataset_name: "test_dataset".to_string(),
            episode_id: Some(Uuid::now_v7()),
            id: Uuid::now_v7(),
            is_deleted: false,
            function_name: "test_function".to_string(),
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Hello, world!".to_string(),
            })]),
            tags: None,
            tool_params: None,
        });
        let llm_judge_config = LLMJudgeConfig {
            include: LLMJudgeIncludeConfig {
                reference_output: true,
            },
            optimize: LLMJudgeOptimize::Max,
            output_type: LLMJudgeOutputType::Boolean,
        };
        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "happy_bool",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, Some(json!(true)));

        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "sad_bool",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, Some(json!(false)));

        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "zero",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, Some(json!(0)));

        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "one",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, Some(json!(1)));

        // Try without output
        let datapoint = Datapoint::ChatInference(ChatInferenceDatapoint {
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            auxiliary: "".to_string(),
            dataset_name: "test_dataset".to_string(),
            episode_id: Some(Uuid::now_v7()),
            id: Uuid::now_v7(),
            is_deleted: false,
            function_name: "test_function".to_string(),
            output: None,
            tags: None,
            tool_params: None,
        });

        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "happy_bool",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_run_llm_judge_evaluator_json() {
        let tensorzero_client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(PathBuf::from(&format!(
                "{}/../tensorzero-internal/tests/e2e/tensorzero.toml",
                std::env::var("CARGO_MANIFEST_DIR").unwrap()
            ))),
            clickhouse_url: None,
            timeout: None,
        })
        .build()
        .await
        .unwrap();
        let tensorzero_client = Arc::new(ThrottledTensorZeroClient::new(
            tensorzero_client,
            Semaphore::new(1),
        ));
        let inference_response = InferenceResponse::Json(JsonInferenceResponse {
            output: JsonInferenceOutput {
                parsed: Some(json!({"answer": "LeBron James"})),
                raw: "{\"answer\": \"LeBron James\"}".to_string(),
            },
            original_response: None,
            finish_reason: None,
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
            variant_name: "test_variant".to_string(),
        });
        let datapoint = Datapoint::JsonInference(JsonInferenceDatapoint {
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            auxiliary: "".to_string(),
            dataset_name: "test_dataset".to_string(),
            episode_id: Some(Uuid::now_v7()),
            id: Uuid::now_v7(),
            is_deleted: false,
            function_name: "test_function".to_string(),
            output: Some(JsonInferenceOutput {
                parsed: Some(json!({"answer": "LeBron James"})),
                raw: "{\"answer\": \"LeBron James\"}".to_string(),
            }),
            output_schema: json!({"answer": "string"}),
            tags: None,
        });
        let llm_judge_config = LLMJudgeConfig {
            include: LLMJudgeIncludeConfig {
                reference_output: true,
            },
            optimize: LLMJudgeOptimize::Max,
            output_type: LLMJudgeOutputType::Boolean,
        };
        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "happy_bool",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, Some(json!(true)));

        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "sad_bool",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, Some(json!(false)));

        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "zero",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, Some(json!(0)));

        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "one",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, Some(json!(1)));

        // Try without output
        let datapoint = Datapoint::ChatInference(ChatInferenceDatapoint {
            input: ResolvedInput {
                system: None,
                messages: vec![ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("Hello, world!"),
                    }],
                }],
            },
            auxiliary: "".to_string(),
            dataset_name: "test_dataset".to_string(),
            episode_id: Some(Uuid::now_v7()),
            id: Uuid::now_v7(),
            is_deleted: false,
            function_name: "test_function".to_string(),
            output: None,
            tags: None,
            tool_params: None,
        });

        let result = run_llm_judge_evaluator(
            &inference_response,
            &datapoint,
            &tensorzero_client,
            &llm_judge_config,
            "test_eval",
            "happy_bool",
            Uuid::now_v7(),
        )
        .await
        .unwrap();
        assert_eq!(result, None);
    }
}
