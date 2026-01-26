use std::collections::HashMap;

use anyhow::{Result, bail};
use serde_json::{Value, json};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::{
    ClientInferenceParams, DynamicToolParams, File, InferenceOutput, InferenceParams,
    InferenceResponse, Input, InputMessage, InputMessageContent, Role,
};
use tensorzero_core::db::evaluation_queries::EvaluationQueries;
use tensorzero_core::endpoints::datasets::Datapoint;
use tensorzero_core::evaluations::{
    LLMJudgeConfig, LLMJudgeInputFormat, LLMJudgeOutputType, get_evaluator_metric_name,
    get_llm_judge_function_name,
};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, JsonInferenceOutput, System, Template, Text,
};
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::Clients;
use crate::helpers::get_cache_options;
use crate::merge_tags;

#[derive(Debug)]
pub struct LLMJudgeEvaluationResult {
    pub evaluator_inference_id: Uuid,
    pub value: Value,
    pub human_feedback: bool,
}

impl LLMJudgeEvaluationResult {
    pub fn tags(&self) -> HashMap<String, String> {
        if self.human_feedback {
            HashMap::from([(
                "tensorzero::derived_from_human_feedback".to_string(),
                "true".to_string(),
            )])
        } else {
            HashMap::new()
        }
    }
}

pub struct RunLLMJudgeEvaluatorParams<'a> {
    pub inference_response: &'a InferenceResponse,
    pub datapoint: &'a Datapoint,
    pub clients: &'a Clients,
    pub llm_judge_config: &'a LLMJudgeConfig,
    pub evaluation_name: &'a str,
    pub evaluator_name: &'a str,
    pub evaluation_run_id: Uuid,
    pub input: &'a Input,
    pub inference_cache: CacheEnabledMode,
    pub external_tags: &'a HashMap<String, String>,
}

#[instrument(skip_all, fields(datapoint_id = %params.datapoint.id(), evaluator_name = %params.evaluator_name))]
pub async fn run_llm_judge_evaluator(
    params: RunLLMJudgeEvaluatorParams<'_>,
) -> Result<Option<LLMJudgeEvaluationResult>> {
    let RunLLMJudgeEvaluatorParams {
        inference_response,
        datapoint,
        clients,
        llm_judge_config,
        evaluation_name,
        evaluator_name,
        evaluation_run_id,
        input,
        inference_cache,
        external_tags,
    } = params;
    debug!("Checking for existing human feedback");
    let serialized_output = inference_response.get_serialized_output()?;
    if let Some(human_feedback) = clients
        .clickhouse_client
        .get_inference_evaluation_human_feedback(
            &get_evaluator_metric_name(evaluation_name, evaluator_name),
            &datapoint.id(),
            &serialized_output,
        )
        .await?
    {
        info!("Found existing human feedback, using that instead of LLM judge");
        return Ok(Some(LLMJudgeEvaluationResult {
            evaluator_inference_id: human_feedback.evaluator_inference_id,
            value: human_feedback.value,
            human_feedback: true,
        }));
    }
    debug!("Preparing LLM judge input");
    let judge_input =
        match prepare_llm_judge_input(llm_judge_config, input, inference_response, datapoint)? {
            Some(input) => {
                debug!("LLM judge input prepared successfully");
                input
            }
            None => {
                debug!("Cannot prepare LLM judge input, returning None");
                return Ok(None);
            }
        };

    debug!("Making LLM judge inference request");
    let internal_tags = HashMap::from([
        (
            "tensorzero::evaluation_run_id".to_string(),
            evaluation_run_id.to_string(),
        ),
        (
            "tensorzero::evaluation_name".to_string(),
            evaluation_name.to_string(),
        ),
    ]);
    let tags = merge_tags(external_tags, internal_tags)?;

    let params = ClientInferenceParams {
        function_name: Some(get_llm_judge_function_name(evaluation_name, evaluator_name)),
        model_name: None,
        episode_id: None,
        input: judge_input,
        stream: Some(false),
        include_original_response: false,
        include_raw_response: false,
        include_raw_usage: false,
        params: InferenceParams::default(),
        variant_name: None,
        dryrun: Some(false),
        internal: true,
        tags,
        dynamic_tool_params: DynamicToolParams::default(),
        output_schema: None,
        credentials: HashMap::new(),
        cache_options: get_cache_options(inference_cache),
        extra_body: Default::default(),
        extra_headers: Default::default(),
        internal_dynamic_variant_config: None,
        otlp_traces_extra_headers: HashMap::new(),
        otlp_traces_extra_attributes: HashMap::new(),
        otlp_traces_extra_resources: HashMap::new(),
        api_key: None,
    };
    let result = clients.inference_executor.inference(params).await?;
    let response = match result {
        InferenceOutput::NonStreaming(response) => response,
        InferenceOutput::Streaming(..) => {
            bail!(
                "Streaming not supported for LLM judge evaluations. This is a bug, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."
            )
        }
    };
    let evaluator_inference_id = response.inference_id();
    let output = match response {
        InferenceResponse::Chat(..) => {
            bail!(
                "Chat output not supported for LLM judge evaluations. This is a bug, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."
            )
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
            evaluator_inference_id,
            value,
            human_feedback: false,
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
    input: &Input,
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
) -> Result<Option<Input>> {
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
            Ok(Some(Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        #[expect(clippy::expect_used)]
                        arguments: Arguments(
                            json!({
                                "input": serialized_input,
                                "generated_output": generated_output,
                                "reference_output": reference_output,
                            })
                            .as_object()
                            .expect("Arguments should be an object")
                            .clone(),
                        ),
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
                    messages.push(InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text(Text {
                            text: final_message,
                        })],
                    });
                }
                None => return Ok(None),
            }
            Ok(Some(Input {
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

fn prepare_serialized_input(input: &Input) -> Result<String> {
    for message in &input.messages {
        for content in &message.content {
            match content {
                InputMessageContent::File { .. } => {
                    bail!(
                        "Image content not supported for LLM judge evaluations with `serialized` input format. If you want image evaluations, try the `messages` input format."
                    )
                }
                InputMessageContent::Unknown(_) => {
                    bail!("Unknown content not supported for LLM judge evaluations")
                }
                InputMessageContent::Text { .. }
                | InputMessageContent::Template { .. }
                | InputMessageContent::ToolCall { .. }
                | InputMessageContent::ToolResult { .. }
                | InputMessageContent::RawText(..)
                | InputMessageContent::Thought(..) => {}
            }
        }
    }
    Ok(serde_json::to_string(input)?)
}

fn prepare_messages_input(input: &Input) -> Result<Vec<InputMessage>> {
    let mut messages = Vec::new();
    if let Some(system) = &input.system {
        match system {
            System::Text(text) => {
                messages.push(InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text { text: text.clone() })],
                });
            }
            System::Template(arguments) => {
                let system_message = serde_json::to_string(arguments)?;
                messages.push(InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: system_message,
                    })],
                });
            }
        }
    }
    for message in &input.messages {
        let content = serialize_content_for_messages_input(&message.content)?;
        messages.push(InputMessage {
            role: message.role,
            content,
        });
    }
    Ok(messages)
}

fn serialize_content_for_messages_input(
    content: &Vec<InputMessageContent>,
) -> Result<Vec<InputMessageContent>> {
    let mut serialized_content = Vec::new();
    for content_block in content {
        match content_block {
            InputMessageContent::File(image) => {
                // The image was already converted from a ResolvedImage to a Base64Image before this.
                if let File::Url(..) = image {
                    bail!(
                        "URL images not supported for LLM judge evaluations. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."
                    )
                }
                serialized_content.push(InputMessageContent::File(image.clone()));
            }
            InputMessageContent::Unknown(_) => {
                bail!("Unknown content not supported for LLM judge evaluations")
            }
            InputMessageContent::ToolCall { .. }
            | InputMessageContent::ToolResult { .. }
            | InputMessageContent::RawText { .. }
            | InputMessageContent::Thought(_) => {
                serialized_content.push(content_block.clone());
            }
            InputMessageContent::Template(input) => {
                // Since the LLM Judge does not have the template of the original function,
                // we instead serialize the arguments and send them as a Text block.
                let arguments_string = serde_json::to_string(input)?;
                serialized_content.push(InputMessageContent::Text(Text {
                    text: arguments_string,
                }));
            }
            InputMessageContent::Text(text) => {
                serialized_content.push(InputMessageContent::Text(Text {
                    text: text.text.clone(),
                }));
            }
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
        Datapoint::Chat(chat_datapoint) => match &chat_datapoint.output {
            Some(output) => prepare_serialized_chat_output(output).map(Some),
            None => bail!("Datapoint does not contain an output when this is expected"),
        },
        Datapoint::Json(json_datapoint) => match &json_datapoint.output {
            Some(output) => prepare_serialized_json_output(output).map(Some),
            None => bail!("Datapoint does not contain an output when this is expected"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;
    use tensorzero_core::client::{File, Role, UrlFile};
    use tensorzero_core::endpoints::datasets::ChatInferenceDatapoint;
    use tensorzero_core::endpoints::datasets::JsonInferenceDatapoint;
    use tensorzero_core::endpoints::inference::ChatInferenceResponse;
    use tensorzero_core::endpoints::inference::JsonInferenceResponse;
    use tensorzero_core::evaluations::LLMJudgeIncludeConfig;
    use tensorzero_core::evaluations::LLMJudgeOptimize;
    use tensorzero_core::inference::types::Input;
    use tensorzero_core::inference::types::Usage;
    use tensorzero_core::tool::{ToolCall, ToolCallWrapper};
    use tensorzero_core::{
        inference::types::{ContentBlockChatOutput, RawText, Text, Thought, Unknown},
        tool::{InferenceResponseToolCall, ToolResult},
    };

    use url::Url;

    #[test]
    fn test_prepare_serialized_input() {
        // No system, just a text user message
        let input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
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
        let input = Input {
            system: Some(System::Text("You are a helpful assistant".to_string())),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![
                    InputMessageContent::Text(Text {
                        text: "Hello, world!".to_string(),
                    }),
                    InputMessageContent::ToolResult(ToolResult {
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
        let input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::File(File::Url(UrlFile {
                    url: Url::parse("https://example.com/image.png").unwrap(),
                    mime_type: None,
                    detail: None,
                    filename: None,
                }))],
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
            ContentBlockChatOutput::Unknown(Unknown {
                data: json!({"foo": "bar"}),
                model_name: Some("foo".to_string()),
                provider_name: Some("bar".to_string()),
            }),
        ];
        let serialized_output = prepare_serialized_chat_output(&content).unwrap();
        assert_eq!(
            serialized_output,
            r#"[{"type":"text","text":"Hello, world!"}]"#
        );
        // Tool call and text content blocks
        let content = vec![
            ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
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
            r#"[{"type":"tool_call","id":"foooo","raw_name":"tool","raw_arguments":"{\"foo\": \"bar\"}","name":"tool","arguments":{"foo":"bar"}},{"type":"text","text":"Hello, world!"}]"#
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
            description: None,
        };
        let input = Input {
            system: Some(System::Text("You are a helpful assistant".to_string())),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
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
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: None,
                episode_id: Uuid::now_v7(),
            }),
            &Datapoint::Chat(ChatInferenceDatapoint {
                dataset_name: "foo".to_string(),
                function_name: "foo".to_string(),
                name: None,
                id: Uuid::now_v7(),
                episode_id: Some(Uuid::now_v7()),
                input: Input {
                    // This shouldn't get used
                    system: None,
                    messages: Vec::new(),
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "Hello, world!".to_string(),
                })]),
                tool_params: Default::default(),
                tags: None,
                auxiliary: String::new(),
                is_deleted: false,
                source_inference_id: None,
                staled_at: None,
                updated_at: "2025-10-13T20:17:36Z".to_string(),
                is_custom: true,
            }),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            input,
            Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(serde_json::Map::from_iter([
                            ("input".to_string(), "{\"system\":\"You are a helpful assistant\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"bar\"}]}]}".into()),
                            ("generated_output".to_string(), "[{\"type\":\"text\",\"text\":\"Hi world!\"}]".into()),
                            ("reference_output".to_string(), Value::Null),
                        ])),
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
            description: None,
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
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: None,
                episode_id: Uuid::now_v7(),
            }),
            &Datapoint::Chat(ChatInferenceDatapoint {
                dataset_name: "foo".to_string(),
                function_name: "foo".to_string(),
                name: None,
                id: Uuid::now_v7(),
                episode_id: Some(Uuid::now_v7()),
                input: Input {
                    // This shouldn't get used
                    system: None,
                    messages: Vec::new(),
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "Hello, world!".to_string(),
                })]),
                tool_params: Default::default(),
                tags: None,
                auxiliary: String::new(),
                is_deleted: false,
                source_inference_id: None,
                staled_at: None,
                updated_at: "2025-10-13T20:17:36Z".to_string(),
                is_custom: true,
            }),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            input,
            Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(serde_json::Map::from_iter([
                            ("input".to_string(), "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"template\",\"name\":\"user\",\"arguments\":{\"input\":\"{\\\"system\\\":\\\"You are a helpful assistant\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"bar\\\"}]}]}\",\"generated_output\":\"[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Hi world!\\\"}]\",\"reference_output\":null}}]}]}".into()),
                            ("generated_output".to_string(), "[{\"type\":\"text\",\"text\":\"Hi, world!\"}]".into()),
                            ("reference_output".to_string(), "[{\"type\":\"text\",\"text\":\"Hello, world!\"}]".into()),
                        ])),
                    })],
                }],
            }
        );
    }

    #[test]
    fn test_prepare_messages_input() {
        // Test with simple string system message
        let input = Input {
            system: Some(System::Text("You are a helpful assistant".to_string())),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello!".to_string(),
                })],
            }],
        };
        let messages = prepare_messages_input(&input).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].content.len(), 1);
        if let InputMessageContent::Text(Text { text }) = &messages[0].content[0] {
            assert_eq!(text, "You are a helpful assistant");
        } else {
            panic!("Expected Text");
        }
        assert_eq!(messages[1].role, Role::User);

        // Test with object system message
        let input = Input {
            system: Some(System::Template(Arguments(serde_json::Map::from_iter([
                ("instructions".to_string(), "Be helpful".into()),
                ("persona".to_string(), "assistant".into()),
            ])))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello!".to_string(),
                })],
            }],
        };
        let messages = prepare_messages_input(&input).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        if let InputMessageContent::Text(Text { text }) = &messages[0].content[0] {
            assert_eq!(
                text,
                r#"{"instructions":"Be helpful","persona":"assistant"}"#
            );
        } else {
            panic!("Expected Text");
        }
    }

    #[test]
    fn test_serialize_content_for_messages_input() {
        // Test with Text
        let content = vec![InputMessageContent::Text(Text {
            text: "Hello, world!".to_string(),
        })];
        let serialized = serialize_content_for_messages_input(&content).unwrap();
        assert_eq!(serialized.len(), 1);
        if let InputMessageContent::Text(Text { text }) = &serialized[0] {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected Text");
        }

        // Test with Template
        let content = vec![InputMessageContent::Template(Template {
            name: "test".to_string(),
            arguments: Arguments(serde_json::Map::from_iter([(
                "key".to_string(),
                "value".into(),
            )])),
        })];
        let serialized = serialize_content_for_messages_input(&content).unwrap();
        assert_eq!(serialized.len(), 1);
        if let InputMessageContent::Text(Text { text }) = &serialized[0] {
            assert!(text.contains("test") && text.contains("key"));
        } else {
            panic!("Expected Text");
        }

        // Test with ToolCall, ToolResult, etc. (should pass through)
        let content = vec![
            InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(ToolCall {
                id: "toolid".to_string(),
                name: "tool".to_string(),
                arguments: json!({"arg": "value"}).to_string(),
            })),
            InputMessageContent::ToolResult(ToolResult {
                name: "tool".to_string(),
                result: "result".to_string(),
                id: "toolid".to_string(),
            }),
            InputMessageContent::RawText(RawText {
                value: "raw text".to_string(),
            }),
            InputMessageContent::Thought(Thought {
                text: Some("thought".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
                extra_data: None,
            }),
        ];
        let serialized = serialize_content_for_messages_input(&content).unwrap();
        assert_eq!(serialized.len(), 4);

        // Test with Unknown content (should error)
        let content = vec![InputMessageContent::Unknown(Unknown {
            data: json!({"unknown": "data"}),
            model_name: Some("model".to_string()),
            provider_name: Some("provider".to_string()),
        })];
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
            description: None,
        };
        let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
            dataset_name: "dataset".to_string(),
            function_name: "function".to_string(),
            name: None,
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: None,
                messages: Vec::new(),
            },
            output: None,
            tool_params: Default::default(),
            tags: None,
            auxiliary: String::new(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
            updated_at: "2025-10-13T20:17:36Z".to_string(),
            is_custom: true,
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
            description: None,
        };
        let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
            dataset_name: "dataset".to_string(),
            function_name: "function".to_string(),
            name: None,
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: None,
                messages: Vec::new(),
            },
            output: None,
            tool_params: Default::default(),
            tags: None,
            auxiliary: String::new(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
            updated_at: "2025-10-13T20:17:36Z".to_string(),
            is_custom: true,
        });
        let err = handle_reference_output(&config, &datapoint).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Datapoint does not contain an output when this is expected"
        );

        // Test with reference output enabled and present (chat)
        let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
            dataset_name: "dataset".to_string(),
            function_name: "function".to_string(),
            name: None,
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
                system: None,
                messages: Vec::new(),
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Reference text".to_string(),
            })]),
            tool_params: Default::default(),
            tags: None,
            auxiliary: String::new(),
            is_deleted: false,
            source_inference_id: None,
            staled_at: None,
            updated_at: "2025-10-13T20:17:36Z".to_string(),
            is_custom: true,
        });
        let result = handle_reference_output(&config, &datapoint)
            .unwrap()
            .unwrap();
        assert_eq!(result, r#"[{"type":"text","text":"Reference text"}]"#);

        // Test with reference output enabled and present (json)
        let datapoint = Datapoint::Json(JsonInferenceDatapoint {
            dataset_name: "dataset".to_string(),
            function_name: "function".to_string(),
            name: None,
            id: Uuid::now_v7(),
            episode_id: Some(Uuid::now_v7()),
            input: Input {
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
            staled_at: None,
            updated_at: "2025-10-13T20:17:36Z".to_string(),
            is_custom: true,
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
            description: None,
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
            description: None,
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
            description: None,
        };
        let input = Input {
            system: Some(System::Text("System instruction".to_string())),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
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
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: None,
                episode_id: Uuid::now_v7(),
            }),
            &Datapoint::Chat(ChatInferenceDatapoint {
                dataset_name: "dataset".to_string(),
                function_name: "function".to_string(),
                name: None,
                id: Uuid::now_v7(),
                episode_id: Some(Uuid::now_v7()),
                input: Input {
                    system: None,
                    messages: Vec::new(),
                },
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: "Reference output".to_string(),
                })]),
                tool_params: Default::default(),
                tags: None,
                auxiliary: String::new(),
                is_deleted: false,
                source_inference_id: None,
                staled_at: None,
                updated_at: "2025-10-13T20:17:36Z".to_string(),
                is_custom: true,
            }),
        )
        .unwrap()
        .unwrap();

        // Check structure of prepared input
        assert_eq!(prepared_input.system, None);
        assert_eq!(prepared_input.messages.len(), 3); // System converted to user + original user + final message

        // First message should be system converted to user
        assert_eq!(prepared_input.messages[0].role, Role::User);
        if let InputMessageContent::Text(Text { text }) = &prepared_input.messages[0].content[0] {
            assert_eq!(text, "System instruction");
        } else {
            panic!("Expected Text");
        }

        // Second message should be original user message
        assert_eq!(prepared_input.messages[1].role, Role::User);
        if let InputMessageContent::Text(Text { text }) = &prepared_input.messages[1].content[0] {
            assert_eq!(text, "User message");
        } else {
            panic!("Expected Text");
        }

        // Third message should contain the generated output
        assert_eq!(prepared_input.messages[2].role, Role::User);
        if let InputMessageContent::Text(Text { text }) = &prepared_input.messages[2].content[0] {
            let expected = format!(
                include_str!("message_output_template_without_reference.txt"),
                "[{\"type\":\"text\",\"text\":\"Generated output\"}]"
            );
            assert_eq!(text, &expected);
        } else {
            panic!("Expected Text");
        }
    }

    #[test]
    fn test_prepare_serialized_chat_output_error_cases() {
        // Test with only Unknown blocks
        let content = vec![ContentBlockChatOutput::Unknown(Unknown {
            data: json!({"foo": "bar"}),
            model_name: Some("model".to_string()),
            provider_name: Some("provider".to_string()),
        })];
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
            description: None,
        };
        let input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
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
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: None,
                episode_id: Uuid::now_v7(),
            }),
            &Datapoint::Json(JsonInferenceDatapoint {
                dataset_name: "dataset".to_string(),
                function_name: "function".to_string(),
                name: None,
                id: Uuid::now_v7(),
                episode_id: Some(Uuid::now_v7()),
                input: Input {
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
                staled_at: None,
                updated_at: "2025-10-13T20:17:36Z".to_string(),
                is_custom: true,
            }),
        )
        .unwrap()
        .unwrap();

        // Check the prepared input
        assert_eq!(prepared_input.system, None);
        assert_eq!(prepared_input.messages.len(), 1);
        assert_eq!(prepared_input.messages[0].role, Role::User);
        if let InputMessageContent::Template(Template { name, arguments }) =
            &prepared_input.messages[0].content[0]
        {
            assert_eq!(name, "user");
            assert_eq!(
                arguments.0.get("input").and_then(|v| v.as_str()).unwrap(),
                r#"{"messages":[{"role":"user","content":[{"type":"text","text":"Query"}]}]}"#
            );
            assert_eq!(
                arguments
                    .0
                    .get("generated_output")
                    .and_then(|v| v.as_str())
                    .unwrap(),
                r#"{"result":"json output"}"#
            );
            assert!(arguments.0.get("reference_output").unwrap().is_null());
        } else {
            panic!("Expected Template");
        }
    }
}
