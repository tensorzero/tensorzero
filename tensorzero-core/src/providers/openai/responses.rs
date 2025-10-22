use std::borrow::Cow;
use std::time::Duration;

use crate::inference::types::{ProviderInferenceResponseStreamInner, ThoughtSummaryBlock};
use crate::providers::helpers::convert_stream_error;
use crate::{error::IMPOSSIBLE_ERROR_MESSAGE, inference::TensorZeroEventError};
use futures::StreamExt;
use futures::{future::try_join_all, Stream};
use reqwest_eventsource::Event;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use crate::{
    error::{warn_discarded_thought_block, Error, ErrorDetails},
    inference::types::{
        ContentBlock, ContentBlockChunk, ContentBlockOutput, FinishReason, FlattenUnknown, Latency,
        ModelInferenceRequest, ModelInferenceRequestJsonMode, ProviderInferenceResponse,
        ProviderInferenceResponseArgs, ProviderInferenceResponseChunk, RequestMessage, Role, Text,
        TextChunk, Thought, ThoughtChunk, Usage,
    },
    model::fully_qualified_name,
    providers::openai::{
        prepare_file_message, prepare_system_or_developer_message_helper, OpenAIContentBlock,
        OpenAIFile, OpenAIMessagesConfig, OpenAITool, OpenAIToolType, SystemOrDeveloper,
        PROVIDER_TYPE,
    },
    tool::{ToolCall, ToolCallChunk, ToolChoice},
};

#[derive(Serialize, Debug)]
pub struct OpenAIResponsesRequest<'a> {
    model: &'a str,
    input: Vec<OpenAIResponsesInput<'a>>,
    text: OpenAIResponsesTextConfig,
    tools: Vec<OpenAIResponsesTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIResponsesToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<OpenAIResponsesInclude>>,
    stream: bool,
}

#[derive(Serialize, Debug, Clone)]
pub enum OpenAIResponsesInclude {
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

#[derive(Serialize, Debug, Clone)]
pub struct OpenAIResponsesTextConfig {
    format: OpenAIResponsesTextConfigFormat,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum OpenAIResponsesTextConfigFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

#[derive(Deserialize, Debug, Clone)]
pub(super) struct OpenAIResponsesResponse<'a> {
    #[serde(borrow)]
    pub(super) output: Vec<OpenAIResponsesOutput<'a>>,
    pub(super) usage: OpenAIResponsesUsage,
    pub incomplete_details: Option<OpenAIResponsesIncompleteDetails>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIResponsesIncompleteDetails {
    reason: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl From<OpenAIResponsesUsage> for Usage {
    fn from(usage: OpenAIResponsesUsage) -> Self {
        Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        }
    }
}

impl OpenAIResponsesResponse<'_> {
    pub fn into_provider_response(
        self,
        latency: Latency,
        raw_request: String,
        raw_response: String,
        generic_request: &ModelInferenceRequest<'_>,
        model_name: &str,
        provider_name: &str,
    ) -> Result<ProviderInferenceResponse, Error> {
        let mut output = Vec::new();
        for message in self.output {
            match message {
                FlattenUnknown::Normal(OpenAIResponsesOutputInner::Message(message)) => {
                    if message.role != "assistant" {
                        return Err(Error::new(ErrorDetails::InferenceServer {
                            message:
                                "Only assistant messages are supported in responses API output"
                                    .to_string(),
                            provider_type: PROVIDER_TYPE.to_string(),
                            raw_request: Some(raw_request.clone()),
                            raw_response: Some(raw_response.clone()),
                        }));
                    }
                    for block in message.content {
                        match block {
                            OpenAIResponsesInputMessageContent::OutputText { text } => {
                                output.push(ContentBlockOutput::Text(Text {
                                    text: text.to_string(),
                                }));
                            }
                            _ => {
                                return Err(Error::new(ErrorDetails::InferenceServer {
                                    message:
                                        "Only output text is supported in responses API output"
                                            .to_string(),
                                    provider_type: PROVIDER_TYPE.to_string(),
                                    raw_request: Some(raw_request.clone()),
                                    raw_response: Some(raw_response.clone()),
                                }));
                            }
                        }
                    }
                }
                FlattenUnknown::Normal(OpenAIResponsesOutputInner::FunctionCall(function_call)) => {
                    output.push(ContentBlockOutput::ToolCall(ToolCall {
                        id: function_call.call_id.to_string(),
                        arguments: function_call.arguments.to_string(),
                        name: function_call.name.to_string(),
                    }));
                }

                FlattenUnknown::Normal(OpenAIResponsesOutputInner::Reasoning {
                    encrypted_content,
                    summary,
                }) => {
                    let mut thought = Thought {
                        text: None,
                        signature: None,
                        provider_type: Some(PROVIDER_TYPE.to_string()),
                        summary: None,
                    };

                    if let Some(encrypted_content) = encrypted_content {
                        thought.signature = Some(encrypted_content);
                    }

                    let tensorzero_summary = summary
                        .into_iter()
                        .map(|summary| match summary {
                            OpenAIResponsesReasoningSummary::SummaryText { text } => {
                                ThoughtSummaryBlock::SummaryText {
                                    text: text.to_string(),
                                }
                            }
                        })
                        .collect::<Vec<ThoughtSummaryBlock>>();
                    thought.summary = Some(tensorzero_summary);
                    output.push(ContentBlockOutput::Thought(thought));
                }
                FlattenUnknown::Unknown(data) => {
                    output.push(ContentBlockOutput::Unknown {
                        data: data.into_owned(),
                        model_provider_name: Some(fully_qualified_name(model_name, provider_name)),
                    });
                }
            }
        }

        let finish_reason = match self.incomplete_details {
            Some(incomplete_details) => {
                // The contents of the 'reason' field is undocumented,
                // but OpenAI appears to set it to 'max_output_tokens' when the 'max_output_tokens'
                // field is provided and the response is incomplete.
                if incomplete_details.reason == "max_output_tokens" {
                    Some(FinishReason::Length)
                } else {
                    None
                }
            }
            None => None,
        };

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output,
                system: generic_request.system.clone(),
                input_messages: generic_request.messages.clone(),
                raw_request,
                raw_response: raw_response.clone(),
                usage: self.usage.into(),
                latency,
                finish_reason,
            },
        ))
    }
}

pub(super) fn get_responses_url(base_url: &Url) -> Result<Url, Error> {
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    url.join("responses").map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}

impl<'a> OpenAITool<'a> {
    pub fn into_openai_responses_tool(self) -> OpenAIResponsesTool<'a> {
        OpenAIResponsesTool::Function(OpenAIResponsesFunctionTool {
            r#type: self.r#type,
            name: self.function.name,
            description: self.function.description,
            parameters: self.function.parameters,
            strict: self.strict,
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum OpenAIResponsesTool<'a> {
    Function(OpenAIResponsesFunctionTool<'a>),
    BuiltIn(&'a Value),
}

#[derive(Serialize, Debug)]
pub struct OpenAIResponsesFunctionTool<'a> {
    r#type: OpenAIToolType,
    name: &'a str,
    description: Option<&'a str>,
    parameters: &'a Value,
    strict: bool,
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum OpenAIResponsesToolChoice {
    String(OpenAIResponsesToolChoiceString),
    AllowedTools(OpenAIResponsesAllowedTools),
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIResponsesAllowedToolsMode {
    Required,
}

#[derive(Serialize, Debug)]
pub struct OpenAIResponsesAllowedTools {
    mode: OpenAIResponsesAllowedToolsMode,
    tools: Vec<OpenAIResponsesToolReference>,
    r#type: StaticTypeAllowedTools,
}

#[derive(Debug)]
struct StaticTypeAllowedTools;
impl Serialize for StaticTypeAllowedTools {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str("allowed_tools")
    }
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIResponsesToolReference {
    Function { name: String },
}

impl<'a> OpenAIResponsesRequest<'a> {
    pub async fn new(
        openai_model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
        include_encrypted_reasoning: bool,
        built_in_tools: &'a [Value],
        tensorzero_model_name: &str,
        tensorzero_model_provider_name: &str,
    ) -> Result<OpenAIResponsesRequest<'a>, Error> {
        let mut tools: Vec<OpenAIResponsesTool> = request
            .tool_config
            .as_ref()
            .map(|tool_config| {
                tool_config
                    .tools_available
                    .iter()
                    .map(|tool| OpenAITool::from(tool).into_openai_responses_tool())
                    .collect()
            })
            .unwrap_or_default();
        // If we have built_in_tools we should extend the list with them
        tools.extend(built_in_tools.iter().map(OpenAIResponsesTool::BuiltIn));
        if let Some(tc) = request.tool_config.as_ref() {
            let provider_tools =
                tc.get_scoped_provider_tools(tensorzero_model_name, tensorzero_model_provider_name);
            tools.extend(
                provider_tools
                    .iter()
                    .map(|t| OpenAIResponsesTool::BuiltIn(&t.tool)),
            );
        }

        // For now, we don't allow selecting any built-in tools
        let tool_choice =
            request
                .tool_config
                .as_ref()
                .map(|tool_config| match &tool_config.tool_choice {
                    ToolChoice::None => {
                        OpenAIResponsesToolChoice::String(OpenAIResponsesToolChoiceString::None)
                    }
                    ToolChoice::Auto => {
                        OpenAIResponsesToolChoice::String(OpenAIResponsesToolChoiceString::Auto)
                    }
                    ToolChoice::Required => {
                        OpenAIResponsesToolChoice::String(OpenAIResponsesToolChoiceString::Required)
                    }
                    ToolChoice::Specific(tool_name) => {
                        OpenAIResponsesToolChoice::AllowedTools(OpenAIResponsesAllowedTools {
                            mode: OpenAIResponsesAllowedToolsMode::Required,
                            tools: vec![OpenAIResponsesToolReference::Function {
                                name: tool_name.clone(),
                            }],
                            r#type: StaticTypeAllowedTools,
                        })
                    }
                });

        let mut parallel_tool_calls = request
            .tool_config
            .as_ref()
            .and_then(|config| config.parallel_tool_calls);
        if openai_model.to_lowercase().starts_with("o1") && parallel_tool_calls == Some(false) {
            parallel_tool_calls = None;
        }

        if request.borrow_stop_sequences().is_some() {
            tracing::warn!("Stop sequences are not supported in the OpenAI Responses API");
        }

        let text = match request.json_mode {
            ModelInferenceRequestJsonMode::On => OpenAIResponsesTextConfigFormat::JsonObject,
            ModelInferenceRequestJsonMode::Off => OpenAIResponsesTextConfigFormat::Text,
            ModelInferenceRequestJsonMode::Strict => {
                if let Some(output_schema) = request.output_schema {
                    OpenAIResponsesTextConfigFormat::JsonSchema {
                        // TODO - should we allow users to customize this name?
                        name: "response_schema".to_string(),
                        schema: output_schema.clone(),
                        strict: true,
                    }
                } else {
                    OpenAIResponsesTextConfigFormat::JsonObject
                }
            }
        };

        Ok(Self {
            model: openai_model,
            input: prepare_openai_responses_messages(
                request
                    .system
                    .as_deref()
                    .map(|m| SystemOrDeveloper::System(Cow::Borrowed(m))),
                &request.messages,
                OpenAIMessagesConfig {
                    json_mode: Some(&request.json_mode),
                    provider_type: PROVIDER_TYPE,
                    fetch_and_encode_input_files_before_inference: request
                        .fetch_and_encode_input_files_before_inference,
                },
            )
            .await?,
            text: OpenAIResponsesTextConfig { format: text },
            tools,
            parallel_tool_calls,
            tool_choice,
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            seed: request.seed,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            include: if include_encrypted_reasoning {
                Some(vec![OpenAIResponsesInclude::ReasoningEncryptedContent])
            } else {
                None
            },
            stream: request.stream,
        })
    }
}

pub type OpenAIResponsesOutput<'a> = FlattenUnknown<'a, OpenAIResponsesOutputInner<'a>>;

#[derive(Clone, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIResponsesOutputInner<'a> {
    #[serde(borrow)]
    Message(OpenAIResponsesInputMessage<'a>),
    #[serde(borrow)]
    FunctionCall(OpenAIResponsesFunctionCall<'a>),
    Reasoning {
        #[serde(default)]
        encrypted_content: Option<String>,
        summary: Vec<OpenAIResponsesReasoningSummary<'a>>,
    },
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIResponsesReasoningSummary<'a> {
    SummaryText { text: Cow<'a, str> },
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIResponsesInputInner<'a> {
    #[serde(borrow)]
    Message(OpenAIResponsesInputMessage<'a>),
    #[serde(borrow)]
    FunctionCallOutput(OpenAIResponsesFunctionCallOutput<'a>),
    #[serde(borrow)]
    FunctionCall(OpenAIResponsesFunctionCall<'a>),
    #[serde(borrow)]
    Reasoning(OpenAIResponsesReasoning<'a>),
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(untagged)]
pub enum OpenAIResponsesInput<'a> {
    #[serde(borrow)]
    Known(OpenAIResponsesInputInner<'a>),
    Unknown(Cow<'a, Value>),
}

impl OpenAIResponsesInput<'_> {
    pub fn content_contains_case_insensitive(&self, value: &str) -> bool {
        match self {
            OpenAIResponsesInput::Known(OpenAIResponsesInputInner::Message(msg)) => {
                for block in &msg.content {
                    if let OpenAIResponsesInputMessageContent::InputText { text } = block {
                        if text.to_lowercase().contains(value) {
                            return true;
                        }
                    }
                }
                false
            }
            // Don't consider the content of non-text blocks
            OpenAIResponsesInput::Known(
                OpenAIResponsesInputInner::FunctionCallOutput(_)
                | OpenAIResponsesInputInner::FunctionCall(_)
                | OpenAIResponsesInputInner::Reasoning(_),
            )
            | OpenAIResponsesInput::Unknown(_) => false,
        }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct OpenAIResponsesFunctionCallOutput<'a> {
    call_id: Cow<'a, str>,
    output: Cow<'a, str>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct OpenAIResponsesFunctionCall<'a> {
    call_id: Cow<'a, str>,
    name: Cow<'a, str>,
    arguments: Cow<'a, str>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct OpenAIResponsesInputMessage<'a> {
    pub role: &'a str,
    pub id: Option<Cow<'a, str>>,
    pub content: Vec<OpenAIResponsesInputMessageContent<'a>>,
}

#[derive(Clone, Deserialize, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIResponsesInputMessageContent<'a> {
    InputText {
        text: Cow<'a, str>,
    },
    InputImage {
        image_url: Cow<'a, str>,
    },
    InputFile {
        #[serde(flatten)]
        file: OpenAIFile<'a>,
    },
    OutputText {
        text: Cow<'a, str>,
    },
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct OpenAIResponsesReasoning<'a> {
    encrypted_content: Cow<'a, str>,
    summary: Vec<OpenAIResponsesReasoningSummary<'a>>,
}

impl Serialize for OpenAIResponsesInputMessageContent<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Helper<'a> {
            InputText {
                text: &'a str,
            },
            InputImage {
                image_url: &'a str,
            },
            InputFile {
                #[serde(flatten)]
                file: &'a OpenAIFile<'a>,
            },
            OutputText {
                text: &'a str,
            },
        }
        match self {
            OpenAIResponsesInputMessageContent::InputText { text } => {
                Helper::InputText { text }.serialize(serializer)
            }
            OpenAIResponsesInputMessageContent::InputImage { image_url } => {
                Helper::InputImage { image_url }.serialize(serializer)
            }
            OpenAIResponsesInputMessageContent::InputFile { file } => {
                Helper::InputFile { file }.serialize(serializer)
            }
            OpenAIResponsesInputMessageContent::OutputText { text } => {
                Helper::OutputText { text }.serialize(serializer)
            }
        }
    }
}

fn should_add_json_instruction_responses(
    content: &str,
    messages: &[OpenAIResponsesInput<'_>],
) -> bool {
    !content.to_lowercase().contains("json")
        && !messages
            .iter()
            .any(|msg| msg.content_contains_case_insensitive("json"))
}

pub async fn prepare_openai_responses_messages<'a>(
    system_or_developer: Option<SystemOrDeveloper<'a>>,
    messages: &'a [RequestMessage],
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<OpenAIResponsesInput<'a>>, Error> {
    let mut openai_messages: Vec<_> = try_join_all(
        messages
            .iter()
            .map(|msg| tensorzero_to_openai_responses_messages(msg, messages_config)),
    )
    .await?
    .into_iter()
    .flatten()
    .collect();

    if let Some(system_msg) = prepare_system_or_developer_message_helper(
        system_or_developer,
        messages_config.json_mode,
        |content| should_add_json_instruction_responses(content, &openai_messages),
    ) {
        openai_messages.insert(0, system_msg.into_openai_responses_input());
    }

    Ok(openai_messages)
}

pub(super) async fn tensorzero_to_openai_responses_messages<'a>(
    message: &'a RequestMessage,
    messages_config: OpenAIMessagesConfig<'_>,
) -> Result<Vec<OpenAIResponsesInput<'a>>, Error> {
    match message.role {
        Role::User => {
            tensorzero_to_openai_responses_user_messages(&message.content, messages_config).await
        }
        Role::Assistant => tensorzero_to_openai_responses_assistant_message(
            Cow::Borrowed(&message.content),
            messages_config.provider_type,
        ),
    }
}

async fn tensorzero_to_openai_responses_user_messages<'a>(
    content_blocks: &'a [ContentBlock],
    messages_config: OpenAIMessagesConfig<'_>,
) -> Result<Vec<OpenAIResponsesInput<'a>>, Error> {
    // We need to separate the tool result messages from the user content blocks.

    let mut messages = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                messages.push(OpenAIResponsesInput::Known(
                    OpenAIResponsesInputInner::Message(OpenAIResponsesInputMessage {
                        id: None,
                        role: "user",
                        content: vec![OpenAIResponsesInputMessageContent::InputText {
                            text: Cow::Borrowed(text),
                        }],
                    }),
                ));
            }
            ContentBlock::ToolCall(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool calls are not supported in user messages".to_string(),
                }));
            }
            ContentBlock::ToolResult(tool_result) => {
                messages.push(OpenAIResponsesInput::Known(
                    OpenAIResponsesInputInner::FunctionCallOutput(
                        OpenAIResponsesFunctionCallOutput {
                            output: Cow::Borrowed(&tool_result.result),
                            call_id: Cow::Borrowed(&tool_result.id),
                        },
                    ),
                ));
            }
            ContentBlock::File(file) => {
                let content_block = prepare_file_message(file, messages_config).await?;
                match content_block {
                    OpenAIContentBlock::ImageUrl { image_url } => {
                        messages.push(OpenAIResponsesInput::Known(
                            OpenAIResponsesInputInner::Message(OpenAIResponsesInputMessage {
                                id: None,
                                role: "user",
                                content: vec![OpenAIResponsesInputMessageContent::InputImage {
                                    image_url: Cow::Owned(image_url.url),
                                }],
                            }),
                        ));
                    }
                    OpenAIContentBlock::File { file } => {
                        messages.push(OpenAIResponsesInput::Known(
                            OpenAIResponsesInputInner::Message(OpenAIResponsesInputMessage {
                                id: None,
                                role: "user",
                                content: vec![OpenAIResponsesInputMessageContent::InputFile {
                                    file,
                                }],
                            }),
                        ));
                    }
                    _ => {
                        return Err(Error::new(ErrorDetails::InternalError {
                            message: format!("`prepare_file_message` produced an unexpected content block. {IMPOSSIBLE_ERROR_MESSAGE}")
                        }));
                    }
                }
            }
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(messages_config.provider_type, thought);
            }
            ContentBlock::Unknown {
                data,
                model_provider_name: _,
            } => {
                // The user included an 'unknown' content block inside of the user message,
                // so push a new user message that includes their custom JSON value
                messages.push(OpenAIResponsesInput::Unknown(Cow::Borrowed(data)));
            }
        };
    }

    Ok(messages)
}

pub fn tensorzero_to_openai_responses_assistant_message<'a>(
    content_blocks: Cow<'a, [ContentBlock]>,
    _provider_type: &str,
) -> Result<Vec<OpenAIResponsesInput<'a>>, Error> {
    let mut output = Vec::new();
    let content_block_cows: Vec<Cow<'_, ContentBlock>> = match content_blocks {
        Cow::Borrowed(content_blocks) => content_blocks.iter().map(Cow::Borrowed).collect(),
        Cow::Owned(content_blocks) => content_blocks.into_iter().map(Cow::Owned).collect(),
    };

    for block in content_block_cows {
        match block {
            Cow::Borrowed(ContentBlock::Text(Text { text })) => {
                output.push(OpenAIResponsesInput::Known(
                    OpenAIResponsesInputInner::Message(OpenAIResponsesInputMessage {
                        id: None,
                        role: "assistant",
                        content: vec![OpenAIResponsesInputMessageContent::OutputText {
                            text: Cow::Borrowed(text),
                        }],
                    }),
                ));
            }
            Cow::Owned(ContentBlock::Text(Text { text })) => {
                output.push(OpenAIResponsesInput::Known(
                    OpenAIResponsesInputInner::Message(OpenAIResponsesInputMessage {
                        id: None,
                        role: "assistant",
                        content: vec![OpenAIResponsesInputMessageContent::OutputText {
                            text: Cow::Owned(text),
                        }],
                    }),
                ));
            }
            Cow::Borrowed(ContentBlock::ToolCall(tool_call)) => {
                output.push(OpenAIResponsesInput::Known(
                    OpenAIResponsesInputInner::FunctionCall(OpenAIResponsesFunctionCall {
                        call_id: Cow::Borrowed(&tool_call.id),
                        name: Cow::Borrowed(&tool_call.name),
                        arguments: Cow::Borrowed(&tool_call.arguments),
                    }),
                ));
            }
            Cow::Owned(ContentBlock::ToolCall(tool_call)) => {
                output.push(OpenAIResponsesInput::Known(
                    OpenAIResponsesInputInner::FunctionCall(OpenAIResponsesFunctionCall {
                        call_id: Cow::Owned(tool_call.id),
                        name: Cow::Owned(tool_call.name),
                        arguments: Cow::Owned(tool_call.arguments),
                    }),
                ));
            }
            Cow::Borrowed(ContentBlock::ToolResult(_))
            | Cow::Owned(ContentBlock::ToolResult(_)) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool results are not supported in assistant messages".to_string(),
                }));
            }
            Cow::Borrowed(ContentBlock::File(_)) | Cow::Owned(ContentBlock::File(_)) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Files are not supported in assistant messages".to_string(),
                }));
            }
            Cow::Borrowed(ContentBlock::Thought(ref thought))
            | Cow::Owned(ContentBlock::Thought(ref thought)) => {
                if let Some(encrypted_content) = &thought.signature {
                    output.push(OpenAIResponsesInput::Known(
                        OpenAIResponsesInputInner::Reasoning(OpenAIResponsesReasoning {
                            encrypted_content: Cow::Owned(encrypted_content.clone()),
                            summary: thought
                                .summary
                                .as_ref()
                                .map(|summary| {
                                    summary
                                        .iter()
                                        .map(|block| match block {
                                            ThoughtSummaryBlock::SummaryText { text } => {
                                                OpenAIResponsesReasoningSummary::SummaryText {
                                                    text: Cow::Owned(text.to_string()),
                                                }
                                            }
                                        })
                                        .collect()
                                })
                                .unwrap_or_default(),
                        }),
                    ));
                }
            }

            Cow::Borrowed(ContentBlock::Unknown {
                data,
                model_provider_name: _,
            }) => {
                output.push(OpenAIResponsesInput::Unknown(Cow::Borrowed(data)));
            }
            Cow::Owned(ContentBlock::Unknown {
                data,
                model_provider_name: _,
            }) => {
                output.push(OpenAIResponsesInput::Unknown(Cow::Owned(data)));
            }
        }
    }

    Ok(output)
}

// Streaming types for OpenAI Responses API
// Based on: https://platform.openai.com/docs/api-reference/responses-streaming

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
#[expect(dead_code)]
pub(super) enum OpenAIResponsesStreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated { response: Value },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress { response: Value },
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: Value },
    #[serde(rename = "response.failed")]
    ResponseFailed { response: Value },
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete { response: Value },
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded { item: Value, output_index: u32 },
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone { item: Value, output_index: u32 },
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        part: Value,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        part: Value,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        delta: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        text: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    #[serde(rename = "response.refusal.delta")]
    ResponseRefusalDelta {
        delta: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    #[serde(rename = "response.refusal.done")]
    ResponseRefusalDone {
        refusal: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ResponseReasoningSummaryTextDelta {
        delta: String,
        item_id: String,
        output_index: u32,
        summary_index: u32,
    },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ResponseReasoningSummaryTextDone {
        text: String,
        item_id: String,
        output_index: u32,
        summary_index: u32,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        delta: String,
        item_id: String,
        output_index: u32,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        arguments: String,
        item_id: String,
        output_index: u32,
    },
    #[serde(rename = "error")]
    Error { error: Value },
    #[serde(other)]
    Unknown,
}

/// Stream function for OpenAI Responses API
/// Similar to stream_openai but uses the Responses API streaming format
pub fn stream_openai_responses(
    provider_type: String,
    event_source: impl Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static,
    start_time: Instant,
) -> ProviderInferenceResponseStreamInner {
    let mut current_tool_id: Option<String> = None;
    let mut current_tool_name: Option<String> = None;

    Box::pin(async_stream::stream! {
        futures::pin_mut!(event_source);
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    match e {
                        TensorZeroEventError::TensorZero(e) => {
                            yield Err(e);
                        }
                        TensorZeroEventError::EventSource(e) => {
                            yield Err(convert_stream_error(provider_type.clone(), e).await);
                        }
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        // OpenAI Responses API does not send [DONE] marker
                        // Instead, we check for terminal events: completed, failed, or incomplete
                        let data: Result<OpenAIResponsesStreamEvent, serde_json::Error> =
                            serde_json::from_str(&message.data);

                        // If we can't parse the event at all, log and skip it
                        let event = match data {
                            Ok(event) => event,
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to parse OpenAI Responses stream event, skipping. Error: {}, Data: {}",
                                    e,
                                    message.data
                                );
                                continue;
                            }
                        };

                        // Check if this is a terminal event
                        let is_terminal = matches!(
                            &event,
                            OpenAIResponsesStreamEvent::ResponseCompleted { .. }
                                | OpenAIResponsesStreamEvent::ResponseIncomplete { .. }
                                | OpenAIResponsesStreamEvent::ResponseFailed { .. }
                        );

                        let latency = start_time.elapsed();
                        let stream_message = openai_responses_to_tensorzero_chunk(
                            message.data,
                            event,
                            latency,
                            &mut current_tool_id,
                            &mut current_tool_name,
                        );

                        match stream_message {
                            Ok(Some(chunk)) => {
                                yield Ok(chunk);
                                // Break after yielding terminal events
                                if is_terminal {
                                    break;
                                }
                            }
                            Ok(None) => continue, // Skip lifecycle events
                            Err(e) => {
                                yield Err(e);
                                // Break on error events too
                                break;
                            }
                        }
                    }
                },
            }
        }
    })
}

/// Maps an OpenAI Responses API stream event to a TensorZero chunk
/// Similar to anthropic_to_tensorzero_stream_message in anthropic.rs
/// Tool calls require tracking the current tool ID and name across chunks
pub(super) fn openai_responses_to_tensorzero_chunk(
    raw_message: String,
    event: OpenAIResponsesStreamEvent,
    message_latency: Duration,
    current_tool_id: &mut Option<String>,
    current_tool_name: &mut Option<String>,
) -> Result<Option<ProviderInferenceResponseChunk>, Error> {
    match event {
        // Text delta - the main content streaming event
        OpenAIResponsesStreamEvent::ResponseOutputTextDelta {
            delta,
            content_index,
            ..
        } => Ok(Some(ProviderInferenceResponseChunk::new(
            vec![ContentBlockChunk::Text(TextChunk {
                text: delta,
                id: content_index.to_string(),
            })],
            None,
            raw_message,
            message_latency,
            None,
        ))),

        // Reasoning (thought) delta
        OpenAIResponsesStreamEvent::ResponseReasoningSummaryTextDelta {
            delta,
            output_index,
            summary_index,
            ..
        } => Ok(Some(ProviderInferenceResponseChunk::new(
            vec![ContentBlockChunk::Thought(ThoughtChunk {
                text: None,
                signature: None,
                id: output_index.to_string(),
                summary_id: Some(summary_index.to_string()),
                summary_text: Some(delta),
                provider_type: Some(PROVIDER_TYPE.to_string()),
            })],
            None,
            raw_message,
            message_latency,
            None,
        ))),

        // Function call arguments delta
        OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDelta { delta, .. } => {
            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: current_tool_id.clone().ok_or_else(|| {
                        Error::new(ErrorDetails::InferenceServer {
                            message: "Got function_call_arguments.delta without current tool id"
                                .to_string(),
                            provider_type: PROVIDER_TYPE.to_string(),
                            raw_request: None,
                            raw_response: Some(raw_message.clone()),
                        })
                    })?,
                    raw_name: None, // Name already sent in the 'done' event
                    raw_arguments: delta,
                })],
                None,
                raw_message,
                message_latency,
                None,
            )))
        }

        // Function call done - marks the end of the function call arguments streaming
        // Don't send the name again (it was already sent in output_item.added)
        // Don't send the arguments (they were already sent via deltas)
        OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDone { item_id, .. } => {
            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: item_id,
                    raw_name: None,
                    raw_arguments: String::new(),
                })],
                None,
                raw_message,
                message_latency,
                None,
            )))
        }

        // Output item added - captures tool metadata when it's a function_call
        // This is where we emit the tool name to the client
        OpenAIResponsesStreamEvent::ResponseOutputItemAdded { item, .. } => {
            // Check if this is a function_call item
            if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
                if item_type == "function_call" {
                    // Extract the tool name and ID
                    if let (Some(name), Some(id)) = (
                        item.get("name").and_then(|n| n.as_str()),
                        item.get("id").and_then(|i| i.as_str()),
                    ) {
                        *current_tool_id = Some(id.to_string());
                        *current_tool_name = Some(name.to_string());

                        // Emit a chunk with the tool name and ID
                        return Ok(Some(ProviderInferenceResponseChunk::new(
                            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                                id: id.to_string(),
                                raw_name: Some(name.to_string()),
                                raw_arguments: String::new(),
                            })],
                            None,
                            raw_message,
                            message_latency,
                            None,
                        )));
                    }
                }
            }
            // Don't emit a chunk for this event if it's not a function_call
            Ok(None)
        }

        // Completed event - extract usage and finish reason
        OpenAIResponsesStreamEvent::ResponseCompleted { response } => {
            let usage = response.get("usage").and_then(|u| {
                let input_tokens = u.get("input_tokens")?.as_u64()? as u32;
                let output_tokens = u.get("output_tokens")?.as_u64()? as u32;
                Some(Usage {
                    input_tokens,
                    output_tokens,
                })
            });

            // The incomplete_details field indicates if response was cut short
            let finish_reason = if response.get("incomplete_details").is_some() {
                Some(FinishReason::Length)
            } else {
                Some(FinishReason::Stop)
            };

            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![],
                usage,
                raw_message,
                message_latency,
                finish_reason,
            )))
        }

        // Failed event - return error
        OpenAIResponsesStreamEvent::ResponseFailed { response } => {
            let error_msg = response
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            Err(Error::new(ErrorDetails::InferenceServer {
                message: error_msg.to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_message),
            }))
        }

        // Incomplete event - extract finish reason
        OpenAIResponsesStreamEvent::ResponseIncomplete { response } => {
            let usage = response.get("usage").and_then(|u| {
                let input_tokens = u.get("input_tokens")?.as_u64()? as u32;
                let output_tokens = u.get("output_tokens")?.as_u64()? as u32;
                Some(Usage {
                    input_tokens,
                    output_tokens,
                })
            });

            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![],
                usage,
                raw_message,
                message_latency,
                Some(FinishReason::Length),
            )))
        }

        // Refusal - treat as an error
        OpenAIResponsesStreamEvent::ResponseRefusalDelta { delta, .. }
        | OpenAIResponsesStreamEvent::ResponseRefusalDone { refusal: delta, .. } => {
            Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("Model refused to respond: {delta}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_message),
            }))
        }

        // Error event
        OpenAIResponsesStreamEvent::Error { error } => {
            Err(Error::new(ErrorDetails::InferenceServer {
                message: error.to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_message),
            }))
        }

        // Lifecycle and other events we don't need to process
        OpenAIResponsesStreamEvent::ResponseCreated { .. }
        | OpenAIResponsesStreamEvent::ResponseInProgress { .. }
        | OpenAIResponsesStreamEvent::ResponseOutputItemDone { .. }
        | OpenAIResponsesStreamEvent::ResponseContentPartAdded { .. }
        | OpenAIResponsesStreamEvent::ResponseContentPartDone { .. }
        | OpenAIResponsesStreamEvent::ResponseOutputTextDone { .. }
        | OpenAIResponsesStreamEvent::ResponseReasoningSummaryTextDone { .. } => Ok(None),

        // Unknown event type - log and skip
        OpenAIResponsesStreamEvent::Unknown => {
            tracing::warn!(
                "Received unknown event type in OpenAI Responses stream, skipping. Raw message: {}",
                raw_message
            );
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::types::FunctionType;

    use super::*;
    use std::time::Duration;

    #[test]
    fn test_deserialize_response_created() {
        let json = r#"{
            "type": "response.created",
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 1741487325,
                "status": "in_progress"
            },
            "sequence_number": 1
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            OpenAIResponsesStreamEvent::ResponseCreated { .. }
        ));
    }

    #[test]
    fn test_deserialize_response_output_item_added() {
        let json = r#"{
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "name": "get_weather",
                "id": "fc_abc123",
                "arguments": ""
            },
            "output_index": 0,
            "sequence_number": 4
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            OpenAIResponsesStreamEvent::ResponseOutputItemAdded { item, output_index } => {
                assert_eq!(output_index, 0);
                assert_eq!(item.get("type").unwrap().as_str().unwrap(), "function_call");
                assert_eq!(item.get("name").unwrap().as_str().unwrap(), "get_weather");
                assert_eq!(item.get("id").unwrap().as_str().unwrap(), "fc_abc123");
            }
            _ => panic!("Expected ResponseOutputItemAdded"),
        }
    }

    #[test]
    fn test_deserialize_response_output_text_delta() {
        let json = r#"{
            "type": "response.output_text.delta",
            "item_id": "msg_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
            "sequence_number": 5
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            OpenAIResponsesStreamEvent::ResponseOutputTextDelta {
                delta,
                content_index,
                ..
            } => {
                assert_eq!(delta, "Hello");
                assert_eq!(content_index, 0);
            }
            _ => panic!("Expected ResponseOutputTextDelta"),
        }
    }

    #[test]
    fn test_deserialize_response_reasoning_summary_text_delta() {
        let json = r#"{
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_123",
            "output_index": 0,
            "summary_index": 0,
            "delta": "Thinking about...",
            "sequence_number": 3
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            OpenAIResponsesStreamEvent::ResponseReasoningSummaryTextDelta {
                delta,
                summary_index,
                ..
            } => {
                assert_eq!(delta, "Thinking about...");
                assert_eq!(summary_index, 0);
            }
            _ => panic!("Expected ResponseReasoningSummaryTextDelta"),
        }
    }

    #[test]
    fn test_deserialize_function_call_arguments_delta() {
        let json = r#"{
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_123",
            "output_index": 0,
            "delta": "{\"location\":",
            "sequence_number": 7
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDelta { delta, .. } => {
                assert_eq!(delta, r#"{"location":"#);
            }
            _ => panic!("Expected ResponseFunctionCallArgumentsDelta"),
        }
    }

    #[test]
    fn test_deserialize_function_call_arguments_done() {
        let json = r#"{
            "type": "response.function_call_arguments.done",
            "item_id": "fc_123",
            "output_index": 0,
            "arguments": "{\"location\": \"San Francisco\"}",
            "sequence_number": 10
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
                arguments,
                item_id,
                ..
            } => {
                assert_eq!(arguments, r#"{"location": "San Francisco"}"#);
                assert_eq!(item_id, "fc_123");
            }
            _ => panic!("Expected ResponseFunctionCallArgumentsDone"),
        }
    }

    #[test]
    fn test_deserialize_response_completed() {
        let json = r#"{
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20
                }
            },
            "sequence_number": 15
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            OpenAIResponsesStreamEvent::ResponseCompleted { .. }
        ));
    }

    #[test]
    fn test_deserialize_response_failed() {
        let json = r#"{
            "type": "response.failed",
            "response": {
                "id": "resp_123",
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "Internal error"
                }
            },
            "sequence_number": 5
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            OpenAIResponsesStreamEvent::ResponseFailed { .. }
        ));
    }

    #[test]
    fn test_deserialize_refusal_delta() {
        let json = r#"{
            "type": "response.refusal.delta",
            "item_id": "msg_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "I cannot",
            "sequence_number": 3
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            OpenAIResponsesStreamEvent::ResponseRefusalDelta { delta, .. } => {
                assert_eq!(delta, "I cannot");
            }
            _ => panic!("Expected ResponseRefusalDelta"),
        }
    }

    #[test]
    fn test_deserialize_error_event() {
        let json = r#"{
            "type": "error",
            "error": {
                "code": "invalid_request",
                "message": "Bad request"
            }
        }"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, OpenAIResponsesStreamEvent::Error { .. }));
    }

    #[test]
    fn test_text_delta_conversion() {
        let event = OpenAIResponsesStreamEvent::ResponseOutputTextDelta {
            delta: "Hello world".to_string(),
            item_id: "msg_1".to_string(),
            output_index: 0,
            content_index: 2,
        };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlockChunk::Text(text_chunk) => {
                assert_eq!(text_chunk.text, "Hello world");
                assert_eq!(text_chunk.id, "2");
            }
            _ => panic!("Expected Text chunk"),
        }
    }

    #[test]
    fn test_reasoning_delta_conversion() {
        let event = OpenAIResponsesStreamEvent::ResponseReasoningSummaryTextDelta {
            delta: "Let me think...".to_string(),
            item_id: "rs_1".to_string(),
            output_index: 0,
            summary_index: 3,
        };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlockChunk::Thought(thought_chunk) => {
                assert_eq!(thought_chunk.text, None);
                assert_eq!(thought_chunk.id, "0");
                assert_eq!(thought_chunk.summary_id, Some("3".to_string()));
                assert_eq!(
                    thought_chunk.summary_text,
                    Some("Let me think...".to_string())
                );
                assert_eq!(thought_chunk.provider_type, Some(PROVIDER_TYPE.to_string()));
            }
            _ => panic!("Expected Thought chunk"),
        }
    }

    #[test]
    fn test_output_item_added_conversion() {
        let item_json = serde_json::json!({
            "type": "function_call",
            "name": "get_weather",
            "id": "fc_abc123",
            "arguments": ""
        });

        let event = OpenAIResponsesStreamEvent::ResponseOutputItemAdded {
            item: item_json,
            output_index: 0,
        };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        // Should emit a chunk with tool name and ID
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "fc_abc123");
                assert_eq!(tool_call.raw_name, Some("get_weather".to_string()));
                assert_eq!(tool_call.raw_arguments, "");
            }
            _ => panic!("Expected ToolCall chunk"),
        }

        // And should have captured tool ID and name in state
        assert_eq!(tool_id, Some("fc_abc123".to_string()));
        assert_eq!(tool_name, Some("get_weather".to_string()));
    }

    #[test]
    fn test_function_call_done_conversion() {
        let event = OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "fc_123".to_string(),
            arguments: r#"{"location": "NYC"}"#.to_string(),
            output_index: 0,
        };

        // Set up tool name from state (as it would be from output_item.added event)
        let mut tool_id = Some("fc_123".to_string());
        let mut tool_name = Some("get_weather".to_string());

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "fc_123");
                // The done event doesn't send the raw_name (it was sent in output_item.added)
                assert_eq!(tool_call.raw_name, None);
                // The done event doesn't send arguments since they were already sent via deltas
                assert_eq!(tool_call.raw_arguments, "");
            }
            _ => panic!("Expected ToolCall chunk"),
        }
    }

    #[test]
    fn test_function_call_arguments_delta_conversion() {
        let event = OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            delta: r#""value"}"#.to_string(),
            item_id: "fc_123".to_string(),
            output_index: 0,
        };

        // Set up the tool ID as it would be from a previous output_item.added event
        let mut tool_id = Some("fc_123".to_string());
        let mut tool_name = Some("my_function".to_string());

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "fc_123");
                assert_eq!(tool_call.raw_name, None);
                assert_eq!(tool_call.raw_arguments, r#""value"}"#);
            }
            _ => panic!("Expected ToolCall chunk"),
        }
    }

    #[test]
    fn test_function_call_streaming_sequence() {
        // This test verifies the correct event sequence for tool call streaming:
        // 1. output_item.added - emits tool name and ID chunk
        // 2. function_call_arguments.delta - streams argument chunks
        // 3. function_call_arguments.done - marks the end (no name or arguments)

        let mut tool_id = None;
        let mut tool_name = None;

        // Step 1: output_item.added event
        let item_added_event = OpenAIResponsesStreamEvent::ResponseOutputItemAdded {
            item: serde_json::json!({
                "type": "function_call",
                "name": "get_weather",
                "id": "fc_xyz789",
                "arguments": ""
            }),
            output_index: 0,
        };

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json_1".to_string(),
            item_added_event,
            Duration::from_millis(10),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        // Should emit a chunk with the tool name and ID
        match &result.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "fc_xyz789");
                assert_eq!(tool_call.raw_name, Some("get_weather".to_string()));
                assert_eq!(tool_call.raw_arguments, "");
            }
            _ => panic!("Expected ToolCall chunk"),
        }
        // And should have captured metadata in state
        assert_eq!(tool_id, Some("fc_xyz789".to_string()));
        assert_eq!(tool_name, Some("get_weather".to_string()));

        // Step 2: function_call_arguments.delta event
        let delta_event = OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            delta: r#"{"location": "#.to_string(),
            item_id: "fc_xyz789".to_string(),
            output_index: 0,
        };

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json_2".to_string(),
            delta_event,
            Duration::from_millis(20),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        // Should emit a tool call chunk with the delta
        match &result.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "fc_xyz789");
                assert_eq!(tool_call.raw_name, None);
                assert_eq!(tool_call.raw_arguments, r#"{"location": "#);
            }
            _ => panic!("Expected ToolCall chunk"),
        }

        // Step 3: Another delta
        let delta_event2 = OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            delta: r#""San Francisco"}"#.to_string(),
            item_id: "fc_xyz789".to_string(),
            output_index: 0,
        };

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json_3".to_string(),
            delta_event2,
            Duration::from_millis(30),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        match &result.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "fc_xyz789");
                assert_eq!(tool_call.raw_arguments, r#""San Francisco"}"#);
            }
            _ => panic!("Expected ToolCall chunk"),
        }

        // Step 4: function_call_arguments.done event
        let done_event = OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDone {
            item_id: "fc_xyz789".to_string(),
            arguments: r#"{"location": "San Francisco"}"#.to_string(),
            output_index: 0,
        };

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json_4".to_string(),
            done_event,
            Duration::from_millis(40),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        // Should emit a chunk marking the end of the function call
        // No name (already sent in output_item.added) and no arguments (already sent in deltas)
        match &result.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "fc_xyz789");
                // No name - it was already sent in step 1
                assert_eq!(tool_call.raw_name, None);
                // No arguments - they were already sent in steps 2 and 3
                assert_eq!(tool_call.raw_arguments, "");
            }
            _ => panic!("Expected ToolCall chunk"),
        }
    }

    #[test]
    fn test_function_call_arguments_delta_without_tool_id() {
        let event = OpenAIResponsesStreamEvent::ResponseFunctionCallArgumentsDelta {
            delta: "delta".to_string(),
            item_id: "fc_123".to_string(),
            output_index: 0,
        };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_response_completed_conversion() {
        let response_json = serde_json::json!({
            "id": "resp_123",
            "status": "completed",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 25
            }
        });

        let event = OpenAIResponsesStreamEvent::ResponseCompleted {
            response: response_json,
        };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.content.len(), 0); // No content, just metadata
        assert_eq!(
            result.usage,
            Some(Usage {
                input_tokens: 15,
                output_tokens: 25
            })
        );
        assert_eq!(result.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_response_incomplete_conversion() {
        let response_json = serde_json::json!({
            "id": "resp_123",
            "status": "incomplete",
            "incomplete_details": {
                "reason": "max_output_tokens"
            },
            "usage": {
                "input_tokens": 10,
                "output_tokens": 100
            }
        });

        let event = OpenAIResponsesStreamEvent::ResponseIncomplete {
            response: response_json,
        };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.finish_reason, Some(FinishReason::Length));
        assert_eq!(
            result.usage,
            Some(Usage {
                input_tokens: 10,
                output_tokens: 100
            })
        );
    }

    #[test]
    fn test_response_failed_conversion() {
        let response_json = serde_json::json!({
            "id": "resp_123",
            "status": "failed",
            "error": {
                "code": "server_error",
                "message": "Internal server error"
            }
        });

        let event = OpenAIResponsesStreamEvent::ResponseFailed {
            response: response_json,
        };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_refusal_delta_conversion() {
        let event = OpenAIResponsesStreamEvent::ResponseRefusalDelta {
            delta: "I cannot help with that".to_string(),
            item_id: "msg_1".to_string(),
            output_index: 0,
            content_index: 0,
        };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_error_event_conversion() {
        let error_json = serde_json::json!({
            "code": "bad_request",
            "message": "Invalid parameters"
        });

        let event = OpenAIResponsesStreamEvent::Error { error: error_json };

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            "raw_json".to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_lifecycle_events_return_none() {
        let events = vec![
            OpenAIResponsesStreamEvent::ResponseCreated {
                response: serde_json::json!({}),
            },
            OpenAIResponsesStreamEvent::ResponseInProgress {
                response: serde_json::json!({}),
            },
            // Note: ResponseOutputItemAdded with function_call type now emits a chunk
            // so we test with a non-function_call item
            OpenAIResponsesStreamEvent::ResponseOutputItemAdded {
                item: serde_json::json!({"type": "message"}),
                output_index: 0,
            },
            OpenAIResponsesStreamEvent::ResponseOutputItemDone {
                item: serde_json::json!({}),
                output_index: 0,
            },
        ];

        let mut tool_id = None;
        let mut tool_name = None;

        for event in events {
            let result = openai_responses_to_tensorzero_chunk(
                "raw_json".to_string(),
                event,
                Duration::from_millis(100),
                &mut tool_id,
                &mut tool_name,
            )
            .unwrap();

            assert!(result.is_none(), "Lifecycle events should return None");
        }
    }

    #[test]
    fn test_unknown_event_type() {
        // Test that unknown event types are handled gracefully
        let json = r#"{"type": "response.some_new_event", "data": "foo"}"#;

        let event: OpenAIResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, OpenAIResponsesStreamEvent::Unknown));

        let mut tool_id = None;
        let mut tool_name = None;

        let result = openai_responses_to_tensorzero_chunk(
            json.to_string(),
            event,
            Duration::from_millis(100),
            &mut tool_id,
            &mut tool_name,
        )
        .unwrap();

        // Unknown events should return None (skip them)
        assert!(result.is_none(), "Unknown events should return None");
    }

    #[test]
    fn test_malformed_json_in_stream() {
        // Test that completely malformed JSON doesn't parse but won't crash
        let json = r#"{"type": "response.created", "invalid json here"#;

        let result: Result<OpenAIResponsesStreamEvent, serde_json::Error> =
            serde_json::from_str(json);

        // Should fail to parse
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_unknown_output_block() {
        // Test that unknown output block types (like web_search_call) are deserialized as FlattenUnknown::Unknown
        let json = r#"{
            "type": "web_search_call",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "test query"
            }
        }"#;

        let result: OpenAIResponsesOutput = serde_json::from_str(json).unwrap();

        match result {
            FlattenUnknown::Unknown(data) => {
                assert_eq!(
                    data.get("type").and_then(|v| v.as_str()),
                    Some("web_search_call")
                );
                assert_eq!(
                    data.get("status").and_then(|v| v.as_str()),
                    Some("completed")
                );
            }
            FlattenUnknown::Normal(_) => panic!("Expected FlattenUnknown::Unknown variant"),
        }
    }

    #[test]
    fn test_convert_unknown_block_to_content_block_output() {
        // Test that unknown blocks are converted to ContentBlockOutput::Unknown with proper model_provider_name
        let json = r#"{
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_123",
                    "status": "completed"
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        }"#;

        let response: OpenAIResponsesResponse = serde_json::from_str(json).unwrap();
        let generic_request = ModelInferenceRequest {
            inference_id: uuid::Uuid::new_v4(),
            messages: vec![],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop_sequences: None,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            extra_cache_key: None,
        };

        let result = response.into_provider_response(
            Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            "test_request".to_string(),
            "test_response".to_string(),
            &generic_request,
            "test-model",
            "test-provider",
        );

        assert!(result.is_ok());
        let provider_response = result.unwrap();
        assert_eq!(provider_response.output.len(), 1);

        match &provider_response.output[0] {
            ContentBlockOutput::Unknown {
                data,
                model_provider_name,
            } => {
                assert_eq!(
                    data.get("type").and_then(|v| v.as_str()),
                    Some("web_search_call")
                );
                assert_eq!(
                    model_provider_name.as_deref(),
                    Some("tensorzero::model_name::test-model::provider_name::test-provider")
                );
            }
            _ => panic!("Expected ContentBlockOutput::Unknown"),
        }
    }

    #[test]
    fn test_mixed_output_with_known_and_unknown_blocks() {
        // Test that responses with both known and unknown blocks are handled correctly
        let json = r#"{
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_1",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello, world!"
                        }
                    ]
                },
                {
                    "type": "web_search_call",
                    "id": "ws_123",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "test"
                    }
                },
                {
                    "type": "function_call",
                    "call_id": "fc_456",
                    "name": "get_weather",
                    "arguments": "{}"
                },
                {
                    "type": "another_unknown_type",
                    "custom_field": "custom_value"
                }
            ],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 100
            }
        }"#;

        let response: OpenAIResponsesResponse = serde_json::from_str(json).unwrap();
        let generic_request = ModelInferenceRequest {
            inference_id: uuid::Uuid::new_v4(),
            messages: vec![],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop_sequences: None,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            extra_cache_key: None,
        };

        let result = response.into_provider_response(
            Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            "test_request".to_string(),
            "test_response".to_string(),
            &generic_request,
            "gpt-5-nano",
            "openai",
        );

        assert!(result.is_ok());
        let provider_response = result.unwrap();

        // Should have 4 output blocks: 1 text, 2 unknown, 1 tool call
        assert_eq!(provider_response.output.len(), 4);

        // First should be text
        match &provider_response.output[0] {
            ContentBlockOutput::Text(text) => {
                assert_eq!(text.text, "Hello, world!");
            }
            _ => panic!("Expected ContentBlockOutput::Text"),
        }

        // Second should be unknown (web_search_call)
        match &provider_response.output[1] {
            ContentBlockOutput::Unknown { data, .. } => {
                assert_eq!(
                    data.get("type").and_then(|v| v.as_str()),
                    Some("web_search_call")
                );
            }
            _ => panic!("Expected ContentBlockOutput::Unknown for web_search_call"),
        }

        // Third should be tool call
        match &provider_response.output[2] {
            ContentBlockOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.name, "get_weather");
                assert_eq!(tool_call.id, "fc_456");
            }
            _ => panic!("Expected ContentBlockOutput::ToolCall"),
        }

        // Fourth should be unknown (another_unknown_type)
        match &provider_response.output[3] {
            ContentBlockOutput::Unknown {
                data,
                model_provider_name,
            } => {
                assert_eq!(
                    data.get("type").and_then(|v| v.as_str()),
                    Some("another_unknown_type")
                );
                assert_eq!(
                    data.get("custom_field").and_then(|v| v.as_str()),
                    Some("custom_value")
                );
                assert_eq!(
                    model_provider_name.as_deref(),
                    Some("tensorzero::model_name::gpt-5-nano::provider_name::openai")
                );
            }
            _ => panic!("Expected ContentBlockOutput::Unknown for another_unknown_type"),
        }
    }
}
