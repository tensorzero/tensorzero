use std::mem::uninitialized;

use axum::body::Body;
use axum::debug_handler;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

use crate::endpoints::inference::{
    inference, ChatCompletionInferenceParams, InferenceParams, Params,
};
use crate::error::Error;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::Input;
use crate::tool::{DynamicToolParams, Tool, ToolChoice};

use super::inference::InferenceCredentials;

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct OpenAICompatibleFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct OpenAICompatibleToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only `function` is supported.
    pub r#type: String,
    /// The function that the model called.
    pub function: OpenAICompatibleFunctionCall,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
struct OpenAICompatibleSystemMessage {
    content: Value,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
struct OpenAICompatibleUserMessage {
    content: Value,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
struct OpenAICompatibleAssistantMessage {
    content: Option<Value>,
    tool_calls: Option<Vec<OpenAICompatibleToolCall>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
struct OpenAICompatibleToolMessage {
    content: Option<Value>,
    tool_call_id: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
enum OpenAICompatibleMessage {
    System(OpenAICompatibleSystemMessage),
    User(OpenAICompatibleUserMessage),
    Assistant(OpenAICompatibleAssistantMessage),
    Tool(OpenAICompatibleToolMessage),
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum OpenAICompatibleResponseFormat {
    Text,
    JsonObject { json_schema: Option<Value> },
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", content = "function")]
#[serde(rename_all = "snake_case")]
enum OpenAICompatibleTool {
    Function {
        description: Option<String>,
        name: String,
        parameters: Option<Value>,
        strict: Option<bool>,
    },
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
struct FunctionName {
    name: String,
}

/// Specifies a tool the model should use. Use to force the model to call a specific function.
#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAICompatibleNamedToolChoice {
    /// The type of the tool. Currently, only `function` is supported.
    r#type: String,
    function: FunctionName,
}

/// Controls which (if any) tool is called by the model.
/// `none` means the model will not call any tool and instead generates a message.
/// `auto` means the model can pick between generating a message or calling one or more tools.
/// `required` means the model must call one or more tools.
/// Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.
///
/// `none` is the default when no tools are present. `auto` is the default if tools are present.present.
#[derive(Clone, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum ChatCompletionToolChoiceOption {
    #[default]
    None,
    Auto,
    Required,
    #[serde(untagged)]
    Named(OpenAICompatibleNamedToolChoice),
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct OpenAICompatibleParams {
    messages: Vec<OpenAICompatibleMessage>,
    model: String,
    frequency_penalty: Option<f32>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    presence_penalty: Option<f32>,
    response_format: Option<OpenAICompatibleResponseFormat>,
    seed: Option<u32>,
    stream: Option<bool>,
    temperature: Option<f32>,
    tools: Option<Vec<OpenAICompatibleTool>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    top_p: Option<f32>,
    parallel_tool_calls: Option<bool>,
}

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn openai_compatible_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    headers: HeaderMap,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleParams>,
) -> Result<Response<Body>, Error> {
    let params = (headers, openai_compatible_params).try_into()?;
    let response = inference(config, http_client, clickhouse_connection_info, params).await?;
    // TODO: Convert the response to the OpenAI compatible response
    unimplemented!()
}

impl TryFrom<(HeaderMap, OpenAICompatibleParams)> for Params<'static> {
    type Error = Error;
    fn try_from(
        (headers, openai_compatible_params): (HeaderMap, OpenAICompatibleParams),
    ) -> Result<Self, Self::Error> {
        let function_name = headers
            .get("function_name")
            .ok_or(Error::InvalidOpenAICompatibleRequest {
                message: "function_name header is required".to_string(),
            })?
            .to_str()
            .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                message: "function_name header is not valid UTF-8".to_string(),
            })?
            .to_string();
        let episode_id = headers
            .get("episode_id")
            .map(|h| {
                h.to_str()
                    .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                        message: "episode_id header is not valid UTF-8".to_string(),
                    })
                    .and_then(|s| {
                        Uuid::parse_str(s).map_err(|_| Error::InvalidEpisodeId {
                            message: "episode_id header is not a valid UUID".to_string(),
                        })
                    })
            })
            .transpose()?;
        let input = Input::from(openai_compatible_params.messages);
        let chat_completion_inference_params = ChatCompletionInferenceParams {
            temperature: openai_compatible_params.temperature,
            max_tokens: openai_compatible_params.max_tokens,
            seed: openai_compatible_params.seed,
            top_p: openai_compatible_params.top_p,
            presence_penalty: openai_compatible_params.presence_penalty,
            frequency_penalty: openai_compatible_params.frequency_penalty,
        };
        let inference_params = InferenceParams {
            chat_completion: chat_completion_inference_params,
        };
        let variant_name = headers
            .get("variant_name")
            .map(|h| {
                h.to_str()
                    .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                        message: "variant_name header is not valid UTF-8".to_string(),
                    })
                    .map(|s| s.to_string())
            })
            .transpose()?;
        let dryrun = headers
            .get("dryrun")
            .map(|h| {
                h.to_str()
                    .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                        message: "dryrun header is not valid UTF-8".to_string(),
                    })
                    .and_then(|s| {
                        s.parse::<bool>()
                            .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                                message: "dryrun header is not a valid boolean".to_string(),
                            })
                    })
            })
            .transpose()?;
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: openai_compatible_params
                .tools
                .map(|tools| tools.into_iter().map(|tool| tool.into()).collect()),
            tool_choice: openai_compatible_params
                .tool_choice
                .map(|tool_choice| tool_choice.into()),
            parallel_tool_calls: openai_compatible_params.parallel_tool_calls,
        };
        let output_schema = match openai_compatible_params.response_format {
            Some(OpenAICompatibleResponseFormat::JsonObject { json_schema }) => json_schema,
            _ => None,
        };
        Ok(Params {
            function_name,
            episode_id,
            input,
            stream: openai_compatible_params.stream,
            params: inference_params,
            variant_name,
            dryrun,
            dynamic_tool_params,
            output_schema,
            credentials: InferenceCredentials::default(),
        })
    }
}

impl From<Vec<OpenAICompatibleMessage>> for Input {
    fn from(messages: Vec<OpenAICompatibleMessage>) -> Self {
        unimplemented!()
    }
}

impl From<OpenAICompatibleTool> for Tool {
    fn from(tool: OpenAICompatibleTool) -> Self {
        unimplemented!()
    }
}

impl From<ChatCompletionToolChoiceOption> for ToolChoice {
    fn from(tool_choice: ChatCompletionToolChoiceOption) -> Self {
        unimplemented!()
    }
}
