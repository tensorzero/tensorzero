use axum::body::Body;
use axum::debug_handler;
use axum::extract::State;
use axum::response::Response;
use serde::Deserialize;
use serde_json::Value;

use crate::endpoints::inference::{inference_handler, Params};
use crate::error::Error;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};

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
    stream: Option<bool>,
    temperature: Option<f32>,
    tools: Option<Vec<OpenAICompatibleTool>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    parallel_tool_calls: Option<bool>,
}

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn openai_compatible_handler(
    state: AppState,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleParams>,
) -> Result<Response<Body>, Error> {
    unimplemented!();

    // let response = inference_handler(state, openai_compatible_params).await?;
    // Ok(response)
}
