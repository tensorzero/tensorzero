use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fmt,
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::{
    error::{Error, ResultExt},
    jsonschema_util::JSONSchemaFromPath,
};

/// Data flow in TensorZero
///
/// The flow of an inference request through TensorZero can be viewed as a series of transformations between types.
/// Most of them are defined below.
///
/// A request is made that contains a list of InputMessages.
/// These are validated against the input schema of the Function
/// and then templated and transformed into InferenceRequestMessages for a particular Variant.
/// These InferenceRequestMessages are collected into a ModelInferenceRequest,
/// which should contain all information needed by a ModelProvider to perform the
/// inference that is called for.
///
/// Each provider transforms a ModelInferenceRequest into a provider-specific (private) inference request type
/// that is suitable for serialization directly into a request to the provider.
///
/// In both non-streaming and streaming inference, each ModelProvider recieves data from the provider in a
/// a (private) provider-specific format that is then transformed into a ModelInferenceResponse (non-streaming)
/// or a stream of ModelInferenceResponseChunks (streaming).
/// As a Variant might make use of multiple model inferences, we then combine
/// one or more ModelInferenceResponses into a single InferenceResponse (but we keep the original ModelInferenceResponses around).
/// In the non-streaming case, this InferenceResponse is serialized into the TensorZero response format.
/// In the streaming case we convert ModelInferenceResponseChunks into serialized InferenceResponseChunks to the client.
/// We then collect all the InferenceResponseChunks into an InferenceResponse for validation and storage after the fact.
///
/// Alongside the response, we also store information about what happened during the request.
/// For this we convert the InferenceResponse into an Inference and ModelInferences, which are written to ClickHouse asynchronously.

/// InputMessage and InputMessageRole are our representation of the input sent by the client
/// prior to any processing into LLM representations below.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMessageRole {
    System,
    User,
    Assistant,
    // TODO: add Tool
}

impl fmt::Display for InputMessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap_or_default())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputMessage {
    pub role: InputMessageRole,
    pub content: serde_json::Value,
}

/// Top-level TensorZero type for an inference request to a particular model.
/// This should contain all the information required to make a valid inference request
/// for a provider, except for information about what model to actually request,
/// and to convert it back to the appropriate response format.
/// An example of the latter is that we might have prepared a request with Tools available
/// but the client actually just wants a chat response.
#[derive(Builder, Clone, Debug, Default, PartialEq)]
#[builder(setter(into, strip_option), default)]
pub struct ModelInferenceRequest<'a> {
    pub messages: Vec<InferenceRequestMessage>,
    pub tools_available: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: bool,
    pub json_mode: bool,
    pub function_type: FunctionType,
    pub output_schema: Option<&'a Value>,
}

/// FunctionType denotes whether the request is a chat or tool call.
/// By keeping this enum separately from whether tools are available, we
/// allow the request to use tools to enforce an output schema without necessarily
/// exposing that to the client (unless they requested a tool call themselves).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum FunctionType {
    Chat,
    Tool,
}

/// The default FunctionType is Chat
impl Default for FunctionType {
    fn default() -> Self {
        FunctionType::Chat
    }
}

/// Most inference providers allow the user to force a tool to be used
/// and even specify which tool to be used.
///
/// This enum is used to denote this tool choice.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Tool(String), // Forces the LLM to call a particular tool, the String is the name of the Tool
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolType {
    Function,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub r#type: ToolType,
    pub description: Option<String>,
    pub name: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UserInferenceRequestMessage {
    pub content: String, // TODO: for now, we don't support image input. This would be the place to start.
}

#[derive(Clone, Debug, PartialEq)]
pub struct SystemInferenceRequestMessage {
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssistantInferenceRequestMessage {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolInferenceRequestMessage {
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InferenceRequestMessage {
    User(UserInferenceRequestMessage),
    System(SystemInferenceRequestMessage),
    Assistant(AssistantInferenceRequestMessage),
    Tool(ToolInferenceRequestMessage),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

// TODO: use this and write to DB somehow
#[derive(Clone, Debug, PartialEq)]
pub enum Latency {
    Streaming { ttft: Duration, ttd: Duration },
    NonStreaming { ttd: Duration },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInferenceResponse {
    pub id: Uuid,
    pub created: u64,
    pub content: Option<String>,
    #[serde(skip)]
    pub tool_calls: Option<Vec<ToolCall>>,
    pub raw: String,
    pub usage: Usage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelInference {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub input: String,
    pub output: String,
    pub raw_response: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl ModelInference {
    pub fn new(response: ModelInferenceResponse, input: String, inference_id: Uuid) -> Self {
        // TODO: deal with tools
        Self {
            id: Uuid::now_v7(),
            inference_id,
            input,
            output: response.content.unwrap_or_default(),
            raw_response: response.raw,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
        }
    }
}

impl ModelInferenceResponse {
    pub fn new(
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        raw: String,
        usage: Usage,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            content,
            tool_calls,
            raw,
            usage,
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct ChatInferenceResponse {
    inference_id: Uuid,
    created: u64,
    pub content: Option<Value>,
    pub raw_content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub usage: Usage,
    #[serde(skip_serializing)]
    pub model_inference_responses: Vec<ModelInferenceResponse>,
}

impl ChatInferenceResponse {
    pub fn new(
        inference_id: Uuid,
        created: u64,
        raw_content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        usage: Usage,
        model_inference_responses: Vec<ModelInferenceResponse>,
        output_schema: Option<&JSONSchemaFromPath>,
    ) -> Self {
        let content = Self::parse_output(raw_content.as_deref(), output_schema).ok_or_log();
        Self {
            inference_id,
            created,
            content,
            raw_content,
            tool_calls,
            usage,
            model_inference_responses,
        }
    }

    fn parse_output(
        content: Option<&str>,
        output_schema: Option<&JSONSchemaFromPath>,
    ) -> Result<Value, Error> {
        let content = content.ok_or(Error::OutputParsing {
            raw_output: "".to_string(),
            message: "Output parsing failed due to None content".to_string(),
        })?;
        match output_schema {
            Some(schema) => {
                let output_value =
                    serde_json::from_str(content).map_err(|e| Error::OutputParsing {
                        raw_output: content.to_string(),
                        message: e.to_string(),
                    })?;
                schema
                    .validate(&output_value)
                    .map_err(|e| Error::OutputValidation {
                        source: Box::new(e),
                    })?;
                Ok(output_value)
            }
            None => Ok(Value::String(content.to_string())),
        }
    }

    pub fn get_serialized_model_inferences(&self, input: &str) -> Vec<serde_json::Value> {
        self.model_inference_responses
            .iter()
            .map(|r| {
                let model_inference =
                    ModelInference::new(r.clone(), input.to_string(), self.inference_id);
                serde_json::to_value(model_inference).unwrap_or_default()
            })
            .collect()
    }
}

#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
}

// TODO: handle references and stuff
#[derive(Serialize, Debug)]
pub struct Inference {
    pub id: Uuid,
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: String,
    pub output: Option<Value>,
    pub raw_output: String,
}

impl Inference {
    pub fn new(
        inference_response: InferenceResponse,
        input: String,
        episode_id: Uuid,
        function_name: String,
        variant_name: String,
    ) -> Self {
        match inference_response {
            InferenceResponse::Chat(chat_response) => Self {
                id: chat_response.inference_id,
                function_name,
                variant_name,
                episode_id,
                input,
                raw_output: chat_response.raw_content.unwrap_or_default(),
                output: chat_response.content,
            },
        }
    }
}

// Function to get the current timestamp in seconds
fn current_timestamp() -> u64 {
    #[allow(clippy::expect_used)]
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatInferenceResponseChunk {
    pub inference_id: Uuid,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallChunk>>,
    pub created: u64,
    pub usage: Option<Usage>,
}

impl From<ModelInferenceResponseChunk> for ChatInferenceResponseChunk {
    fn from(chunk: ModelInferenceResponseChunk) -> Self {
        Self {
            inference_id: chunk.inference_id,
            content: chunk.content,
            tool_calls: chunk.tool_calls,
            created: chunk.created,
            usage: chunk.usage,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum InferenceResponseChunk {
    Chat(ChatInferenceResponseChunk),
}

#[derive(Debug, Clone)]
pub struct ModelInferenceResponseChunk {
    pub inference_id: Uuid,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallChunk>>,
    pub created: u64,
    pub usage: Option<Usage>,
    pub raw: String,
}

impl ModelInferenceResponseChunk {
    pub fn new(
        inference_id: Uuid,
        content: Option<String>,
        tool_calls: Option<Vec<ToolCallChunk>>,
        usage: Option<Usage>,
        raw: String,
    ) -> Self {
        Self {
            inference_id,
            content,
            tool_calls,
            created: current_timestamp(),
            usage,
            raw,
        }
    }
}

pub fn collect_chunks(
    value: Vec<ModelInferenceResponseChunk>,
    output_schema: Option<&JSONSchemaFromPath>,
) -> Result<InferenceResponse, Error> {
    // TODO: we will eventually need this to be per-inference-response-type
    // and sensitive to the type of variant and function being called.

    let inference_id = value
        .first()
        .ok_or(Error::TypeConversion {
            message: "Attempted to create an InferenceResponse from an empty response chunk vector"
                .to_string(),
        })?
        .inference_id;
    let created = value
        .first()
        .ok_or(Error::TypeConversion {
            message: "Attempted to create an InferenceResponse from an empty response chunk vector"
                .to_string(),
        })?
        .created;
    let mut content: Option<String> = None;
    let mut tool_calls: Option<Vec<ToolCall>> = None;
    let raw: String = value
        .iter()
        .map(|chunk| chunk.raw.as_str())
        .collect::<Vec<&str>>()
        .join("\n");
    let mut usage: Usage = Usage::default();
    for chunk in value {
        content = match content {
            Some(c) => Some(c + &chunk.content.unwrap_or_default()),
            None => chunk.content,
        };
        tool_calls = match tool_calls {
            Some(_) => {
                unimplemented!()
                // TODO: when we add this code back, make _ into mut t
                // for (j, tool_call) in chunk.tool_calls.unwrap_or_default().iter().enumerate() {
                //     if let Some(existing_tool_call) = t.get_mut(j) {
                //         existing_tool_call
                //             .arguments
                //             .push_str(tool_call.arguments.as_deref().unwrap_or_default());
                //     }
                // }
                // Some(t)
            }
            None => chunk.tool_calls.map(|tool_calls| {
                tool_calls
                    .into_iter()
                    .map(|tool_call| tool_call.into())
                    .collect()
            }),
        };
        if let Some(chunk_usage) = chunk.usage {
            usage.prompt_tokens = usage
                .prompt_tokens
                .saturating_add(chunk_usage.prompt_tokens);
            usage.completion_tokens = usage
                .completion_tokens
                .saturating_add(chunk_usage.completion_tokens);
        }
    }
    let model_response = ModelInferenceResponse::new(
        content.clone(),
        tool_calls.clone(),
        raw.clone(),
        usage.clone(),
    );

    Ok(InferenceResponse::Chat(ChatInferenceResponse::new(
        inference_id,
        created,
        content,
        tool_calls,
        usage,
        vec![model_response],
        output_schema,
    )))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallChunk {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

impl From<ToolCallChunk> for ToolCall {
    fn from(tool_call: ToolCallChunk) -> Self {
        // TODO: explicitly handle tools both for streaming and non-streaming
        // as well as for Chat and Tool-style Functions
        Self {
            id: tool_call.id.unwrap_or_default(),
            name: tool_call.name.unwrap_or_default(),
            arguments: tool_call.arguments.unwrap_or_default(),
        }
    }
}

pub type InferenceResponseStream =
    Pin<Box<dyn Stream<Item = Result<ModelInferenceResponseChunk, Error>> + Send>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_chat_inference_response() {
        // TODO: handle the tool call case here. For now, we will always set those values to None.
        // Case 1: No output schema
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let content = Some("Hello, world!".to_string());
        let tool_calls = None;
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
        };
        let model_inference_responses = vec![ModelInferenceResponse::new(
            content.clone(),
            tool_calls.clone(),
            "".to_string(),
            usage.clone(),
        )];
        let chat_inference_response = ChatInferenceResponse::new(
            inference_id,
            created,
            content.clone(),
            tool_calls.clone(),
            usage.clone(),
            model_inference_responses,
            None,
        );
        let parsed_content = content.as_ref().map(|c| Value::String(c.clone()));
        assert_eq!(chat_inference_response.inference_id, inference_id);
        assert_eq!(chat_inference_response.created, created);
        assert_eq!(chat_inference_response.content, parsed_content);
        assert_eq!(chat_inference_response.tool_calls, tool_calls);
        assert_eq!(chat_inference_response.usage, usage);

        // Case 2: a JSON string that passes validation
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let json_content = r#"{"name": "John", "age": 30}"#.to_string();
        let tool_calls = None;
        let usage = Usage {
            prompt_tokens: 15,
            completion_tokens: 25,
        };
        let model_inference_responses = vec![ModelInferenceResponse::new(
            Some(json_content.clone()),
            tool_calls.clone(),
            json_content.clone(),
            usage.clone(),
        )];

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let output_schema = JSONSchemaFromPath::from_value(&schema);

        let chat_inference_response = ChatInferenceResponse::new(
            inference_id,
            created,
            Some(json_content.clone()),
            tool_calls.clone(),
            usage.clone(),
            model_inference_responses,
            Some(&output_schema),
        );

        assert_eq!(chat_inference_response.inference_id, inference_id);
        assert_eq!(chat_inference_response.created, created);
        // Content will be the parsed Value if the JSON passes validation
        assert_eq!(
            chat_inference_response.content,
            Some(serde_json::from_str(&json_content).unwrap())
        );
        assert_eq!(chat_inference_response.raw_content, Some(json_content));
        assert_eq!(chat_inference_response.tool_calls, tool_calls);
        assert_eq!(chat_inference_response.usage, usage);

        // TODO: assert that the appropriate errors were logged in the next two test cases
        // Case 3: a JSON string that fails validation
        let invalid_json_content = r#"{"name": "John", "age": "thirty"}"#.to_string();
        let model_inference_responses = vec![ModelInferenceResponse::new(
            Some(invalid_json_content.clone()),
            None,
            invalid_json_content.clone(),
            usage.clone(),
        )];

        let chat_inference_response = ChatInferenceResponse::new(
            inference_id,
            created,
            Some(invalid_json_content.clone()),
            None,
            usage.clone(),
            model_inference_responses,
            Some(&output_schema),
        );

        assert_eq!(chat_inference_response.inference_id, inference_id);
        assert_eq!(chat_inference_response.created, created);
        // Content will be None if the validation fails
        assert_eq!(chat_inference_response.content, None);
        assert_eq!(
            chat_inference_response.raw_content,
            Some(invalid_json_content)
        );
        assert_eq!(chat_inference_response.tool_calls, None);
        assert_eq!(chat_inference_response.usage, usage);

        // Case 4: a malformed JSON
        let malformed_json = r#"{"name": "John", "age": 30,"#.to_string();
        let model_inference_responses = vec![ModelInferenceResponse::new(
            Some(malformed_json.clone()),
            None,
            malformed_json.clone(),
            usage.clone(),
        )];

        let chat_inference_response = ChatInferenceResponse::new(
            inference_id,
            created,
            Some(malformed_json.clone()),
            None,
            usage.clone(),
            model_inference_responses,
            Some(&output_schema),
        );

        assert_eq!(chat_inference_response.inference_id, inference_id);
        assert_eq!(chat_inference_response.created, created);
        // Content will be None if the JSON is malformed
        assert_eq!(chat_inference_response.content, None);
        assert_eq!(chat_inference_response.raw_content, Some(malformed_json));
        assert_eq!(chat_inference_response.tool_calls, None);
        assert_eq!(chat_inference_response.usage, usage);
    }

    #[test]
    fn test_collect_chunks() {
        // Test case 1: empty chunks (should error)
        let chunks = vec![];
        let output_schema = None;
        let result = collect_chunks(chunks, output_schema);
        assert_eq!(
            result.unwrap_err(),
            Error::TypeConversion {
                message:
                    "Attempted to create an InferenceResponse from an empty response chunk vector"
                        .to_string(),
            }
        );

        // Test case 2: non-empty chunks with no tool calls but content exists
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let content = Some("Hello,".to_string());
        let chunks = vec![
            ModelInferenceResponseChunk {
                inference_id,
                content,
                tool_calls: None,
                created,
                usage: None,
                raw: "{\"message\": \"Hello}".to_string(),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: Some(" world!".to_string()),
                tool_calls: None,
                created,
                usage: Some(Usage {
                    prompt_tokens: 2,
                    completion_tokens: 4,
                }),
                raw: ", world!\"}".to_string(),
            },
        ];
        let response = collect_chunks(chunks, None).unwrap();
        let InferenceResponse::Chat(chat_response) = response;
        assert_eq!(chat_response.inference_id, inference_id);
        assert_eq!(chat_response.created, created);
        assert_eq!(
            chat_response.content,
            Some(serde_json::Value::String("Hello, world!".to_string()))
        );
        assert_eq!(chat_response.raw_content, Some("Hello, world!".to_string()));
        assert_eq!(chat_response.tool_calls, None);
        assert_eq!(
            chat_response.usage,
            Usage {
                prompt_tokens: 2,
                completion_tokens: 4,
            }
        );

        // Test Case 3: a JSON string that passes validation and also include usage in each chunk
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let schema = JSONSchemaFromPath::from_value(&serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let usage1 = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        let usage2 = Usage {
            prompt_tokens: 5,
            completion_tokens: 10,
        };
        let chunks = vec![
            ModelInferenceResponseChunk {
                inference_id,
                content: Some("{\"name\":".to_string()),
                tool_calls: None,
                created,
                usage: Some(usage1.clone()),
                raw: "{\"name\":".to_string(),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: Some("\"John\",\"age\":30}".to_string()),
                tool_calls: None,
                created,
                usage: Some(usage2.clone()),
                raw: "\"John\",\"age\":30}".to_string(),
            },
        ];
        let response = collect_chunks(chunks, Some(&schema)).unwrap();
        let InferenceResponse::Chat(chat_response) = response;
        assert_eq!(chat_response.inference_id, inference_id);
        assert_eq!(chat_response.created, created);
        assert_eq!(
            chat_response.content,
            Some(serde_json::json!({"name": "John", "age": 30}))
        );
        assert_eq!(
            chat_response.raw_content,
            Some("{\"name\":\"John\",\"age\":30}".to_string())
        );
        assert_eq!(chat_response.tool_calls, None);
        assert_eq!(
            chat_response.usage,
            Usage {
                prompt_tokens: 15,
                completion_tokens: 15,
            }
        );

        // Test Case 4: a JSON string that fails validation and usage only in last chunk
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let schema = JSONSchemaFromPath::from_value(&serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        let chunks = vec![
            ModelInferenceResponseChunk {
                inference_id,
                content: Some("{\"name\":".to_string()),
                tool_calls: None,
                created,
                usage: Some(usage.clone()),
                raw: "{\"name\":".to_string(),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: Some("\"John\"}".to_string()),
                tool_calls: None,
                created,
                usage: None,
                raw: "\"John\"}".to_string(),
            },
        ];
        let result = collect_chunks(chunks, Some(&schema));
        assert!(result.is_ok());
        if let Ok(InferenceResponse::Chat(chat_response)) = result {
            assert_eq!(chat_response.inference_id, inference_id);
            assert_eq!(chat_response.created, created);
            // Content is None when we fail validation
            assert_eq!(chat_response.content, None);
            assert_eq!(
                chat_response.raw_content,
                Some("{\"name\":\"John\"}".to_string())
            );
            assert_eq!(chat_response.tool_calls, None);
            assert_eq!(chat_response.usage, usage);
        } else {
            unreachable!("Expected Ok(InferenceResponse::Chat), got {:?}", result);
        }

        // Test case 5: chunks with some None content
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let schema = JSONSchemaFromPath::from_value(&serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let usage = Usage {
            prompt_tokens: 15,
            completion_tokens: 10,
        };
        let chunks = vec![
            ModelInferenceResponseChunk {
                inference_id,
                content: Some("{\"name\":\"John\",".to_string()),
                tool_calls: None,
                created,
                usage: Some(usage.clone()),
                raw: "{\"name\":\"John\",".to_string(),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: None,
                tool_calls: None,
                created,
                usage: None,
                raw: "".to_string(),
            },
            ModelInferenceResponseChunk {
                inference_id,
                content: Some("\"age\":30}".to_string()),
                tool_calls: None,
                created,
                usage: None,
                raw: "\"age\":30}".to_string(),
            },
        ];
        let result = collect_chunks(chunks, Some(&schema));
        assert!(result.is_ok());
        if let Ok(InferenceResponse::Chat(chat_response)) = result {
            assert_eq!(chat_response.inference_id, inference_id);
            assert_eq!(chat_response.created, created);
            assert_eq!(
                chat_response.content,
                Some(serde_json::json!({"name": "John", "age": 30}))
            );
            assert_eq!(
                chat_response.raw_content,
                Some("{\"name\":\"John\",\"age\":30}".to_string())
            );
            assert_eq!(chat_response.tool_calls, None);
            assert_eq!(chat_response.usage, usage);
        } else {
            unreachable!("Expected Ok(InferenceResponse::Chat), got {:?}", result);
        }
    }
}
