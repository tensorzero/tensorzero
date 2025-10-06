//! Types and utilities for streaming inference responses

use crate::endpoints::inference::{InferenceIds, InferenceParams};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::{
    ContentBlockOutput, ContentBlockOutputType, FinishReason, FunctionConfigType, InferenceConfig,
    Latency, ModelInferenceResponse, ModelInferenceResponseWithMetadata, ProviderInferenceResponse,
    ProviderInferenceResponseArgs, RequestMessage, Text, Thought, ToolCall, Usage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::tool::{ToolCallChunk, ToolCallConfig};
use futures::stream::Peekable;
use futures::Stream;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProviderInferenceResponseChunk {
    pub content: Vec<ContentBlockChunk>,
    pub created: u64,
    pub usage: Option<Usage>,
    pub raw_response: String,
    pub latency: Duration,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockChunk {
    Text(TextChunk),
    ToolCall(ToolCallChunk),
    Thought(ThoughtChunk),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextChunk {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThoughtChunk {
    pub id: String,
    pub text: Option<String>,
    pub signature: Option<String>,
    /// See `Thought.provider_type`
    #[serde(
        rename = "_internal_provider_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_type: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatInferenceResultChunk {
    pub content: Vec<ContentBlockChunk>,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub latency: Duration,
    pub raw_response: String,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JsonInferenceResultChunk {
    pub raw: Option<String>,
    pub thought: Option<String>,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub latency: Duration,
    pub raw_response: String,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum InferenceResultChunk {
    Chat(ChatInferenceResultChunk),
    Json(JsonInferenceResultChunk),
}

impl InferenceResultChunk {
    pub fn latency(&self) -> Duration {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.latency,
            InferenceResultChunk::Json(chunk) => chunk.latency,
        }
    }

    pub fn usage(&self) -> Option<&Usage> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.usage.as_ref(),
            InferenceResultChunk::Json(chunk) => chunk.usage.as_ref(),
        }
    }

    pub fn raw_response(&self) -> &str {
        match self {
            InferenceResultChunk::Chat(chunk) => &chunk.raw_response,
            InferenceResultChunk::Json(chunk) => &chunk.raw_response,
        }
    }

    pub fn finish_reason(&self) -> Option<&FinishReason> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.finish_reason.as_ref(),
            InferenceResultChunk::Json(chunk) => chunk.finish_reason.as_ref(),
        }
    }

    pub fn new(chunk: ProviderInferenceResponseChunk, function: FunctionConfigType) -> Self {
        match function {
            FunctionConfigType::Chat => Self::Chat(chunk.into()),
            FunctionConfigType::Json => Self::Json(chunk.into()),
        }
    }
}

impl From<ProviderInferenceResponseChunk> for ChatInferenceResultChunk {
    fn from(chunk: ProviderInferenceResponseChunk) -> Self {
        Self {
            content: chunk.content,
            created: chunk.created,
            usage: chunk.usage,
            latency: chunk.latency,
            finish_reason: chunk.finish_reason,
            raw_response: chunk.raw_response,
        }
    }
}

/// We use best-effort to reconstruct the raw response for JSON functions
/// They might either return a ToolCallChunk or a TextChunk
/// We take the string from either of these (from the last block if there are multiple)
/// and use that as the raw response.
impl From<ProviderInferenceResponseChunk> for JsonInferenceResultChunk {
    fn from(chunk: ProviderInferenceResponseChunk) -> Self {
        let mut raw = None;
        let mut thought = None;
        for content in chunk.content.into_iter() {
            match content {
                ContentBlockChunk::ToolCall(tool_call) => {
                    raw = Some(tool_call.raw_arguments.to_owned());
                }
                ContentBlockChunk::Text(text_chunk) => raw = Some(text_chunk.text.to_owned()),
                ContentBlockChunk::Thought(thought_chunk) => {
                    thought = thought_chunk.text;
                }
            }
        }
        Self {
            raw,
            thought,
            created: chunk.created,
            usage: chunk.usage,
            latency: chunk.latency,
            raw_response: chunk.raw_response,
            finish_reason: chunk.finish_reason,
        }
    }
}

// Define the CollectChunksArgs struct with existing and new fields
pub struct CollectChunksArgs<'a, 'b> {
    pub value: Vec<InferenceResultChunk>,
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub function: Arc<FunctionConfig>,
    pub model_name: Arc<str>,
    pub model_provider_name: Arc<str>,
    pub raw_request: String,
    /// We may sometimes construct a fake stream from a non-streaming response
    /// (e.g. in `mixture_of_n` if we have a successful non-streaming candidate, but
    /// a streaming fuser request fails).
    /// In this case, we want to store the original `raw_response`, instead of building
    /// it up from the chunks.
    pub raw_response: Option<String>,
    pub inference_params: InferenceParams,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub function_name: &'b str,
    pub variant_name: &'b str,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
    pub templates: &'b TemplateConfig<'a>,
    pub tool_config: Option<&'b ToolCallConfig>,
    pub cached: bool,
    pub extra_body: UnfilteredInferenceExtraBody,
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    pub fetch_and_encode_input_files_before_inference: bool,
}

// Modify the collect_chunks function to accept CollectChunksArgs
// 'a ends up as static and 'b ends up as stack allocated in the caller (endpoints::inference::create_stream)
pub async fn collect_chunks(
    args: CollectChunksArgs<'_, '_>,
) -> Result<crate::inference::types::InferenceResult, Error> {
    let CollectChunksArgs {
        value,
        inference_id,
        episode_id,
        function,
        model_name,
        model_provider_name,
        raw_request,
        raw_response,
        inference_params,
        system,
        input_messages,
        function_name,
        variant_name,
        dynamic_output_schema,
        templates,
        tool_config,
        cached,
        fetch_and_encode_input_files_before_inference,
        extra_body,
        extra_headers,
    } = args;

    // NOTE: We will eventually need this to be per-inference-response-type and sensitive to the type of variant and function being called.
    // We preserve the order of chunks in the stream when combining them into `ContentBlockOutput`, except when
    // the same id is used by non-adjacent blocks.
    // For example, the following chunks:
    // `[TextChunk(id=0, content="Hello ""), ThoughtChunk(id=0, content=Something), TextChunk(id=0, content=World)]``
    // will be collected into the content block list: `[Text("Hello World"), Thought("Something"))]`
    //
    // All chunks with the same type and id (in this case, TextChunk id=0) are combined into a single content
    // block at the first occurrence of that type and id.
    // We use an 'IndexMap' to preserve the insertion order, so that newly-seen type/id combinations
    // are not reordered.
    let mut blocks: IndexMap<(ContentBlockOutputType, String), ContentBlockOutput> =
        IndexMap::new();
    // If the variant gave us an explicit 'raw_response', use that.
    // Otherwise, concatenate the raw_response from each chunk.
    let raw_response = raw_response.unwrap_or_else(|| {
        value
            .iter()
            .map(InferenceResultChunk::raw_response)
            .collect::<Vec<&str>>()
            .join("\n")
    });
    let mut usage: Usage = Usage::default();
    let mut ttft: Option<Duration> = None;
    let response_time = value
        .last()
        .ok_or_else(|| {
            Error::new(ErrorDetails::TypeConversion {
                message:
                    "Attempted to create an InferenceResult from an empty response chunk vector"
                        .to_string(),
            })
        })?
        .latency();
    // We'll take the finish reason from the last chunk
    let mut finish_reason: Option<FinishReason> = None;
    for chunk in value {
        if let Some(chunk_usage) = chunk.usage() {
            usage.input_tokens = usage.input_tokens.saturating_add(chunk_usage.input_tokens);
            usage.output_tokens = usage
                .output_tokens
                .saturating_add(chunk_usage.output_tokens);
        }
        match chunk {
            InferenceResultChunk::Chat(chunk) => {
                if let Some(chunk_finish_reason) = chunk.finish_reason {
                    finish_reason = Some(chunk_finish_reason);
                }
                for content in chunk.content {
                    match content {
                        ContentBlockChunk::Text(text) => {
                            handle_textual_content_block(
                                &mut blocks,
                                (ContentBlockOutputType::Text, text.id),
                                text.text,
                                &mut ttft,
                                chunk.latency,
                                Into::into,
                                |block, text| {
                                    if let ContentBlockOutput::Text(Text {
                                        text: existing_text,
                                    }) = block
                                    {
                                        existing_text.push_str(text);
                                    }
                                },
                            );
                        }
                        ContentBlockChunk::Thought(thought) => {
                            // We check for both 'text' and 'signature', in case a provider produces
                            // both in the same chunk.
                            // These two cases update different fields ('text' vs 'signature') on the
                            // thought with id 'thought.id' - this is how providers attach a signature
                            // to a thought.
                            if let Some(text) = thought.text {
                                handle_textual_content_block(
                                    &mut blocks,
                                    (ContentBlockOutputType::Thought, thought.id.clone()),
                                    text,
                                    &mut ttft,
                                    chunk.latency,
                                    |text| {
                                        ContentBlockOutput::Thought(Thought {
                                            text: Some(text),
                                            signature: None,
                                            provider_type: thought.provider_type.clone(),
                                        })
                                    },
                                    |block, text| {
                                        if let ContentBlockOutput::Thought(thought) = block {
                                            thought.text.get_or_insert_default().push_str(text);
                                        }
                                    },
                                );
                            }
                            if let Some(signature) = thought.signature {
                                handle_textual_content_block(
                                    &mut blocks,
                                    (ContentBlockOutputType::Thought, thought.id),
                                    signature,
                                    &mut ttft,
                                    chunk.latency,
                                    |signature| {
                                        ContentBlockOutput::Thought(Thought {
                                            text: None,
                                            signature: Some(signature),
                                            provider_type: thought.provider_type,
                                        })
                                    },
                                    |block, signature| {
                                        if let ContentBlockOutput::Thought(thought) = block {
                                            match &mut thought.signature {
                                                Some(existing) => existing.push_str(signature),
                                                None => {
                                                    thought.signature = Some(signature.to_string());
                                                }
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        ContentBlockChunk::ToolCall(tool_call) => {
                            match blocks
                                .get_mut(&(ContentBlockOutputType::ToolCall, tool_call.id.clone()))
                            {
                                // If there is already a tool call block with this id, append to it
                                Some(ContentBlockOutput::ToolCall(existing_tool_call)) => {
                                    // We assume that the ID is present and complete in the first chunk
                                    // and that the name and arguments are accumulated with more chunks.
                                    if let Some(raw_name) = tool_call.raw_name {
                                        existing_tool_call.name.push_str(&raw_name);
                                    }
                                    existing_tool_call
                                        .arguments
                                        .push_str(&tool_call.raw_arguments);
                                }
                                // If there is no tool call block, create one
                                _ => {
                                    if ttft.is_none() {
                                        ttft = Some(chunk.latency);
                                    }
                                    blocks.insert(
                                        (ContentBlockOutputType::ToolCall, tool_call.id.clone()),
                                        ContentBlockOutput::ToolCall(tool_call_chunk_to_tool_call(
                                            tool_call,
                                        )),
                                    );
                                }
                            }
                        }
                    }
                }
            }
            InferenceResultChunk::Json(chunk) => {
                if let Some(chunk_finish_reason) = chunk.finish_reason {
                    finish_reason = Some(chunk_finish_reason);
                }
                match blocks.get_mut(&(ContentBlockOutputType::Text, String::new())) {
                    // If there is already a text block, append to it
                    Some(ContentBlockOutput::Text(Text {
                        text: existing_text,
                    })) => {
                        if let Some(raw) = chunk.raw {
                            existing_text.push_str(&raw);
                        }
                    }
                    // If there is no text block, create one
                    _ => {
                        // We put this here and below rather than in the loop start because we
                        // only want to set TTFT if there is some real content
                        if ttft.is_none() {
                            ttft = Some(chunk.latency);
                        }
                        if let Some(raw) = chunk.raw {
                            blocks
                                .insert((ContentBlockOutputType::Text, String::new()), raw.into());
                        }
                    }
                }
                if let Some(thought) = chunk.thought {
                    match blocks.get_mut(&(ContentBlockOutputType::Thought, String::new())) {
                        // If there is already a thought block, append to it
                        Some(ContentBlockOutput::Thought(existing_thought)) => {
                            existing_thought
                                .text
                                .get_or_insert_default()
                                .push_str(&thought);
                        }
                        // If there is no thought block, create one
                        _ => {
                            blocks.insert(
                                (ContentBlockOutputType::Thought, String::new()),
                                ContentBlockOutput::Thought(Thought {
                                    text: Some(thought),
                                    signature: None,
                                    provider_type: None,
                                }),
                            );
                        }
                    }
                }
            }
        }
    }
    let ttft = ttft.ok_or_else(|| {
        Error::new(ErrorDetails::TypeConversion {
            message: "Never got TTFT because there was never content in the response.".to_string(),
        })
    })?;
    let latency = Latency::Streaming {
        ttft,
        response_time,
    };
    let content_blocks: Vec<_> = blocks.into_values().collect();
    let model_response = ProviderInferenceResponse::new(ProviderInferenceResponseArgs {
        output: content_blocks.clone(),
        system,
        input_messages,
        raw_request,
        raw_response,
        usage,
        latency: latency.clone(),
        finish_reason,
    });
    let model_inference_response =
        ModelInferenceResponse::new(model_response, model_provider_name, cached);
    let original_response = model_inference_response.raw_response.clone();
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, model_name);
    let inference_config = InferenceConfig {
        ids: InferenceIds {
            inference_id,
            episode_id,
        },
        function_name,
        variant_name,
        tool_config,
        templates,
        dynamic_output_schema: dynamic_output_schema.as_ref(),
        fetch_and_encode_input_files_before_inference,
        extra_body: Cow::Borrowed(&extra_body),
        extra_headers: Cow::Borrowed(&extra_headers),
        extra_cache_key: None,
    };
    function
        .prepare_response(
            inference_id,
            content_blocks,
            vec![model_inference_result],
            &inference_config,
            inference_params,
            Some(original_response),
        )
        .await
}

fn tool_call_chunk_to_tool_call(tool_call: ToolCallChunk) -> ToolCall {
    ToolCall {
        id: tool_call.id,
        name: tool_call.raw_name.unwrap_or_default(), // Since we are accumulating tool call names, we can start with "" if missing and hopefully accumulate with more chunks.
        arguments: tool_call.raw_arguments,
    }
}

// We use a very specific combination of `Pin` and `Peekable` here, due to a combination of several requirements:
// * Inside of a model provider (e.g. anthropic), we may want to peek and modify the first chunk
//   to fix the start of a JSON response.
// * Outside of a model provider, we always want to peek at the first chunk to make sure that the HTTP request
//   actually succeeded.
// * The model providers produce distinct stream types (arising from different `async_stream` calls), so we
//   need to use a trait object.
//
// Combining all of these requirements, we need to wrap the entire `Pin<Box<dyn Stream>>` in `Peekable`.
// The `Peekable` type needs to be 'visible' (not erased inside the trait object), so that we can
// check the first chunk with 'peek()' outside of a model provider implementation. While we could have
// two `Peekable` types (one erased inside the trait object, one visible outside), this would add
// additional runtime overhead, and make things more difficult to reason about.
//
// We cannot write `Peekable<dyn Stream>`, since `Peekable` does not support the special unsized coercions that standard
// library types support (e.g. `Box<MyStreamType>` -> `Box<dyn Stream>`)'
// We also cannot write `Pin<Peekable>`, since the argument to `Pin` needs to implement `DerefMut`.
// This gives us the particular combination of types below.
//
// We split this into an 'inner' type to make it easier to write `stream_<provider>` functions
// (e.g. `stream_anthropic`). These functions can return `ProviderInferenceResponseStreamInner`,
// which will cause the compiler to coerce `Pin<Box<SomeUnderlyingStreamType>>` into
// `Pin<Box<dyn Stream>>`. The caller than then write `stream_anthropic().peekable()` to produce
// a `PeekableProviderInferenceResponseStream`. If we attempted to directly return a `Peekable<Pin<Box<dyn Stream>>>`,
// the compiler would fail to coerce `Peekable<Pin<Box<SomeUnderlyingStreamType>>>` into `Peekable<Pin<Box<dyn Stream>>>`.
// (due to the fact that unsized coercions are not supported on `Peekable` or other user-defined types).
// This would require `stream_<provider>` functions to first introduce a local variable with the correct
// `Pin<Box<dyn Stream>>` type, and then call `.peekable()` on that.
pub type ProviderInferenceResponseStreamInner =
    Pin<Box<dyn Stream<Item = Result<ProviderInferenceResponseChunk, Error>> + Send>>;

pub type PeekableProviderInferenceResponseStream = Peekable<ProviderInferenceResponseStreamInner>;

pub type InferenceResultStream =
    Pin<Box<dyn Stream<Item = Result<InferenceResultChunk, Error>> + Send>>;

/// Handles a textual content block (text or thought)
/// It checks if there is already a block with the given id, and if so, appends the text to it.
/// Otherwise, it creates a new block and inserts it into the map.
/// It also updates the TTFT if it hasn't been set
fn handle_textual_content_block<F, A>(
    blocks: &mut IndexMap<(ContentBlockOutputType, String), ContentBlockOutput>,
    key: (ContentBlockOutputType, String),
    text: String,
    ttft: &mut Option<Duration>,
    chunk_latency: Duration,
    create_block: F,
    append_text: A,
) where
    F: FnOnce(String) -> ContentBlockOutput,
    A: FnOnce(&mut ContentBlockOutput, &str),
{
    match blocks.get_mut(&key) {
        // If there is already a block, append to it
        Some(existing_block) => append_text(existing_block, &text),
        // If there is no block, create one
        _ => {
            // We only want to set TTFT if there is some real content
            if ttft.is_none() {
                *ttft = Some(chunk_latency);
            }
            if !text.is_empty() {
                blocks.insert(key, create_block(text));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::{ContentBlockOutputType, Text, Thought};

    #[test]
    fn test_handle_textual_content_block() {
        let mut blocks: IndexMap<(ContentBlockOutputType, String), ContentBlockOutput> =
            IndexMap::new();
        let mut ttft: Option<Duration> = None;
        let chunk_latency = Duration::from_millis(100);

        // Test case 1: Create new text block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Text, "1".to_string()),
            "Hello".to_string(),
            &mut ttft,
            chunk_latency,
            |text| ContentBlockOutput::Text(Text { text }),
            |block, text| {
                if let ContentBlockOutput::Text(Text {
                    text: existing_text,
                }) = block
                {
                    existing_text.push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(ttft, Some(chunk_latency));
        match blocks
            .get(&(ContentBlockOutputType::Text, "1".to_string()))
            .unwrap()
        {
            ContentBlockOutput::Text(Text { text }) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text block"),
        }

        // Test case 2: Append to existing text block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Text, "1".to_string()),
            " World".to_string(),
            &mut ttft,
            chunk_latency,
            |text| ContentBlockOutput::Text(Text { text }),
            |block, text| {
                if let ContentBlockOutput::Text(Text {
                    text: existing_text,
                }) = block
                {
                    existing_text.push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 1);
        match blocks
            .get(&(ContentBlockOutputType::Text, "1".to_string()))
            .unwrap()
        {
            ContentBlockOutput::Text(Text { text }) => assert_eq!(text, "Hello World"),
            _ => panic!("Expected text block"),
        }

        // Test case 3: Empty text should not create block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Text, "2".to_string()),
            String::new(),
            &mut ttft,
            chunk_latency,
            |text| ContentBlockOutput::Text(Text { text }),
            |block, text| {
                if let ContentBlockOutput::Text(Text {
                    text: existing_text,
                }) = block
                {
                    existing_text.push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 1); // Should still only have the first block

        // Test case 4: Create thought block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Thought, "3".to_string()),
            "Thinking...".to_string(),
            &mut ttft,
            chunk_latency,
            |text| {
                ContentBlockOutput::Thought(Thought {
                    text: Some(text),
                    signature: None,
                    provider_type: None,
                })
            },
            |block, text| {
                if let ContentBlockOutput::Thought(thought) = block {
                    thought.text.get_or_insert_default().push_str(text);
                }
            },
        );

        assert_eq!(blocks.len(), 2);
        match blocks
            .get(&(ContentBlockOutputType::Thought, "3".to_string()))
            .unwrap()
        {
            ContentBlockOutput::Thought(Thought {
                text,
                signature: _,
                provider_type: _,
            }) => {
                assert_eq!(text, &Some("Thinking...".to_string()));
            }
            _ => panic!("Expected thought block"),
        }
    }
}
