//! Types and utilities for streaming inference responses

use crate::endpoints::inference::{InferenceIds, InferenceParams};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::{
    ContentBlockOutput, ContentBlockOutputType, FinishReason, FunctionConfigType, InferenceConfig,
    Latency, ModelInferenceResponse, ModelInferenceResponseWithMetadata, ProviderInferenceResponse,
    ProviderInferenceResponseArgs, RawResponseEntry, RawUsageEntry, RequestMessage, Text, Thought,
    ThoughtSummaryBlock, ToolCall, Unknown, Usage,
};
use crate::jsonschema_util::JSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::tool::{ToolCallChunk, ToolCallConfig};
use futures::Stream;
use futures::stream::Peekable;
use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

use super::InferenceResult;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProviderInferenceResponseChunk {
    pub content: Vec<ContentBlockChunk>,
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    pub raw_response: String,
    /// Time elapsed between making the request to the model provider and receiving this chunk.
    /// Important: this is NOT latency from the start of the TensorZero request.
    pub provider_latency: Duration,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentBlockChunk {
    Text(TextChunk),
    ToolCall(ToolCallChunk),
    Thought(ThoughtChunk),
    Unknown(UnknownChunk),
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
    pub summary_id: Option<String>,
    pub summary_text: Option<String>,

    /// See `Thought.provider_type`
    #[serde(
        // This alias is written to the database, so we cannot remove it.
        alias = "_internal_provider_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_type: Option<String>,
    /// Provider-specific opaque data for multi-turn reasoning support.
    /// See `Thought.extra_data`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_data: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnknownChunk {
    pub id: String,
    pub data: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_name: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatInferenceResultChunk {
    pub content: Vec<ContentBlockChunk>,
    pub usage: Option<Usage>,
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// Raw responses from previous model inferences (e.g., best-of-n candidates).
    /// Used for artificial chunks emitted at stream start.
    pub raw_response: Option<Vec<RawResponseEntry>>,
    /// Time elapsed between making the request to the model provider and receiving this chunk.
    /// Important: this is NOT latency from the start of the TensorZero request.
    /// None for artificial chunks created by TensorZero (e.g., usage/finish_reason chunks).
    pub provider_latency: Option<Duration>,
    /// Raw response string for the current chunk from the model provider.
    /// Empty for fake streams (non-streaming converted to streaming).
    pub raw_chunk: String,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct JsonInferenceResultChunk {
    pub raw: Option<String>,
    /// Thought content from reasoning models (e.g., Claude's extended thinking).
    /// Populated during streaming from `ContentBlockChunk::Thought` variants.
    /// Used internally to collect the final `Thought` block; not streamed directly to clients.
    pub thought_chunks: Vec<ThoughtChunk>,
    pub usage: Option<Usage>,
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// Raw responses from previous model inferences (e.g., best-of-n candidates).
    /// Used for artificial chunks emitted at stream start.
    pub raw_response: Option<Vec<RawResponseEntry>>,
    /// Time elapsed between making the request to the model provider and receiving this chunk.
    /// Important: this is NOT latency from the start of the TensorZero request.
    /// None for artificial chunks created by TensorZero (e.g., usage/finish_reason chunks).
    pub provider_latency: Option<Duration>,
    /// Raw response string for the current chunk from the model provider.
    /// Empty for fake streams (non-streaming converted to streaming).
    pub raw_chunk: String,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InferenceResultChunk {
    Chat(ChatInferenceResultChunk),
    Json(JsonInferenceResultChunk),
}

impl InferenceResultChunk {
    pub fn provider_latency(&self) -> Option<Duration> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.provider_latency,
            InferenceResultChunk::Json(chunk) => chunk.provider_latency,
        }
    }

    pub fn usage(&self) -> Option<&Usage> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.usage.as_ref(),
            InferenceResultChunk::Json(chunk) => chunk.usage.as_ref(),
        }
    }

    pub fn set_usage(&mut self, usage: Option<Usage>) {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.usage = usage,
            InferenceResultChunk::Json(chunk) => chunk.usage = usage,
        }
    }

    pub fn raw_usage(&self) -> Option<&Vec<RawUsageEntry>> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.raw_usage.as_ref(),
            InferenceResultChunk::Json(chunk) => chunk.raw_usage.as_ref(),
        }
    }

    pub fn raw_response(&self) -> Option<&Vec<RawResponseEntry>> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.raw_response.as_ref(),
            InferenceResultChunk::Json(chunk) => chunk.raw_response.as_ref(),
        }
    }

    pub fn raw_chunk(&self) -> &str {
        match self {
            InferenceResultChunk::Chat(chunk) => &chunk.raw_chunk,
            InferenceResultChunk::Json(chunk) => &chunk.raw_chunk,
        }
    }

    pub fn finish_reason(&self) -> Option<&FinishReason> {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.finish_reason.as_ref(),
            InferenceResultChunk::Json(chunk) => chunk.finish_reason.as_ref(),
        }
    }

    pub fn set_finish_reason(&mut self, finish_reason: Option<FinishReason>) {
        match self {
            InferenceResultChunk::Chat(chunk) => chunk.finish_reason = finish_reason,
            InferenceResultChunk::Json(chunk) => chunk.finish_reason = finish_reason,
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
            usage: chunk.usage,
            raw_usage: chunk.raw_usage,
            raw_response: None, // Only populated via artificial chunks from TensorZero
            provider_latency: Some(chunk.provider_latency),
            finish_reason: chunk.finish_reason,
            raw_chunk: chunk.raw_response,
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
        let mut thought_chunks = Vec::new();
        for content in chunk.content {
            match content {
                ContentBlockChunk::ToolCall(tool_call) => {
                    raw = Some(tool_call.raw_arguments.to_owned());
                }
                ContentBlockChunk::Text(text_chunk) => raw = Some(text_chunk.text.to_owned()),
                ContentBlockChunk::Thought(thought_chunk) => {
                    thought_chunks.push(thought_chunk);
                }
                ContentBlockChunk::Unknown(_) => {
                    // Unknown chunks are ignored for JSON functions
                    // They don't contribute to the JSON output
                }
            }
        }
        Self {
            raw,
            thought_chunks,
            usage: chunk.usage,
            raw_usage: chunk.raw_usage,
            raw_response: None, // Only populated via artificial chunks from TensorZero
            provider_latency: Some(chunk.provider_latency),
            raw_chunk: chunk.raw_response,
            finish_reason: chunk.finish_reason,
        }
    }
}

pub struct CollectChunksArgs {
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
    pub function_name: Arc<str>,
    pub variant_name: Arc<str>,
    pub dynamic_output_schema: Option<Arc<JSONSchema>>,
    pub templates: Arc<TemplateConfig<'static>>,
    pub tool_config: Option<Arc<ToolCallConfig>>,
    pub cached: bool,
    pub extra_body: UnfilteredInferenceExtraBody,
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    pub fetch_and_encode_input_files_before_inference: bool,
    /// Pre-generated model inference ID for streaming (to match raw_usage entries)
    pub model_inference_id: Uuid,
    /// The `Usage` for the final streamed inference
    pub model_inference_usage: Usage,
    /// The final `FinishReason` we streamed to the client
    pub finish_reason: Option<FinishReason>,
}

pub async fn collect_chunks(args: CollectChunksArgs) -> Result<InferenceResult, Error> {
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
        model_inference_id,
        model_inference_usage,
        finish_reason,
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
    // Otherwise, concatenate the raw_chunk from each chunk.
    let raw_response = raw_response.unwrap_or_else(|| {
        value
            .iter()
            .map(InferenceResultChunk::raw_chunk)
            .filter(|s| !s.is_empty()) // remove artificial chunks (e.g. our final chunk with usage and finish reason)
            .collect::<Vec<&str>>()
            .join("\n")
    });
    // Collect raw_usage entries from chunks (relay or provider streaming)
    let mut raw_usage: Option<Vec<RawUsageEntry>> = None;

    // Extract provider-level TTFT from the first real chunk (with provider_latency set).
    // This is used for ModelInference timing.
    let provider_ttft = value.iter().find_map(|chunk| chunk.provider_latency());

    // Extract provider-level response_time from the last real chunk (with provider_latency set).
    // This is used for ModelInference timing.
    let provider_response_time = value
        .iter()
        .rev()
        .find_map(|chunk| chunk.provider_latency());

    // Maps a chunk id to a map of summary ids to summary texts
    // This is used to build up a thought summary list for each thought,
    // which is used to construct the final 'summary' field on Thought.
    let mut thought_summaries: IndexMap<String, IndexSet<String>> = IndexMap::new();

    for chunk in value {
        // Only collect raw_usage from real provider chunks (not artificial chunks).
        // Artificial chunks have provider_latency: None.
        if chunk.provider_latency().is_some()
            && let Some(chunk_raw_usage) = chunk.raw_usage()
        {
            raw_usage
                .get_or_insert_with(Vec::new)
                .extend(chunk_raw_usage.iter().cloned());
        }
        match chunk {
            InferenceResultChunk::Chat(chunk) => {
                for content in chunk.content {
                    match content {
                        ContentBlockChunk::Text(text) => {
                            handle_textual_content_block(
                                &mut blocks,
                                (ContentBlockOutputType::Text, text.id),
                                text.text,
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
                            let ThoughtChunk {
                                id,
                                text,
                                signature,
                                summary_id,
                                summary_text,
                                provider_type,
                                extra_data,
                            } = thought;
                            // We check for both 'text' and 'signature', in case a provider produces
                            // both in the same chunk.
                            // These two cases update different fields ('text' vs 'signature') on the
                            // thought with id 'thought.id' - this is how providers attach a signature
                            // to a thought.
                            if let Some(text) = text {
                                handle_textual_content_block(
                                    &mut blocks,
                                    (ContentBlockOutputType::Thought, id.clone()),
                                    text,
                                    |text| {
                                        ContentBlockOutput::Thought(Thought {
                                            text: Some(text),
                                            signature: None,
                                            summary: None,
                                            provider_type: provider_type.clone(),
                                            extra_data: extra_data.clone(),
                                        })
                                    },
                                    |block, text| {
                                        if let ContentBlockOutput::Thought(thought) = block {
                                            thought.text.get_or_insert_default().push_str(text);
                                        }
                                    },
                                );
                            }
                            if let Some(signature) = signature {
                                handle_textual_content_block(
                                    &mut blocks,
                                    (ContentBlockOutputType::Thought, id.clone()),
                                    signature,
                                    |signature| {
                                        ContentBlockOutput::Thought(Thought {
                                            text: None,
                                            signature: Some(signature),
                                            summary: None,
                                            provider_type: provider_type.clone(),
                                            extra_data: extra_data.clone(),
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
                            if summary_id.is_some() && summary_text.is_none() {
                                tracing::error!(
                                    "Summary id is present but summary text is missing for thought {}",
                                    id
                                );
                            }
                            if summary_id.is_none() && summary_text.is_some() {
                                tracing::error!(
                                    "Summary text is present but summary id is missing for thought {}",
                                    id
                                );
                            }
                            if let (Some(summary_id), Some(summary_text)) =
                                (summary_id, summary_text)
                            {
                                // Determine the index to use for our thought summary id
                                // The string ids are mapped to indices in the order that we first encounter them.
                                // There's an edge case here - if see a chunk with summary id "1", then "0",
                                // then the concatenated summary with id "1" will come first in the array
                                // (even though the underlying provider was presumably providing integer ids).
                                // Thought summaries are not used by models in input, so this should be fine.
                                let summary_set = thought_summaries.entry(id.clone()).or_default();
                                let (index, _) = summary_set.insert_full(summary_id.clone());

                                handle_textual_content_block(
                                    &mut blocks,
                                    (ContentBlockOutputType::Thought, id),
                                    summary_text,
                                    |summary_text| {
                                        ContentBlockOutput::Thought(Thought {
                                            text: None,
                                            signature: None,
                                            summary: Some(vec![ThoughtSummaryBlock::SummaryText {
                                                text: summary_text,
                                            }]),
                                            provider_type,
                                            extra_data,
                                        })
                                    },
                                    |block, summary_text| {
                                        if let ContentBlockOutput::Thought(thought) = block {
                                            match &mut thought.summary {
                                                Some(existing) => {
                                                    // Create entries in the array for the missing index
                                                    // (since we get the index from our `IndexSet`, there should be at most one missing entry)
                                                    if index >= existing.len() {
                                                        existing.resize(
                                                            index + 1,
                                                            ThoughtSummaryBlock::SummaryText {
                                                                text: String::new(),
                                                            },
                                                        );
                                                    }
                                                    // Concatenate our summary text to the (possibly freshly created) summary entry.
                                                    match &mut existing[index] {
                                                        ThoughtSummaryBlock::SummaryText {
                                                            text,
                                                        } => {
                                                            text.push_str(summary_text);
                                                        }
                                                    }
                                                }
                                                None => {
                                                    thought.summary = Some(vec![
                                                        ThoughtSummaryBlock::SummaryText {
                                                            text: summary_text.to_string(),
                                                        },
                                                    ]);
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
                                    blocks.insert(
                                        (ContentBlockOutputType::ToolCall, tool_call.id.clone()),
                                        ContentBlockOutput::ToolCall(tool_call_chunk_to_tool_call(
                                            tool_call,
                                        )),
                                    );
                                }
                            }
                        }
                        ContentBlockChunk::Unknown(UnknownChunk {
                            id,
                            data,
                            model_name,
                            provider_name,
                        }) => {
                            blocks.insert(
                                (ContentBlockOutputType::Unknown, id.clone()),
                                ContentBlockOutput::Unknown(Unknown {
                                    data: data.clone(),
                                    model_name: model_name.clone(),
                                    provider_name: provider_name.clone(),
                                }),
                            );
                        }
                    }
                }
            }
            InferenceResultChunk::Json(chunk) => {
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
                        if let Some(raw) = chunk.raw {
                            blocks
                                .insert((ContentBlockOutputType::Text, String::new()), raw.into());
                        }
                    }
                }
                for thought_chunk in chunk.thought_chunks {
                    let ThoughtChunk {
                        id: _,
                        text,
                        signature,
                        summary_id,
                        summary_text,
                        provider_type,
                        extra_data,
                    } = thought_chunk;

                    // Handle text if present
                    if let Some(text) = text {
                        match blocks.get_mut(&(ContentBlockOutputType::Thought, String::new())) {
                            Some(ContentBlockOutput::Thought(existing_thought)) => {
                                existing_thought
                                    .text
                                    .get_or_insert_default()
                                    .push_str(&text);
                            }
                            _ => {
                                blocks.insert(
                                    (ContentBlockOutputType::Thought, String::new()),
                                    ContentBlockOutput::Thought(Thought {
                                        text: Some(text),
                                        summary: None,
                                        signature: None,
                                        provider_type: provider_type.clone(),
                                        extra_data: extra_data.clone(),
                                    }),
                                );
                            }
                        }
                    }

                    // Handle signature if present
                    if let Some(signature) = signature {
                        match blocks.get_mut(&(ContentBlockOutputType::Thought, String::new())) {
                            Some(ContentBlockOutput::Thought(existing_thought)) => {
                                match &mut existing_thought.signature {
                                    Some(existing) => existing.push_str(&signature),
                                    None => {
                                        existing_thought.signature = Some(signature);
                                    }
                                }
                            }
                            _ => {
                                blocks.insert(
                                    (ContentBlockOutputType::Thought, String::new()),
                                    ContentBlockOutput::Thought(Thought {
                                        text: None,
                                        summary: None,
                                        signature: Some(signature),
                                        provider_type: provider_type.clone(),
                                        extra_data: extra_data.clone(),
                                    }),
                                );
                            }
                        }
                    }

                    // Handle summary if present (summary_id and summary_text both required)
                    if summary_id.is_some() && summary_text.is_none() {
                        tracing::error!(
                            "Summary id is present but summary text is missing for JSON thought"
                        );
                    }
                    if summary_id.is_none() && summary_text.is_some() {
                        tracing::error!(
                            "Summary text is present but summary id is missing for JSON thought"
                        );
                    }
                    if let (Some(summary_id), Some(summary_text)) = (summary_id, summary_text) {
                        // For JSON mode, we use a single thought block with id ""
                        // so we track summary ids in thought_summaries with key ""
                        let summary_set = thought_summaries.entry(String::new()).or_default();
                        let (index, _) = summary_set.insert_full(summary_id);

                        match blocks.get_mut(&(ContentBlockOutputType::Thought, String::new())) {
                            Some(ContentBlockOutput::Thought(existing_thought)) => {
                                match &mut existing_thought.summary {
                                    Some(existing) => {
                                        if index >= existing.len() {
                                            existing.resize(
                                                index + 1,
                                                ThoughtSummaryBlock::SummaryText {
                                                    text: String::new(),
                                                },
                                            );
                                        }
                                        match &mut existing[index] {
                                            ThoughtSummaryBlock::SummaryText { text } => {
                                                text.push_str(&summary_text);
                                            }
                                        }
                                    }
                                    None => {
                                        existing_thought.summary =
                                            Some(vec![ThoughtSummaryBlock::SummaryText {
                                                text: summary_text,
                                            }]);
                                    }
                                }
                            }
                            _ => {
                                blocks.insert(
                                    (ContentBlockOutputType::Thought, String::new()),
                                    ContentBlockOutput::Thought(Thought {
                                        text: None,
                                        summary: Some(vec![ThoughtSummaryBlock::SummaryText {
                                            text: summary_text,
                                        }]),
                                        signature: None,
                                        provider_type,
                                        extra_data,
                                    }),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    // For ModelInference, use provider-level timing (excludes TensorZero overhead).
    // We should always have at least one real provider chunk with provider_latency set.
    let provider_ttft = provider_ttft.ok_or_else(|| {
        Error::new(ErrorDetails::TypeConversion {
            message: "No provider-level TTFT found - all chunks were artificial".to_string(),
        })
    })?;
    let provider_response_time = provider_response_time.ok_or_else(|| {
        Error::new(ErrorDetails::TypeConversion {
            message: "No provider-level response time found - all chunks were artificial"
                .to_string(),
        })
    })?;
    let provider_latency = Latency::Streaming {
        ttft: provider_ttft,
        response_time: provider_response_time,
    };
    let content_blocks: Vec<_> = blocks.into_values().collect();
    let model_response = ProviderInferenceResponse::new(ProviderInferenceResponseArgs {
        output: content_blocks.clone(),
        system,
        input_messages,
        raw_request,
        raw_response,
        usage: model_inference_usage,
        raw_usage,
        relay_raw_response: None,
        provider_latency,
        finish_reason,
        id: model_inference_id,
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
        dynamic_output_schema,
        fetch_and_encode_input_files_before_inference,
        extra_body,
        extra_headers,
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

type InferenceResultStreamInner =
    Pin<Box<dyn Stream<Item = Result<InferenceResultChunk, Error>> + Send>>;

pub type InferenceResultStream = Peekable<InferenceResultStreamInner>;

/// Handles a textual content block (text or thought)
/// It checks if there is already a block with the given id, and if so, appends the text to it.
/// Otherwise, it creates a new block and inserts it into the map.
/// It also updates the TTFT if it hasn't been set
fn handle_textual_content_block<F, A>(
    blocks: &mut IndexMap<(ContentBlockOutputType, String), ContentBlockOutput>,
    key: (ContentBlockOutputType, String),
    text: String,
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
            if !text.is_empty() {
                blocks.insert(key, create_block(text));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::{
        config::SchemaData,
        experimentation::ExperimentationConfig,
        function::{FunctionConfigChat, FunctionConfigJson},
        inference::types::{
            ContentBlockChatOutput, ContentBlockOutputType, InferenceResult, Text, Thought,
            current_timestamp,
        },
        jsonschema_util::JSONSchema,
        tool::InferenceResponseToolCall,
    };

    #[test]
    fn test_handle_textual_content_block() {
        let mut blocks: IndexMap<(ContentBlockOutputType, String), ContentBlockOutput> =
            IndexMap::new();

        // Test case 1: Create new text block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Text, "1".to_string()),
            "Hello".to_string(),
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
            ContentBlockOutput::Text(Text { text }) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text block"),
        }

        // Test case 2: Append to existing text block
        handle_textual_content_block(
            &mut blocks,
            (ContentBlockOutputType::Text, "1".to_string()),
            " World".to_string(),
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
            |text| {
                ContentBlockOutput::Thought(Thought {
                    text: Some(text),
                    signature: None,
                    summary: None,
                    provider_type: None,
                    extra_data: None,
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
                summary: _,
                provider_type: _,
                extra_data: _,
            }) => {
                assert_eq!(text, &Some("Thinking...".to_string()));
            }
            _ => panic!("Expected thought block"),
        }
    }

    #[tokio::test]
    async fn test_collect_chunks() {
        // Test case 1: empty chunks (should error)
        let templates = Arc::new(TemplateConfig::default());

        let chunks = vec![];
        let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let model_name = "test_model";
        let model_provider_name = "test_provider";
        let raw_request = "raw request".to_string();
        let collect_chunks_args = CollectChunksArgs {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            value: chunks,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage::default(),
            finish_reason: None,
        };
        let result = collect_chunks(collect_chunks_args).await;
        assert_eq!(
            result.unwrap_err(),
            ErrorDetails::TypeConversion {
                message: "No provider-level TTFT found - all chunks were artificial".to_string(),
            }
            .into()
        );

        // Test case 2: non-empty chunks with no tool calls but content exists
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let created = current_timestamp();
        let content = vec![ContentBlockChunk::Text(TextChunk {
            text: "Hello,".to_string(),
            id: "0".to_string(),
        })];
        let provider_latency = Some(Duration::from_millis(150));
        let chunks = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content,
                usage: None,
                raw_usage: None,
                raw_response: None,
                raw_chunk: "{\"message\": \"Hello}".to_string(),
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: " world!".to_string(),
                    id: "0".to_string(),
                })],
                usage: Some(Usage {
                    input_tokens: Some(2),
                    output_tokens: Some(4),
                }),
                raw_usage: None,
                raw_response: None,
                raw_chunk: ", world!\"}".to_string(),
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage {
                input_tokens: Some(2),
                output_tokens: Some(4),
            },
            finish_reason: Some(FinishReason::Stop),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };
        assert_eq!(chat_result.inference_id, inference_id);
        // We make a new timestamp for `chat_result.created`, so just check that it's at least
        // the timestamp of the first chunk.
        assert!(
            chat_result.created >= created,
            "Chat result was created at {:?}, before the first chunk was created at {:?}",
            chat_result.created,
            created
        );
        assert_eq!(chat_result.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            chat_result.content,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(chat_result.model_inference_results.len(), 1);
        let model_inference_result = chat_result.model_inference_results.first().unwrap();
        assert_eq!(&*model_inference_result.model_name, model_name);
        assert_eq!(
            &*model_inference_result.model_provider_name,
            model_provider_name
        );
        assert_eq!(model_inference_result.raw_request, raw_request);
        // Test Case 3: a JSON string that passes validation and also include usage in each chunk
        let inference_id = Uuid::now_v7();
        let output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = JSONSchema::from_value(output_schema).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            json_mode_tool_call_config,
            output_schema,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));
        let usage1 = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
        };
        let usage2 = Usage {
            input_tokens: Some(5),
            output_tokens: Some(10),
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("{\"name\":".to_string()),
                thought_chunks: vec![ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("Thought 1".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                }],
                usage: Some(usage1),
                raw_usage: None,
                raw_response: None,
                raw_chunk: "{\"name\":".to_string(),
                provider_latency: Some(Duration::from_millis(150)),
                finish_reason: Some(FinishReason::ToolCall),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought_chunks: vec![ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("Thought 2".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                }],
                usage: Some(usage2),
                raw_usage: None,
                raw_response: None,
                raw_chunk: "\"John\",\"age\":30}".to_string(),
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            value: chunks,
            system: None,
            inference_id,
            episode_id,
            input_messages: vec![],
            function: json_function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage {
                input_tokens: Some(15),
                output_tokens: Some(15),
            },
            finish_reason: Some(FinishReason::Stop),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(
            response.usage_considering_cached(),
            Usage {
                input_tokens: Some(15),
                output_tokens: Some(15),
            }
        );
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    Some("{\"name\":\"John\",\"age\":30}".to_string())
                );
                assert_eq!(json_result.finish_reason, Some(FinishReason::Stop));
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    &*model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
        }

        // Test Case 4: a JSON string that fails validation and usage only in last chunk
        let inference_id = Uuid::now_v7();
        let created = current_timestamp();
        let model_inference_usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("{\"name\":".to_string()),
                thought_chunks: vec![ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("Thought 1".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                }],
                usage: Some(model_inference_usage),
                raw_usage: None,
                raw_response: None,
                raw_chunk: "{\"name\":".to_string(),
                provider_latency: Some(Duration::from_millis(100)),
                finish_reason: Some(FinishReason::ToolCall),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\"}".to_string()),
                thought_chunks: vec![],
                usage: None,
                raw_usage: None,
                raw_response: None,
                raw_chunk: "\"John\"}".to_string(),
                provider_latency: Some(Duration::from_millis(200)),
                finish_reason: None,
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            value: chunks,
            inference_id,
            episode_id,
            system: None,
            input_messages: vec![],
            function: json_function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage,
            finish_reason: Some(FinishReason::ToolCall),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(result.usage_considering_cached(), model_inference_usage);
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                // We make a new timestamp for `json_result.created`, so just check that it's at least
                // the timestamp of the first chunk.
                assert!(
                    json_result.created >= created,
                    "Json result was created at {:?}, before the first chunk was created at {:?}",
                    json_result.created,
                    created
                );
                assert_eq!(json_result.output.parsed, None);
                assert_eq!(
                    json_result.output.raw,
                    Some("{\"name\":\"John\"}".to_string())
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    &*model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
        }

        // Test case 5: chunks with some None content
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let created = current_timestamp();
        let model_inference_usage = Usage {
            input_tokens: Some(15),
            output_tokens: Some(10),
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("{\"name\":\"John\",".to_string()),
                thought_chunks: vec![],
                usage: Some(model_inference_usage),
                raw_usage: None,
                raw_response: None,
                raw_chunk: "{\"name\":\"John\",".to_string(),
                provider_latency: Some(Duration::from_millis(100)),
                finish_reason: None,
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some(String::new()),
                thought_chunks: vec![ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("Thought 2".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                }],
                usage: None,
                raw_usage: None,
                raw_response: None,
                raw_chunk: String::new(),
                provider_latency: Some(Duration::from_millis(200)),
                finish_reason: None,
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"age\":30}".to_string()),
                thought_chunks: vec![],
                usage: None,
                raw_usage: None,
                raw_response: None,
                raw_chunk: "\"age\":30}".to_string(),
                provider_latency: Some(Duration::from_millis(300)),
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            value: chunks,
            inference_id,
            episode_id,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage,
            finish_reason: Some(FinishReason::Stop),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(result.usage_considering_cached(), model_inference_usage);
        match result {
            InferenceResult::Chat(chat_response) => {
                assert_eq!(chat_response.inference_id, inference_id);
                // We make a new timestamp for `chat_response.created`, so just check that it's at least
                // the timestamp of the first chunk.
                assert!(
                    chat_response.created >= created,
                    "Chat result was created at {:?}, before the first chunk was created at {:?}",
                    chat_response.created,
                    created
                );
                assert_eq!(
                    chat_response.content,
                    vec![
                        ContentBlockChatOutput::Text(Text {
                            text: "{\"name\":\"John\",\"age\":30}".to_string()
                        }),
                        ContentBlockChatOutput::Thought(Thought {
                            text: Some("Thought 2".to_string()),
                            summary: None,
                            signature: None,
                            provider_type: None,
                            extra_data: None,
                        }),
                    ]
                );
                assert_eq!(chat_response.model_inference_results.len(), 1);
                let model_inference_result = chat_response.model_inference_results.first().unwrap();
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(chat_response.finish_reason, Some(FinishReason::Stop));
                assert_eq!(
                    model_inference_result.finish_reason,
                    Some(FinishReason::Stop)
                );
                assert_eq!(
                    &*model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        }

        // Test Case 6: a JSON function with implicit tool call config
        let inference_id = Uuid::now_v7();

        let output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = JSONSchema::from_value(output_schema).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            json_mode_tool_call_config,
            output_schema,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));
        let usage1 = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
        };
        let usage2 = Usage {
            input_tokens: Some(5),
            output_tokens: Some(10),
        };
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("{\"name\":".to_string()),
                thought_chunks: vec![ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("Thought 1".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                }],
                usage: Some(usage1),
                raw_usage: None,
                raw_response: None,
                raw_chunk: "{\"name\":".to_string(),
                provider_latency: Some(Duration::from_millis(150)),
                finish_reason: Some(FinishReason::ToolCall),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought_chunks: vec![ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("Thought 2".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                }],
                usage: Some(usage2),
                raw_usage: None,
                raw_response: None,
                raw_chunk: "\"John\",\"age\":30}".to_string(),
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks,
            system: None,
            input_messages: vec![],
            function: json_function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage {
                input_tokens: Some(15),
                output_tokens: Some(15),
            },
            finish_reason: Some(FinishReason::Stop),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(
            response.usage_considering_cached(),
            Usage {
                input_tokens: Some(15),
                output_tokens: Some(15),
            }
        );
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    Some("{\"name\":\"John\",\"age\":30}".to_string())
                );
                assert_eq!(json_result.finish_reason, Some(FinishReason::Stop));
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    &*model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
        }
        // Test Case 7: a JSON string with a dynamic schema that passes validation and also include usage in each chunk
        let inference_id = Uuid::now_v7();
        let static_output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"]
        });
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&static_output_schema);
        let output_schema = JSONSchema::from_value(static_output_schema).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            json_mode_tool_call_config,
            output_schema,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));
        let usage1 = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
        };
        let usage2 = Usage {
            input_tokens: Some(5),
            output_tokens: Some(10),
        };
        let dynamic_output_schema = JSONSchema::compile_background(serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }));
        let templates = TemplateConfig::default();
        let chunks = vec![
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("{\"name\":".to_string()),
                thought_chunks: vec![ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("Thought 1".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                }],
                usage: Some(usage1),
                raw_usage: None,
                raw_response: None,
                finish_reason: Some(FinishReason::Stop),
                raw_chunk: "{\"name\":".to_string(),
                provider_latency: Some(Duration::from_millis(150)),
            }),
            InferenceResultChunk::Json(JsonInferenceResultChunk {
                raw: Some("\"John\",\"age\":30}".to_string()),
                thought_chunks: vec![ThoughtChunk {
                    id: "0".to_string(),
                    text: Some("Thought 2".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                }],
                usage: Some(usage2),
                raw_usage: None,
                raw_response: None,
                finish_reason: Some(FinishReason::ToolCall),
                raw_chunk: "\"John\",\"age\":30}".to_string(),
                provider_latency: Some(Duration::from_millis(250)),
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks,
            system: None,
            input_messages: vec![],
            function: json_function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: Some(dynamic_output_schema.into()),
            templates: Arc::new(templates),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage {
                input_tokens: Some(15),
                output_tokens: Some(15),
            },
            finish_reason: Some(FinishReason::ToolCall),
        };
        let response = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(
            response.usage_considering_cached(),
            Usage {
                input_tokens: Some(15),
                output_tokens: Some(15),
            }
        );
        match response {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.inference_id, inference_id);
                assert_eq!(
                    json_result.output.parsed,
                    Some(serde_json::json!({"name": "John", "age": 30}))
                );
                assert_eq!(
                    json_result.output.raw,
                    Some("{\"name\":\"John\",\"age\":30}".to_string())
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                let model_inference_result = json_result.model_inference_results.first().unwrap();
                assert_eq!(&*model_inference_result.model_name, model_name);
                assert_eq!(
                    model_inference_result.finish_reason,
                    Some(FinishReason::ToolCall)
                );
                assert_eq!(
                    &*model_inference_result.model_provider_name,
                    model_provider_name
                );
                assert_eq!(model_inference_result.raw_request, raw_request);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
        }
    }

    #[tokio::test]
    async fn test_collect_interleaved_chunks() {
        let templates = TemplateConfig::default();
        let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let model_name = "test_model";
        let model_provider_name = "test_provider";
        let raw_request = "raw request".to_string();
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let created = current_timestamp();
        let provider_latency = Some(Duration::from_millis(150));
        let chunks = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Text(TextChunk {
                        text: "Hello ".to_string(),
                        id: "0".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "0".to_string(),
                        raw_name: Some("my_tool_call".to_string()),
                        raw_arguments: "true".to_string(),
                    }),
                ],
                usage: None,
                raw_usage: None,
                raw_response: None,
                raw_chunk: "{\"message\": \"Hello}".to_string(),
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Thought(ThoughtChunk {
                        text: Some("Some thou".to_string()),
                        id: "0".to_string(),
                        summary_id: None,
                        summary_text: None,
                        signature: None,
                        provider_type: None,
                        extra_data: None,
                    }),
                    ContentBlockChunk::Thought(ThoughtChunk {
                        text: Some("My other interleaved thought".to_string()),
                        id: "1".to_string(),
                        summary_id: Some("abc".to_string()),
                        summary_text: Some("Inline summary".to_string()),
                        signature: None,
                        provider_type: None,
                        extra_data: None,
                    }),
                    ContentBlockChunk::Thought(ThoughtChunk {
                        text: None,
                        id: "0".to_string(),
                        summary_id: Some("abc".to_string()),
                        summary_text: Some("First summary".to_string()),
                        signature: None,
                        provider_type: None,
                        extra_data: None,
                    }),
                    ContentBlockChunk::Thought(ThoughtChunk {
                        text: None,
                        id: "0".to_string(),
                        summary_id: Some("2".to_string()),
                        summary_text: Some("Second summary".to_string()),
                        signature: None,
                        provider_type: None,
                        extra_data: None,
                    }),
                    ContentBlockChunk::Thought(ThoughtChunk {
                        text: None,
                        id: "0".to_string(),
                        summary_id: Some("abc".to_string()),
                        summary_text: Some(" content.".to_string()),
                        signature: None,
                        provider_type: None,
                        extra_data: None,
                    }),
                    ContentBlockChunk::Thought(ThoughtChunk {
                        text: None,
                        id: "0".to_string(),
                        summary_id: Some("2".to_string()),
                        summary_text: Some(" message.".to_string()),
                        signature: None,
                        provider_type: None,
                        extra_data: None,
                    }),
                ],
                usage: None,
                raw_usage: None,
                raw_response: None,
                raw_chunk: "my raw thought".to_string(),
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::Text(TextChunk {
                    text: "world!".to_string(),
                    id: "0".to_string(),
                })],
                usage: Some(Usage {
                    input_tokens: Some(2),
                    output_tokens: Some(4),
                }),
                raw_usage: None,
                raw_response: None,
                raw_chunk: ", world!\"}".to_string(),
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::Stop),
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::Thought(ThoughtChunk {
                    text: Some("ght".to_string()),
                    id: "0".to_string(),
                    summary_id: None,
                    summary_text: None,
                    signature: None,
                    provider_type: None,
                    extra_data: None,
                })],
                usage: None,
                raw_usage: None,
                raw_response: None,
                raw_chunk: "my other raw thought".to_string(),
                provider_latency,
                finish_reason: None,
            }),
        ];
        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: Arc::new(templates),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage {
                input_tokens: Some(2),
                output_tokens: Some(4),
            },
            finish_reason: Some(FinishReason::Stop),
        };
        let result = collect_chunks(collect_chunks_args).await.unwrap();
        assert_eq!(
            result.usage_considering_cached(),
            Usage {
                input_tokens: Some(2),
                output_tokens: Some(4),
            }
        );
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };
        assert_eq!(chat_result.inference_id, inference_id);
        // We make a new timestamp for `chat_result.created`, so just check that it's at least
        // the timestamp of the first chunk.
        assert!(
            chat_result.created >= created,
            "Chat result was created at {:?}, before the first chunk was created at {:?}",
            chat_result.created,
            created
        );
        assert_eq!(chat_result.finish_reason, Some(FinishReason::Stop));

        let expected_content = vec![
            ContentBlockChatOutput::Text(Text {
                text: "Hello world!".to_string(),
            }),
            ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
                name: None,
                raw_name: "my_tool_call".to_string(),
                raw_arguments: "true".to_string(),
                arguments: None,
                id: "0".to_string(),
            }),
            ContentBlockChatOutput::Thought(Thought {
                text: Some("Some thought".to_string()),
                summary: Some(vec![
                    ThoughtSummaryBlock::SummaryText {
                        text: "First summary content.".to_string(),
                    },
                    ThoughtSummaryBlock::SummaryText {
                        text: "Second summary message.".to_string(),
                    },
                ]),
                signature: None,
                provider_type: None,
                extra_data: None,
            }),
            ContentBlockChatOutput::Thought(Thought {
                text: Some("My other interleaved thought".to_string()),
                summary: Some(vec![ThoughtSummaryBlock::SummaryText {
                    text: "Inline summary".to_string(),
                }]),
                signature: None,
                provider_type: None,
                extra_data: None,
            }),
        ];
        assert_eq!(chat_result.content, expected_content);

        assert_eq!(chat_result.model_inference_results.len(), 1);
        let model_inference_result = chat_result.model_inference_results.first().unwrap();
        assert_eq!(&*model_inference_result.model_name, model_name);
        assert_eq!(
            &*model_inference_result.model_provider_name,
            model_provider_name
        );
        assert_eq!(model_inference_result.raw_request, raw_request);
    }

    #[tokio::test]
    async fn test_collect_chunks_tool_name_accumulation() {
        let templates = Arc::new(TemplateConfig::default());
        let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let model_name = "test_model";
        let model_provider_name = "test_provider";
        let raw_request = "raw request".to_string();
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let provider_latency = Some(Duration::from_millis(150));

        // Test case 1: Tool name sent in first chunk, then arguments accumulated
        let chunks_case1 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: Some("get_weather".to_string()),
                    raw_arguments: "{\"loca".to_string(),
                })],
                usage: None,
                raw_usage: None,
                raw_chunk: "chunk1".to_string(),
                raw_response: None,
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: None, // No name in subsequent chunks
                    raw_arguments: "tion\": \"San Francisco\", \"unit\": \"celsius\"}".to_string(),
                })],
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                }),
                raw_usage: None,
                raw_chunk: "chunk2".to_string(),
                raw_response: None,
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case1,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage::default(),
            finish_reason: None,
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 1);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_weather");
                assert_eq!(
                    tool_call.raw_arguments,
                    r#"{"location": "San Francisco", "unit": "celsius"}"#
                );
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected tool call block"),
        }

        // Test case 2: Multiple tool calls with different IDs and name accumulation
        let chunks_case2 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("get_wea".to_string()),
                        raw_arguments: "{\"loc".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: Some("calculate".to_string()),
                        raw_arguments: "{\"expr".to_string(),
                    }),
                ],
                usage: None,
                raw_usage: None,
                raw_chunk: "chunk1".to_string(),
                raw_response: None,
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("ther".to_string()), // Continue accumulating name
                        raw_arguments: "ation\": \"NYC\"}".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: None, // No more name for tool_2
                        raw_arguments: "ession\": \"2+2\"}".to_string(),
                    }),
                ],
                usage: Some(Usage {
                    input_tokens: Some(15),
                    output_tokens: Some(25),
                }),
                raw_usage: None,
                raw_chunk: "chunk2".to_string(),
                raw_response: None,
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case2,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage::default(),
            finish_reason: None,
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 2);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_weather");
                assert_eq!(tool_call.raw_arguments, r#"{"location": "NYC"}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected first tool call block"),
        }
        match &chat_result.content[1] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "calculate");
                assert_eq!(tool_call.raw_arguments, r#"{"expression": "2+2"}"#);
                assert_eq!(tool_call.id, "tool_2");
            }
            _ => panic!("Expected second tool call block"),
        }

        // Test case 3: Tool call with no name in first chunk (should start with empty name)
        let chunks_case3 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: None, // No name in first chunk
                    raw_arguments: "{\"key\":".to_string(),
                })],
                usage: None,
                raw_usage: None,
                raw_chunk: "chunk1".to_string(),
                raw_response: None,
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "tool_1".to_string(),
                    raw_name: Some("my_function".to_string()), // Name comes later
                    raw_arguments: " \"value\"}".to_string(),
                })],
                usage: Some(Usage {
                    input_tokens: Some(5),
                    output_tokens: Some(10),
                }),
                raw_usage: None,
                raw_chunk: "chunk2".to_string(),
                raw_response: None,
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case3,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage::default(),
            finish_reason: None,
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 1);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "my_function"); // Should accumulate to the full name
                assert_eq!(tool_call.raw_arguments, r#"{"key": "value"}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected tool call block"),
        }

        // Test case 4: Mixed content with text and tool calls preserving order
        let chunks_case4 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Text(TextChunk {
                        text: "I'll help you with that. ".to_string(),
                        id: "0".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("search".to_string()),
                        raw_arguments: "{\"query\"".to_string(),
                    }),
                ],
                usage: None,
                raw_usage: None,
                raw_chunk: "chunk1".to_string(),
                raw_response: None,
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Text(TextChunk {
                        text: "Let me search for information.".to_string(),
                        id: "0".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: None,
                        raw_arguments: ": \"weather today\"}".to_string(),
                    }),
                ],
                usage: Some(Usage {
                    input_tokens: Some(20),
                    output_tokens: Some(15),
                }),
                raw_usage: None,
                raw_chunk: "chunk2".to_string(),
                raw_response: None,
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case4,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage::default(),
            finish_reason: None,
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 2);
        // Order should be preserved: text first, then tool call
        match &chat_result.content[0] {
            ContentBlockChatOutput::Text(text) => {
                assert_eq!(
                    text.text,
                    "I'll help you with that. Let me search for information."
                );
            }
            _ => panic!("Expected text block first"),
        }
        match &chat_result.content[1] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "search");
                assert_eq!(tool_call.raw_arguments, r#"{"query": "weather today"}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected tool call block second"),
        }

        // Test case 5: Tool call with empty name parts that should result in empty final name
        let chunks_case5 = vec![InferenceResultChunk::Chat(ChatInferenceResultChunk {
            content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "tool_1".to_string(),
                raw_name: None,
                raw_arguments: "{\"test\": true}".to_string(),
            })],
            usage: Some(Usage {
                input_tokens: Some(5),
                output_tokens: Some(5),
            }),
            raw_usage: None,
            raw_response: None,
            raw_chunk: "chunk1".to_string(),
            provider_latency,
            finish_reason: Some(FinishReason::ToolCall),
        })];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case5,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage::default(),
            finish_reason: None,
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 1);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, ""); // Should be empty string when no name provided
                assert_eq!(tool_call.raw_arguments, r#"{"test": true}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected tool call block"),
        }

        // Test case 6: Complex multi-tool name accumulation across multiple chunks
        let chunks_case6 = vec![
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("get_".to_string()),
                        raw_arguments: "{\"lo".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: Some("cal".to_string()),
                        raw_arguments: "{\"op".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_3".to_string(),
                        raw_name: Some("send_".to_string()),
                        raw_arguments: "{\"me".to_string(),
                    }),
                ],
                usage: None,
                raw_usage: None,
                raw_chunk: "chunk1".to_string(),
                raw_response: None,
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("wea".to_string()),
                        raw_arguments: "cation\": ".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: Some("cul".to_string()),
                        raw_arguments: "eration\": ".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_3".to_string(),
                        raw_name: Some("email".to_string()),
                        raw_arguments: "ssage\": ".to_string(),
                    }),
                ],
                usage: None,
                raw_usage: None,
                raw_chunk: "chunk2".to_string(),
                raw_response: None,
                provider_latency,
                finish_reason: None,
            }),
            InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_1".to_string(),
                        raw_name: Some("ther".to_string()),
                        raw_arguments: "\"Paris\"}".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_2".to_string(),
                        raw_name: Some("ate".to_string()),
                        raw_arguments: "\"5*5\"}".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "tool_3".to_string(),
                        raw_name: None, // No more name parts
                        raw_arguments: "\"Hello world\"}".to_string(),
                    }),
                ],
                usage: Some(Usage {
                    input_tokens: Some(20),
                    output_tokens: Some(30),
                }),
                raw_usage: None,
                raw_response: None,
                raw_chunk: "chunk3".to_string(),
                provider_latency: Some(Duration::from_millis(250)),
                finish_reason: Some(FinishReason::ToolCall),
            }),
        ];

        let collect_chunks_args = CollectChunksArgs {
            inference_id,
            episode_id,
            value: chunks_case6,
            system: None,
            input_messages: vec![],
            function: function_config.clone(),
            model_name: model_name.into(),
            model_provider_name: model_provider_name.into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            inference_params: InferenceParams::default(),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            templates: templates.clone(),
            tool_config: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            model_inference_id: Uuid::now_v7(),
            model_inference_usage: Usage::default(),
            finish_reason: None,
        };

        let result = collect_chunks(collect_chunks_args).await.unwrap();
        let chat_result = match result {
            InferenceResult::Chat(chat_result) => chat_result,
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        };

        assert_eq!(chat_result.content.len(), 3);
        match &chat_result.content[0] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "get_weather"); // "get_" + "wea" + "ther"
                assert_eq!(tool_call.raw_arguments, r#"{"location": "Paris"}"#);
                assert_eq!(tool_call.id, "tool_1");
            }
            _ => panic!("Expected first tool call block"),
        }
        match &chat_result.content[1] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "calculate"); // "cal" + "cul" + "ate"
                assert_eq!(tool_call.raw_arguments, r#"{"operation": "5*5"}"#);
                assert_eq!(tool_call.id, "tool_2");
            }
            _ => panic!("Expected second tool call block"),
        }
        match &chat_result.content[2] {
            ContentBlockChatOutput::ToolCall(tool_call) => {
                assert_eq!(tool_call.raw_name, "send_email"); // "send_" + "email"
                assert_eq!(tool_call.raw_arguments, r#"{"message": "Hello world"}"#);
                assert_eq!(tool_call.id, "tool_3");
            }
            _ => panic!("Expected third tool call block"),
        }
    }

    #[test]
    fn test_json_inference_result_chunk_from_provider_chunk() {
        use std::time::Duration;

        // Test case for ToolCall content
        let tool_chunk = ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "123".to_string(),
                raw_arguments: "{\"key\": \"value\"}".to_string(),
                raw_name: Some("test_tool".to_string()),
            })],
            usage: Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            }),
            raw_usage: None,
            raw_response: "raw response".to_string(),
            provider_latency: Duration::from_secs(1),
            finish_reason: Some(FinishReason::ToolCall),
        };

        let result = JsonInferenceResultChunk::from(tool_chunk);
        assert_eq!(result.raw, Some("{\"key\": \"value\"}".to_string()));
        assert!(result.thought_chunks.is_empty());
        assert_eq!(result.raw_response, None);
        assert_eq!(result.raw_chunk, "raw response");
        assert_eq!(result.provider_latency, Some(Duration::from_secs(1)));
        assert_eq!(
            result.usage,
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            })
        );
        assert_eq!(result.finish_reason, Some(FinishReason::ToolCall));
        // Test case for Text content
        let text_chunk = ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::Text(TextChunk {
                id: "123".to_string(),
                text: "some text".to_string(),
            })],
            usage: None,
            raw_usage: None,
            raw_response: "raw response".to_string(),
            provider_latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(text_chunk);
        assert_eq!(result.raw, Some("some text".to_string()));
        assert!(result.thought_chunks.is_empty());

        // Test case for Thought content
        let thought_chunk = ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::Thought(ThoughtChunk {
                id: "123".to_string(),
                text: Some("thinking...".to_string()),
                summary_id: None,
                summary_text: None,
                signature: None,
                provider_type: None,
                extra_data: None,
            })],
            usage: None,
            raw_usage: None,
            raw_response: "raw response".to_string(),
            provider_latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(thought_chunk);
        assert_eq!(result.raw, None);
        assert_eq!(
            result.thought_chunks,
            vec![ThoughtChunk {
                id: "123".to_string(),
                text: Some("thinking...".to_string()),
                summary_id: None,
                summary_text: None,
                signature: None,
                provider_type: None,
                extra_data: None,
            }]
        );
        assert_eq!(result.finish_reason, None);
        // Test case for multiple content blocks - should use last raw content
        let mixed_chunk = ProviderInferenceResponseChunk {
            content: vec![
                ContentBlockChunk::Text(TextChunk {
                    id: "123".to_string(),
                    text: "first text".to_string(),
                }),
                ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "456".to_string(),
                    raw_arguments: "final content".to_string(),
                    raw_name: Some("test_tool".to_string()),
                }),
                ContentBlockChunk::Thought(ThoughtChunk {
                    id: "789".to_string(),
                    text: Some("final thought".to_string()),
                    summary_id: None,
                    summary_text: None,
                    signature: None,
                    provider_type: None,
                    extra_data: None,
                }),
            ],
            usage: None,
            raw_usage: None,
            raw_response: "raw response".to_string(),
            provider_latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(mixed_chunk);
        assert_eq!(result.raw, Some("final content".to_string()));
        assert_eq!(
            result.thought_chunks,
            vec![ThoughtChunk {
                id: "789".to_string(),
                text: Some("final thought".to_string()),
                summary_id: None,
                summary_text: None,
                signature: None,
                provider_type: None,
                extra_data: None,
            }]
        );

        // Test case for empty content
        let empty_chunk = ProviderInferenceResponseChunk {
            content: vec![],
            usage: None,
            raw_usage: None,
            raw_response: "raw response".to_string(),
            provider_latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(empty_chunk);
        assert_eq!(result.raw, None);
        assert!(result.thought_chunks.is_empty());
        assert_eq!(result.finish_reason, None);

        // Test case: ThoughtChunk with only summary_text (no text)
        // This is the format OpenRouter uses for reasoning.summary type
        let summary_chunk = ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::Thought(ThoughtChunk {
                id: "0".to_string(),
                text: None,
                summary_id: Some("0".to_string()),
                summary_text: Some("This is a summary".to_string()),
                signature: None,
                provider_type: Some("openrouter".to_string()),
                extra_data: None,
            })],
            usage: None,
            raw_usage: None,
            raw_response: "raw response".to_string(),
            provider_latency: Duration::from_secs(1),
            finish_reason: None,
        };

        let result = JsonInferenceResultChunk::from(summary_chunk);
        assert_eq!(
            result.thought_chunks,
            vec![ThoughtChunk {
                id: "0".to_string(),
                text: None,
                summary_id: Some("0".to_string()),
                summary_text: Some("This is a summary".to_string()),
                signature: None,
                provider_type: Some("openrouter".to_string()),
                extra_data: None,
            }],
            "ThoughtChunk should be preserved when text is None"
        );
    }
}
