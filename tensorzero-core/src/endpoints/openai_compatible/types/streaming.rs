//! Streaming response types and logic for OpenAI-compatible API.
//!
//! This module provides types and functions for streaming chat completion responses
//! in Server-Sent Events (SSE) format, compatible with OpenAI's streaming API.
//! It handles chunk formatting, delta updates, and usage reporting for streaming responses.

use axum::response::sse::Event;
use futures::Stream;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use tokio_stream::StreamExt;

use crate::error::{Error, ErrorDetails};
use crate::inference::types::streams::{ThoughtChunk, UnknownChunk};
use crate::inference::types::usage::{RawResponseEntry, RawUsageEntry};
use crate::inference::types::{ContentBlockChunk, FinishReason, current_timestamp};

use crate::endpoints::inference::{InferenceResponseChunk, InferenceStream};

use super::chat_completions::OpenAICompatibleFinishReason;
use super::is_none_or_empty;
use super::tool::{OpenAICompatibleToolCallChunk, OpenAICompatibleToolCallDelta};
use super::usage::OpenAICompatibleUsage;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleResponseChunk {
    pub id: String,
    pub episode_id: String,
    pub choices: Vec<OpenAICompatibleChoiceChunk>,
    pub created: u32,
    pub model: String,
    pub system_fingerprint: String,
    pub service_tier: Option<String>,
    pub object: String,
    // OpenAI spec requires `usage` to be "object or null", not omitted
    pub usage: Option<OpenAICompatibleUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_usage: Option<Vec<RawUsageEntry>>,
    /// Raw responses from previous model inferences (e.g., best-of-n candidates).
    /// Emitted in the first chunk of a streaming response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_response: Option<Vec<RawResponseEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `tensorzero_raw_chunk` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_original_chunk: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_chunk: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleChoiceChunk {
    pub index: u32,
    pub finish_reason: Option<OpenAICompatibleFinishReason>,
    pub logprobs: Option<()>, // This is always set to None for now
    pub delta: OpenAICompatibleDelta,
}

/// Extra content block chunk for streaming responses (Thought or Unknown)
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExtraContentBlockChunk {
    Thought {
        insert_index: usize,
        #[serde(flatten)]
        chunk: ThoughtChunk,
    },
    Unknown {
        insert_index: usize,
        #[serde(flatten)]
        chunk: UnknownChunk,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "is_none_or_empty")]
    pub tool_calls: Option<Vec<OpenAICompatibleToolCallChunk>>,
    #[serde(skip_serializing_if = "is_none_or_empty")]
    pub tensorzero_extra_content_experimental: Option<Vec<ExtraContentBlockChunk>>,
}

/// State maintained across streaming chunks to track content block positions.
///
/// This tracks all content block IDs (text, tool calls, extra content) to ensure
/// `insert_index` values correctly reflect the position in the full content array,
/// matching the behavior of non-streaming responses.
#[derive(Debug, Default)]
pub struct StreamingContentState {
    /// Maps tool call IDs to their index in the tool_calls array
    pub tool_id_to_index: HashMap<String, usize>,
    /// Maps extra content IDs (thoughts, unknowns) to their insert_index in the full content array
    pub extra_id_to_index: HashMap<String, usize>,
    /// Tracks which text block IDs we've seen (to know when a new text block starts)
    pub text_ids_seen: HashSet<String>,
    /// Counter for the total number of distinct content blocks seen
    pub content_block_count: usize,
}

#[expect(clippy::too_many_arguments)]
pub fn convert_inference_response_chunk_to_openai_compatible(
    chunk: InferenceResponseChunk,
    state: &mut StreamingContentState,
    response_model_prefix: &str,
    is_first_chunk: bool,
    include_usage: bool,
    include_raw_usage: bool,
    include_original_response: bool,
    include_raw_response: bool,
) -> Vec<OpenAICompatibleResponseChunk> {
    // OpenAI includes "assistant" role in the first chunk but not in subsequent chunks
    let role = if is_first_chunk {
        Some("assistant".to_string())
    } else {
        None
    };

    let response_chunk = match chunk {
        InferenceResponseChunk::Chat(c) => {
            let (content, tool_calls, extra_content) = process_chat_content_chunk(c.content, state);
            let usage = if include_usage {
                c.usage.map(OpenAICompatibleUsage::from)
            } else {
                None
            };
            let tensorzero_raw_usage = if include_raw_usage { c.raw_usage } else { None };
            let tensorzero_raw_response = if include_raw_response {
                c.raw_response
            } else {
                None
            };
            // Compute chunk fields based on which request flags were set
            let tensorzero_original_chunk = if include_original_response {
                c.original_chunk.clone()
            } else {
                None
            };
            let tensorzero_raw_chunk = if include_raw_response {
                c.raw_chunk.or(c.original_chunk)
            } else {
                None
            };
            OpenAICompatibleResponseChunk {
                id: c.inference_id.to_string(),
                episode_id: c.episode_id.to_string(),
                choices: vec![OpenAICompatibleChoiceChunk {
                    index: 0,
                    finish_reason: c.finish_reason.map(FinishReason::into),
                    logprobs: None,
                    delta: OpenAICompatibleDelta {
                        role: role.clone(),
                        content,
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                        tensorzero_extra_content_experimental: if extra_content.is_empty() {
                            None
                        } else {
                            Some(extra_content)
                        },
                    },
                }],
                created: current_timestamp() as u32,
                service_tier: None,
                model: format!("{response_model_prefix}{}", c.variant_name),
                system_fingerprint: String::new(),
                object: "chat.completion.chunk".to_string(),
                usage,
                tensorzero_raw_usage,
                tensorzero_raw_response: tensorzero_raw_response.clone(),
                tensorzero_original_chunk,
                tensorzero_raw_chunk,
            }
        }
        InferenceResponseChunk::Json(c) => {
            let usage = if include_usage {
                c.usage.map(OpenAICompatibleUsage::from)
            } else {
                None
            };
            let tensorzero_raw_usage = if include_raw_usage { c.raw_usage } else { None };
            let tensorzero_raw_response = if include_raw_response {
                c.raw_response
            } else {
                None
            };
            // Compute chunk fields based on which request flags were set
            let tensorzero_original_chunk = if include_original_response {
                c.original_chunk.clone()
            } else {
                None
            };
            let tensorzero_raw_chunk = if include_raw_response {
                c.raw_chunk.or(c.original_chunk)
            } else {
                None
            };
            OpenAICompatibleResponseChunk {
                id: c.inference_id.to_string(),
                episode_id: c.episode_id.to_string(),
                choices: vec![OpenAICompatibleChoiceChunk {
                    index: 0,
                    finish_reason: c.finish_reason.map(FinishReason::into),
                    logprobs: None,
                    delta: OpenAICompatibleDelta {
                        role,
                        content: Some(c.raw),
                        tool_calls: None,
                        tensorzero_extra_content_experimental: None,
                    },
                }],
                created: current_timestamp() as u32,
                service_tier: None,
                model: format!("{response_model_prefix}{}", c.variant_name),
                system_fingerprint: String::new(),
                object: "chat.completion.chunk".to_string(),
                usage,
                tensorzero_raw_usage,
                tensorzero_raw_response,
                tensorzero_original_chunk,
                tensorzero_raw_chunk,
            }
        }
    };

    vec![response_chunk]
}

pub fn process_chat_content_chunk(
    content: Vec<ContentBlockChunk>,
    state: &mut StreamingContentState,
) -> (
    Option<String>,
    Vec<OpenAICompatibleToolCallChunk>,
    Vec<ExtraContentBlockChunk>,
) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    let mut extra_content = Vec::new();
    for block in content {
        match block {
            ContentBlockChunk::Text(text) => {
                // Track new text blocks to maintain correct content_block_count
                if state.text_ids_seen.insert(text.id.clone()) {
                    state.content_block_count += 1;
                }
                match content_str {
                    Some(ref mut content) => content.push_str(&text.text),
                    None => content_str = Some(text.text),
                }
            }
            ContentBlockChunk::ToolCall(tool_call) => {
                use std::collections::hash_map::Entry;
                let next_tool_index = state.tool_id_to_index.len();
                let (index, is_new) = match state.tool_id_to_index.entry(tool_call.id.clone()) {
                    Entry::Occupied(e) => (*e.get(), false),
                    Entry::Vacant(e) => {
                        // New tool call: assign next tool index and increment content block count
                        e.insert(next_tool_index);
                        state.content_block_count += 1;
                        (next_tool_index, true)
                    }
                };
                tool_calls.push(OpenAICompatibleToolCallChunk {
                    id: if is_new { Some(tool_call.id) } else { None },
                    index,
                    r#type: "function".to_string(),
                    function: OpenAICompatibleToolCallDelta {
                        name: tool_call.raw_name.unwrap_or_default(),
                        arguments: tool_call.raw_arguments,
                    },
                });
            }
            ContentBlockChunk::Thought(chunk) => {
                use std::collections::hash_map::Entry;
                let insert_index = match state.extra_id_to_index.entry(chunk.id.clone()) {
                    Entry::Occupied(e) => *e.get(),
                    Entry::Vacant(e) => {
                        // New thought: use current content_block_count as insert_index
                        let idx = state.content_block_count;
                        e.insert(idx);
                        state.content_block_count += 1;
                        idx
                    }
                };
                extra_content.push(ExtraContentBlockChunk::Thought {
                    insert_index,
                    chunk,
                });
            }
            ContentBlockChunk::Unknown(chunk) => {
                use std::collections::hash_map::Entry;
                let insert_index = match state.extra_id_to_index.entry(chunk.id.clone()) {
                    Entry::Occupied(e) => *e.get(),
                    Entry::Vacant(e) => {
                        // New unknown: use current content_block_count as insert_index
                        let idx = state.content_block_count;
                        e.insert(idx);
                        state.content_block_count += 1;
                        idx
                    }
                };
                extra_content.push(ExtraContentBlockChunk::Unknown {
                    insert_index,
                    chunk,
                });
            }
        }
    }
    (content_str, tool_calls, extra_content)
}

/// Prepares an Event for SSE on the way out of the gateway.
/// Converts each InferenceResponseChunk to OpenAI-compatible format and streams it.
/// Usage, raw_usage, and original_chunk are passed through from the upstream `create_stream` based on flags.
pub fn prepare_serialized_openai_compatible_events(
    mut stream: InferenceStream,
    response_model_prefix: String,
    include_usage: bool,
    include_raw_usage: bool,
    include_original_response: bool,
    include_raw_response: bool,
) -> impl Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        let mut state = StreamingContentState::default();
        let mut is_first_chunk = true;

        while let Some(chunk) = stream.next().await {
            // NOTE: in the future, we may want to end the stream early if we get an error
            // For now, we just ignore the error and try to get more chunks
            let Ok(chunk) = chunk else {
                continue;
            };

            let openai_compatible_chunks = convert_inference_response_chunk_to_openai_compatible(
                chunk,
                &mut state,
                &response_model_prefix,
                is_first_chunk,
                include_usage,
                include_raw_usage,
                include_original_response,
                include_raw_response,
            );
            is_first_chunk = false;

            for chunk in openai_compatible_chunks {
                yield Event::default().json_data(&chunk).map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert chunk to Event: {e}"),
                    })
                })
            }
        }

        yield Ok(Event::default().data("[DONE]"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::inference::{ChatInferenceResponseChunk, JsonInferenceResponseChunk};
    use crate::inference::types::TextChunk;
    use crate::inference::types::usage::{ApiType, RawUsageEntry, Usage};
    use crate::tool::ToolCallChunk;
    use uuid::Uuid;

    #[test]
    fn test_convert_chunk_with_usage_passthrough() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let chunk = InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![],
            usage: Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            }),
            raw_usage: None,
            finish_reason: None,
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            true,  // include_usage
            false, // include_raw_usage
            false, // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.usage.is_some(),
            "usage should be passed through when include_usage is true"
        );
        let usage = chunk.usage.as_ref().unwrap();
        assert_eq!(
            usage.prompt_tokens,
            Some(10),
            "prompt_tokens should match input_tokens"
        );
        assert_eq!(
            usage.completion_tokens,
            Some(20),
            "completion_tokens should match output_tokens"
        );
        assert_eq!(
            usage.total_tokens,
            Some(30),
            "total_tokens should be sum of input and output"
        );
    }

    #[test]
    fn test_convert_chunk_without_usage() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let chunk = InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "Hello".to_string(),
            })],
            usage: None,
            raw_usage: None,
            finish_reason: None,
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            true,  // include_usage
            false, // include_raw_usage
            false, // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.usage.is_none(),
            "usage should be None when input chunk has no usage"
        );
    }

    #[test]
    fn test_convert_chunk_raw_usage_included_when_enabled() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let model_inference_id = Uuid::now_v7();
        let raw_usage_entry = RawUsageEntry {
            model_inference_id,
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: serde_json::json!({"total_tokens": 100}),
        };
        let chunk = InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![],
            usage: Some(Usage {
                input_tokens: Some(50),
                output_tokens: Some(50),
            }),
            raw_usage: Some(vec![raw_usage_entry.clone()]),
            finish_reason: None,
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            true,  // include_usage
            true,  // include_raw_usage
            false, // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.tensorzero_raw_usage.is_some(),
            "tensorzero_raw_usage should be included when include_raw_usage is true"
        );
        let raw_usage = chunk.tensorzero_raw_usage.as_ref().unwrap();
        assert_eq!(
            raw_usage.len(),
            1,
            "raw_usage should have one entry from input"
        );
        assert_eq!(
            raw_usage[0].model_inference_id, model_inference_id,
            "raw_usage entry should match input"
        );
    }

    #[test]
    fn test_convert_chunk_raw_usage_omitted_when_disabled() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let model_inference_id = Uuid::now_v7();
        let raw_usage_entry = RawUsageEntry {
            model_inference_id,
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: serde_json::json!({"total_tokens": 100}),
        };
        let chunk = InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![],
            usage: Some(Usage {
                input_tokens: Some(50),
                output_tokens: Some(50),
            }),
            raw_usage: Some(vec![raw_usage_entry]),
            finish_reason: None,
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            true,  // include_usage
            false, // include_raw_usage
            false, // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.tensorzero_raw_usage.is_none(),
            "tensorzero_raw_usage should be None when include_raw_usage is false"
        );
    }

    #[test]
    fn test_convert_json_chunk_with_usage_passthrough() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let chunk = InferenceResponseChunk::Json(JsonInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            raw: r#"{"key": "value"}"#.to_string(),
            usage: Some(Usage {
                input_tokens: Some(15),
                output_tokens: Some(25),
            }),
            raw_usage: None,
            finish_reason: None,
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            true,  // include_usage
            false, // include_raw_usage
            false, // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.usage.is_some(),
            "usage should be passed through for JSON chunks when include_usage is true"
        );
        let usage = chunk.usage.as_ref().unwrap();
        assert_eq!(
            usage.prompt_tokens,
            Some(15),
            "prompt_tokens should match input_tokens for JSON chunk"
        );
        assert_eq!(
            usage.completion_tokens,
            Some(25),
            "completion_tokens should match output_tokens for JSON chunk"
        );
    }

    #[test]
    fn test_convert_chunk_usage_stripped_when_disabled() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let chunk = InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![],
            usage: Some(Usage {
                input_tokens: Some(100),
                output_tokens: Some(200),
            }),
            raw_usage: None,
            finish_reason: None,
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            false, // include_usage = false
            false, // include_raw_usage
            false, // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.usage.is_none(),
            "usage should be stripped when include_usage is false, even if input chunk has usage"
        );
    }

    #[test]
    fn test_convert_chunk_original_response_included_when_enabled() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let raw_response =
            r#"{"id": "chatcmpl-123", "object": "chat.completion.chunk"}"#.to_string();
        let chunk = InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "Hello".to_string(),
            })],
            usage: None,
            raw_usage: None,
            finish_reason: None,
            original_chunk: Some(raw_response.clone()),
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            false, // include_usage
            false, // include_raw_usage
            true,  // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.tensorzero_original_chunk.is_some(),
            "tensorzero_original_chunk should be included when include_original_response is true"
        );
        assert_eq!(
            chunk.tensorzero_original_chunk.as_ref().unwrap(),
            &raw_response,
            "tensorzero_original_chunk should match the input original_chunk"
        );
    }

    #[test]
    fn test_convert_chunk_original_response_omitted_when_disabled() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let raw_response =
            r#"{"id": "chatcmpl-123", "object": "chat.completion.chunk"}"#.to_string();
        let chunk = InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "Hello".to_string(),
            })],
            usage: None,
            raw_usage: None,
            finish_reason: None,
            original_chunk: Some(raw_response),
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            false, // include_usage
            false, // include_raw_usage
            false, // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.tensorzero_original_chunk.is_none(),
            "tensorzero_original_chunk should be None when include_original_response is false"
        );
    }

    #[test]
    fn test_convert_json_chunk_original_response_included_when_enabled() {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let raw_response =
            r#"{"id": "chatcmpl-456", "object": "chat.completion.chunk"}"#.to_string();
        let chunk = InferenceResponseChunk::Json(JsonInferenceResponseChunk {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            raw: r#"{"key": "value"}"#.to_string(),
            usage: None,
            raw_usage: None,
            finish_reason: None,
            original_chunk: Some(raw_response.clone()),
            raw_chunk: None,
            raw_response: None,
        });

        let mut state = StreamingContentState::default();
        let result = convert_inference_response_chunk_to_openai_compatible(
            chunk,
            &mut state,
            "test_prefix::",
            true,
            false, // include_usage
            false, // include_raw_usage
            true,  // include_original_response
            false, // include_raw_response
        );

        assert_eq!(result.len(), 1, "should produce one chunk");
        let chunk = &result[0];
        assert!(
            chunk.tensorzero_original_chunk.is_some(),
            "tensorzero_original_chunk should be included for JSON chunks when include_original_response is true"
        );
        assert_eq!(
            chunk.tensorzero_original_chunk.as_ref().unwrap(),
            &raw_response,
            "tensorzero_original_chunk should match the input original_chunk for JSON chunks"
        );
    }

    #[test]
    fn test_process_chat_content_chunk() {
        let content = vec![
            ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "Hello".to_string(),
            }),
            ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "1".to_string(),
                raw_name: Some("test_tool".to_string()),
                raw_arguments: "{}".to_string(),
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "2".to_string(),
                text: ", world!".to_string(),
            }),
        ];
        let mut state = StreamingContentState::default();
        let (content_str, tool_calls, extra_content) =
            process_chat_content_chunk(content, &mut state);
        assert_eq!(content_str, Some("Hello, world!".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, Some("1".to_string()));
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[0].function.name, "test_tool".to_string());
        assert_eq!(tool_calls[0].function.arguments, "{}");
        assert!(extra_content.is_empty());

        let content: Vec<ContentBlockChunk> = vec![];
        let (content_str, tool_calls, extra_content) =
            process_chat_content_chunk(content, &mut state);
        assert_eq!(content_str, None);
        assert!(tool_calls.is_empty());
        assert!(extra_content.is_empty());

        let content = vec![
            ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "First part".to_string(),
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "2".to_string(),
                text: " second part".to_string(),
            }),
            ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "123".to_string(),
                raw_name: Some("middle_tool".to_string()),
                raw_arguments: "{\"key\": \"value\"}".to_string(),
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "3".to_string(),
                text: " third part".to_string(),
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "4".to_string(),
                text: " fourth part".to_string(),
            }),
            ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "5".to_string(),
                raw_name: Some("last_tool".to_string()),
                raw_arguments: "{\"key\": \"value\"}".to_string(),
            }),
        ];
        let mut state = StreamingContentState::default();
        let (content_str, tool_calls, extra_content) =
            process_chat_content_chunk(content, &mut state);
        assert_eq!(
            content_str,
            Some("First part second part third part fourth part".to_string())
        );
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].id, Some("123".to_string()));
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[0].function.name, "middle_tool".to_string());
        assert_eq!(tool_calls[0].function.arguments, "{\"key\": \"value\"}");
        assert_eq!(tool_calls[1].id, Some("5".to_string()));
        assert_eq!(tool_calls[1].index, 1);
        assert_eq!(tool_calls[1].function.name, "last_tool".to_string());
        assert_eq!(tool_calls[1].function.arguments, "{\"key\": \"value\"}");
        assert!(extra_content.is_empty());
    }

    #[test]
    fn test_process_chat_content_chunk_with_extra_content() {
        // Content order: [Thought(0), Text(1), Unknown(2), Thought(same ID), Thought(3)]
        // The insert_index now reflects position in the FULL content array
        let content = vec![
            ContentBlockChunk::Thought(ThoughtChunk {
                id: "thought_1".to_string(),
                text: Some("Let me think...".to_string()),
                signature: Some("sig123".to_string()),
                summary_id: None,
                summary_text: None,
                provider_type: Some("anthropic".to_string()),
                extra_data: None,
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "text_1".to_string(),
                text: "Response text".to_string(),
            }),
            ContentBlockChunk::Unknown(UnknownChunk {
                id: "unknown_1".to_string(),
                data: serde_json::json!({"custom": "data"}),
                model_name: Some("test_model".to_string()),
                provider_name: None,
            }),
            ContentBlockChunk::Thought(ThoughtChunk {
                id: "thought_1".to_string(), // Same ID, new chunk
                text: Some(" more thinking".to_string()),
                signature: None,
                summary_id: None,
                summary_text: None,
                provider_type: None,
                extra_data: None,
            }),
            ContentBlockChunk::Thought(ThoughtChunk {
                id: "thought_2".to_string(), // New thought
                text: Some("Different thought".to_string()),
                signature: None,
                summary_id: None,
                summary_text: None,
                provider_type: None,
                extra_data: None,
            }),
        ];

        let mut state = StreamingContentState::default();
        let (content_str, tool_calls, extra_content) =
            process_chat_content_chunk(content, &mut state);

        assert_eq!(
            content_str,
            Some("Response text".to_string()),
            "Text content should be extracted"
        );
        assert!(tool_calls.is_empty(), "No tool calls expected");
        assert_eq!(
            extra_content.len(),
            4,
            "Should have four extra content chunks"
        );

        // First: Thought chunk - position 0 in full content array
        match &extra_content[0] {
            ExtraContentBlockChunk::Thought {
                insert_index,
                chunk,
            } => {
                assert_eq!(
                    *insert_index, 0,
                    "First thought should have insert_index 0 (first in content array)"
                );
                assert_eq!(chunk.id, "thought_1");
                assert_eq!(chunk.text, Some("Let me think...".to_string()));
                assert_eq!(chunk.signature, Some("sig123".to_string()));
                assert_eq!(chunk.provider_type, Some("anthropic".to_string()));
            }
            ExtraContentBlockChunk::Unknown { .. } => panic!("Expected Thought at position 0"),
        }

        // Second: Unknown chunk - position 2 in full content array (after Thought@0 and Text@1)
        match &extra_content[1] {
            ExtraContentBlockChunk::Unknown {
                insert_index,
                chunk,
            } => {
                assert_eq!(
                    *insert_index, 2,
                    "Unknown should have insert_index 2 (after thought_1 and text_1)"
                );
                assert_eq!(chunk.id, "unknown_1");
                assert_eq!(chunk.data, serde_json::json!({"custom": "data"}));
                assert_eq!(chunk.model_name, Some("test_model".to_string()));
            }
            ExtraContentBlockChunk::Thought { .. } => panic!("Expected Unknown at position 1"),
        }

        // Third: Second Thought chunk (same ID, same insert_index)
        match &extra_content[2] {
            ExtraContentBlockChunk::Thought {
                insert_index,
                chunk,
            } => {
                assert_eq!(
                    *insert_index, 0,
                    "Second chunk of thought_1 should still have insert_index 0"
                );
                assert_eq!(chunk.id, "thought_1");
                assert_eq!(chunk.text, Some(" more thinking".to_string()));
            }
            ExtraContentBlockChunk::Unknown { .. } => panic!("Expected Thought at position 2"),
        }

        // Fourth: New Thought chunk - position 3 in full content array
        match &extra_content[3] {
            ExtraContentBlockChunk::Thought {
                insert_index,
                chunk,
            } => {
                assert_eq!(
                    *insert_index, 3,
                    "New thought_2 should have insert_index 3 (after thought_1, text_1, unknown_1)"
                );
                assert_eq!(chunk.id, "thought_2");
                assert_eq!(chunk.text, Some("Different thought".to_string()));
            }
            ExtraContentBlockChunk::Unknown { .. } => panic!("Expected Thought at position 3"),
        }

        // Verify the state tracked the IDs correctly
        assert_eq!(
            state.extra_id_to_index.get("thought_1"),
            Some(&0),
            "thought_1 should be mapped to index 0"
        );
        assert_eq!(
            state.extra_id_to_index.get("unknown_1"),
            Some(&2),
            "unknown_1 should be mapped to index 2"
        );
        assert_eq!(
            state.extra_id_to_index.get("thought_2"),
            Some(&3),
            "thought_2 should be mapped to index 3"
        );
    }

    /// Test that demonstrates the bug fix: text before thought should give thought insert_index > 0.
    ///
    /// Previously, the streaming code tracked insert_index only by counting extra content IDs,
    /// so a stream of [Text, Thought] would give Thought insert_index=0 instead of 1.
    /// This test verifies the fix: insert_index now reflects position in the full content array.
    #[test]
    fn test_streaming_insert_index_text_before_thought() {
        // Scenario: Text appears before Thought in the stream
        // Expected: Thought should have insert_index=1 (not 0)
        let content = vec![
            ContentBlockChunk::Text(TextChunk {
                id: "text_1".to_string(),
                text: "Hello ".to_string(),
            }),
            ContentBlockChunk::Thought(ThoughtChunk {
                id: "thought_1".to_string(),
                text: Some("Thinking...".to_string()),
                signature: None,
                summary_id: None,
                summary_text: None,
                provider_type: None,
                extra_data: None,
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "text_1".to_string(), // Same text block, more content
                text: "world!".to_string(),
            }),
        ];

        let mut state = StreamingContentState::default();
        let (content_str, tool_calls, extra_content) =
            process_chat_content_chunk(content, &mut state);

        assert_eq!(
            content_str,
            Some("Hello world!".to_string()),
            "Text content should be concatenated"
        );
        assert!(tool_calls.is_empty(), "No tool calls expected");
        assert_eq!(
            extra_content.len(),
            1,
            "Should have one extra content chunk"
        );

        // The key assertion: Thought should have insert_index=1 because Text came first
        match &extra_content[0] {
            ExtraContentBlockChunk::Thought {
                insert_index,
                chunk,
            } => {
                assert_eq!(
                    *insert_index, 1,
                    "Thought should have insert_index=1 (after Text at position 0)"
                );
                assert_eq!(chunk.id, "thought_1");
            }
            ExtraContentBlockChunk::Unknown { .. } => panic!("Expected Thought"),
        }

        // Verify state tracking
        assert_eq!(
            state.content_block_count, 2,
            "Should have counted 2 content blocks (text + thought)"
        );
        assert!(
            state.text_ids_seen.contains("text_1"),
            "Text ID should be tracked"
        );
        assert_eq!(
            state.extra_id_to_index.get("thought_1"),
            Some(&1),
            "thought_1 should be mapped to index 1"
        );
    }

    #[test]
    fn test_extra_content_block_chunk_serialization() {
        // Test Thought variant serialization
        let thought_chunk = ExtraContentBlockChunk::Thought {
            insert_index: 2,
            chunk: ThoughtChunk {
                id: "chunk_id".to_string(),
                text: Some("Thinking...".to_string()),
                signature: Some("sig_chunk".to_string()),
                summary_id: Some("summary_1".to_string()),
                summary_text: Some("Summary text".to_string()),
                provider_type: Some("anthropic".to_string()),
                extra_data: None,
            },
        };
        let json = serde_json::to_value(&thought_chunk).unwrap();

        // With tagged enum + flatten, type tag and flattened fields at top level
        assert_eq!(json["type"], "thought");
        assert_eq!(json["insert_index"], 2);
        assert_eq!(json["id"], "chunk_id");
        assert_eq!(json["text"], "Thinking...");
        assert_eq!(json["signature"], "sig_chunk");
        assert_eq!(json["summary_id"], "summary_1");
        assert_eq!(json["summary_text"], "Summary text");
        assert_eq!(json["provider_type"], "anthropic");

        // Test Unknown variant serialization
        let unknown_chunk = ExtraContentBlockChunk::Unknown {
            insert_index: 5,
            chunk: UnknownChunk {
                id: "unknown_id".to_string(),
                data: serde_json::json!({"custom": "data"}),
                model_name: Some("test_model".to_string()),
                provider_name: None,
            },
        };
        let json = serde_json::to_value(&unknown_chunk).unwrap();

        assert_eq!(json["type"], "unknown");
        assert_eq!(json["insert_index"], 5);
        assert_eq!(json["id"], "unknown_id");
        assert_eq!(json["data"], serde_json::json!({"custom": "data"}));
        assert_eq!(json["model_name"], "test_model");
    }
}
