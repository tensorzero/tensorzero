//! Streaming response types and logic for Anthropic-compatible API.
//!
//! This module provides types and functions for streaming message responses in
//! Server-Sent Events (SSE) format, compatible with Anthropic's Messages API.
//!
//! # Event Types
//!
//! Anthropic uses multiple event types for streaming (unlike OpenAI's single `chunk` event):
//! - `message_start`: Initial metadata (id, type, role, model)
//! - `content_block_start`: Beginning of a content block (text or tool_use)
//! - `content_block_delta`: Incremental content (text or partial JSON)
//! - `content_block_stop`: End of a content block
//! - `message_delta`: Final metadata (stop_reason, usage)
//! - `message_stop`: Stream complete
//!
//! # Example
//!
//! ```rust
//! use tensorzero_core::endpoints::anthropic_compatible::types::streaming::prepare_serialized_anthropic_events;
//!
//! let stream = prepare_serialized_anthropic_events(
//!     inference_stream,
//!     "tensorzero::function_name::".to_string(),
//!     true,  // include_usage
//!     false, // include_raw_usage
//!     false, // include_raw_response
//! );
//! ```

use axum::response::sse::Event;
use futures::Stream;
use serde::Serialize;
use std::collections::HashMap;
use tokio_stream::StreamExt;

use crate::error::{Error, ErrorDetails};
use crate::inference::types::ContentBlockChunk;

use crate::endpoints::anthropic_compatible::types::messages::AnthropicOutputContentBlock;
use crate::endpoints::anthropic_compatible::types::messages::finish_reason_to_anthropic;
use crate::endpoints::anthropic_compatible::types::usage::AnthropicStreamingUsage;
use crate::endpoints::inference::{InferenceResponseChunk, InferenceStream};

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum AnthropicStreamingEventData {
    MessageStart {
        message: AnthropicMessageStart,
    },
    ContentBlockStart {
        content_block: AnthropicContentBlockStart,
        index: u32,
    },
    ContentBlockDelta {
        delta: AnthropicDelta,
        index: u32,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: Option<AnthropicStreamingUsage>,
    },
    MessageStop,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnthropicMessageStart {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: Vec<AnthropicOutputContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum AnthropicContentBlockStart {
    Text {
        index: u32,
    },
    ToolUse {
        id: String,
        name: String,
        index: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnthropicDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub text: Option<String>,
    pub partial_json: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnthropicMessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

/// Converts a TensorZero inference chunk to Anthropic-compatible streaming events.
///
/// # Arguments
/// * `chunk` - The inference response chunk to convert
/// * `response_model_prefix` - Prefix to prepend to the model name in responses
/// * `is_first_chunk` - Whether this is the first chunk (triggers message_start event)
/// * `include_usage` - Whether to include usage information in message_delta
/// * `_include_raw_usage` - Unused (reserved for future use)
/// * `_include_raw_response` - Unused (reserved for future use)
///
/// # Returns
/// A vector of streaming event data that can be serialized to SSE format
///
/// # Event Flow
/// 1. First chunk: `message_start` event
/// 2. Each new content block: `content_block_start` event
/// 3. Content deltas: `content_block_delta` events
/// 4. Each block end: `content_block_stop` event
/// 5. Final chunk: `message_delta` event followed by `message_stop`
pub fn convert_inference_response_chunk_to_anthropic(
    chunk: InferenceResponseChunk,
    response_model_prefix: &str,
    is_first_chunk: bool,
    include_usage: bool,
    _include_raw_usage: bool,
    _include_raw_response: bool,
) -> Vec<AnthropicStreamingEventData> {
    let mut events = Vec::new();

    match chunk {
        InferenceResponseChunk::Chat(c) => {
            // Generate message_start event for first chunk
            if is_first_chunk {
                events.push(AnthropicStreamingEventData::MessageStart {
                    message: AnthropicMessageStart {
                        id: c.inference_id.to_string(),
                        message_type: "message".to_string(),
                        role: "assistant".to_string(),
                        content: Vec::new(), // Will be populated as blocks arrive
                        model: format!("{response_model_prefix}{}", c.variant_name),
                        stop_reason: None,
                    },
                });
            }

            // Process content blocks
            let (text_deltas, tool_calls, has_new_block) = process_chat_content_chunk(c.content);

            // Generate content_block_start for new blocks
            if has_new_block && !text_deltas.is_empty() {
                events.push(AnthropicStreamingEventData::ContentBlockStart {
                    content_block: AnthropicContentBlockStart::Text { index: 0 },
                    index: 0,
                });
            }

            // Generate content_block_delta events
            for (index, text_delta) in text_deltas.iter().enumerate() {
                if !text_delta.is_empty() {
                    events.push(AnthropicStreamingEventData::ContentBlockDelta {
                        delta: AnthropicDelta {
                            delta_type: "text_delta".to_string(),
                            text: Some(text_delta.clone()),
                            partial_json: None,
                        },
                        index: index as u32,
                    });
                }
            }

            // Handle tool calls
            for (index, tool_call) in tool_calls.iter().enumerate() {
                if tool_call.is_new {
                    events.push(AnthropicStreamingEventData::ContentBlockStart {
                        content_block: AnthropicContentBlockStart::ToolUse {
                            id: tool_call.id.clone(),
                            name: tool_call.name.clone(),
                            index: index as u32 + 1,
                        },
                        index: index as u32 + 1,
                    });
                }

                let arguments_delta = if tool_call.arguments_delta.is_empty() {
                    None
                } else {
                    Some(tool_call.arguments_delta.clone())
                };

                if arguments_delta.is_some() {
                    events.push(AnthropicStreamingEventData::ContentBlockDelta {
                        delta: AnthropicDelta {
                            delta_type: "input_json_delta".to_string(),
                            text: None,
                            partial_json: arguments_delta,
                        },
                        index: index as u32 + 1,
                    });
                }
            }

            // Generate message_delta for final chunk (when finish_reason is present)
            if c.finish_reason.is_some() {
                let stop_reason = c.finish_reason.map(finish_reason_to_anthropic);

                let usage = if include_usage {
                    c.usage.map(|u| AnthropicStreamingUsage {
                        input_tokens: Some(u.input_tokens.unwrap_or(0)),
                        output_tokens: Some(u.output_tokens.unwrap_or(0)),
                    })
                } else {
                    None
                };

                events.push(AnthropicStreamingEventData::MessageDelta {
                    delta: AnthropicMessageDelta {
                        stop_reason,
                        stop_sequence: None,
                    },
                    usage,
                });

                events.push(AnthropicStreamingEventData::MessageStop);
            }
        }
        InferenceResponseChunk::Json(c) => {
            // JSON mode - similar to chat but simpler
            if is_first_chunk {
                events.push(AnthropicStreamingEventData::MessageStart {
                    message: AnthropicMessageStart {
                        id: c.inference_id.to_string(),
                        message_type: "message".to_string(),
                        role: "assistant".to_string(),
                        content: vec![],
                        model: format!("{response_model_prefix}{}", c.variant_name),
                        stop_reason: None,
                    },
                });

                events.push(AnthropicStreamingEventData::ContentBlockStart {
                    content_block: AnthropicContentBlockStart::Text { index: 0 },
                    index: 0,
                });
            }

            // Add text delta
            if !c.raw.is_empty() {
                events.push(AnthropicStreamingEventData::ContentBlockDelta {
                    delta: AnthropicDelta {
                        delta_type: "text_delta".to_string(),
                        text: Some(c.raw),
                        partial_json: None,
                    },
                    index: 0,
                });
            }

            if c.finish_reason.is_some() {
                let stop_reason = c.finish_reason.map(finish_reason_to_anthropic);

                let usage = if include_usage {
                    c.usage.map(|u| AnthropicStreamingUsage {
                        input_tokens: Some(u.input_tokens.unwrap_or(0)),
                        output_tokens: Some(u.output_tokens.unwrap_or(0)),
                    })
                } else {
                    None
                };

                events.push(AnthropicStreamingEventData::MessageDelta {
                    delta: AnthropicMessageDelta {
                        stop_reason,
                        stop_sequence: None,
                    },
                    usage,
                });

                events.push(AnthropicStreamingEventData::MessageStop);
            }
        }
    }

    events
}

struct ToolCallDelta {
    id: String,
    name: String,
    arguments_delta: String,
    is_new: bool,
}

fn process_chat_content_chunk(
    content: Vec<ContentBlockChunk>,
) -> (Vec<String>, Vec<ToolCallDelta>, bool) {
    let mut text_deltas = Vec::new();
    let mut tool_calls = HashMap::new();

    for block in content {
        match block {
            ContentBlockChunk::Text(text) => {
                text_deltas.push(text.text);
            }
            ContentBlockChunk::ToolCall(tool_call) => {
                let entry =
                    tool_calls
                        .entry(tool_call.id.clone())
                        .or_insert_with(|| ToolCallDelta {
                            id: tool_call.id,
                            name: tool_call.raw_name.unwrap_or_default(),
                            arguments_delta: String::new(),
                            is_new: true,
                        });

                if !tool_call.raw_arguments.is_empty() {
                    entry.arguments_delta.push_str(&tool_call.raw_arguments);
                    entry.is_new = false;
                }
            }
            ContentBlockChunk::Thought(_) => {
                tracing::warn!(
                    "Ignoring 'thought' content block chunk when constructing Anthropic-compatible response"
                );
            }
            ContentBlockChunk::Unknown(_) => {
                tracing::warn!(
                    "Ignoring 'unknown' content block chunk when constructing Anthropic-compatible response"
                );
            }
        }
    }

    let has_new_block = !text_deltas.is_empty();
    let tool_calls_vec = tool_calls.into_values().collect();

    (text_deltas, tool_calls_vec, has_new_block)
}

/// Prepares an Event for SSE on the way out of the gateway.
/// Converts each InferenceResponseChunk to Anthropic-compatible format and streams it.
pub fn prepare_serialized_anthropic_events(
    mut stream: InferenceStream,
    response_model_prefix: String,
    include_usage: bool,
    include_raw_usage: bool,
    include_raw_response: bool,
) -> impl Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        let mut is_first_chunk = true;

        while let Some(chunk) = stream.next().await {
            let Ok(chunk) = chunk else {
                continue;
            };

            let anthropic_events = convert_inference_response_chunk_to_anthropic(
                chunk,
                &response_model_prefix,
                is_first_chunk,
                include_usage,
                include_raw_usage,
                include_raw_response,
            );

            is_first_chunk = false;

            for event_data in anthropic_events {
                let event_type = get_event_type(&event_data);
                yield Event::default()
                    .event(event_type)
                    .json_data(&event_data)
                    .map_err(|e| {
                        Error::new(ErrorDetails::Inference {
                            message: format!("Failed to convert chunk to Event: {e}"),
                        })
                    });
            }
        }
    }
}

fn get_event_type(event: &AnthropicStreamingEventData) -> &str {
    match event {
        AnthropicStreamingEventData::MessageStart { .. } => "message_start",
        AnthropicStreamingEventData::ContentBlockStart { .. } => "content_block_start",
        AnthropicStreamingEventData::ContentBlockDelta { .. } => "content_block_delta",
        AnthropicStreamingEventData::ContentBlockStop { .. } => "content_block_stop",
        AnthropicStreamingEventData::MessageDelta { .. } => "message_delta",
        AnthropicStreamingEventData::MessageStop => "message_stop",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::inference::ChatInferenceResponseChunk;
    use crate::inference::types::usage::Usage;
    use crate::inference::types::{FinishReason, TextChunk};
    use uuid::Uuid;

    #[test]
    fn test_convert_chat_chunk_first() {
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

        let events = convert_inference_response_chunk_to_anthropic(
            chunk,
            "test_prefix::",
            true,  // is_first_chunk
            true,  // include_usage
            false, // include_raw_usage
            false, // include_raw_response
        );

        assert!(!events.is_empty());
        assert!(matches!(
            events[0],
            AnthropicStreamingEventData::MessageStart { .. }
        ));
    }

    #[test]
    fn test_convert_chat_chunk_final() {
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
            finish_reason: Some(FinishReason::Stop),
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
        });

        let events = convert_inference_response_chunk_to_anthropic(
            chunk,
            "test_prefix::",
            false, // is_first_chunk
            true,  // include_usage
            false, // include_raw_usage
            false, // include_raw_response
        );

        assert!(!events.is_empty());
        let has_delta = events
            .iter()
            .any(|e| matches!(e, AnthropicStreamingEventData::MessageDelta { .. }));
        assert!(has_delta);

        let has_stop = events
            .iter()
            .any(|e| matches!(e, AnthropicStreamingEventData::MessageStop));
        assert!(has_stop);
    }

    #[test]
    fn test_event_type_mapping() {
        assert_eq!(
            get_event_type(&AnthropicStreamingEventData::MessageStart {
                message: AnthropicMessageStart {
                    id: "test".to_string(),
                    message_type: "message".to_string(),
                    role: "assistant".to_string(),
                    content: vec![],
                    model: "test".to_string(),
                    stop_reason: None,
                }
            }),
            "message_start"
        );

        assert_eq!(
            get_event_type(&AnthropicStreamingEventData::MessageStop),
            "message_stop"
        );
    }
}
