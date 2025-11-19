//! Streaming response types and logic for OpenAI-compatible API.
//!
//! This module provides types and functions for streaming chat completion responses
//! in Server-Sent Events (SSE) format, compatible with OpenAI's streaming API.
//! It handles chunk formatting, delta updates, and usage reporting for streaming responses.

use axum::response::sse::Event;
use futures::Stream;
use serde::Serialize;
use std::collections::HashMap;
use tokio_stream::StreamExt;

use crate::error::{Error, ErrorDetails};
use crate::inference::types::{current_timestamp, ContentBlockChunk, FinishReason};

use crate::endpoints::inference::{InferenceResponseChunk, InferenceStream};

use super::chat_completions::{OpenAICompatibleFinishReason, OpenAICompatibleStreamOptions};
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
    pub usage: Option<OpenAICompatibleUsage>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleChoiceChunk {
    pub index: u32,
    pub finish_reason: Option<OpenAICompatibleFinishReason>,
    pub logprobs: Option<()>, // This is always set to None for now
    pub delta: OpenAICompatibleDelta,
}

fn is_none_or_empty<T>(v: &Option<Vec<T>>) -> bool {
    // if it's None -> skip, or if the Vec is empty -> skip
    v.as_ref().is_none_or(Vec::is_empty)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "is_none_or_empty")]
    pub tool_calls: Option<Vec<OpenAICompatibleToolCallChunk>>,
}

pub fn convert_inference_response_chunk_to_openai_compatible(
    chunk: InferenceResponseChunk,
    tool_id_to_index: &mut HashMap<String, usize>,
    response_model_prefix: &str,
) -> Vec<OpenAICompatibleResponseChunk> {
    let response_chunk = match chunk {
        InferenceResponseChunk::Chat(c) => {
            let (content, tool_calls) = process_chat_content_chunk(c.content, tool_id_to_index);
            OpenAICompatibleResponseChunk {
                id: c.inference_id.to_string(),
                episode_id: c.episode_id.to_string(),
                choices: vec![OpenAICompatibleChoiceChunk {
                    index: 0,
                    finish_reason: c.finish_reason.map(FinishReason::into),
                    logprobs: None,
                    delta: OpenAICompatibleDelta {
                        content,
                        tool_calls: Some(tool_calls),
                    },
                }],
                created: current_timestamp() as u32,
                service_tier: None,
                model: format!("{response_model_prefix}{}", c.variant_name),
                system_fingerprint: String::new(),
                object: "chat.completion.chunk".to_string(),
                // We emit a single chunk containing 'usage' at the end of the stream
                usage: None,
            }
        }
        InferenceResponseChunk::Json(c) => OpenAICompatibleResponseChunk {
            id: c.inference_id.to_string(),
            episode_id: c.episode_id.to_string(),
            choices: vec![OpenAICompatibleChoiceChunk {
                index: 0,
                finish_reason: c.finish_reason.map(FinishReason::into),
                logprobs: None,
                delta: OpenAICompatibleDelta {
                    content: Some(c.raw),
                    tool_calls: None,
                },
            }],
            created: current_timestamp() as u32,
            service_tier: None,
            model: format!("{response_model_prefix}{}", c.variant_name),
            system_fingerprint: String::new(),
            object: "chat.completion.chunk".to_string(),
            // We emit a single chunk containing 'usage' at the end of the stream
            usage: None,
        },
    };

    vec![response_chunk]
}

pub fn process_chat_content_chunk(
    content: Vec<ContentBlockChunk>,
    tool_id_to_index: &mut HashMap<String, usize>,
) -> (Option<String>, Vec<OpenAICompatibleToolCallChunk>) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    for block in content {
        match block {
            ContentBlockChunk::Text(text) => match content_str {
                Some(ref mut content) => content.push_str(&text.text),
                None => content_str = Some(text.text),
            },
            ContentBlockChunk::ToolCall(tool_call) => {
                let len = tool_id_to_index.len();
                let is_new = !tool_id_to_index.contains_key(&tool_call.id);
                let index = tool_id_to_index.entry(tool_call.id.clone()).or_insert(len);
                tool_calls.push(OpenAICompatibleToolCallChunk {
                    id: if is_new { Some(tool_call.id) } else { None },
                    index: *index,
                    r#type: "function".to_string(),
                    function: OpenAICompatibleToolCallDelta {
                        name: tool_call.raw_name.unwrap_or_default(),
                        arguments: tool_call.raw_arguments,
                    },
                });
            }
            ContentBlockChunk::Thought(_thought) => {
                // OpenAI compatible endpoint does not support thought blocks
                // Users of this endpoint will need to check observability to see them
                tracing::warn!(
                    "Ignoring 'thought' content block chunk when constructing OpenAI-compatible response"
                );
            }
            ContentBlockChunk::Unknown(_) => {
                // OpenAI compatible endpoint does not support unknown blocks
                // Users of this endpoint will need to check observability to see them
                tracing::warn!(
                    "Ignoring 'unknown' content block chunk when constructing OpenAI-compatible response"
                );
            }
        }
    }
    (content_str, tool_calls)
}

/// Prepares an Event for SSE on the way out of the gateway
/// When None is passed in, we send "[DONE]" to the client to signal the end of the stream
pub fn prepare_serialized_openai_compatible_events(
    mut stream: InferenceStream,
    response_model_prefix: String,
    stream_options: Option<OpenAICompatibleStreamOptions>,
) -> impl Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        let mut tool_id_to_index = HashMap::new();
        let mut is_first_chunk = true;
        // `total_usage` is `None` until we receive a chunk with usage information
        let mut total_usage: Option<OpenAICompatibleUsage> = None;
        let mut inference_id = None;
        let mut episode_id = None;
        let mut variant_name = None;
        while let Some(chunk) = stream.next().await {
            // NOTE: in the future, we may want to end the stream early if we get an error
            // For now, we just ignore the error and try to get more chunks
            let Ok(chunk) = chunk else {
                continue;
            };
            inference_id = Some(chunk.inference_id());
            episode_id = Some(chunk.episode_id());
            variant_name = Some(chunk.variant_name().to_string());
            let chunk_usage = match &chunk {
                InferenceResponseChunk::Chat(c) => {
                    &c.usage
                }
                InferenceResponseChunk::Json(c) => {
                    &c.usage
                }
            };
            if let Some(chunk_usage) = chunk_usage {
                // `total_usage` will be `None` if this is the first chunk with usage information....
                if total_usage.is_none() {
                    // ... so initialize it to zero ...
                    total_usage = Some(OpenAICompatibleUsage::zero());
                }
                // ...and then add the chunk usage to it (handling `None` fields)
                if let Some(ref mut u) = total_usage { u.sum_usage_strict(chunk_usage); }
            }
            let openai_compatible_chunks = convert_inference_response_chunk_to_openai_compatible(chunk, &mut tool_id_to_index, &response_model_prefix);
            for chunk in openai_compatible_chunks {
                let mut chunk_json = serde_json::to_value(chunk).map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert chunk to JSON: {e}"),
                    })
                })?;
                if is_first_chunk {
                    // OpenAI includes "assistant" role in the first chunk but not in the subsequent chunks
                    chunk_json["choices"][0]["delta"]["role"] = serde_json::Value::String("assistant".to_string());
                    is_first_chunk = false;
                }

                yield Event::default().json_data(chunk_json).map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert Value to Event: {e}"),
                    })
                })
            }
        }

        // If we don't see a chunk with usage information, set `total_usage` to the default value (fields as `None`)
        let total_usage = total_usage.unwrap_or_default();

        if stream_options.map(|s| s.include_usage).unwrap_or(false) {
            let episode_id = episode_id.ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "Cannot find episode_id - no chunks were produced by TensorZero".to_string(),
                })
            })?;
            let inference_id = inference_id.ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "Cannot find inference_id - no chunks were produced by TensorZero".to_string(),
                })
            })?;
            let variant_name = variant_name.ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "Cannot find variant_name - no chunks were produced by TensorZero".to_string(),
                })
            })?;
            let usage_chunk = OpenAICompatibleResponseChunk {
                id: inference_id.to_string(),
                episode_id: episode_id.to_string(),
                choices: vec![],
                created: current_timestamp() as u32,
                model: format!("{response_model_prefix}{variant_name}"),
                system_fingerprint: String::new(),
                object: "chat.completion.chunk".to_string(),
                service_tier: None,
                usage: Some(OpenAICompatibleUsage {
                    prompt_tokens: total_usage.prompt_tokens,
                    completion_tokens: total_usage.completion_tokens,
                    total_tokens: total_usage.total_tokens,
                }),
            };
            yield Event::default().json_data(
                usage_chunk)
                .map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert usage chunk to JSON: {e}"),
                    })
                });
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::TextChunk;
    use crate::tool::ToolCallChunk;

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
        let mut tool_id_to_index = HashMap::new();
        let (content_str, tool_calls) = process_chat_content_chunk(content, &mut tool_id_to_index);
        assert_eq!(content_str, Some("Hello, world!".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, Some("1".to_string()));
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[0].function.name, "test_tool".to_string());
        assert_eq!(tool_calls[0].function.arguments, "{}");

        let content: Vec<ContentBlockChunk> = vec![];
        let (content_str, tool_calls) = process_chat_content_chunk(content, &mut tool_id_to_index);
        assert_eq!(content_str, None);
        assert!(tool_calls.is_empty());

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
        let mut tool_id_to_index = HashMap::new();
        let (content_str, tool_calls) = process_chat_content_chunk(content, &mut tool_id_to_index);
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
    }
}
