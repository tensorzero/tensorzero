use serde_json::{json, Value};
use uuid::Uuid;

/// Shared types and utilities for generating mock batch responses

#[derive(Debug, Clone)]
pub enum MockResponseType {
    SimpleText,
    JsonObject,
    SingleToolCall { name: String, args: Value },
    ParallelToolCalls(Vec<ToolCallSpec>),
}

#[derive(Debug, Clone)]
pub struct ToolCallSpec {
    pub name: String,
    pub args: Value,
}

/// Generate appropriate tool arguments based on tool name
pub fn generate_tool_args(tool_name: &str) -> Value {
    match tool_name {
        "get_temperature" => json!({
            "location": "Tokyo",
            "units": "celsius"
        }),
        "get_humidity" => json!({
            "location": "Tokyo"
        }),
        "self_destruct" => json!({}),
        _ => json!({}),
    }
}

/// Generate a simple text response based on context
pub fn generate_simple_text_from_request(request: &serde_json::Value) -> String {
    // Check if this is a "What is your name?" type question (auto_unused test)
    if let Some(messages) = extract_last_user_message(request) {
        if messages.to_lowercase().contains("name") {
            return "I am Dr. Mehta, a helpful assistant.".to_string();
        }
        if messages.to_lowercase().contains("tokyo") {
            return "Tokyo is the capital city of Japan, known for its blend of traditional culture and modern technology.".to_string();
        }
        // Handle image-related questions (e.g., "What kind of animal is in this image?")
        if messages.to_lowercase().contains("animal") || messages.to_lowercase().contains("image") {
            return "This is a cartoon crab, specifically Ferris the Rust mascot.".to_string();
        }
    }

    // Default response
    "This is a mock response for batch inference.".to_string()
}

/// Generate a simple text response (generic fallback)
pub fn generate_simple_text() -> String {
    "This is a mock response for batch inference.".to_string()
}

/// Extract the last user message content from a request
fn extract_last_user_message(request: &serde_json::Value) -> Option<String> {
    // Try OpenAI format first (messages array)
    if let Some(messages) = request.get("messages").and_then(|m| m.as_array()) {
        let user_msg = messages
            .iter()
            .rev()
            .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"));

        if let Some(msg) = user_msg {
            if let Some(content) = msg.get("content") {
                // Handle string content
                if let Some(text) = content.as_str() {
                    return Some(text.to_string());
                }
                // Handle array content (multimodal messages with text and images)
                if let Some(content_array) = content.as_array() {
                    // Extract text from content parts
                    let text_parts: Vec<String> = content_array
                        .iter()
                        .filter_map(|part| {
                            if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                                part.get("text").and_then(|t| t.as_str()).map(String::from)
                            } else {
                                None
                            }
                        })
                        .collect();

                    if !text_parts.is_empty() {
                        return Some(text_parts.join(" "));
                    }
                }
            }
        }
        return None;
    }

    // Try GCP format (contents array)
    if let Some(contents) = request.get("contents").and_then(|c| c.as_array()) {
        return contents
            .iter()
            .rev()
            .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"))
            .and_then(|msg| msg.get("parts"))
            .and_then(|parts| parts.as_array())
            .and_then(|parts_arr| {
                // Find the first part that has a "text" field (skip image/file parts)
                parts_arr
                    .iter()
                    .find_map(|part| part.get("text").and_then(|t| t.as_str()))
            })
            .map(String::from);
    }

    None
}

/// Generate a JSON object response
pub fn generate_json_object() -> Value {
    json!({
        "answer": "Tokyo"
    })
}

/// OpenAI-specific response generation
pub mod openai {
    use super::*;

    /// Generate an OpenAI message for simple text
    pub fn generate_text_message(text: &str) -> Value {
        json!({
            "role": "assistant",
            "content": text
        })
    }

    /// Generate an OpenAI message for JSON mode
    pub fn generate_json_message(json_obj: &Value) -> Value {
        json!({
            "role": "assistant",
            "content": json_obj.to_string()
        })
    }

    /// Generate an OpenAI message with tool calls
    pub fn generate_tool_call_message(tool_calls: &[ToolCallSpec]) -> Value {
        let tool_calls_array: Vec<Value> = tool_calls
            .iter()
            .map(|spec| {
                json!({
                    "id": format!("call_{}", Uuid::now_v7()),
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "arguments": spec.args.to_string()
                    }
                })
            })
            .collect();

        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls_array
        })
    }

    /// Wrap a message in a full OpenAI batch response
    pub fn wrap_batch_response(custom_id: &str, message: Value, finish_reason: &str) -> Value {
        json!({
            "id": format!("batch_req_{}", Uuid::now_v7()),
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": format!("req_{}", Uuid::now_v7()),
                "body": {
                    "id": format!("chatcmpl-{}", Uuid::now_v7()),
                    "object": "chat.completion",
                    "created": chrono::Utc::now().timestamp(),
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20
                    }
                }
            },
            "error": null
        })
    }
}

/// GCP Vertex-specific response generation
pub mod gcp {
    use super::*;

    /// Generate GCP Vertex parts for simple text
    pub fn generate_text_parts(text: &str) -> Vec<Value> {
        vec![json!({
            "text": text
        })]
    }

    /// Generate GCP Vertex parts for JSON mode (text containing JSON)
    pub fn generate_json_parts(json_obj: &Value) -> Vec<Value> {
        vec![json!({
            "text": json_obj.to_string()
        })]
    }

    /// Generate GCP Vertex parts for function calls
    pub fn generate_function_call_parts(tool_calls: &[ToolCallSpec]) -> Vec<Value> {
        tool_calls
            .iter()
            .map(|spec| {
                json!({
                    "functionCall": {
                        "name": spec.name,
                        "args": spec.args
                    }
                })
            })
            .collect()
    }

    /// Wrap parts in a full GCP Vertex batch response
    pub fn wrap_batch_response(request: &Value, parts: Vec<Value>) -> Value {
        json!({
            "request": request,
            "response": {
                "candidates": [{
                    "content": {
                        "parts": parts,
                        "role": "model"
                    },
                    "finishReason": "STOP",
                    "index": 0
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 20
                }
            }
        })
    }
}
