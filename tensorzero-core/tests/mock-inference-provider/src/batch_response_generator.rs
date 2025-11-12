use serde_json::{json, Map, Value};
use uuid::Uuid;

use std::collections::HashSet;

/// Shared types and utilities for generating mock batch responses

#[derive(Debug, Clone)]
pub struct ToolCallSpec {
    pub name: String,
    pub args: Value,
}

type ToolResults = (Vec<(String, String)>, Option<String>);

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
        "self_destruct" => json!({
            "fast": true
        }),
        _ => json!({}),
    }
}

/// Generate a simple text response based on context
pub fn generate_simple_text_from_request(request: &serde_json::Value) -> String {
    if let Some(summary) = generate_tool_result_summary(request) {
        return summary;
    }

    if let Some(messages) = extract_last_user_message_text(request) {
        let lower = messages.to_lowercase();
        if lower.contains("capital city of japan") {
            return "Tokyo is the capital city of Japan, known for its vibrant culture and history.".to_string();
        }
        if lower.contains("tokyo") {
            return "Tokyo is the capital city of Japan, known for its blend of traditional culture and modern technology.".to_string();
        }
        if lower.contains("name") {
            return "I am Dr. Mehta, a helpful assistant.".to_string();
        }
        if lower.contains("animal") || lower.contains("image") {
            return "This is a cartoon crab, specifically Ferris the Rust mascot.".to_string();
        }
        if lower.contains("weather") {
            return "Tokyo typically experiences mild weather with moderate humidity around 30%."
                .to_string();
        }
    }

    "This is a mock response for batch inference.".to_string()
}

/// Generate a simple text response (generic fallback)
pub fn generate_simple_text() -> String {
    "This is a mock response for batch inference.".to_string()
}

/// Extract the last user message content from a request
fn extract_last_user_message_text(request: &serde_json::Value) -> Option<String> {
    if let Some(messages) = request.get("messages").and_then(|m| m.as_array()) {
        let user_msg = messages
            .iter()
            .rev()
            .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"));

        if let Some(msg) = user_msg {
            if let Some(content) = msg.get("content") {
                if let Some(text) = extract_text_from_openai_content(content) {
                    return Some(text);
                }
            }
        }
        return None;
    }

    if let Some(contents) = request.get("contents").and_then(|c| c.as_array()) {
        return contents
            .iter()
            .rev()
            .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"))
            .and_then(|msg| msg.get("parts"))
            .and_then(|parts| parts.as_array())
            .and_then(|parts_arr| collect_text_from_gcp_parts(parts_arr))
            .map(|texts| texts.join(" "));
    }

    None
}

fn extract_text_from_openai_content(content: &Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }

    if let Some(content_array) = content.as_array() {
        let mut text_parts: Vec<String> = Vec::new();
        for part in content_array {
            match part.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        text_parts.push(text.to_string());
                    }
                }
                Some("raw_text") => {
                    if let Some(text) = part.get("value").and_then(|t| t.as_str()) {
                        text_parts.push(text.to_string());
                    }
                }
                _ => {}
            }
        }

        if !text_parts.is_empty() {
            return Some(text_parts.join(" "));
        }
    }

    None
}

fn collect_text_from_gcp_parts(parts: &[Value]) -> Option<Vec<String>> {
    let mut collected = Vec::new();
    for part in parts {
        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            collected.push(text.to_string());
        }
    }
    if collected.is_empty() {
        None
    } else {
        Some(collected)
    }
}

fn extract_tool_results(request: &Value) -> Option<ToolResults> {
    if let Some(messages) = request.get("messages").and_then(|m| m.as_array()) {
        if let Some(result) = extract_tool_results_from_openai(messages) {
            return Some(result);
        }
    }

    if let Some(contents) = request.get("contents").and_then(|c| c.as_array()) {
        if let Some(result) = extract_tool_results_from_gcp(contents) {
            return Some(result);
        }
    }

    None
}

fn extract_tool_results_from_openai(messages: &[Value]) -> Option<ToolResults> {
    let mut results = Vec::new();

    // Check for OpenAI format: role="tool" messages
    for msg in messages.iter().rev() {
        if msg.get("role").and_then(|r| r.as_str()) == Some("tool") {
            // OpenAI format: {"role": "tool", "content": "result", "tool_call_id": "123"}
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            // Try to find the tool name from the corresponding tool call
            let tool_call_id = msg.get("tool_call_id").and_then(|id| id.as_str());
            let name =
                find_tool_name_by_id(messages, tool_call_id).unwrap_or_else(|| "tool".to_string());
            results.push((name, content.to_string()));
        }
    }

    if results.is_empty() {
        return None;
    }

    let location = extract_location_from_openai_messages(messages);
    Some((results, location))
}

fn find_tool_name_by_id(messages: &[Value], tool_call_id: Option<&str>) -> Option<String> {
    let tool_call_id = tool_call_id?;

    for msg in messages.iter().rev() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }

        if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
            for call in tool_calls {
                if call.get("id").and_then(|id| id.as_str()) == Some(tool_call_id) {
                    if let Some(function) = call.get("function") {
                        if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                            return Some(name.to_string());
                        }
                    }
                }
            }
        }
    }

    None
}

fn extract_location_from_openai_messages(messages: &[Value]) -> Option<String> {
    for message in messages.iter().rev() {
        if message.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }

        if let Some(content_array) = message.get("content").and_then(|c| c.as_array()) {
            for block in content_array {
                if block.get("type").and_then(|t| t.as_str()) == Some("tool_call") {
                    if let Some(location) = extract_location_from_tool_call_block(block) {
                        return Some(location);
                    }
                }
            }
        }

        if let Some(tool_calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tool_call in tool_calls {
                if let Some(function) = tool_call.get("function") {
                    if let Some(arguments) = function.get("arguments") {
                        if let Some(location) = extract_location_from_arguments(arguments) {
                            return Some(location);
                        }
                    }
                }
            }
        }
    }
    None
}

fn extract_tool_results_from_gcp(contents: &[Value]) -> Option<ToolResults> {
    let last_user = contents
        .iter()
        .rev()
        .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"))?;

    let parts = last_user.get("parts").and_then(|p| p.as_array())?;
    let mut results = Vec::new();
    for part in parts {
        if let Some(function_response) = part.get("function_response") {
            let name = function_response
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("tool")
                .to_string();
            let response = function_response.get("response");
            let content_value = response
                .and_then(|r| r.get("content"))
                .cloned()
                .unwrap_or_else(|| response.cloned().unwrap_or(Value::Null));
            let value_string = value_to_string(&content_value);
            results.push((name, value_string));
        }
    }

    if results.is_empty() {
        return None;
    }

    let location = extract_location_from_gcp_contents(contents);
    Some((results, location))
}

fn extract_location_from_gcp_contents(contents: &[Value]) -> Option<String> {
    for message in contents.iter().rev() {
        if message.get("role").and_then(|r| r.as_str()) != Some("model") {
            continue;
        }

        if let Some(parts) = message.get("parts").and_then(|p| p.as_array()) {
            for part in parts {
                if let Some(function_call) = part.get("functionCall") {
                    if let Some(args) = function_call.get("args") {
                        if let Some(location) = args.get("location").and_then(|l| l.as_str()) {
                            return Some(location.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

fn extract_location_from_tool_call_block(block: &Value) -> Option<String> {
    if let Some(arguments) = block.get("arguments") {
        if let Some(location) = arguments.get("location").and_then(|l| l.as_str()) {
            return Some(location.to_string());
        }
    }

    if let Some(raw_arguments) = block.get("raw_arguments").and_then(|r| r.as_str()) {
        if let Ok(parsed) = serde_json::from_str::<Value>(raw_arguments) {
            if let Some(location) = parsed.get("location").and_then(|l| l.as_str()) {
                return Some(location.to_string());
            }
        }
    }

    None
}

fn extract_location_from_arguments(arguments: &Value) -> Option<String> {
    if let Some(obj) = arguments.as_object() {
        if let Some(location) = obj.get("location").and_then(|l| l.as_str()) {
            return Some(location.to_string());
        }
    }

    if let Some(text) = arguments.as_str() {
        if let Ok(parsed) = serde_json::from_str::<Value>(text) {
            if let Some(location) = parsed.get("location").and_then(|l| l.as_str()) {
                return Some(location.to_string());
            }
        }
    }

    None
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Array(arr) => arr
            .iter()
            .map(value_to_string)
            .collect::<Vec<_>>()
            .join(", "),
        Value::Object(obj) => {
            if let Some(result) = obj.get("result") {
                value_to_string(result)
            } else {
                serde_json::to_string(obj).unwrap_or_default()
            }
        }
        Value::Null => String::new(),
    }
}

fn format_tool_summary(results: Vec<(String, String)>, location: Option<String>) -> String {
    let location = location.unwrap_or_else(|| "Tokyo".to_string());

    let mut phrases = Vec::new();
    for (name, value) in results {
        let phrase = match name.as_str() {
            "get_temperature" => format!("temperature is {value}"),
            "get_humidity" => format!("humidity is {value}"),
            _ => format!("{name} result is {value}"),
        };
        phrases.push(phrase);
    }

    let summary_body = match phrases.len() {
        0 => String::new(),
        1 => phrases[0].clone(),
        _ => {
            let last = phrases.pop().unwrap();
            format!("{} and {last}", phrases.join(", "))
        }
    };

    if summary_body.is_empty() {
        format!("In {location}, the results look good.")
    } else {
        format!("In {location}, the {summary_body}.")
    }
}

pub fn generate_tool_result_summary(request: &Value) -> Option<String> {
    let (results, location) = extract_tool_results(request)?;
    Some(format_tool_summary(results, location))
}

/// Generate a JSON object response
pub fn generate_json_object() -> Value {
    json!({
        "answer": "Tokyo"
    })
}

pub fn generate_json_object_from_schema(schema: Option<&Value>) -> Value {
    let Some(schema) = schema else {
        return generate_json_object();
    };

    let properties = schema.get("properties").and_then(|p| p.as_object());
    if properties.is_none() {
        return generate_json_object();
    }
    let properties = properties.unwrap();

    let mut required_keys: HashSet<&str> = HashSet::new();
    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for key in required {
            if let Some(key) = key.as_str() {
                required_keys.insert(key);
            }
        }
    }

    let mut output = Map::new();
    if properties.contains_key("response") || required_keys.contains("response") {
        output.insert(
            "response".to_string(),
            Value::String("Tokyo is the capital city of Japan.".to_string()),
        );
    }

    if properties.contains_key("answer") && !output.contains_key("answer") {
        output.insert("answer".to_string(), Value::String("Tokyo".to_string()));
    }

    if output.is_empty() {
        return generate_json_object();
    }

    Value::Object(output)
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
