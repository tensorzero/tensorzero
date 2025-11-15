use axum::{
    extract::Path,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use uuid::Uuid;

use crate::{apply_delay, error::Error};

// Storage for uploaded files (file_id -> content)
static OPENAI_FILES: OnceLock<Mutex<HashMap<String, FileData>>> = OnceLock::new();

// Storage for batch jobs (batch_id -> BatchJob)
static OPENAI_BATCHES: OnceLock<Mutex<HashMap<String, BatchJob>>> = OnceLock::new();

#[expect(dead_code)]
#[derive(Clone)]
pub struct FileData {
    pub content: String,
    pub filename: String,
    pub purpose: String,
    pub bytes: usize,
    pub created_at: i64,
}

#[derive(Clone)]
struct BatchJob {
    id: String,
    input_file_id: String,
    endpoint: String,
    created_at: i64,
    status: BatchStatus,
    output_file_id: Option<String>,
    request_counts: RequestCounts,
    // Timestamp when this batch should transition to completed
    complete_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum BatchStatus {
    Validating,
    #[serde(rename = "in_progress")]
    InProgress,
    Completed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RequestCounts {
    total: usize,
    completed: usize,
    failed: usize,
}

#[derive(Deserialize)]
pub struct CreateBatchRequest {
    input_file_id: String,
    endpoint: String,
    completion_window: String,
}

/// POST /openai/batches - Create a batch job
pub async fn create_batch(
    Json(request): Json<CreateBatchRequest>,
) -> Result<Json<serde_json::Value>, Error> {
    apply_delay().await;

    // Check if the input file exists
    let files = OPENAI_FILES.get_or_init(Default::default).lock().unwrap();
    let input_file = files.get(&request.input_file_id).ok_or_else(|| {
        Error::new(
            format!("File not found: {}", request.input_file_id),
            StatusCode::NOT_FOUND,
        )
    })?;

    // Parse the input file to count requests
    let line_count = input_file.content.lines().count();
    drop(files);

    let batch_id = format!("batch-{}", Uuid::now_v7());
    let now = chrono::Utc::now().timestamp();

    // Batches complete after 2 seconds
    let complete_at = now + 2;

    let batch = BatchJob {
        id: batch_id.clone(),
        input_file_id: request.input_file_id.clone(),
        endpoint: request.endpoint.clone(),
        created_at: now,
        status: BatchStatus::Validating,
        output_file_id: None,
        request_counts: RequestCounts {
            total: line_count,
            completed: 0,
            failed: 0,
        },
        complete_at,
    };

    let mut batches = OPENAI_BATCHES.get_or_init(Default::default).lock().unwrap();
    batches.insert(batch_id.clone(), batch);
    drop(batches);

    Ok(Json(json!({
        "id": batch_id,
        "object": "batch",
        "endpoint": request.endpoint,
        "errors": null,
        "input_file_id": request.input_file_id,
        "completion_window": request.completion_window,
        "status": "validating",
        "output_file_id": null,
        "error_file_id": null,
        "created_at": now,
        "in_progress_at": null,
        "expires_at": null,
        "finalizing_at": null,
        "completed_at": null,
        "failed_at": null,
        "expired_at": null,
        "cancelling_at": null,
        "cancelled_at": null,
        "request_counts": {
            "total": line_count,
            "completed": 0,
            "failed": 0
        },
        "metadata": {}
    })))
}

/// GET /openai/batches/{batch_id} - Get batch status
pub async fn get_batch(Path(batch_id): Path<String>) -> Result<Json<serde_json::Value>, Error> {
    apply_delay().await;

    let mut batches = OPENAI_BATCHES.get_or_init(Default::default).lock().unwrap();
    let batch = batches.get_mut(&batch_id).ok_or_else(|| {
        Error::new(
            format!("Batch not found: {batch_id}"),
            StatusCode::NOT_FOUND,
        )
    })?;

    let now = chrono::Utc::now().timestamp();

    // Update batch status based on time
    if now >= batch.complete_at && !matches!(batch.status, BatchStatus::Completed) {
        // Generate output file
        let files = OPENAI_FILES.get_or_init(Default::default).lock().unwrap();
        let input_file = files.get(&batch.input_file_id).unwrap();

        // Generate mock output
        let output_content = generate_batch_output(&input_file.content);
        drop(files);

        let output_file_id = format!("file-{}", Uuid::now_v7());
        let mut files = OPENAI_FILES.get_or_init(Default::default).lock().unwrap();
        files.insert(
            output_file_id.clone(),
            FileData {
                content: output_content,
                filename: "batch_output.jsonl".to_string(),
                purpose: "batch_output".to_string(),
                bytes: batch.request_counts.total * 100, // Approximate
                created_at: now,
            },
        );
        drop(files);

        batch.status = BatchStatus::Completed;
        batch.output_file_id = Some(output_file_id.clone());
        batch.request_counts.completed = batch.request_counts.total;
    } else if now >= batch.complete_at - 1 && matches!(batch.status, BatchStatus::Validating) {
        // After 1 second, transition to in_progress
        batch.status = BatchStatus::InProgress;
    }

    let (status_str, output_file_id, completed_at) = match &batch.status {
        BatchStatus::Validating => ("validating", None, None),
        BatchStatus::InProgress => ("in_progress", None, None),
        BatchStatus::Completed => ("completed", batch.output_file_id.as_ref(), Some(now)),
    };

    Ok(Json(json!({
        "id": batch.id,
        "object": "batch",
        "endpoint": batch.endpoint,
        "errors": null,
        "input_file_id": batch.input_file_id,
        "completion_window": "24h",
        "status": status_str,
        "output_file_id": output_file_id,
        "error_file_id": null,
        "created_at": batch.created_at,
        "in_progress_at": if matches!(batch.status, BatchStatus::InProgress | BatchStatus::Completed) { Some(batch.created_at + 1) } else { None },
        "expires_at": null,
        "finalizing_at": if matches!(batch.status, BatchStatus::Completed) { Some(now) } else { None },
        "completed_at": completed_at,
        "failed_at": null,
        "expired_at": null,
        "cancelling_at": null,
        "cancelled_at": null,
        "request_counts": batch.request_counts,
        "metadata": {}
    })))
}

/// GET /openai/files/{file_id}/content - Download file content
pub async fn get_file_content(Path(file_id): Path<String>) -> Response {
    apply_delay().await;

    let files = OPENAI_FILES.get_or_init(Default::default).lock().unwrap();
    let file = match files.get(&file_id) {
        Some(f) => f.clone(),
        None => {
            return Error::new(format!("File not found: {file_id}"), StatusCode::NOT_FOUND)
                .into_response()
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(file.content.into())
        .unwrap()
}

/// Generate mock batch output from input
fn generate_batch_output(input_content: &str) -> String {
    use crate::batch_response_generator::openai::wrap_batch_response;
    use serde_json::Value;

    let mut output_lines = Vec::new();

    for line in input_content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        // Parse the input line
        let input: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Failed to parse input line: {}: {}", e, line);
                continue;
            }
        };

        // DEBUG: Print the body to see the response_format
        let body = input.get("body");
        if let Some(body_val) = body {
            if let Some(response_format) = body_val.get("response_format") {
                tracing::warn!(
                    "OpenAI batch response_format: {}",
                    serde_json::to_string_pretty(response_format).unwrap_or_default()
                );
            }
        }

        let custom_id = input
            .get("custom_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Detect response type and generate appropriate message
        let (message, finish_reason) = detect_and_generate_response(body);

        let response = wrap_batch_response(custom_id, message, &finish_reason);
        output_lines.push(serde_json::to_string(&response).unwrap());
    }

    output_lines.join("\n")
}

/// Detect the type of response needed and generate it
fn detect_and_generate_response(body: Option<&serde_json::Value>) -> (serde_json::Value, String) {
    use crate::batch_response_generator::{
        generate_json_object, generate_json_object_from_schema, generate_simple_text,
        generate_simple_text_from_request, generate_tool_args, generate_tool_result_summary,
        openai::{generate_json_message, generate_text_message, generate_tool_call_message},
        ToolCallSpec,
    };
    use serde_json::Value;

    let Some(body) = body else {
        return (
            generate_text_message(&generate_simple_text()),
            "stop".to_string(),
        );
    };

    if let Some(summary) = generate_tool_result_summary(body) {
        return (generate_text_message(&summary), "stop".to_string());
    }

    // Check for JSON mode (response_format)
    if let Some(response_format) = body.get("response_format") {
        if let Some(format_type) = response_format.get("type").and_then(|v| v.as_str()) {
            if format_type == "json_object" {
                return (
                    generate_json_message(&generate_json_object()),
                    "stop".to_string(),
                );
            }
            if format_type == "json_schema" {
                let schema = response_format
                    .get("json_schema")
                    .and_then(|json_schema| json_schema.get("schema").or(Some(json_schema)));
                let json_obj = generate_json_object_from_schema(schema);
                return (generate_json_message(&json_obj), "stop".to_string());
            }
        }
    }

    // Check for tool use
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        if !tools.is_empty() {
            // Check tool_choice
            let tool_choice = body.get("tool_choice");

            match tool_choice {
                // Explicit "none" - no tool calls
                Some(Value::String(s)) if s == "none" => {
                    return (
                        generate_text_message(&generate_simple_text_from_request(body)),
                        "stop".to_string(),
                    );
                }
                // "required" - must use tools
                Some(Value::String(s)) if s == "required" => {
                    let tool_name = tools[0]
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown");
                    let args = generate_tool_args(tool_name);
                    return (
                        generate_tool_call_message(&[ToolCallSpec {
                            name: tool_name.to_string(),
                            args,
                        }]),
                        "tool_calls".to_string(),
                    );
                }
                // Specific tool choice
                Some(Value::Object(obj)) if obj.contains_key("function") => {
                    let tool_name = obj
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown");
                    let args = generate_tool_args(tool_name);
                    return (
                        generate_tool_call_message(&[ToolCallSpec {
                            name: tool_name.to_string(),
                            args,
                        }]),
                        "tool_calls".to_string(),
                    );
                }
                // "auto" or default - use heuristics
                _ => {
                    if should_use_tools(body, tools) {
                        // Check if parallel tool calls
                        let parallel = body
                            .get("parallel_tool_calls")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);

                        if parallel && tools.len() > 1 {
                            // Generate parallel tool calls
                            let tool_specs: Vec<ToolCallSpec> = tools
                                .iter()
                                .take(2) // Use first 2 tools for parallel
                                .filter_map(|t| {
                                    t.get("function")
                                        .and_then(|f| f.get("name"))
                                        .and_then(|n| n.as_str())
                                })
                                .map(|name| ToolCallSpec {
                                    name: name.to_string(),
                                    args: generate_tool_args(name),
                                })
                                .collect();
                            return (
                                generate_tool_call_message(&tool_specs),
                                "tool_calls".to_string(),
                            );
                        } else {
                            // Single tool call
                            let tool_name = tools[0]
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");
                            let args = generate_tool_args(tool_name);
                            return (
                                generate_tool_call_message(&[ToolCallSpec {
                                    name: tool_name.to_string(),
                                    args,
                                }]),
                                "tool_calls".to_string(),
                            );
                        }
                    }
                }
            }
        }
    }

    // Default: simple text response
    (
        generate_text_message(&generate_simple_text_from_request(body)),
        "stop".to_string(),
    )
}

/// Heuristic to determine if tools should be used in "auto" mode
fn should_use_tools(body: &serde_json::Value, tools: &[serde_json::Value]) -> bool {
    // Get the last user message
    let messages = body.get("messages").and_then(|m| m.as_array());
    if messages.is_none() {
        return false;
    }

    let last_message = messages.and_then(|msgs| {
        msgs.iter()
            .rev()
            .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"))
    });

    if last_message.is_none() {
        return false;
    }

    let content = last_message
        .and_then(|msg| msg.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");

    let content_lower = content.to_lowercase();

    // Keywords suggesting tool use
    let tool_keywords = [
        "use the",
        "call the",
        "use both",
        "call both",
        "temperature",
        "humidity",
        "weather",
    ];

    for keyword in &tool_keywords {
        if content_lower.contains(keyword) {
            return true;
        }
    }

    // Check if any tool names are mentioned
    for tool in tools {
        if let Some(tool_name) = tool
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
        {
            if content_lower.contains(tool_name) {
                return true;
            }
        }
    }

    false
}

/// Store file data for testing
pub fn store_file(file_id: String, data: FileData) {
    let mut files = OPENAI_FILES.get_or_init(Default::default).lock().unwrap();
    files.insert(file_id, data);
}

/// Export FileData for use in main.rs
pub use FileData as OpenAIFileData;
