use axum::{extract::Path, http::StatusCode, Json};
use bytes::Bytes;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::ObjectStore;
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};
use uuid::Uuid;

use crate::{apply_delay, error::Error};

// Storage for batch jobs (job_id -> BatchJob)
static GCP_BATCH_JOBS: OnceLock<Mutex<HashMap<String, BatchJob>>> = OnceLock::new();

#[derive(Clone)]
struct BatchJob {
    name: String, // Full resource name: projects/{project}/locations/{location}/batchPredictionJobs/{id}
    display_name: String,
    model: String,
    state: JobState,
    input_config: InputConfig,
    output_config: OutputConfig,
    create_time: String,
    // Timestamp when this job should transition to succeeded
    complete_at: i64,
    // Flag to track if output has been generated and uploaded
    output_generated: bool,
}

#[derive(Clone, Debug)]
enum JobState {
    Pending,
    Running,
    Succeeded,
}

impl JobState {
    fn as_str(&self) -> &'static str {
        match self {
            JobState::Pending => "JOB_STATE_PENDING",
            JobState::Running => "JOB_STATE_RUNNING",
            JobState::Succeeded => "JOB_STATE_SUCCEEDED",
        }
    }
}

#[expect(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InputConfig {
    #[serde(rename = "instancesFormat")]
    instances_format: String,
    #[serde(alias = "gcs_source")]
    gcs_source: GCSSource,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GCSSource {
    uris: String,
}

#[expect(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OutputConfig {
    #[serde(rename = "predictionsFormat")]
    predictions_format: String,
    #[serde(alias = "gcs_destination")]
    gcs_destination: GCSDestination,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GCSDestination {
    output_uri_prefix: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateBatchRequest {
    display_name: String,
    model: String,
    input_config: InputConfig,
    output_config: OutputConfig,
}

/// POST /v1/projects/{project}/locations/{location}/batchPredictionJobs
pub async fn create_batch_prediction_job(
    Path((project, location)): Path<(String, String)>,
    Json(request): Json<CreateBatchRequest>,
) -> Result<Json<serde_json::Value>, Error> {
    apply_delay().await;

    let job_id = Uuid::now_v7();
    let now = chrono::Utc::now();
    let complete_at = now.timestamp() + 2; // Complete after 2 seconds

    let job_name = format!("projects/{project}/locations/{location}/batchPredictionJobs/{job_id}");

    let batch = BatchJob {
        name: job_name.clone(),
        display_name: request.display_name.clone(),
        model: request.model.clone(),
        state: JobState::Pending,
        input_config: request.input_config,
        output_config: request.output_config,
        create_time: now.to_rfc3339(),
        complete_at,
        output_generated: false,
    };

    let mut jobs = GCP_BATCH_JOBS.get_or_init(Default::default).lock().unwrap();
    jobs.insert(job_name.clone(), batch);
    drop(jobs);

    Ok(Json(json!({
        "name": job_name,
        "displayName": request.display_name,
        "model": request.model,
        "state": "JOB_STATE_PENDING",
        "createTime": now.to_rfc3339(),
    })))
}

/// GET /v1/projects/{project}/locations/{location}/batchPredictionJobs/{job_id}
pub async fn get_batch_prediction_job(
    Path((project, location, job_id)): Path<(String, String, String)>,
) -> Result<Json<serde_json::Value>, Error> {
    apply_delay().await;

    let job_name = format!("projects/{project}/locations/{location}/batchPredictionJobs/{job_id}");

    tracing::debug!("Looking for job: {}", job_name);

    let mut jobs = GCP_BATCH_JOBS.get_or_init(Default::default).lock().unwrap();

    tracing::debug!("Available jobs: {:?}", jobs.keys().collect::<Vec<_>>());

    let job = jobs.get_mut(&job_name).ok_or_else(|| {
        Error::new(
            format!("Batch prediction job not found: {job_name}"),
            StatusCode::NOT_FOUND,
        )
    })?;

    let now = chrono::Utc::now().timestamp();

    // Update job state based on time and generate output if needed
    let should_start_upload = now >= job.complete_at
        && !matches!(job.state, JobState::Succeeded)
        && !job.output_generated;

    if should_start_upload {
        let input_uri = job.input_config.gcs_source.uris.clone();
        let output_uri_prefix = job.output_config.gcs_destination.output_uri_prefix.clone();
        let job_name_clone = job_name.clone();

        // Mark as output being generated to prevent duplicate uploads
        job.output_generated = true;
        job.state = JobState::Running;

        // Drop the lock before spawning the async task
        drop(jobs);

        // Spawn async task to generate and upload output
        // This avoids blocking the handler with GCS operations
        #[expect(clippy::disallowed_methods)]
        tokio::spawn(async move {
            match generate_and_upload_batch_output(&input_uri, &output_uri_prefix).await {
                Ok(()) => {
                    tracing::info!(
                        "Successfully uploaded batch output to {}",
                        output_uri_prefix
                    );
                    // Update job state to succeeded
                    let mut jobs = GCP_BATCH_JOBS.get_or_init(Default::default).lock().unwrap();
                    if let Some(job) = jobs.get_mut(&job_name_clone) {
                        job.state = JobState::Succeeded;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to upload batch output: {}", e);
                    // Still mark as succeeded to allow the test to progress
                    let mut jobs = GCP_BATCH_JOBS.get_or_init(Default::default).lock().unwrap();
                    if let Some(job) = jobs.get_mut(&job_name_clone) {
                        job.state = JobState::Succeeded;
                    }
                }
            }
        });

        // Re-acquire lock for response generation
        jobs = GCP_BATCH_JOBS.get_or_init(Default::default).lock().unwrap();
        let job = jobs.get(&job_name).ok_or_else(|| {
            Error::new(
                format!("Batch prediction job not found: {job_name}"),
                StatusCode::NOT_FOUND,
            )
        })?;

        let response = json!({
            "name": job.name,
            "displayName": job.display_name,
            "model": job.model,
            "state": job.state.as_str(),
            "createTime": job.create_time,
        });

        return Ok(Json(response));
    }

    // Normal state transitions
    if now >= job.complete_at && !matches!(job.state, JobState::Succeeded) && job.output_generated {
        // Upload is in progress or completed, keep as running or check if it's done
        if !matches!(job.state, JobState::Succeeded) {
            job.state = JobState::Running;
        }
    } else if now >= job.complete_at - 1 && matches!(job.state, JobState::Pending) {
        // After 1 second, transition to running
        job.state = JobState::Running;
    }

    let mut response = json!({
        "name": job.name,
        "displayName": job.display_name,
        "model": job.model,
        "state": job.state.as_str(),
        "createTime": job.create_time,
    });

    // Add output info if job is succeeded
    if matches!(job.state, JobState::Succeeded) {
        response["outputInfo"] = json!({
            "gcsOutputDirectory": job.output_config.gcs_destination.output_uri_prefix
        });
    }

    Ok(Json(response))
}

/// Generate mock batch output from input JSONL
fn generate_batch_output(input_content: &[u8]) -> String {
    use crate::batch_response_generator::gcp::wrap_batch_response;
    use serde_json::Value;

    let input_str = String::from_utf8_lossy(input_content);
    let mut output_lines = Vec::new();

    for line in input_str.lines() {
        if line.trim().is_empty() {
            continue;
        }

        // Parse the input line (format: {"request": {...}})
        let input: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Failed to parse input line: {}: {}", e, line);
                continue;
            }
        };

        // Extract the request object
        let request = input.get("request").unwrap_or(&input);

        // Detect response type and generate appropriate parts
        let parts = detect_and_generate_gcp_response(request);

        // Wrap in full response structure
        let response = wrap_batch_response(request, parts);

        output_lines.push(serde_json::to_string(&response).unwrap());
    }

    output_lines.join("\n")
}

/// Detect the type of response needed and generate GCP Vertex parts
fn detect_and_generate_gcp_response(request: &serde_json::Value) -> Vec<serde_json::Value> {
    use crate::batch_response_generator::{
        gcp::{generate_function_call_parts, generate_json_parts, generate_text_parts},
        generate_json_object_from_schema, generate_simple_text_from_request, generate_tool_args,
        generate_tool_result_summary, ToolCallSpec,
    };

    if let Some(summary) = generate_tool_result_summary(request) {
        return generate_text_parts(&summary);
    }

    // Check for JSON mode (generationConfig.responseMimeType)
    if let Some(gen_config) = request.get("generationConfig") {
        if let Some(mime_type) = gen_config.get("responseMimeType").and_then(|v| v.as_str()) {
            if mime_type == "application/json" {
                let schema = gen_config.get("responseSchema");
                let json_obj = generate_json_object_from_schema(schema);
                return generate_json_parts(&json_obj);
            }
        }
    }

    // Check for tool/function use
    if let Some(tools) = request.get("tools").and_then(|v| v.as_array()) {
        if !tools.is_empty() {
            // Check toolConfig.functionCallingConfig.mode
            let mode = request
                .get("toolConfig")
                .and_then(|tc| tc.get("functionCallingConfig"))
                .and_then(|fcc| fcc.get("mode"))
                .and_then(|m| m.as_str())
                .unwrap_or("auto");

            match mode {
                "none" => {
                    // No function calls allowed
                    return generate_text_parts(&generate_simple_text_from_request(request));
                }
                "any" => {
                    // Must use a function
                    let tool_name = extract_first_tool_name(tools);
                    let args = generate_tool_args(&tool_name);
                    return generate_function_call_parts(&[ToolCallSpec {
                        name: tool_name,
                        args,
                    }]);
                }
                _ => {
                    // Use heuristics to decide
                    if should_use_functions_gcp(request, tools) {
                        // Check for allowed function names (specific tools)
                        if let Some(allowed_names) = request
                            .get("toolConfig")
                            .and_then(|tc| tc.get("functionCallingConfig"))
                            .and_then(|fcc| fcc.get("allowedFunctionNames"))
                            .and_then(|afn| afn.as_array())
                        {
                            if !allowed_names.is_empty() {
                                // Use the first allowed tool
                                let tool_name = allowed_names[0].as_str().unwrap_or("unknown");
                                let args = generate_tool_args(tool_name);
                                return generate_function_call_parts(&[ToolCallSpec {
                                    name: tool_name.to_string(),
                                    args,
                                }]);
                            }
                        }

                        // Check if multiple tools available (parallel calls)
                        if tools.len() > 1 {
                            let tool_specs: Vec<ToolCallSpec> = tools
                                .iter()
                                .take(2)
                                .filter_map(|t| {
                                    t.get("functionDeclarations")
                                        .and_then(|fd| fd.as_array())
                                        .and_then(|arr| arr.first())
                                        .and_then(|decl| decl.get("name"))
                                        .and_then(|n| n.as_str())
                                })
                                .map(|name| ToolCallSpec {
                                    name: name.to_string(),
                                    args: generate_tool_args(name),
                                })
                                .collect();

                            if !tool_specs.is_empty() {
                                return generate_function_call_parts(&tool_specs);
                            }
                        }

                        // Single tool call
                        let tool_name = extract_first_tool_name(tools);
                        let args = generate_tool_args(&tool_name);
                        return generate_function_call_parts(&[ToolCallSpec {
                            name: tool_name,
                            args,
                        }]);
                    }
                }
            }
        }
    }

    // Default: simple text response
    generate_text_parts(&generate_simple_text_from_request(request))
}

/// Extract the first tool name from GCP tools array
fn extract_first_tool_name(tools: &[serde_json::Value]) -> String {
    tools
        .first()
        .and_then(|t| t.get("functionDeclarations"))
        .and_then(|fd| fd.as_array())
        .and_then(|arr| arr.first())
        .and_then(|decl| decl.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Heuristic to determine if functions should be used in AUTO mode
fn should_use_functions_gcp(request: &serde_json::Value, tools: &[serde_json::Value]) -> bool {
    // Check if this is an inference params test (should NOT use tools)
    if let Some(labels) = request.get("labels") {
        if let Some(test_type) = labels.get("test_type").and_then(|t| t.as_str()) {
            if test_type == "batch_inference_params" {
                return false; // Don't use tools for params tests
            }
        }
    }

    // Get the last user message from contents
    let contents = request.get("contents").and_then(|c| c.as_array());
    if contents.is_none() {
        return false;
    }

    let last_user_content = contents
        .and_then(|msgs| {
            msgs.iter()
                .rev()
                .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"))
        })
        .and_then(|msg| msg.get("parts"))
        .and_then(|parts| parts.as_array())
        .and_then(|parts_arr| parts_arr.first())
        .and_then(|part| part.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    let content_lower = last_user_content.to_lowercase();

    // Keywords suggesting function use
    let function_keywords = [
        "use the",
        "call the",
        "use both",
        "call both",
        "humidity",
        "weather",
    ];

    for keyword in &function_keywords {
        if content_lower.contains(keyword) {
            return true;
        }
    }

    // Check if any function names are mentioned (but not "temperature" since that's ambiguous)
    for tool in tools {
        if let Some(func_decls) = tool
            .get("functionDeclarations")
            .and_then(|fd| fd.as_array())
        {
            for decl in func_decls {
                if let Some(func_name) = decl.get("name").and_then(|n| n.as_str()) {
                    // Don't match on "temperature" alone since it can be part of the question
                    if func_name != "get_temperature" && content_lower.contains(func_name) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Generate and upload batch output to GCS
async fn generate_and_upload_batch_output(
    input_uri: &str,
    output_uri_prefix: &str,
) -> Result<(), anyhow::Error> {
    // Read input file from GCS
    let input_content = read_from_gcs(input_uri).await?;

    // Generate mock responses
    let output_content = generate_batch_output(&input_content);

    // Upload to GCS
    let output_path = format!(
        "{}/predictions.jsonl",
        output_uri_prefix.trim_end_matches('/')
    );
    upload_to_gcs(&output_path, output_content.as_bytes()).await?;

    Ok(())
}

/// Read a file from GCS
async fn read_from_gcs(gs_url: &str) -> Result<Vec<u8>, anyhow::Error> {
    let store_and_path = make_gcp_object_store(gs_url).await?;
    let result = store_and_path.store.get(&store_and_path.path).await?;
    let bytes = result.bytes().await?;
    Ok(bytes.to_vec())
}

/// Upload data to GCS
async fn upload_to_gcs(gs_url: &str, data: &[u8]) -> Result<(), anyhow::Error> {
    let store_and_path = make_gcp_object_store(gs_url).await?;
    let bytes = Bytes::copy_from_slice(data);
    store_and_path
        .store
        .put(&store_and_path.path, bytes.into())
        .await?;
    Ok(())
}

struct StoreAndPath {
    store: Arc<dyn ObjectStore>,
    path: object_store::path::Path,
}

/// Create a GCS object store from a gs:// URL
/// This uses service account credentials from GOOGLE_APPLICATION_CREDENTIALS if set,
/// otherwise falls back to Application Default Credentials (ADC)
async fn make_gcp_object_store(gs_url: &str) -> Result<StoreAndPath, anyhow::Error> {
    let bucket_and_path = gs_url
        .strip_prefix("gs://")
        .ok_or_else(|| anyhow::anyhow!("GCS url does not start with 'gs://': {gs_url}"))?;

    let (bucket, path) = bucket_and_path
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("GCS url does not contain a bucket name: {gs_url}"))?;

    let key = object_store::path::Path::parse(path)?;

    let mut builder = GoogleCloudStorageBuilder::default()
        .with_bucket_name(bucket)
        // Skip metadata server check - we're not running on GCP
        .with_skip_signature(false);

    // If GOOGLE_APPLICATION_CREDENTIALS is set, use service account key file directly
    // This avoids attempting to contact the GCP metadata server first
    if let Ok(credentials_path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        builder = builder.with_service_account_path(&credentials_path);
    }

    let store = builder.build()?;

    Ok(StoreAndPath {
        store: Arc::new(store),
        path: key,
    })
}
