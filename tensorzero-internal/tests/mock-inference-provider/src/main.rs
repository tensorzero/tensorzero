// This project is used only for testing, so it's fine if it panics
#![allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]

mod error;
mod fireworks;

use async_stream::try_stream;
use axum::http::StatusCode;
use axum::{
    body::Body,
    extract::{Json, Multipart, Path},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
};
use error::Error;
use futures::Stream;
use mimalloc::MiMalloc;
use rand::distr::{Alphanumeric, SampleString};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Mutex, OnceLock},
    time::Duration,
};
use tokio::signal;
use tower_http::trace::TraceLayer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Clone)]
struct FineTuningJob {
    num_polls: usize,
    val: serde_json::Value,
    finish_at: Option<chrono::DateTime<chrono::Utc>>,
}

static OPENAI_FINE_TUNING_JOBS: OnceLock<Mutex<HashMap<String, FineTuningJob>>> = OnceLock::new();

/// Get the socket address for the mock inference provider from the CLI arguments.
/// Defaults to 0.0.0.0:3030 if no address is provided.
fn get_listener_address() -> SocketAddr {
    match std::env::args().nth(1) {
        Some(path) => path
            .parse()
            .unwrap_or_else(|e| panic!("Invalid address: {path}: {e}")),
        None => SocketAddr::from(([0, 0, 0, 0], 3030)),
    }
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(unix)]
    let hangup = async {
        signal::unix::signal(signal::unix::SignalKind::hangup())
            .expect("Failed to install SIGHUP handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C signal");
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM signal");
        }
        _ = hangup => {
            tokio::time::sleep(Duration::from_secs(1)).await;
            tracing::info!("Received SIGHUP signal");
        }
    };
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let listener_address = get_listener_address();
    let listener = tokio::net::TcpListener::bind(listener_address)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {listener_address}: {e}"));
    tracing::info!("Listening on on {}", listener.local_addr().unwrap());

    axum::serve(listener, make_router())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

fn make_router() -> axum::Router {
    axum::Router::new()
        .route(
            "/openai/chat/completions",
            axum::routing::post(completions_handler),
        )
        .route(
            "/openai/fine_tuning/jobs",
            axum::routing::post(create_openai_fine_tuning_job),
        )
        .route(
            "/openai/fine_tuning/jobs/{job_id}",
            axum::routing::get(get_openai_fine_tuning_job),
        )
        .route("/openai/files", axum::routing::post(create_openai_file))
        .route(
            "/azure/openai/deployments/{deployment}/chat/completions",
            axum::routing::post(completions_handler),
        )
        .route(
            "/fireworks/v1/accounts/{account_id}/datasets",
            axum::routing::post(fireworks::create_dataset),
        )
        .route(
            "/fireworks/v1/accounts/{account_id}/datasets/{dataset_id}",
            axum::routing::post(fireworks::upload_to_dataset),
        )
        .route(
            "/fireworks/v1/accounts/{account_id}/datasets/{dataset_id}",
            axum::routing::get(fireworks::get_dataset),
        )
        .route(
            "/fireworks/v1/accounts/{account_id}/fineTuningJobs",
            axum::routing::post(fireworks::create_fine_tuning_job),
        )
        .route(
            "/fireworks/v1/accounts/{account_id}/fineTuningJobs/{job_id}",
            axum::routing::get(fireworks::get_fine_tuning_job),
        )
        .route(
            "/fireworks/v1/accounts/{account_id}/deployedModels",
            axum::routing::post(fireworks::create_deployed_model),
        )
        .route("/status", axum::routing::get(status_handler))
        .layer(TraceLayer::new_for_http())
}

#[derive(Deserialize)]
struct FineTuningGetParams {
    job_id: String,
}

const SHOW_PROGRESS_AT: usize = 1;

async fn get_openai_fine_tuning_job(
    Path(params): Path<FineTuningGetParams>,
) -> Json<serde_json::Value> {
    let job_id = params.job_id.clone();
    let mut fine_tuning_jobs = OPENAI_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    let job = fine_tuning_jobs.get_mut(&job_id);
    if let Some(job) = job {
        job.num_polls += 1;
        if job.num_polls == SHOW_PROGRESS_AT {
            let finish_at = chrono::Utc::now() + chrono::Duration::seconds(2);
            job.finish_at = Some(finish_at);
            job.val["estimated_finish"] = finish_at.timestamp().into();
        }
        if let Some(finish_at) = job.finish_at {
            if chrono::Utc::now() >= finish_at {
                job.val["status"] = "succeeded".into();
                job.val["fine_tuned_model"] = "mock-inference-finetune-1234".into();
            }
        }
        Json(serde_json::to_value(&job.val).unwrap())
    } else {
        Json(json!({
            "error": {
                "message": format!("Could not find fine tune: {job_id}"),
                "type": "invalid_request_error",
                "param": "fine_tune_id",
                "code": "fine_tune_not_found"
            }
        }))
    }
}

async fn create_openai_fine_tuning_job(
    Json(params): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let mut fine_tuning_jobs = OPENAI_FINE_TUNING_JOBS
        .get_or_init(Default::default)
        .lock()
        .unwrap();

    let job_id =
        "mock-inference-finetune-".to_string() + &Alphanumeric.sample_string(&mut rand::rng(), 10);
    let job = FineTuningJob {
        num_polls: 0,
        finish_at: None,
        val: serde_json::json!({
            "object": "fine_tuning.job",
            "id": job_id,
            "model": params["model"],
            "created_at": chrono::Utc::now().timestamp(),
            "fine_tuned_model": null,
            "organization_id": "my-fake-org",
            "result_files": [],
            "status": "queued",
            "validation_file": params.get("validation_file").unwrap_or(&serde_json::Value::Null),
            "training_file": params.get("training_file"),
            "method": {
                "type": "supervised",
                "hyperparameters": {
                    "batch_size": "auto",
                    "learning_rate_multiplier": "auto",
                    "n_epochs": "auto",
                  }
            }
        }),
    };
    fine_tuning_jobs.insert(job_id.clone(), job.clone());
    Json(job.val.clone())
}

async fn create_openai_file(mut form: Multipart) -> Result<Json<serde_json::Value>, Error> {
    let file_id =
        "mock-inference-file-".to_string() + &Alphanumeric.sample_string(&mut rand::rng(), 10);

    let mut file_len = None;
    let mut filename = None;
    let mut purpose = None;
    while let Some(field) = form.next_field().await.unwrap() {
        if field.name() == Some("file") {
            let bytes = field.bytes().await.unwrap();
            file_len = Some(bytes.len());

            // Try to parse the file as JSONL
            if let Ok(content) = std::str::from_utf8(&bytes) {
                // Check if it's valid JSONL by attempting to parse each line
                for line in content.lines() {
                    let result = serde_json::from_str::<OpenAIFineTuningRow>(line);
                    match result {
                        Ok(row) => {
                            for message in row.messages.iter() {
                                let Some(content_array) = message.content.as_array() else {
                                    continue;
                                };
                                for content in content_array.iter() {
                                    let object = content.as_object().unwrap();
                                    let content_type = object.get("type").unwrap();
                                    if content_type == "image_url" {
                                        let url_data = object.get("image_url").unwrap();
                                        let url =
                                            url_data.get("url").and_then(|v| v.as_str()).unwrap();
                                        if !url.starts_with("data:") {
                                            return Err(Error::new(
                                                format!("Invalid JSONL line (\"{line}\"): image_url is not a data URL"),
                                                StatusCode::BAD_REQUEST,
                                            ));
                                        }
                                        if url.len() < 100 {
                                            return Err(Error::new(
                                                format!("Invalid JSONL line (\"{line}\"): image_url is too short"),
                                                StatusCode::BAD_REQUEST,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            return Err(Error::new(
                                format!("Invalid JSONL line (\"{line}\"): {e}"),
                                StatusCode::BAD_REQUEST,
                            ));
                        }
                    }
                }
            }
        } else if field.name() == Some("filename") {
            filename = Some(field.text().await.unwrap());
        } else if field.name() == Some("purpose") {
            purpose = Some(field.text().await.unwrap());
        }
    }

    Ok(Json(json!({
        "id": file_id,
        "object": "file",
        "bytes": file_len,
        "created_at": chrono::Utc::now().timestamp(),
        "filename": filename,
        "purpose": purpose,
    })))
}

#[derive(Debug, Deserialize)]
struct OpenAIFineTuningRow {
    messages: Vec<OpenAIFineTuningMessage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIFineTuningMessage {
    // role is not used
    content: Value,
}

async fn completions_handler(Json(params): Json<serde_json::Value>) -> Response<Body> {
    let stream = match params.get("stream") {
        Some(stream) => stream.as_bool().unwrap_or(false),
        None => false,
    };
    let json_mode = matches!(
        params["response_format"]["type"].as_str(),
        Some("json_object")
    );
    let function_call = params.get("tools").is_some();

    if stream {
        Sse::new(create_stream(json_mode, function_call))
            .keep_alive(axum::response::sse::KeepAlive::new())
            .into_response()
    } else {
        // TODO (#82): map fixtures to functions in config
        let response = if function_call {
            include_str!("../fixtures/openai/chat_completions_function_example.json")
        } else if json_mode {
            include_str!("../fixtures/openai/chat_completions_json_example.json")
        } else {
            include_str!("../fixtures/openai/chat_completions_example.json")
        };

        let response_json = serde_json::from_str::<serde_json::Value>(response).unwrap();

        axum::response::Json(response_json).into_response()
    }
}

fn create_stream(
    json_mode: bool,
    function_call: bool,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    try_stream! {
        // TODO (#82): map fixtures to functions in config
        let lines = if function_call {
            include_str!("../fixtures/openai/chat_completions_streaming_function_example.jsonl")
        } else if json_mode {
            include_str!("../fixtures/openai/chat_completions_streaming_json_example.jsonl")
        } else {
            include_str!("../fixtures/openai/chat_completions_streaming_example.jsonl")
        };

        for line in lines.trim().split('\n') {
            let event = Event::default().data(line);
            yield event;
        }

        yield Event::default().data("[DONE]");
    }
}

pub async fn status_handler() -> Json<serde_json::Value> {
    axum::response::Json(json!({ "status": "ok" }))
}

#[cfg(test)]
mod tests {

    use super::*;
    use eventsource_stream::Eventsource;
    use tokio_stream::StreamExt as _; // for `next`
    use tower::ServiceExt; // for `oneshot`

    #[tokio::test]
    async fn test_openai_completions_handler() {
        let payload = json!({
            "model": "gpt-3.5-turbo",
            "prompt": "Is Santa real?",
            "max_tokens": 5,
        });

        let router = make_router();

        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/openai/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        // Check the response headers and status
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );

        // Check the response body
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();

        assert_eq!(
            body_json["choices"][0]["message"]["content"],
            "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
        );
    }

    #[tokio::test]
    async fn test_openai_completions_handler_json() {
        let payload = json!({
            "model": "gpt-3.5-turbo",
            "prompt": "Is Santa real?",
            "max_tokens": 5,
            "response_format": {
                "type": "json_object"
            }
        });

        let router = make_router();

        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/openai/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        // Check the response headers and status
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );

        // Check the response body
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();

        assert_eq!(
            body_json["choices"][0]["message"]["content"],
            "{\"answer\": \"The 2020 World Series was played in Texas at Globe Life Field in Arlington.\"}",
        );
    }

    #[tokio::test]
    async fn test_openai_completions_handler_function() {
        let payload = json!({
                  "model": "gpt-3.5-turbo",
                  "prompt": "Is Santa real?",
                  "max_tokens": 5,
                  "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "description": "Get the current temperature in a given location",
              "parameters": {
                "type": "object",
                "properties": {
                  "answer": {
                    "type": "string"
                  }
                },
                "required": ["answer"]
              }
            }
          }
        ],
              });

        let router = make_router();

        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/openai/chat/completions")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        // Check the response headers and status
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );

        // Check the response body
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();

        let arguments = body_json["choices"][0]["message"]["tool_calls"][0]["function"]
            ["arguments"]
            .as_str()
            .unwrap();
        assert_eq!(
            arguments,
            "{\n\"answer\": \"The 2020 World Series was played in Texas at Globe Life Field in Arlington.\"}",
        );
    }

    #[tokio::test]
    async fn test_openai_completions_handler_stream() {
        let payload = json!({
            "model": "gpt-3.5-turbo",
            "prompt": "Is Santa real?",
            "max_tokens": 5,
            "stream": true,
        });

        // Launch the test server
        let router = make_router();
        let listener = tokio::net::TcpListener::bind("localhost:0").await.unwrap();

        let url = format!(
            "http://{}/openai/chat/completions",
            listener.local_addr().unwrap()
        );

        tokio::spawn(async {
            axum::serve(listener, router).await.unwrap();
        });

        // Make the request
        let response = reqwest::Client::new()
            .post(&url)
            .json(&payload)
            .send()
            .await
            .unwrap();

        // Check the response headers and status
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let mut event_stream = response.bytes_stream().eventsource();

        // Check we receive all client chunks correctly
        for _ in 0..10 {
            let event_data: serde_json::Value =
                serde_json::from_str(&event_stream.next().await.unwrap().unwrap().data).unwrap();

            // Includes some OpenAI fields
            assert!(event_data.get("choices").is_some());
        }

        // Ensure we get the final chunk
        assert_eq!(event_stream.next().await.unwrap().unwrap().data, "[DONE]");

        // Check we've exhausted the client stream
        assert!(event_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_openai_completions_handler_stream_json() {
        let payload = json!({
            "model": "gpt-3.5-turbo",
            "prompt": "Is Santa real?",
            "max_tokens": 5,
            "stream": true,
            "response_format": {
                "type": "json_object"
            }
        });

        // Launch the test server
        let router = make_router();
        let listener = tokio::net::TcpListener::bind("localhost:0").await.unwrap();

        let url = format!(
            "http://{}/openai/chat/completions",
            listener.local_addr().unwrap()
        );

        tokio::spawn(async {
            axum::serve(listener, router).await.unwrap();
        });

        // Make the request
        let response = reqwest::Client::new()
            .post(&url)
            .json(&payload)
            .send()
            .await
            .unwrap();

        // Check the response headers and status
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let mut event_stream = response.bytes_stream().eventsource();

        // Check we receive all client chunks correctly
        for _ in 0..16 {
            let event_data: serde_json::Value =
                serde_json::from_str(&event_stream.next().await.unwrap().unwrap().data).unwrap();

            // Includes some OpenAI fields
            assert!(event_data.get("choices").is_some());
        }

        // Ensure we get the final chunk
        assert_eq!(event_stream.next().await.unwrap().unwrap().data, "[DONE]");

        // Check we've exhausted the client stream
        assert!(event_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_openai_completions_handler_stream_function() {
        let payload = json!({
            "model": "gpt-3.5-turbo",
            "prompt": "Is Santa real?",
            "max_tokens": 5,
            "stream": true,
                  "tools": [
          {
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "description": "Get the current temperature in a given location",
              "parameters": {
                "type": "object",
                "properties": {
                  "answer": {
                    "type": "string"
                  }
                },
                "required": ["answer"]
              }
            }
          }
        ],
        });

        // Launch the test server
        let router = make_router();
        let listener = tokio::net::TcpListener::bind("localhost:0").await.unwrap();

        let url = format!(
            "http://{}/openai/chat/completions",
            listener.local_addr().unwrap()
        );

        tokio::spawn(async {
            axum::serve(listener, router).await.unwrap();
        });

        // Make the request
        let response = reqwest::Client::new()
            .post(&url)
            .json(&payload)
            .send()
            .await
            .unwrap();

        // Check the response headers and status
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let mut event_stream = response.bytes_stream().eventsource();

        // Check we receive all client chunks correctly
        for _ in 0..6 {
            let event_data: serde_json::Value =
                serde_json::from_str(&event_stream.next().await.unwrap().unwrap().data).unwrap();

            // Includes some OpenAI fields
            assert!(
                event_data["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .is_string()
            );
        }
        // this one will just have a finish reason
        let event_data: serde_json::Value =
            serde_json::from_str(&event_stream.next().await.unwrap().unwrap().data).unwrap();
        assert!(event_data["choices"][0]["finish_reason"].is_string());

        // Ensure we get the final chunk
        assert_eq!(event_stream.next().await.unwrap().unwrap().data, "[DONE]");

        // Check we've exhausted the client stream
        assert!(event_stream.next().await.is_none());
    }
}
