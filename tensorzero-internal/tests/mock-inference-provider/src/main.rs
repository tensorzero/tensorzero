use async_stream::try_stream;
use axum::{
    body::Body,
    extract::Json,
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
};
use futures::Stream;
use mimalloc::MiMalloc;
use serde_json::json;
use std::net::SocketAddr;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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

#[tokio::main]
async fn main() {
    let listener_address = get_listener_address();
    let listener = tokio::net::TcpListener::bind(listener_address)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {listener_address}: {e}"));

    axum::serve(listener, make_router()).await.unwrap();
}

fn make_router() -> axum::Router {
    axum::Router::new()
        .route(
            "/openai/chat/completions",
            axum::routing::post(completions_handler),
        )
        .route(
            "/azure/openai/deployments/{deployment}/chat/completions",
            axum::routing::post(completions_handler),
        )
        .route("/status", axum::routing::get(status_handler))
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
