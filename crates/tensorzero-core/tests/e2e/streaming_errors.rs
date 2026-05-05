use std::future::IntoFuture;
use std::net::SocketAddr;

use axum::Router;
use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use futures::StreamExt;
use http::StatusCode;
use serde_json::json;
use tensorzero::{
    Client, ClientInferenceParams, InferenceOutput, InferenceResponseChunk, Input, InputMessage,
    InputMessageContent, Role, TensorZeroError,
};
use tensorzero_core::inference::types::{Arguments, System, Text};
use tensorzero_error::ErrorDetails;

use crate::common::get_gateway_endpoint;
use reqwest_sse_stream::{Event, RequestBuilderExt};

#[tokio::test]
async fn test_client_stream_with_error_http_gateway() {
    test_client_stream_with_error(tensorzero::test_helpers::make_http_gateway().await).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_client_stream_with_error_embedded_gateway() {
    test_client_stream_with_error(tensorzero::test_helpers::make_embedded_gateway().await).await;
}

async fn test_client_stream_with_error(client: Client) {
    let res = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("err_in_stream".to_string()),
            input: Input {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "AskJeeves".into(),
                )])))),
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Please write me a sentence about Megumin making an explosion."
                            .into(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await
        .unwrap();
    let InferenceOutput::Streaming(stream) = res else {
        panic!("Expected a stream");
    };
    let stream = stream.enumerate().collect::<Vec<_>>().await;
    assert_eq!(stream.len(), 17);

    for (i, chunk) in stream {
        if i == 3 {
            let err = chunk
                .expect_err("Expected error after 3 chunks")
                .to_string();
            assert!(
                err.contains("Dummy error in stream"),
                "Unexpected error: `{err}`"
            );
        } else {
            chunk.expect("Expected first few chunks to be Ok");
        }
    }
}

#[tokio::test]
async fn test_stream_with_error() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "err_in_stream",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Please write me a sentence about Megumin making an explosion."
                }
            ]},
        "stream": true,
    });

    let mut event_stream = reqwest::Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut good_chunks = 0;
    // Check we receive all client chunks correctly
    loop {
        match event_stream.next().await {
            Some(Ok(e)) => match e {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        break;
                    }
                    let obj: serde_json::Value = serde_json::from_str(&message.data).unwrap();
                    if let Some(error) = obj.get("error") {
                        let error_str: &str = error.as_str().unwrap();
                        assert!(
                            error_str.contains("Dummy error in stream"),
                            "Unexpected error: {error_str}"
                        );
                        assert_eq!(good_chunks, 3);
                    } else {
                        let _chunk: InferenceResponseChunk =
                            serde_json::from_str(&message.data).unwrap();
                    }
                    good_chunks += 1;
                }
            },
            Some(Err(e)) => {
                panic!("Unexpected error: {e:?}");
            }
            None => {
                break;
            }
        }
    }
    assert_eq!(good_chunks, 17);
}

/// Spawn a tiny mock OpenAI-compatible server that replies with `status` (and a JSON error body)
/// to any POST to `/chat/completions`, before any SSE bytes are written. This exercises the
/// failure path inside `inject_extra_request_data_and_send_eventsource_with_headers`, which
/// must produce a `FatalStreamError` carrying the upstream HTTP status code.
async fn make_failing_openai_server(
    status: StatusCode,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    let real_addr = listener.local_addr().unwrap();

    let app = Router::new().route(
        "/chat/completions",
        post(move || async move {
            Response::builder()
                .status(status)
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"error":{"message":"mock upstream rejected request","type":"invalid_request_error"}}"#,
                ))
                .unwrap()
        }),
    );

    let (send, recv) = tokio::sync::oneshot::channel::<()>();
    let shutdown_fut = async move {
        let _ = recv.await;
    };

    #[expect(clippy::disallowed_methods, reason = "test code")]
    tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_fut)
            .into_future(),
    );

    (real_addr, send)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_fatal_error_propagates_status_code() {
    // Upstream returns 401 Unauthorized to the streaming chat-completions request before
    // any SSE bytes are emitted. The OpenAI provider's
    // `inject_extra_request_data_and_send_eventsource_with_headers` call should produce a
    // `FatalStreamError` whose `status_code` is `Some(401)`, which we observe via
    // `Error::underlying_status_code()`.
    let (addr, _shutdown) = make_failing_openai_server(StatusCode::UNAUTHORIZED).await;

    let config = format!(
        r#"
[models.upstream-401]
routing = ["fake-openai"]

[models.upstream-401.providers.fake-openai]
type = "openai"
api_base = "http://{addr}/"
api_key_location = "none"
model_name = "gpt-4.1-mini"
"#
    );

    let client = tensorzero::test_helpers::make_embedded_gateway_with_config(&config).await;

    let res = client
        .inference(ClientInferenceParams {
            model_name: Some("upstream-401".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "hello".into(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await;

    let err = res.expect_err("expected the streaming request to fail before any chunks");
    let source = match err {
        TensorZeroError::Http { source, .. } => source,
        TensorZeroError::Other { source } => source,
        other => panic!("unexpected error variant: {other:?}"),
    };
    let inner = &source.0;

    assert_eq!(
        inner.underlying_status_code(),
        Some(StatusCode::UNAUTHORIZED),
        "underlying_status_code() should surface the upstream 401 from FatalStreamError"
    );

    // Walk the error tree to find the FatalStreamError and verify the field directly.
    fn find_fatal_status(details: &ErrorDetails) -> Option<StatusCode> {
        match details {
            ErrorDetails::FatalStreamError { status_code, .. } => *status_code,
            ErrorDetails::AllVariantsFailed { errors } => errors
                .values()
                .find_map(|e| find_fatal_status(e.get_details())),
            ErrorDetails::AllModelProvidersFailed { provider_errors } => provider_errors
                .values()
                .find_map(|e| find_fatal_status(e.get_details())),
            ErrorDetails::AllRetriesFailed { errors } => errors
                .iter()
                .find_map(|e| find_fatal_status(e.get_details())),
            ErrorDetails::AllCandidatesFailed { candidate_errors } => candidate_errors
                .values()
                .find_map(|e| find_fatal_status(e.get_details())),
            _ => None,
        }
    }
    assert_eq!(
        find_fatal_status(inner.get_details()),
        Some(StatusCode::UNAUTHORIZED),
        "FatalStreamError nested inside the wrapper should carry status_code = Some(401)"
    );
}
