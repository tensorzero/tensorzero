use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use bytes::Bytes;
use futures::StreamExt;
use http::StatusCode;
use rcgen::generate_simple_self_signed;
use rustls::ServerConfig;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde_json::json;
use tensorzero::{
    Client, ClientInferenceParams, InferenceOutput, InferenceResponseChunk, Input, InputMessage,
    InputMessageContent, Role, TensorZeroError,
};
use tensorzero_core::inference::types::usage::ApiType;
use tensorzero_core::inference::types::{Arguments, System, Text};
use tensorzero_error::ErrorDetails;
use tokio_rustls::TlsAcceptor;

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

/// Spin up a TLS server on 127.0.0.1 that advertises `h2` via ALPN, accept one
/// connection, perform the h2 handshake, send response headers + one data
/// frame, then `RST_STREAM(INTERNAL_ERROR)` mid-body. Mirrors the in-the-wild
/// OpenAI failure mode.
///
/// Returns the listener's address and a oneshot sender that callers must signal
/// once the client has observed the first data frame — that signal triggers the
/// `RST_STREAM` and prevents a race where the reset arrives before the client
/// has finished reading headers.
async fn spawn_h2_rst_stream_server() -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    // Self-signed cert valid for 127.0.0.1 + localhost. The client's
    // `danger_accept_invalid_certs(true)` (set by tensorzero-http when
    // `TENSORZERO_E2E_PROXY` is present under the `e2e_tests` feature) is what
    // lets reqwest trust this cert.
    let cert = generate_simple_self_signed(vec!["127.0.0.1".to_string(), "localhost".to_string()])
        .expect("generating self-signed cert should succeed");
    let cert_der = CertificateDer::from(cert.cert.der().to_vec());
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()));

    let mut server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .expect("building rustls server config should succeed");
    // Advertise h2 in ALPN so reqwest negotiates HTTP/2 the way it does
    // against real LLM providers.
    server_config.alpn_protocols = vec![b"h2".to_vec()];
    let acceptor = TlsAcceptor::from(Arc::new(server_config));

    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .expect("binding the mock TLS server should succeed");
    let addr = listener.local_addr().unwrap();

    let (client_ready_tx, client_ready_rx) = tokio::sync::oneshot::channel::<()>();

    #[expect(clippy::disallowed_methods, reason = "test code")]
    tokio::spawn(async move {
        let (socket, _peer) = listener
            .accept()
            .await
            .expect("accept of the mock server connection should succeed");
        let tls = acceptor
            .accept(socket)
            .await
            .expect("TLS handshake should succeed");
        let mut conn = h2::server::handshake(tls)
            .await
            .expect("HTTP/2 handshake should succeed");

        let result = conn
            .accept()
            .await
            .expect("at least one h2 stream should be accepted")
            .expect("the inbound stream should be Ok");
        let (request, mut respond) = result;

        // The h2 connection only writes frames while `accept`/`poll_close` is
        // being polled. We hand the request off to a spawned handler and keep
        // polling `accept` here so that the response + data + RST_STREAM
        // frames actually get flushed to the wire.
        #[expect(clippy::disallowed_methods, reason = "test code")]
        let handler = tokio::spawn(async move {
            let mut body = request.into_body();
            while let Some(chunk) = body.data().await {
                let _ = chunk;
            }
            let response = http::Response::builder()
                .status(http::StatusCode::OK)
                .header(http::header::CONTENT_TYPE, "text/event-stream")
                .body(())
                .unwrap();
            let mut send = respond
                .send_response(response, false)
                .expect("sending response headers should succeed");
            send.send_data(Bytes::from_static(b":ok\n\n"), false)
                .expect("sending first data frame should succeed");
            let _ = client_ready_rx.await;
            send.send_reset(h2::Reason::INTERNAL_ERROR);
        });

        while let Some(next) = conn.accept().await {
            let _ = next;
        }
        let _ = handler.await;
    });

    (addr, client_ready_tx)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_h2_rst_stream_is_fatal() {
    // End-to-end check that an HTTP/2 `RST_STREAM(INTERNAL_ERROR)` arriving
    // mid-body — the exact wire-level failure that prompted introducing the
    // body-error fatal classification — produces a `FatalStreamError` whose
    // `Display` message is verbatim the chain that surfaces in production
    // logs. Drives the request through `TensorzeroHttpClient` (the same client
    // used in production for SSE), and relies on the existing
    // `e2e_tests`-feature-flagged path in `tensorzero-http` to enable
    // `danger_accept_invalid_certs(true)` so we can use a self-signed cert.
    //
    // We require `TENSORZERO_E2E_PROXY` to be set: it's what flips on
    // `danger_accept_invalid_certs(true)`. CI sets it; local e2e runs are
    // expected to as well.
    assert!(
        std::env::var("TENSORZERO_E2E_PROXY").is_ok(),
        "TENSORZERO_E2E_PROXY must be set for this test — it's what enables \
         danger_accept_invalid_certs(true) in tensorzero-http so the self-signed \
         TLS cert is trusted by reqwest. CI sets it; for local runs, export e.g. \
         TENSORZERO_E2E_PROXY=http://localhost:3003 before running."
    );

    // Force the e2e debug flag off *before any error is constructed*, so that
    // `format_error_chain` renders each source link via `Display` (what users
    // actually see in production) rather than the e2e-default `Debug`. This
    // lets us pin the exact production-shape message below.
    tensorzero_error::force_set_debug(false)
        .expect("force_set_debug should run before anything reads is_debug()");

    // rustls 0.23 requires a process-wide CryptoProvider. The workspace builds
    // rustls with the `aws_lc_rs` feature, so we install that provider. It's a
    // no-op on subsequent calls within the same process.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let (addr, client_ready_tx) = spawn_h2_rst_stream_server().await;

    // Build the TensorZero HTTP client exactly the way production builds it.
    // Under `e2e_tests`, this picks up `TENSORZERO_E2E_PROXY` and flips on
    // `danger_accept_invalid_certs(true)`. 127.0.0.1 is in the no-proxy list,
    // so requests still go direct to our mock server.
    let client = tensorzero_http::TensorzeroHttpClient::new_testing()
        .expect("building TensorzeroHttpClient should succeed");

    let response = client
        .get(format!("https://127.0.0.1:{}/stream", addr.port()))
        .send()
        .await
        .expect("opening the TLS stream to the in-process h2 server should succeed");

    // Drive the body until it errors. The first chunk unblocks the server to
    // fire RST_STREAM; the next read surfaces a real `reqwest::Error` whose
    // chain terminates in h2's `Reset(INTERNAL_ERROR, Remote)`.
    let mut body_stream = response.bytes_stream();
    let mut client_ready_tx = Some(client_ready_tx);
    let reqwest_err = loop {
        match body_stream.next().await {
            Some(Ok(_)) => {
                if let Some(tx) = client_ready_tx.take() {
                    let _ = tx.send(());
                }
            }
            Some(Err(e)) => break e,
            None => panic!("body ended cleanly; expected a stream-reset error"),
        }
    };
    assert!(
        reqwest_err.is_body() || reqwest_err.is_decode(),
        "test precondition: the produced reqwest::Error should be a body/decode error, got: {reqwest_err:?}"
    );

    // Wrap the reqwest::Error the same way `reqwest-sse-stream` would (via
    // its `Body` SSE variant) and feed it through the production
    // `convert_stream_error` to assert the classification + exact message.
    let sse_err = reqwest_sse_stream::ReqwestSseStreamError::SseError(
        reqwest_sse_stream::SseStreamError::Body(Box::new(reqwest_err)),
    );
    let err = tensorzero_core::providers::helpers::convert_stream_error(
        "raw req".to_string(),
        "openai".to_string(),
        ApiType::ChatCompletions,
        sse_err,
        Some("req_abc123"),
    )
    .await;
    match err.get_details() {
        ErrorDetails::FatalStreamError {
            message,
            provider_type,
            status_code,
            api_type,
            raw_request,
            raw_response,
        } => {
            assert_eq!(provider_type, "openai");
            assert_eq!(*status_code, None);
            assert_eq!(*api_type, ApiType::ChatCompletions);
            assert_eq!(raw_request.as_deref(), Some("raw req"));
            assert_eq!(raw_response.as_deref(), None);
            // Exact, locked-down `Display` message. The chain matches the
            // in-the-wild OpenAI failure that motivated this classification
            // verbatim: SSE error wrapping a `body error` wrapping reqwest
            // Decode → Body → hyper → h2 `RST_STREAM(INTERNAL_ERROR)`. If
            // `reqwest`/`hyper`/`h2` change their `Display` representation in
            // a future bump, update this literal. (Driven by
            // `force_set_debug(false)` above, which switches
            // `DisplayOrDebugGateway` to its `Display` branch.)
            assert_eq!(
                message,
                "SSE error; caused by: body error: error decoding response body; caused by: error decoding response body; caused by: request or response body error; caused by: error reading a body from connection; caused by: stream error received: unexpected internal error encountered [request_id: req_abc123]",
            );
        }
        other => panic!(
            "expected FatalStreamError for an SSE body error wrapping a reqwest body/decode failure, got: {other:?}"
        ),
    }
}
