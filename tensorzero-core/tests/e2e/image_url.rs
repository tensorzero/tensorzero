#![expect(clippy::print_stdout)]

use axum::body::Body;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::time::Duration;
use tensorzero::{
    CacheParamsOptions, ClientInferenceParams, ClientInput, ClientInputMessage,
    ClientInputMessageContent, File, InferenceOutput, InferenceResponse, Role,
};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse,
};
use tensorzero_core::inference::types::TextKind;
use url::Url;
use uuid::Uuid;

use crate::providers::common::FERRIS_PNG;

/// Spawn a temporary HTTP server that serves the test image
async fn make_temp_image_server() -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    let real_addr = listener.local_addr().unwrap();

    async fn get_ferris_png() -> impl IntoResponse {
        // Return wrong Content-Type to test MIME type detection
        Response::builder()
            .header(http::header::CONTENT_TYPE, "text/plain")
            .body(Body::from(FERRIS_PNG.to_vec()))
            .unwrap()
    }

    let app = Router::new().route("/ferris.png", get(get_ferris_png));

    let (send, recv) = tokio::sync::oneshot::channel::<()>();
    let shutdown_fut = async move {
        let _ = recv.await;
    };

    // test code
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_fut)
            .into_future(),
    );

    (real_addr, send)
}

/// Test config with fetch_and_encode_input_files_before_inference = true (default)
const CONFIG_WITH_FETCH_TRUE: &str = r#"
gateway.fetch_and_encode_input_files_before_inference = true

[object_storage]
type = "disabled"

[functions.describe_image]
type = "chat"

[functions.describe_image.variants.openai]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"
"#;

/// Test config with fetch_and_encode_input_files_before_inference = false
const CONFIG_WITH_FETCH_FALSE: &str = r#"
gateway.fetch_and_encode_input_files_before_inference = false

[object_storage]
type = "disabled"

[functions.describe_image]
type = "chat"

[functions.describe_image.variants.openai]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"
"#;

/// Base64 encoded 1x1 red pixel PNG (same as Python test)
const IMAGE_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

#[tokio::test]
async fn test_image_url_with_fetch_true() {
    let episode_id = Uuid::now_v7();

    // The '_shutdown_sender' will wake up the receiver on drop
    let (server_addr, _shutdown_sender) = make_temp_image_server().await;
    let image_url = Url::parse(&format!("http://{server_addr}/ferris.png")).unwrap();

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(CONFIG_WITH_FETCH_TRUE).await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("describe_image".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "What's in this image?".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Url {
                            url: image_url.clone(),
                            mime_type: None,
                        }),
                    ],
                }],
            },
            cache_options: CacheParamsOptions {
                enabled: CacheEnabledMode::On,
                max_age_s: Some(10),
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    // Verify IDs are valid
    let inference_id = response.inference_id();

    // Verify response structure
    let InferenceResponse::Chat(chat_response) = &response else {
        panic!("Expected chat response");
    };

    assert!(
        !chat_response.content.is_empty(),
        "Response content should not be empty"
    );
    assert!(
        chat_response.usage.input_tokens > 0,
        "Input tokens should be > 0"
    );
    assert!(
        chat_response.usage.output_tokens > 0,
        "Output tokens should be > 0"
    );

    // Sleep to allow ClickHouse write
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Verify ClickHouse data
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_some(), "Inference should be in ClickHouse");

    println!("✓ Test passed: Image URL with fetch_and_encode_input_files_before_inference = true");
}

#[tokio::test]
async fn test_image_url_with_fetch_false() {
    let episode_id = Uuid::now_v7();

    // The '_shutdown_sender' will wake up the receiver on drop
    let (server_addr, _shutdown_sender) = make_temp_image_server().await;
    let image_url = Url::parse(&format!("http://{server_addr}/ferris.png")).unwrap();

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(CONFIG_WITH_FETCH_FALSE).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("describe_image".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "What's in this image?".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Url {
                            url: image_url.clone(),
                            mime_type: None,
                        }),
                    ],
                }],
            },
            cache_options: CacheParamsOptions {
                enabled: CacheEnabledMode::On,
                max_age_s: Some(10),
            },
            ..Default::default()
        })
        .await;

    // When fetch_and_encode_input_files_before_inference = false, OpenAI cannot access localhost URLs
    // so the inference should fail with an error about downloading the image
    assert!(
        result.is_err(),
        "Expected error when OpenAI cannot access localhost URL"
    );

    let err = result.unwrap_err();
    let err_msg = format!("{err:?}");

    // The error should indicate that OpenAI couldn't download the image from localhost
    assert!(
        err_msg.contains("Error while downloading") || err_msg.contains("invalid_image_url"),
        "Expected error about downloading localhost URL, got: {err_msg}"
    );

    println!(
        "✓ Test passed: Image URL with fetch_and_encode_input_files_before_inference = false \
         (correctly fails when OpenAI cannot access localhost)"
    );
}

#[tokio::test]
async fn test_base64_image_with_fetch_true() {
    let episode_id = Uuid::now_v7();

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(CONFIG_WITH_FETCH_TRUE).await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("describe_image".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "Describe this image briefly.".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Base64 {
                            mime_type: mime::IMAGE_PNG,
                            data: IMAGE_BASE64.to_string(),
                        }),
                    ],
                }],
            },
            cache_options: CacheParamsOptions {
                enabled: CacheEnabledMode::On,
                max_age_s: Some(10),
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    // Verify IDs are valid
    let inference_id = response.inference_id();

    // Verify response structure
    let InferenceResponse::Chat(chat_response) = &response else {
        panic!("Expected chat response");
    };

    assert!(
        !chat_response.content.is_empty(),
        "Response content should not be empty"
    );
    assert!(
        chat_response.usage.input_tokens > 0,
        "Input tokens should be > 0"
    );
    assert!(
        chat_response.usage.output_tokens > 0,
        "Output tokens should be > 0"
    );

    // Sleep to allow ClickHouse write
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Verify ClickHouse data
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_some(), "Inference should be in ClickHouse");

    println!(
        "✓ Test passed: Base64 image with fetch_and_encode_input_files_before_inference = true"
    );
}

#[tokio::test]
async fn test_base64_image_with_fetch_false() {
    let episode_id = Uuid::now_v7();

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(CONFIG_WITH_FETCH_FALSE).await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("describe_image".to_string()),
            variant_name: Some("openai".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "Describe this image briefly.".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Base64 {
                            mime_type: mime::IMAGE_PNG,
                            data: IMAGE_BASE64.to_string(),
                        }),
                    ],
                }],
            },
            cache_options: CacheParamsOptions {
                enabled: CacheEnabledMode::On,
                max_age_s: Some(10),
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(response) = response else {
        panic!("Expected non-streaming inference response");
    };

    // Verify IDs are valid
    let inference_id = response.inference_id();

    // Verify response structure
    let InferenceResponse::Chat(chat_response) = &response else {
        panic!("Expected chat response");
    };

    assert!(
        !chat_response.content.is_empty(),
        "Response content should not be empty"
    );
    assert!(
        chat_response.usage.input_tokens > 0,
        "Input tokens should be > 0"
    );
    assert!(
        chat_response.usage.output_tokens > 0,
        "Output tokens should be > 0"
    );

    // Sleep to allow ClickHouse write
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Verify ClickHouse data
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_some(), "Inference should be in ClickHouse");

    println!(
        "✓ Test passed: Base64 image with fetch_and_encode_input_files_before_inference = false"
    );
}
