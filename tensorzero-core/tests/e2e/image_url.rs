#![expect(clippy::print_stdout)]

use axum::body::Body;
use axum::extract::Request;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::time::Duration;
use tensorzero::{
    CacheParamsOptions, ClientInferenceParams, ClientInput, ClientInputMessage,
    ClientInputMessageContent, InferenceOutput, InferenceResponse, Role,
};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse,
};
use tensorzero_core::endpoints::status::TENSORZERO_VERSION;
use tensorzero_core::inference::types::TextKind;
use tensorzero_core::inference::types::{Base64File, File, UrlFile};
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

    async fn get_ferris_png(req: Request) -> impl IntoResponse {
        // Assert that the User-Agent header is set correctly
        let user_agent = req
            .headers()
            .get(http::header::USER_AGENT)
            .expect("User-Agent header should be present")
            .to_str()
            .expect("User-Agent should be valid UTF-8");
        let expected_user_agent = format!("TensorZero/{TENSORZERO_VERSION}");
        assert_eq!(
            user_agent, expected_user_agent,
            "User-Agent should be TensorZero/{TENSORZERO_VERSION}"
        );

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

/// Spawn a temporary HTTP server that returns 403 Forbidden with a custom error message
async fn make_temp_403_image_server() -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    let real_addr = listener.local_addr().unwrap();

    async fn get_forbidden_image() -> impl IntoResponse {
        Response::builder()
            .status(http::StatusCode::FORBIDDEN)
            .header(http::header::CONTENT_TYPE, "text/plain")
            .body(Body::from(
                "Access denied: You do not have permission to view this image",
            ))
            .unwrap()
    }

    let app = Router::new().route("/forbidden.png", get(get_forbidden_image));

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

[functions.describe_image.variants.anthropic]
type = "chat_completion"
model = "anthropic::claude-3-7-sonnet-latest"
"#;

/// Test config with fetch_and_encode_input_files_before_inference = false
const CONFIG_WITH_FETCH_FALSE: &str = r#"
gateway.fetch_and_encode_input_files_before_inference = false

[object_storage]
type = "disabled"

[functions.describe_image]
type = "chat"

[functions.describe_image.variants.anthropic]
type = "chat_completion"
model = "anthropic::claude-3-7-sonnet-latest"
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
            variant_name: Some("anthropic".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "What's in this image?".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Url(UrlFile {
                            url: image_url.clone(),
                            mime_type: None,
                            detail: None,
                            filename: None,
                        })),
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
        chat_response.usage.input_tokens.unwrap() > 0,
        "Input tokens should be > 0"
    );
    assert!(
        chat_response.usage.output_tokens.unwrap() > 0,
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
    let image_url = Url::parse(&format!("https://{server_addr}/ferris.png")).unwrap();

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(CONFIG_WITH_FETCH_FALSE).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("describe_image".to_string()),
            variant_name: Some("anthropic".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "What's in this image?".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Url(UrlFile {
                            url: image_url.clone(),
                            mime_type: Some(mime::IMAGE_PNG),
                            detail: None,
                            filename: None,
                        })),
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

    // When fetch_and_encode_input_files_before_inference = false, Anthropic cannot access localhost URLs
    // so the inference should fail with an error about downloading the image
    let err = result.expect_err("Expected error when Anthropic cannot access localhost URL");
    let err_msg = format!("{err:?}");

    // The error should indicate that Anthropic couldn't download the image from localhost
    assert!(
        err_msg.contains("Unable to download the file"),
        "Expected error about downloading localhost URL, got: {err_msg}"
    );

    println!(
        "✓ Test passed: Image URL with fetch_and_encode_input_files_before_inference = false \
         (correctly fails when Anthropic cannot access localhost)"
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
            variant_name: Some("anthropic".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "Describe this image briefly.".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Base64(
                            Base64File::new(
                                None,
                                mime::IMAGE_PNG,
                                IMAGE_BASE64.to_string(),
                                None,
                                None,
                            )
                            .expect("test data should be valid"),
                        )),
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
        chat_response.usage.input_tokens.unwrap() > 0,
        "Input tokens should be > 0"
    );
    assert!(
        chat_response.usage.output_tokens.unwrap() > 0,
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
            variant_name: Some("anthropic".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "Describe this image briefly.".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Base64(
                            Base64File::new(
                                None,
                                mime::IMAGE_PNG,
                                IMAGE_BASE64.to_string(),
                                None,
                                None,
                            )
                            .expect("test data should be valid"),
                        )),
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
        chat_response.usage.input_tokens.unwrap() > 0,
        "Input tokens should be > 0"
    );
    assert!(
        chat_response.usage.output_tokens.unwrap() > 0,
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

#[tokio::test]
async fn test_wikipedia_image_url_with_fetch_true() {
    let episode_id = Uuid::now_v7();

    let wikipedia_url = Url::parse("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg").unwrap();

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(CONFIG_WITH_FETCH_TRUE).await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("describe_image".to_string()),
            variant_name: Some("anthropic".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "What's in this image?".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Url(UrlFile {
                            url: wikipedia_url.clone(),
                            mime_type: None,
                            detail: None,
                            filename: None,
                        })),
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
        chat_response.usage.input_tokens.unwrap() > 0,
        "Input tokens should be > 0"
    );
    assert!(
        chat_response.usage.output_tokens.unwrap() > 0,
        "Output tokens should be > 0"
    );

    // Sleep to allow ClickHouse write
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Verify ClickHouse data
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_some(), "Inference should be in ClickHouse");

    println!("✓ Test passed: Wikipedia image URL with fetch_and_encode_input_files_before_inference = true");
}

#[tokio::test]
async fn test_wikipedia_image_url_with_fetch_false() {
    let episode_id = Uuid::now_v7();

    let wikipedia_url = Url::parse("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg").unwrap();

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(CONFIG_WITH_FETCH_FALSE).await;

    let response = client
        .inference(ClientInferenceParams {
            function_name: Some("describe_image".to_string()),
            variant_name: Some("anthropic".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "What's in this image?".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Url(UrlFile {
                            url: wikipedia_url.clone(),
                            mime_type: None,
                            detail: None,
                            filename: None,
                        })),
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
        chat_response.usage.input_tokens.unwrap() > 0,
        "Input tokens should be > 0"
    );
    assert!(
        chat_response.usage.output_tokens.unwrap() > 0,
        "Output tokens should be > 0"
    );

    // Sleep to allow ClickHouse write
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Verify ClickHouse data
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_some(), "Inference should be in ClickHouse");

    println!("✓ Test passed: Wikipedia image URL with fetch_and_encode_input_files_before_inference = false");
}

#[tokio::test]
async fn test_image_url_403_error() {
    let episode_id = Uuid::now_v7();

    // The '_shutdown_sender' will wake up the receiver on drop
    let (server_addr, _shutdown_sender) = make_temp_403_image_server().await;
    let image_url = Url::parse(&format!("http://{server_addr}/forbidden.png")).unwrap();

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config(CONFIG_WITH_FETCH_TRUE).await;

    let result = client
        .inference(ClientInferenceParams {
            function_name: Some("describe_image".to_string()),
            variant_name: Some("anthropic".to_string()),
            episode_id: Some(episode_id),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![
                        ClientInputMessageContent::Text(TextKind::Text {
                            text: "What's in this image?".to_string(),
                        }),
                        ClientInputMessageContent::File(File::Url(UrlFile {
                            url: image_url.clone(),
                            mime_type: None,
                            detail: None,
                            filename: None,
                        })),
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

    // The inference should fail with an error about 403 Forbidden
    assert!(
        result.is_err(),
        "Expected error when server returns 403 Forbidden"
    );

    let err = result.unwrap_err();
    let err_msg = format!("{err:?}");

    // The error should contain the server's error message
    assert!(
        err_msg.contains("Access denied: You do not have permission to view this image"),
        "Expected error to contain server's error message, got: {err_msg}"
    );

    println!(
        "✓ Test passed: Image URL 403 error handling (error message includes server response)"
    );
}
