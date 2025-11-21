#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
use reqwest::Url;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent, Role,
};
use tensorzero_core::inference::types::TextKind;

use crate::common::{start_gateway_on_random_port, ChildData};

mod common;

#[tokio::test]
async fn test_base_path_no_trailing_slash() {
    let child_data = start_gateway_on_random_port(
        r#"
        base_path = "/my/prefix"
    "#,
        None,
    )
    .await;

    test_base_path(child_data).await;
}

#[tokio::test]
async fn test_base_path_with_trailing_slash() {
    let child_data = start_gateway_on_random_port(
        r#"
        base_path = "/my/prefix/"
    "#,
        None,
    )
    .await;

    test_base_path(child_data).await;
}

async fn test_base_path(child_data: ChildData) {
    // Prevent cross-container communication issues in CI
    // (the provider-proxy container would try to connect to 'localhost')
    std::env::remove_var("TENSORZERO_E2E_PROXY");
    // The health endpoint should be available at the base path
    let health_response = reqwest::Client::new()
        .get(format!("http://{}/my/prefix/health", child_data.addr))
        .send()
        .await
        .unwrap();
    assert!(health_response.status().is_success());

    let inference_response = reqwest::Client::new()
        .post(format!("http://{}/my/prefix/inference", child_data.addr))
        .body("{}")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert_eq!(inference_response, r#"{"error":"missing field `input`"}"#);

    // The normal endpoints should not be available
    let bad_health_response = reqwest::Client::new()
        .get(format!("http://{}/health", child_data.addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        bad_health_response,
        r#"{"error":"Route not found: GET /health"}"#
    );

    let bad_inference_response = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        bad_inference_response,
        r#"{"error":"Route not found: POST /inference"}"#
    );

    let no_trailing_slash_client =
        tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::HTTPGateway {
            url: Url::parse(&format!("http://{}/my/prefix", child_data.addr)).unwrap(),
        })
        .build()
        .await
        .unwrap();

    no_trailing_slash_client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good_response".to_string()),
            input: ClientInput {
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Hello, world!".to_string(),
                    })],
                }],
                system: None,
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let with_trailing_slash_client =
        tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::HTTPGateway {
            url: Url::parse(&format!("http://{}/my/prefix/", child_data.addr)).unwrap(),
        })
        .build()
        .await
        .unwrap();

    with_trailing_slash_client
        .inference(ClientInferenceParams {
            model_name: Some("dummy::good_response".to_string()),
            input: ClientInput {
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Hello, world!".to_string(),
                    })],
                }],
                system: None,
            },
            ..Default::default()
        })
        .await
        .unwrap();
}
