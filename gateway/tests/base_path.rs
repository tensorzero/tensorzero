#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
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
    // The health endpoint should be available at the base path
    let health_response = reqwest::Client::new()
        .get(format!("http://{}/my/prefix/health", child_data.addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(health_response, r#"{"gateway":"ok","clickhouse":"ok"}"#);

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
}
