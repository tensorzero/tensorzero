use std::str::FromStr;

use http::{Method, StatusCode};
use tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres;
use tensorzero_auth::key::TensorZeroApiKey;

use crate::common::start_gateway_on_random_port;
use secrecy::ExposeSecret;

mod common;

#[tokio::test]
async fn test_simple_tensorzero_missing_auth() {
    let child_data = start_gateway_on_random_port(
        "
    [gateway.auth]
    enabled = true
    ",
        None,
    )
    .await;

    let embedded_client = make_embedded_gateway_with_config_and_postgres(
        "
    [gateway.auth]
    enabled = true
    ",
    )
    .await;

    let postgres_pool = embedded_client
        .get_app_state_data()
        .unwrap()
        .postgres_connection_info
        .get_alpha_pool()
        .unwrap();

    let disabled_key =
        tensorzero_auth::postgres::create_key("my_org", "my_workspace", None, &postgres_pool)
            .await
            .unwrap();

    let disabled_at = tensorzero_auth::postgres::disable_key(
        &TensorZeroApiKey::parse(disabled_key.expose_secret()).unwrap(),
        &postgres_pool,
    )
    .await
    .unwrap();

    // TODO - come up with a way of listing all routes.
    // For now, we just check a handful of routes - our auth middleware is applied at the top level,
    // so it should be very difficult for us to accidentally skip a route
    let auth_required_routes = [
        ("POST", "/inference"),
        ("POST", "/batch_inference"),
        ("GET", "/batch_inference/fake-batch-id"),
        (
            "GET",
            "/batch_inference/fake-batch-id/inference/fake-inference-id",
        ),
        ("POST", "/feedback"),
        ("GET", "/metrics"),
        ("GET", "/internal/object_storage"),
        ("GET", "/v1/datasets/get_datapoints"),
    ];

    for (method, path) in auth_required_routes {
        // Authorization runs before we do any parsing of the request parameters/body,
        // so we don't need to provide a valid request here.
        let response = reqwest::Client::new()
            .request(
                Method::from_str(method).unwrap(),
                format!("http://{}/{}", child_data.addr, path),
            )
            .send()
            .await
            .unwrap();

        let status = response.status();
        let text = response.text().await.unwrap();
        assert_eq!(
            text,
            "{\"error\":\"TensorZero authentication error: Authorization header is required\"}"
        );
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        let bad_auth_response = reqwest::Client::new()
            .request(
                Method::from_str(method).unwrap(),
                format!("http://{}/{}", child_data.addr, path),
            )
            .header(http::header::AUTHORIZATION, "bad-header-value")
            .send()
            .await
            .unwrap();

        let status = bad_auth_response.status();
        let text = bad_auth_response.text().await.unwrap();
        assert_eq!(
            text,
            "{\"error\":\"TensorZero authentication error: Authorization header must start with 'Bearer '\"}"
        );
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        let bad_key_format_response = reqwest::Client::new()
            .request(
                Method::from_str(method).unwrap(),
                format!("http://{}/{}", child_data.addr, path),
            )
            .header(http::header::AUTHORIZATION, "Bearer invalid-key-format")
            .send()
            .await
            .unwrap();

        let status = bad_key_format_response.status();
        let text = bad_key_format_response.text().await.unwrap();
        assert_eq!(
            text,
            "{\"error\":\"TensorZero authentication error: Invalid API key: Invalid format for TensorZero API key: API key must be of the form `sk-t0-<short_id>-<long_key>`\"}"
        );
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        let missing_key_response = reqwest::Client::new()
            .request(
                Method::from_str(method).unwrap(),
                format!("http://{}/{}", child_data.addr, path),
            )
            .header(
                http::header::AUTHORIZATION,
                "Bearer sk-t0-aaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            )
            .send()
            .await
            .unwrap();

        let status = missing_key_response.status();
        let text = missing_key_response.text().await.unwrap();
        assert_eq!(
            text,
            "{\"error\":\"TensorZero authentication error: Provided API key does not exist in the database\"}"
        );
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        let disabled_key_response = reqwest::Client::new()
            .request(
                Method::from_str(method).unwrap(),
                format!("http://{}/{}", child_data.addr, path),
            )
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", disabled_key.expose_secret()),
            )
            .send()
            .await
            .unwrap();

        let status = disabled_key_response.status();
        let text = disabled_key_response.text().await.unwrap();
        assert_eq!(
            text,
            format!("{{\"error\":\"TensorZero authentication error: API key was disabled at: {disabled_at}\"}}"),
        );
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
}
