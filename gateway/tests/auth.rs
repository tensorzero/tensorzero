use std::str::FromStr;

use http::{Method, StatusCode};

use crate::common::start_gateway_on_random_port;

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
        assert_eq!(text, "Dummy error");
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
}
