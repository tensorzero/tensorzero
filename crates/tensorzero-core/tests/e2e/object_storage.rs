use reqwest::StatusCode;
use serde_json::Value;
use urlencoding::encode;

use crate::common::get_gateway_endpoint;

/// The long-running e2e gateway is configured with `[object_storage] type = "disabled"`
/// (see `crates/tensorzero-core/tests/e2e/config/object-storage-disabled.gateway.toml`,
/// the default override pulled in by `get_e2e_config_path`), so any request to
/// `/internal/object_storage` must be rejected with a clear 400 error rather
/// than silently building a fresh store from caller-supplied parameters.
#[tokio::test]
async fn test_object_store_fetch_rejected_when_disabled() {
    let client = reqwest::Client::new();
    let res = client
        .get(get_gateway_endpoint(&format!(
            "/internal/object_storage?path={}",
            encode("fake-tensorzero-file"),
        )))
        .send()
        .await
        .unwrap();

    let status = res.status();
    let res = res.json::<Value>().await.unwrap();
    assert!(
        res["error"]
            .as_str()
            .unwrap()
            .contains("Object storage is not configured"),
        "Unexpected response: {res}"
    );
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Defense-in-depth: even though the gateway's object store is `disabled`, a path
/// containing `..` segments must be rejected at parse time rather than reaching
/// the store layer. This guards against a misconfigured gateway accidentally
/// exposing arbitrary filesystem reads if `[object_storage]` is later enabled.
#[tokio::test]
async fn test_object_store_rejects_path_traversal() {
    let client = reqwest::Client::new();
    let res = client
        .get(get_gateway_endpoint(&format!(
            "/internal/object_storage?path={}",
            encode("../../etc/passwd"),
        )))
        .send()
        .await
        .unwrap();
    let status = res.status();
    let body = res.json::<Value>().await.unwrap();
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("Error parsing object path"),
        "Unexpected response: {body}"
    );
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
