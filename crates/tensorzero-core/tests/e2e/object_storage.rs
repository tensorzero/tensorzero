use reqwest::StatusCode;
use serde_json::Value;
use tensorzero_core::inference::types::storage::{StorageKind, StoragePath};

use crate::common::get_gateway_endpoint;

// We test a successful fetch in `test_image_inference_with_provider_amazon_s3`m
// so we only need to test an invalid fetch here.
#[tokio::test]
async fn test_object_store_fetch_missing() {
    let client = reqwest::Client::new();
    let res = client
        .get(get_gateway_endpoint(&format!(
            "/internal/object_storage?storage_path={}",
            serde_json::to_string(&StoragePath {
                kind: StorageKind::Filesystem {
                    path: "/tmp".to_string()
                },
                path: object_store::path::Path::parse("fake-tensorzero-file").unwrap()
            })
            .unwrap()
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
            .contains("Internal error: Error getting object:"),
        "Unexpected response: {res}"
    );
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
}
