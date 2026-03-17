//! E2E tests for the inference metadata endpoint.

use reqwest::Client;
use tensorzero_core::endpoints::internal::inference_metadata::ListInferenceMetadataResponse;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_no_params() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/inference_metadata");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_inference_metadata request failed: status={:?}",
        resp.status()
    );

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    // Should return inference metadata (may be empty if no inferences exist)
    // Just validate the structure is correct
    for meta in &response.inference_metadata {
        assert!(!meta.function_name.is_empty());
        assert!(!meta.variant_name.is_empty());
        // Verify snapshot_hash is lowercase hex of a 256-bit blake3 hash (64 hex chars)
        if let Some(ref hash) = meta.snapshot_hash {
            assert_eq!(
                hash.len(),
                64,
                "snapshot_hash should be 64 hex chars (256-bit hash), got length {} for: {hash}",
                hash.len()
            );
            assert!(
                hash.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
                "snapshot_hash should be lowercase hex, got: {hash}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_with_limit() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/inference_metadata?limit=5");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_inference_metadata request failed: status={:?}",
        resp.status(),
    );

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    assert!(
        response.inference_metadata.len() <= 5,
        "Should not exceed the requested limit"
    );
    // Verify snapshot_hash is properly parsed for all returned records
    for meta in &response.inference_metadata {
        if let Some(ref hash) = meta.snapshot_hash {
            assert!(
                !hash.is_empty(),
                "snapshot_hash should not be empty if present"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_before_and_after_mutually_exclusive() {
    let http_client = Client::new();
    let id = Uuid::now_v7();
    let url = get_gateway_endpoint(&format!(
        "/internal/inference_metadata?before={id}&after={id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        !resp.status().is_success(),
        "Should fail when both before and after are specified"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_with_before() {
    let http_client = Client::new();
    // Use a UUID that is likely after any existing data
    let cursor = Uuid::now_v7();
    let url = get_gateway_endpoint(&format!(
        "/internal/inference_metadata?before={cursor}&limit=5"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_inference_metadata with before failed: status={:?}",
        resp.status()
    );

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    // All returned IDs should be less than the cursor
    for meta in &response.inference_metadata {
        assert!(meta.id < cursor, "All IDs should be before the cursor");
        // Verify snapshot_hash is properly parsed: if present, it should be non-empty
        if let Some(ref hash) = meta.snapshot_hash {
            assert!(
                !hash.is_empty(),
                "snapshot_hash should not be empty if present"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_with_after() {
    let http_client = Client::new();
    // Use a UUID that is likely before any existing data (nil UUID)
    let cursor = Uuid::nil();
    let url = get_gateway_endpoint(&format!(
        "/internal/inference_metadata?after={cursor}&limit=5"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_inference_metadata with after failed: status={:?}",
        resp.status()
    );

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    // All returned IDs should be greater than the cursor
    for meta in &response.inference_metadata {
        assert!(meta.id > cursor, "All IDs should be after the cursor");
        // Verify snapshot_hash is properly parsed: if present, it should be non-empty
        if let Some(ref hash) = meta.snapshot_hash {
            assert!(
                !hash.is_empty(),
                "snapshot_hash should not be empty if present"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_with_function_name() {
    let http_client = Client::new();
    let url =
        get_gateway_endpoint("/internal/inference_metadata?function_name=basic_test&limit=10");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_inference_metadata with function_name failed: status={:?}",
        resp.status()
    );

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    // All returned records should have the specified function_name
    for meta in &response.inference_metadata {
        assert_eq!(
            meta.function_name, "basic_test",
            "All records should have function_name 'basic_test'"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_with_variant_name() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/inference_metadata?variant_name=test&limit=10");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_inference_metadata with variant_name failed: status={:?}",
        resp.status()
    );

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    // All returned records should have the specified variant_name
    for meta in &response.inference_metadata {
        assert_eq!(
            meta.variant_name, "test",
            "All records should have variant_name 'test'"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_with_episode_id() {
    let http_client = Client::new();
    // First, get some inference metadata to find a valid episode_id
    let url = get_gateway_endpoint("/internal/inference_metadata?limit=1");
    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    if response.inference_metadata.is_empty() {
        // No inferences to test with, skip
        return;
    }

    let episode_id = response.inference_metadata[0].episode_id;

    // Now query with that episode_id
    let url = get_gateway_endpoint(&format!(
        "/internal/inference_metadata?episode_id={episode_id}&limit=10"
    ));
    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_inference_metadata with episode_id failed: status={:?}",
        resp.status()
    );

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    // All returned records should have the specified episode_id
    for meta in &response.inference_metadata {
        assert_eq!(
            meta.episode_id, episode_id,
            "All records should have the specified episode_id"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_inference_metadata_with_multiple_filters() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/inference_metadata?function_name=basic_test&variant_name=test&limit=10",
    );

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_inference_metadata with multiple filters failed: status={:?}",
        resp.status()
    );

    let response: ListInferenceMetadataResponse = resp.json().await.unwrap();
    // All returned records should match both filters
    for meta in &response.inference_metadata {
        assert_eq!(
            meta.function_name, "basic_test",
            "All records should have function_name 'basic_test'"
        );
        assert_eq!(
            meta.variant_name, "test",
            "All records should have variant_name 'test'"
        );
    }
}
