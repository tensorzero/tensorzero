/// Tests for the GET /internal/episode/{episode_id}/inferences endpoint.
use reqwest::Client;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper function to call get_inferences_by_episode via HTTP
async fn get_inferences_by_episode(
    episode_id: Uuid,
    function_name: Option<&str>,
    variant_name: Option<&str>,
    limit: Option<u32>,
    offset: Option<u32>,
    deduplicate: Option<bool>,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let http_client = Client::new();

    let mut url = get_gateway_endpoint(&format!("/internal/episode/{episode_id}/inferences"));

    // Build query parameters
    {
        let mut query_pairs = url.query_pairs_mut();
        if let Some(fn_name) = function_name {
            query_pairs.append_pair("function_name", fn_name);
        }
        if let Some(var_name) = variant_name {
            query_pairs.append_pair("variant_name", var_name);
        }
        if let Some(l) = limit {
            query_pairs.append_pair("limit", &l.to_string());
        }
        if let Some(o) = offset {
            query_pairs.append_pair("offset", &o.to_string());
        }
        if let Some(d) = deduplicate {
            query_pairs.append_pair("deduplicate", &d.to_string());
        }
    }

    let resp = http_client.get(url).send().await?;

    assert!(
        resp.status().is_success(),
        "get_inferences_by_episode request failed: status={:?}, body={:?}",
        resp.status(),
        resp.text().await
    );

    let resp_json: Value = resp.json().await?;
    let inferences = resp_json["inferences"]
        .as_array()
        .expect("Expected 'inferences' array in response")
        .clone();

    Ok(inferences)
}

/// Helper to call list_inferences to find an episode with inferences
async fn find_episode_with_inferences(
    function_name: &str,
) -> Result<Uuid, Box<dyn std::error::Error>> {
    let http_client = Client::new();
    let request = json!({
        "function_name": function_name,
        "output_source": "inference",
        "limit": 1
    });

    let resp = http_client
        .post(get_gateway_endpoint("/v1/inferences/list_inferences"))
        .json(&request)
        .send()
        .await?;

    assert!(resp.status().is_success());

    let resp_json: Value = resp.json().await?;
    let inferences = resp_json["inferences"]
        .as_array()
        .expect("Expected 'inferences' array");

    assert!(
        !inferences.is_empty(),
        "No inferences found for function {function_name}"
    );

    let episode_id_str = inferences[0]["episode_id"]
        .as_str()
        .expect("Expected episode_id");
    Ok(episode_id_str.parse()?)
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inferences_by_episode_basic() {
    // Find an episode with inferences using the existing write_haiku function
    let episode_id = find_episode_with_inferences("write_haiku").await.unwrap();

    let inferences = get_inferences_by_episode(episode_id, None, None, None, None, None)
        .await
        .unwrap();

    // Should return at least one inference
    assert!(
        !inferences.is_empty(),
        "Expected at least one inference for episode"
    );

    // All inferences should belong to this episode
    for inference in &inferences {
        assert_eq!(
            inference["episode_id"].as_str().unwrap(),
            episode_id.to_string()
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inferences_by_episode_with_limit() {
    let episode_id = find_episode_with_inferences("write_haiku").await.unwrap();

    let inferences = get_inferences_by_episode(episode_id, None, None, Some(1), None, None)
        .await
        .unwrap();

    // Should return at most 1 inference
    assert!(inferences.len() <= 1);
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inferences_by_episode_nonexistent() {
    // Use a random UUID that shouldn't exist
    let nonexistent_episode_id = Uuid::now_v7();

    let inferences =
        get_inferences_by_episode(nonexistent_episode_id, None, None, None, None, None)
            .await
            .unwrap();

    // Should return empty for nonexistent episode
    assert!(
        inferences.is_empty(),
        "Expected empty result for nonexistent episode"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inferences_by_episode_with_function_filter() {
    let episode_id = find_episode_with_inferences("write_haiku").await.unwrap();

    // Filter by function name
    let inferences =
        get_inferences_by_episode(episode_id, Some("write_haiku"), None, None, None, None)
            .await
            .unwrap();

    // All returned inferences should be for write_haiku function
    for inference in &inferences {
        assert_eq!(inference["function_name"].as_str().unwrap(), "write_haiku");
    }

    // Try filtering by a different function - should return empty if episode only has write_haiku
    let inferences_other =
        get_inferences_by_episode(episode_id, Some("extract_entities"), None, None, None, None)
            .await
            .unwrap();

    // This might be empty or have inferences depending on the test data
    for inference in &inferences_other {
        assert_eq!(
            inference["function_name"].as_str().unwrap(),
            "extract_entities"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inferences_by_episode_with_offset() {
    let episode_id = find_episode_with_inferences("write_haiku").await.unwrap();

    // Get all inferences first
    let all_inferences = get_inferences_by_episode(episode_id, None, None, Some(100), None, None)
        .await
        .unwrap();

    if all_inferences.len() > 1 {
        // Get with offset=1
        let offset_inferences =
            get_inferences_by_episode(episode_id, None, None, Some(100), Some(1), None)
                .await
                .unwrap();

        // Should have one less inference
        assert_eq!(offset_inferences.len(), all_inferences.len() - 1);
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inferences_by_episode_timestamp_ordering() {
    let episode_id = find_episode_with_inferences("write_haiku").await.unwrap();

    let inferences = get_inferences_by_episode(episode_id, None, None, Some(10), None, None)
        .await
        .unwrap();

    if inferences.len() > 1 {
        // Verify timestamps are in ascending order (default)
        let mut prev_timestamp: Option<String> = None;
        for inference in &inferences {
            let timestamp = inference["timestamp"].as_str().unwrap().to_string();
            if let Some(prev) = &prev_timestamp {
                assert!(
                    timestamp >= *prev,
                    "Timestamps should be in ascending order. Got: {timestamp} < {prev}"
                );
            }
            prev_timestamp = Some(timestamp);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inferences_by_episode_with_deduplicate() {
    let episode_id = find_episode_with_inferences("write_haiku").await.unwrap();

    // Get inferences without deduplication
    let all_inferences =
        get_inferences_by_episode(episode_id, None, None, Some(100), None, Some(false))
            .await
            .unwrap();

    // Get inferences with deduplication
    let dedup_inferences =
        get_inferences_by_episode(episode_id, None, None, Some(100), None, Some(true))
            .await
            .unwrap();

    // Deduplicated should have <= the number of non-deduplicated
    assert!(
        dedup_inferences.len() <= all_inferences.len(),
        "Deduplicated inferences ({}) should be <= all inferences ({})",
        dedup_inferences.len(),
        all_inferences.len()
    );

    // All deduplicated inferences should be in the original set
    for dedup in &dedup_inferences {
        let dedup_id = dedup["inference_id"].as_str().unwrap();
        let found = all_inferences
            .iter()
            .any(|inf| inf["inference_id"].as_str().unwrap() == dedup_id);
        assert!(
            found,
            "Deduplicated inference {dedup_id} not found in original set"
        );
    }
}
