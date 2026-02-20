//! E2E tests for the episodes endpoint.

use reqwest::Client;
use serde_json::json;
use tensorzero_core::db::TableBoundsWithCount;
use tensorzero_core::endpoints::episodes::internal::GetEpisodeInferenceCountResponse;
use tensorzero_core::endpoints::episodes::internal::ListEpisodesResponse;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_query_episode_table_bounds() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/episodes/bounds");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "query_episode_table_bounds request failed: status={:?}",
        resp.status()
    );

    let response: TableBoundsWithCount = resp.json().await.unwrap();

    // The fixture should have some episodes
    assert!(
        response.count > 0,
        "Expected at least one episode in fixtures"
    );
    assert!(response.first_id.is_some(), "Expected first_id to be set");
    assert!(response.last_id.is_some(), "Expected last_id to be set");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_episode_table() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/episodes?limit=10");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "query_episode_table request failed: status={:?}",
        resp.status()
    );

    let response: ListEpisodesResponse = resp.json().await.unwrap();
    let episodes = response.episodes;

    // The fixture should have some episodes
    assert!(
        !episodes.is_empty(),
        "Expected at least one episode in fixtures"
    );

    // Verify each episode has valid data
    for episode in &episodes {
        assert!(
            episode.count > 0,
            "Expected episode to have at least one inference"
        );
        assert!(
            episode.start_time <= episode.end_time,
            "start_time should be <= end_time"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_episode_table_with_pagination() {
    let http_client = Client::new();

    // First, get the bounds to know what episodes exist
    let bounds_url = get_gateway_endpoint("/internal/episodes/bounds");
    let bounds_resp = http_client.get(bounds_url).send().await.unwrap();
    assert!(bounds_resp.status().is_success());
    let bounds: TableBoundsWithCount = bounds_resp.json().await.unwrap();

    // Get the first page
    let first_page_url = get_gateway_endpoint("/internal/episodes?limit=5");
    let first_page_resp = http_client.get(first_page_url).send().await.unwrap();
    assert!(first_page_resp.status().is_success());
    let first_page: ListEpisodesResponse = first_page_resp.json().await.unwrap();

    if first_page.episodes.len() == 5 && bounds.count > 5 {
        // If we have more episodes, test pagination with 'before'
        let last_episode_id = first_page.episodes.last().unwrap().episode_id;
        let second_page_url = get_gateway_endpoint(&format!(
            "/internal/episodes?limit=5&before={last_episode_id}"
        ));
        let second_page_resp = http_client.get(second_page_url).send().await.unwrap();
        assert!(
            second_page_resp.status().is_success(),
            "Pagination request failed"
        );

        let second_page: ListEpisodesResponse = second_page_resp.json().await.unwrap();

        // Ensure no overlap between pages (all IDs in second page should be different)
        for episode in &second_page.episodes {
            assert!(
                !first_page
                    .episodes
                    .iter()
                    .any(|e| e.episode_id == episode.episode_id),
                "Second page should not contain episodes from first page"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_episode_table_limit_zero() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/episodes?limit=0");

    let resp = http_client.get(url).send().await.unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "query_episode_table with limit=0 should return 400"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_episode_table_rejects_limit_over_100() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/episodes?limit=101");

    let resp = http_client.get(url).send().await.unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 Bad Request for limit > 100"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_episode_inference_count() {
    let http_client = Client::new();

    // First get an episode that exists
    let bounds_url = get_gateway_endpoint("/internal/episodes/bounds");
    let bounds_resp = http_client.get(bounds_url).send().await.unwrap();
    assert!(bounds_resp.status().is_success());
    let bounds: TableBoundsWithCount = bounds_resp.json().await.unwrap();

    if let Some(first_id) = bounds.first_id {
        let url = get_gateway_endpoint(&format!("/internal/episodes/{first_id}/inference_count"));
        let resp = http_client.get(url).send().await.unwrap();
        assert!(
            resp.status().is_success(),
            "get_episode_inference_count request failed: status={:?}",
            resp.status()
        );

        let response: GetEpisodeInferenceCountResponse = resp.json().await.unwrap();
        assert!(
            response.inference_count > 0,
            "Expected at least one inference for episode"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_episode_inference_count_nonexistent_episode() {
    let http_client = Client::new();

    // Use a UUID that likely doesn't exist
    let nonexistent_id = Uuid::now_v7();
    let url = get_gateway_endpoint(&format!(
        "/internal/episodes/{nonexistent_id}/inference_count"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "Should return success even for nonexistent episode"
    );

    let response: GetEpisodeInferenceCountResponse = resp.json().await.unwrap();
    assert_eq!(
        response.inference_count, 0,
        "Nonexistent episode should have 0 inferences"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_episode_table_post_with_function_name() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/episodes");

    let body = json!({
        "limit": 10,
        "function_name": "extract_entities"
    });

    let resp = http_client.post(url).json(&body).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "POST episodes with function_name failed: status={:?}",
        resp.status()
    );

    let response: ListEpisodesResponse = resp.json().await.unwrap();
    assert!(
        !response.episodes.is_empty(),
        "Expected at least one episode for function_name `extract_entities`"
    );

    for episode in &response.episodes {
        assert!(
            episode.count > 0,
            "Episode should have at least one inference"
        );
        assert!(
            episode.start_time <= episode.end_time,
            "start_time should be <= end_time"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_episode_table_post_with_function_name_and_filter() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/episodes");

    let body = json!({
        "limit": 10,
        "function_name": "extract_entities",
        "filters": {
            "type": "tag",
            "key": "tensorzero::evaluation_name",
            "value": "entity_extraction",
            "comparison_operator": "="
        }
    });

    let resp = http_client.post(url).json(&body).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "POST episodes with function_name + filter failed: status={:?}",
        resp.status()
    );

    let response: ListEpisodesResponse = resp.json().await.unwrap();
    assert!(
        !response.episodes.is_empty(),
        "Expected at least one episode matching function_name `extract_entities` with tag filter"
    );

    for episode in &response.episodes {
        assert!(
            episode.count > 0,
            "Episode should have at least one inference"
        );
        assert!(
            episode.start_time <= episode.end_time,
            "start_time should be <= end_time"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_episode_table_post_nonexistent_function() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/episodes");

    let body = json!({
        "limit": 10,
        "function_name": "nonexistent_function_12345"
    });

    let resp = http_client.post(url).json(&body).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "POST episodes with nonexistent function_name should succeed: status={:?}",
        resp.status()
    );

    let response: ListEpisodesResponse = resp.json().await.unwrap();
    assert!(
        response.episodes.is_empty(),
        "Expected no episodes for nonexistent function_name"
    );
}
