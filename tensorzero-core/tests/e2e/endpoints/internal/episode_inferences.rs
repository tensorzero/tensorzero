//! E2E tests for the episode inferences endpoint.

use reqwest::Client;
use tensorzero_core::endpoints::internal::episode_inferences::ListEpisodeInferencesResponse;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episode_inferences_requires_episode_id() {
    let http_client = Client::new();
    // Missing episode_id should fail
    let url = get_gateway_endpoint("/internal/episode_inferences");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        !resp.status().is_success(),
        "Should fail when episode_id is not provided"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episode_inferences_with_episode_id() {
    let http_client = Client::new();
    // First, get some inference metadata to find a valid episode_id
    let url = get_gateway_endpoint("/internal/inference_metadata?limit=1");
    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let metadata_response: tensorzero_core::endpoints::internal::inference_metadata::ListInferenceMetadataResponse =
        resp.json().await.unwrap();
    if metadata_response.inference_metadata.is_empty() {
        // No inferences to test with, skip
        return;
    }

    let episode_id = metadata_response.inference_metadata[0].episode_id;

    // Now query episode inferences with that episode_id
    let url = get_gateway_endpoint(&format!(
        "/internal/episode_inferences?episode_id={episode_id}"
    ));
    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_episode_inferences with episode_id failed: status={:?}",
        resp.status()
    );

    let response: ListEpisodeInferencesResponse = resp.json().await.unwrap();
    // All returned inferences should have the specified episode_id
    for inference in &response.inferences {
        let inf_episode_id = match inference {
            tensorzero_core::stored_inference::StoredInference::Chat(chat) => chat.episode_id,
            tensorzero_core::stored_inference::StoredInference::Json(json) => json.episode_id,
        };
        assert_eq!(
            inf_episode_id, episode_id,
            "All inferences should have the specified episode_id"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episode_inferences_with_limit() {
    let http_client = Client::new();
    // First, get some inference metadata to find a valid episode_id
    let url = get_gateway_endpoint("/internal/inference_metadata?limit=1");
    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let metadata_response: tensorzero_core::endpoints::internal::inference_metadata::ListInferenceMetadataResponse =
        resp.json().await.unwrap();
    if metadata_response.inference_metadata.is_empty() {
        // No inferences to test with, skip
        return;
    }

    let episode_id = metadata_response.inference_metadata[0].episode_id;

    // Query with limit
    let url = get_gateway_endpoint(&format!(
        "/internal/episode_inferences?episode_id={episode_id}&limit=5"
    ));
    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_episode_inferences with limit failed: status={:?}",
        resp.status()
    );

    let response: ListEpisodeInferencesResponse = resp.json().await.unwrap();
    assert!(
        response.inferences.len() <= 5,
        "Should not exceed the requested limit"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episode_inferences_before_and_after_mutually_exclusive() {
    let http_client = Client::new();
    let episode_id = Uuid::now_v7();
    let cursor_id = Uuid::now_v7();
    let url = get_gateway_endpoint(&format!(
        "/internal/episode_inferences?episode_id={episode_id}&before={cursor_id}&after={cursor_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        !resp.status().is_success(),
        "Should fail when both before and after are specified"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episode_inferences_with_before() {
    let http_client = Client::new();
    // First, get some inference metadata to find a valid episode_id
    let url = get_gateway_endpoint("/internal/inference_metadata?limit=1");
    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let metadata_response: tensorzero_core::endpoints::internal::inference_metadata::ListInferenceMetadataResponse =
        resp.json().await.unwrap();
    if metadata_response.inference_metadata.is_empty() {
        // No inferences to test with, skip
        return;
    }

    let episode_id = metadata_response.inference_metadata[0].episode_id;
    // Use a UUID that is likely after any existing data
    let cursor = Uuid::now_v7();

    let url = get_gateway_endpoint(&format!(
        "/internal/episode_inferences?episode_id={episode_id}&before={cursor}&limit=5"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_episode_inferences with before failed: status={:?}",
        resp.status()
    );

    let response: ListEpisodeInferencesResponse = resp.json().await.unwrap();
    // All returned inference IDs should be less than the cursor
    for inference in &response.inferences {
        let inf_id = match inference {
            tensorzero_core::stored_inference::StoredInference::Chat(chat) => chat.inference_id,
            tensorzero_core::stored_inference::StoredInference::Json(json) => json.inference_id,
        };
        assert!(
            inf_id < cursor,
            "All inference IDs should be before the cursor"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episode_inferences_with_after() {
    let http_client = Client::new();
    // First, get some inference metadata to find a valid episode_id
    let url = get_gateway_endpoint("/internal/inference_metadata?limit=1");
    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let metadata_response: tensorzero_core::endpoints::internal::inference_metadata::ListInferenceMetadataResponse =
        resp.json().await.unwrap();
    if metadata_response.inference_metadata.is_empty() {
        // No inferences to test with, skip
        return;
    }

    let episode_id = metadata_response.inference_metadata[0].episode_id;
    // Use a UUID that is likely before any existing data (nil UUID)
    let cursor = Uuid::nil();

    let url = get_gateway_endpoint(&format!(
        "/internal/episode_inferences?episode_id={episode_id}&after={cursor}&limit=5"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_episode_inferences with after failed: status={:?}",
        resp.status()
    );

    let response: ListEpisodeInferencesResponse = resp.json().await.unwrap();
    // All returned inference IDs should be greater than the cursor
    for inference in &response.inferences {
        let inf_id = match inference {
            tensorzero_core::stored_inference::StoredInference::Chat(chat) => chat.inference_id,
            tensorzero_core::stored_inference::StoredInference::Json(json) => json.inference_id,
        };
        assert!(
            inf_id > cursor,
            "All inference IDs should be after the cursor"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episode_inferences_nonexistent_episode() {
    let http_client = Client::new();
    // Use a random episode_id that likely doesn't exist
    let episode_id = Uuid::now_v7();

    let url = get_gateway_endpoint(&format!(
        "/internal/episode_inferences?episode_id={episode_id}"
    ));
    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "Should succeed with empty results for nonexistent episode"
    );

    let response: ListEpisodeInferencesResponse = resp.json().await.unwrap();
    assert!(
        response.inferences.is_empty(),
        "Should return empty results for nonexistent episode"
    );
}
