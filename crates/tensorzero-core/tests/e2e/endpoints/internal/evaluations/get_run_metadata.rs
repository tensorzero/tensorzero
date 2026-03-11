//! E2E tests for the `GET /internal/evaluations/run_metadata` endpoint.

use googletest::prelude::*;
use googletest_matchers::{matches_json_literal, partially};
use reqwest::{Client, StatusCode};
use serde_json::Value;

use crate::common::get_gateway_endpoint;

/// Fixture run ID for the `entity_extraction` evaluation (JSON function).
const ENTITY_EXTRACTION_RUN_ID: &str = "0196368f-19bd-7082-a677-1c0bf346ff24";
/// Fixture run ID for the `haiku` evaluation (chat function).
const HAIKU_RUN_ID: &str = "01963691-9d3c-7793-a8be-3937ebb849c1";

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_run_metadata_entity_extraction() {
    let http_client = Client::new();

    let url = format!(
        "{}?evaluation_run_ids={}",
        get_gateway_endpoint("/internal/evaluations/run_metadata"),
        ENTITY_EXTRACTION_RUN_ID,
    );
    let resp = http_client
        .get(&url)
        .send()
        .await
        .expect("run_metadata request failed");

    expect_that!(resp.status(), eq(StatusCode::OK));

    let body: Value = resp.json().await.expect("Failed to parse response");

    let metadata = &body["metadata"][ENTITY_EXTRACTION_RUN_ID];

    expect_that!(
        *metadata,
        partially(matches_json_literal!({
            "evaluation_name": "entity_extraction",
            "function_name": "extract_entities",
            "function_type": "json",
        }))
    );

    // This function also contains metrics from outside evaluators, so we use contains_each!
    // to do a partial match on the vector.
    expect_that!(
        metadata["metrics"].as_array(),
        some(contains_each![
            partially(
                matches_json_literal!({"evaluator_name": "count_sports", "value_type": "float"})
            ),
            partially(
                matches_json_literal!({"evaluator_name": "exact_match", "value_type": "boolean"})
            )
        ])
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_run_metadata_haiku() {
    let http_client = Client::new();

    let url = format!(
        "{}?evaluation_run_ids={}",
        get_gateway_endpoint("/internal/evaluations/run_metadata"),
        HAIKU_RUN_ID,
    );
    let resp = http_client
        .get(&url)
        .send()
        .await
        .expect("run_metadata request failed");

    expect_that!(resp.status(), eq(StatusCode::OK));

    let body: Value = resp.json().await.expect("Failed to parse response");

    let metadata = &body["metadata"][HAIKU_RUN_ID];
    expect_that!(
        *metadata,
        partially(matches_json_literal!({
            "evaluation_name": "haiku",
            "function_name": "write_haiku",
            "function_type": "chat"
        }))
    );

    expect_that!(
        metadata["metrics"].as_array(),
        some(unordered_elements_are![
            partially(
                matches_json_literal!({"evaluator_name": "exact_match", "value_type": "boolean"})
            ),
            partially(
                matches_json_literal!({"evaluator_name": "topic_starts_with_f", "value_type": "boolean"})
            ),
        ])
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_run_metadata_multiple_runs() {
    let http_client = Client::new();

    let url = format!(
        "{}?evaluation_run_ids={},{}",
        get_gateway_endpoint("/internal/evaluations/run_metadata"),
        ENTITY_EXTRACTION_RUN_ID,
        HAIKU_RUN_ID,
    );
    let resp = http_client
        .get(&url)
        .send()
        .await
        .expect("run_metadata request failed");

    expect_that!(resp.status(), eq(StatusCode::OK));

    let body: Value = resp.json().await.expect("Failed to parse response");

    expect_that!(
        body["metadata"],
        partially(matches_json_literal!({
            ENTITY_EXTRACTION_RUN_ID: {"evaluation_name": "entity_extraction"},
            HAIKU_RUN_ID: {"evaluation_name": "haiku"},
        }))
    );
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_run_metadata_nonexistent_run() {
    // This explicitly doesn't return 404: we have historical evaluation runs that
    // do not exist in the database, and we want to best-effort return any ones that exist
    // and fall back to resolving with the config.
    let http_client = Client::new();
    let nonexistent_id = uuid::Uuid::now_v7();

    let url = format!(
        "{}?evaluation_run_ids={}",
        get_gateway_endpoint("/internal/evaluations/run_metadata"),
        nonexistent_id,
    );
    let resp = http_client
        .get(&url)
        .send()
        .await
        .expect("run_metadata request failed");

    expect_that!(resp.status(), eq(StatusCode::OK));

    let body: Value = resp.json().await.expect("Failed to parse response");
    expect_that!(body, matches_json_literal!({"metadata": {}}));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_run_metadata_invalid_uuid() {
    let http_client = Client::new();

    let url = format!(
        "{}?evaluation_run_ids=not-a-uuid",
        get_gateway_endpoint("/internal/evaluations/run_metadata"),
    );
    let resp = http_client
        .get(&url)
        .send()
        .await
        .expect("run_metadata request failed");

    expect_that!(resp.status(), eq(StatusCode::BAD_REQUEST));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_run_metadata_empty_ids() {
    let http_client = Client::new();

    let url = format!(
        "{}?evaluation_run_ids=",
        get_gateway_endpoint("/internal/evaluations/run_metadata"),
    );
    let resp = http_client
        .get(&url)
        .send()
        .await
        .expect("run_metadata request failed");

    expect_that!(resp.status(), eq(StatusCode::BAD_REQUEST));
}
