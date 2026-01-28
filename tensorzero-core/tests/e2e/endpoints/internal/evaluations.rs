//! E2E tests for the evaluation endpoints.

use std::time::Duration;

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::evaluation_queries::EvaluationResultRow;
use tensorzero_core::endpoints::internal::evaluations::types::GetEvaluationStatisticsResponse;
use tensorzero_core::endpoints::internal::evaluations::{
    GetEvaluationResultsResponse, GetEvaluationRunInfosResponse,
};
use tokio::time::sleep;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_endpoint() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Use evaluation run IDs from the test fixture data
    let evaluation_run_id1 = "0196368f-19bd-7082-a677-1c0bf346ff24";
    let evaluation_run_id2 = "0196368e-53a8-7e82-a88d-db7086926d81";

    let url = get_gateway_endpoint("/internal/evaluations/run_infos").to_string()
        + &format!(
            "?evaluation_run_ids={evaluation_run_id1},{evaluation_run_id2}&function_name=extract_entities"
        );

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_run_infos request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationRunInfosResponse = resp.json().await.unwrap();

    assert_eq!(
        response.run_infos.len(),
        2,
        "Expected 2 evaluation run infos"
    );

    // Check that the results have the correct structure
    let first = &response.run_infos[0];
    assert_eq!(
        first.evaluation_run_id,
        Uuid::parse_str(evaluation_run_id1).unwrap()
    );
    assert_eq!(first.variant_name, "gpt4o_mini_initial_prompt");
    assert!(!first.most_recent_inference_date.is_empty());

    let second = &response.run_infos[1];
    assert_eq!(
        second.evaluation_run_id,
        Uuid::parse_str(evaluation_run_id2).unwrap()
    );
    assert_eq!(second.variant_name, "gpt4o_initial_prompt");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_single_run() {
    skip_for_postgres!();
    let http_client = Client::new();

    let evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";

    let url = get_gateway_endpoint("/internal/evaluations/run_infos").to_string()
        + &format!("?evaluation_run_ids={evaluation_run_id}&function_name=extract_entities");

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_run_infos single run request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationRunInfosResponse = resp.json().await.unwrap();

    assert_eq!(
        response.run_infos.len(),
        1,
        "Expected 1 evaluation run info"
    );
    assert_eq!(
        response.run_infos[0].evaluation_run_id,
        Uuid::parse_str(evaluation_run_id).unwrap()
    );
    assert_eq!(
        response.run_infos[0].variant_name,
        "gpt4o_mini_initial_prompt"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_nonexistent_run() {
    skip_for_postgres!();
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/evaluations/run_infos").to_string()
        + "?evaluation_run_ids=00000000-0000-0000-0000-000000000000&function_name=extract_entities";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_run_infos nonexistent request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationRunInfosResponse = resp.json().await.unwrap();

    assert_eq!(
        response.run_infos.len(),
        0,
        "Expected 0 evaluation run infos for nonexistent run"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_wrong_function() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Use a valid evaluation run ID but with wrong function name
    let evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";

    let url = get_gateway_endpoint("/internal/evaluations/run_infos").to_string()
        + &format!("?evaluation_run_ids={evaluation_run_id}&function_name=nonexistent_function");

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_run_infos wrong function request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationRunInfosResponse = resp.json().await.unwrap();

    assert_eq!(
        response.run_infos.len(),
        0,
        "Expected 0 evaluation run infos for wrong function"
    );
}

// ============================================================================
// get_evaluation_run_infos_for_datapoint endpoint tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_for_datapoint_json_function() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Use datapoint ID from the test fixture data for extract_entities function
    let datapoint_id = "0196368e-0b64-7321-ab5b-c32eefbf3e9f";

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/run_infos"
    ))
    .to_string()
        + "?function_name=extract_entities";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_run_infos_for_datapoint json request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationRunInfosResponse = resp.json().await.unwrap();

    assert_eq!(
        response.run_infos.len(),
        1,
        "Expected 1 evaluation run info for json datapoint"
    );

    let run_info = &response.run_infos[0];
    assert_eq!(
        run_info.evaluation_run_id,
        Uuid::parse_str("0196368e-53a8-7e82-a88d-db7086926d81").unwrap()
    );
    assert_eq!(run_info.variant_name, "gpt4o_initial_prompt");
    assert!(!run_info.most_recent_inference_date.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_for_datapoint_chat_function() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Use datapoint ID from the test fixture data for write_haiku function
    let datapoint_id = "0196374a-d03f-7420-9da5-1561cba71ddb";

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/run_infos"
    ))
    .to_string()
        + "?function_name=write_haiku";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_run_infos_for_datapoint chat request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationRunInfosResponse = resp.json().await.unwrap();

    assert_eq!(
        response.run_infos.len(),
        1,
        "Expected 1 evaluation run info for chat datapoint"
    );

    let run_info = &response.run_infos[0];
    assert_eq!(
        run_info.evaluation_run_id,
        Uuid::parse_str("0196374b-04a3-7013-9049-e59ed5fe3f74").unwrap()
    );
    assert_eq!(run_info.variant_name, "better_prompt_haiku_4_5");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_for_datapoint_nonexistent() {
    skip_for_postgres!();
    let http_client = Client::new();

    let datapoint_id = "00000000-0000-0000-0000-000000000000";

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/run_infos"
    ))
    .to_string()
        + "?function_name=extract_entities";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_run_infos_for_datapoint nonexistent request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationRunInfosResponse = resp.json().await.unwrap();

    assert_eq!(
        response.run_infos.len(),
        0,
        "Expected 0 evaluation run infos for nonexistent datapoint"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_for_datapoint_wrong_function() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Use a valid datapoint ID but with wrong function name - this will return an error since
    // the function doesn't exist in the config
    let datapoint_id = "0196368e-0b64-7321-ab5b-c32eefbf3e9f";

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/run_infos"
    ))
    .to_string()
        + "?function_name=nonexistent_function";

    let resp = http_client.get(&url).send().await.unwrap();
    // Now this returns an error because the function doesn't exist in the config
    assert!(
        resp.status().is_client_error(),
        "get_evaluation_run_infos_for_datapoint wrong function request should fail: status={:?}",
        resp.status()
    );
}

// ============================================================================
// get_evaluation_statistics endpoint tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_statistics_endpoint() {
    skip_for_postgres!();
    let http_client = Client::new();

    let evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";
    let metric_names = "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match,tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports";

    let url = get_gateway_endpoint("/internal/evaluations/statistics").to_string()
        + &format!(
            "?function_name=extract_entities&function_type=json&metric_names={metric_names}&evaluation_run_ids={evaluation_run_id}"
        );

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_statistics request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationStatisticsResponse = resp.json().await.unwrap();

    // Should have statistics for the metrics
    assert!(
        !response.statistics.is_empty(),
        "Expected at least one statistics entry"
    );

    // Verify structure
    for stat in &response.statistics {
        assert_eq!(
            stat.evaluation_run_id,
            Uuid::parse_str(evaluation_run_id).unwrap()
        );
        assert!(stat.datapoint_count > 0);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_statistics_multiple_runs() {
    skip_for_postgres!();
    let http_client = Client::new();

    let evaluation_run_id1 = "0196368f-19bd-7082-a677-1c0bf346ff24";
    let evaluation_run_id2 = "0196368e-53a8-7e82-a88d-db7086926d81";
    let metric_names =
        "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match";

    let url = get_gateway_endpoint("/internal/evaluations/statistics").to_string()
        + &format!(
            "?function_name=extract_entities&function_type=json&metric_names={metric_names}&evaluation_run_ids={evaluation_run_id1},{evaluation_run_id2}"
        );

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_statistics multiple runs request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationStatisticsResponse = resp.json().await.unwrap();

    // Should have statistics for at least one run
    assert!(
        !response.statistics.is_empty(),
        "Expected statistics for at least one run"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_statistics_empty_run_ids() {
    skip_for_postgres!();
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/evaluations/statistics").to_string()
        + "?function_name=extract_entities&function_type=json&metric_names=some_metric&evaluation_run_ids=";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_statistics empty run_ids request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationStatisticsResponse = resp.json().await.unwrap();

    assert_eq!(
        response.statistics.len(),
        0,
        "Expected 0 statistics for empty run_ids"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_statistics_nonexistent_run() {
    skip_for_postgres!();
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/evaluations/statistics").to_string()
        + "?function_name=extract_entities&function_type=json&metric_names=some_metric&evaluation_run_ids=00000000-0000-0000-0000-000000000000";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_statistics nonexistent run request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationStatisticsResponse = resp.json().await.unwrap();

    assert_eq!(
        response.statistics.len(),
        0,
        "Expected 0 statistics for nonexistent run"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_statistics_invalid_function_type() {
    skip_for_postgres!();
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/evaluations/statistics").to_string()
        + "?function_name=extract_entities&function_type=invalid&metric_names=some_metric&evaluation_run_ids=0196368f-19bd-7082-a677-1c0bf346ff24";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "get_evaluation_statistics invalid function_type should fail: status={:?}",
        resp.status()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_statistics_invalid_uuid() {
    skip_for_postgres!();
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/evaluations/statistics").to_string()
        + "?function_name=extract_entities&function_type=json&metric_names=some_metric&evaluation_run_ids=not-a-uuid";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "get_evaluation_statistics invalid UUID should fail: status={:?}",
        resp.status()
    );
}

// ==================== Get Evaluation Results Tests ====================

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_haiku() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Use evaluation run ID from the test fixture data for haiku evaluation
    let evaluation_run_id = "01963691-9d3c-7793-a8be-3937ebb849c1";

    let url = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + &format!(
            "?evaluation_name=haiku&evaluation_run_ids={evaluation_run_id}&limit=5&offset=0"
        );

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_results request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationResultsResponse = resp.json().await.unwrap();

    // Should get 10 results (5 datapoints * 2 metrics)
    assert_eq!(
        response.results.len(),
        10,
        "Expected 10 results (5 datapoints * 2 metrics)"
    );

    // Verify all results belong to the correct evaluation run
    let expected_run_id = Uuid::parse_str(evaluation_run_id).unwrap();
    for result in &response.results {
        let EvaluationResultRow::Chat(row) = result else {
            panic!("Expected Chat result, got {result:?}");
        };
        assert_eq!(row.evaluation_run_id, expected_run_id);
        assert_eq!(row.variant_name, "better_prompt_haiku_4_5");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_entity_extraction() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Use evaluation run ID from the test fixture data for entity_extraction (JSON function)
    let evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";

    let url = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + &format!(
            "?evaluation_name=entity_extraction&evaluation_run_ids={evaluation_run_id}&limit=2&offset=0"
        );

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_results request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationResultsResponse = resp.json().await.unwrap();

    // Should get 4 results (2 datapoints * 2 metrics)
    assert_eq!(
        response.results.len(),
        4,
        "Expected 4 results (2 datapoints * 2 metrics)"
    );

    // Verify results are JSON type
    for result in &response.results {
        let EvaluationResultRow::Json(_) = result else {
            panic!("Expected Json result, got {result:?}");
        };
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_multiple_runs() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Use two evaluation run IDs from the test fixture data
    let evaluation_run_id1 = "0196374b-04a3-7013-9049-e59ed5fe3f74";
    let evaluation_run_id2 = "01963691-9d3c-7793-a8be-3937ebb849c1";

    let url = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + &format!(
            "?evaluation_name=haiku&evaluation_run_ids={evaluation_run_id1},{evaluation_run_id2}&limit=5&offset=0"
        );

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_results request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationResultsResponse = resp.json().await.unwrap();

    // With ragged data: 5 datapoints * 2 metrics * 2 runs - some missing = 18
    assert_eq!(
        response.results.len(),
        18,
        "Expected 18 results for ragged evaluation"
    );

    // Verify both evaluation runs are present
    let eval_run_ids: std::collections::HashSet<_> = response
        .results
        .iter()
        .map(EvaluationResultRow::evaluation_run_id)
        .collect();
    assert_eq!(
        eval_run_ids.len(),
        2,
        "Expected results from 2 evaluation runs"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_pagination() {
    skip_for_postgres!();
    let http_client = Client::new();

    let evaluation_run_id = "01963691-9d3c-7793-a8be-3937ebb849c1";

    // Get first page
    let url1 = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + &format!(
            "?evaluation_name=haiku&evaluation_run_ids={evaluation_run_id}&limit=3&offset=0"
        );

    // Get second page
    let url2 = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + &format!(
            "?evaluation_name=haiku&evaluation_run_ids={evaluation_run_id}&limit=3&offset=3"
        );

    let resp1 = http_client.get(&url1).send().await.unwrap();
    let resp2 = http_client.get(&url2).send().await.unwrap();

    assert!(resp1.status().is_success());
    assert!(resp2.status().is_success());

    let page1: GetEvaluationResultsResponse = resp1.json().await.unwrap();
    let page2: GetEvaluationResultsResponse = resp2.json().await.unwrap();

    // Each page should have 6 results (3 datapoints * 2 metrics)
    // since there are many more datapoints than the limit
    assert_eq!(
        page1.results.len(),
        6,
        "First page should have 6 results (3 datapoints * 2 metrics)"
    );
    assert_eq!(
        page2.results.len(),
        6,
        "Second page should have 6 results (3 datapoints * 2 metrics)"
    );

    // Verify no overlap between pages
    let page1_datapoints: std::collections::HashSet<_> = page1
        .results
        .iter()
        .map(EvaluationResultRow::datapoint_id)
        .collect();
    let page2_datapoints: std::collections::HashSet<_> = page2
        .results
        .iter()
        .map(EvaluationResultRow::datapoint_id)
        .collect();

    let overlap: Vec<_> = page1_datapoints.intersection(&page2_datapoints).collect();
    assert!(
        overlap.is_empty(),
        "Pages should not have overlapping datapoints"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_evaluation_not_found() {
    skip_for_postgres!();
    let http_client = Client::new();

    let evaluation_run_id = "01963691-9d3c-7793-a8be-3937ebb849c1";

    let url = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + &format!(
            "?evaluation_name=nonexistent_evaluation&evaluation_run_ids={evaluation_run_id}"
        );

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        !resp.status().is_success(),
        "Expected failure for nonexistent evaluation, got status={:?}",
        resp.status()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_invalid_uuid() {
    skip_for_postgres!();
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + "?evaluation_name=haiku&evaluation_run_ids=not-a-uuid";

    let resp = http_client.get(&url).send().await.unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for invalid evaluation run UUID"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_nonexistent_run() {
    skip_for_postgres!();
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + "?evaluation_name=haiku&evaluation_run_ids=00000000-0000-0000-0000-000000000000";

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "Request should succeed even for nonexistent run, got status={:?}",
        resp.status()
    );

    let response: GetEvaluationResultsResponse = resp.json().await.unwrap();
    assert_eq!(
        response.results.len(),
        0,
        "Expected 0 results for nonexistent evaluation run"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_default_pagination() {
    skip_for_postgres!();
    let http_client = Client::new();

    let evaluation_run_id = "01963691-9d3c-7793-a8be-3937ebb849c1";

    // Don't specify limit/offset - should use defaults (100/0)
    let url = get_gateway_endpoint("/internal/evaluations/results").to_string()
        + &format!("?evaluation_name=haiku&evaluation_run_ids={evaluation_run_id}");

    let resp = http_client.get(&url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_evaluation_results request failed: status={:?}",
        resp.status()
    );

    let response: GetEvaluationResultsResponse = resp.json().await.unwrap();

    // With default limit of 100, should return all results for this evaluation run
    // The haiku evaluation run has results for multiple datapoints
    assert!(
        !response.results.is_empty(),
        "Expected results with default pagination"
    );
}
// ============================================================================
// run_evaluation SSE streaming endpoint tests
// ============================================================================

/// Helper function to create a chat datapoint for testing evaluations
async fn create_test_chat_datapoint(
    client: &Client,
    dataset_name: &str,
    datapoint_id: Uuid,
) -> Value {
    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {
                "system": { "assistant_name": "TestBot" },
                "messages": [{
                    "role": "user",
                    "content": [{ "type": "text", "text": "Hello, write me a haiku" }]
                }]
            },
            "output": [{ "type": "text", "text": "Test output response" }],
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Failed to create datapoint: {:?}",
        resp.status()
    );

    resp.json().await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_evaluation_streaming_success() {
    skip_for_postgres!();
    let http_client = Client::new();
    let _clickhouse = get_clickhouse().await;

    // Create a unique dataset with test datapoints
    let dataset_name = format!("test-eval-dataset-{}", Uuid::now_v7());
    let datapoint_id1 = Uuid::now_v7();
    let datapoint_id2 = Uuid::now_v7();

    // Create test datapoints
    create_test_chat_datapoint(&http_client, &dataset_name, datapoint_id1).await;
    create_test_chat_datapoint(&http_client, &dataset_name, datapoint_id2).await;

    // Wait for data to be available in ClickHouse
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Build the evaluation request payload
    let payload = json!({
        "evaluation_config": {
            "type": "inference",
            "function_name": "basic_test",
            "evaluators": {
                "exact_match_eval": {
                    "type": "exact_match",
                }
            }
        },
        "function_config": { "type": "chat" },
        "evaluation_name": "test_evaluation_streaming",
        "dataset_name": dataset_name,
        "variant_name": "test",
        "concurrency": 1,
        "inference_cache": "off",
        "max_datapoints": 2,
    });

    // Make the SSE request
    let mut event_stream = http_client
        .post(get_gateway_endpoint("/internal/evaluations/run"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut events: Vec<Value> = Vec::new();
    let mut start_received = false;
    let mut complete_received = false;
    let mut success_count = 0;
    let mut error_count = 0;

    // Collect events from the stream
    while let Some(event_result) = event_stream.next().await {
        match event_result {
            Ok(Event::Open) => continue,
            Ok(Event::Message(message)) => {
                if message.data == "[DONE]" {
                    break;
                }

                let event: Value = serde_json::from_str(&message.data).unwrap();
                let event_type = event.get("type").and_then(|t| t.as_str());

                match event_type {
                    Some("start") => {
                        start_received = true;
                        assert!(
                            event.get("evaluation_run_id").is_some(),
                            "Start event should have evaluation_run_id"
                        );
                        assert!(
                            event.get("num_datapoints").is_some(),
                            "Start event should have num_datapoints"
                        );
                    }
                    Some("success") => {
                        success_count += 1;
                        assert!(
                            event.get("datapoint").is_some(),
                            "Success event should have datapoint"
                        );
                        assert!(
                            event.get("response").is_some(),
                            "Success event should have response"
                        );
                        assert!(
                            event.get("evaluations").is_some(),
                            "Success event should have evaluations"
                        );
                    }
                    Some("error") => {
                        error_count += 1;
                    }
                    Some("complete") => {
                        complete_received = true;
                        assert!(
                            event.get("evaluation_run_id").is_some(),
                            "Complete event should have evaluation_run_id"
                        );
                    }
                    Some("fatal_error") => {
                        panic!("Received fatal_error event: {:?}", event.get("message"));
                    }
                    _ => {}
                }

                events.push(event);
            }
            Err(reqwest_eventsource::Error::StreamEnded) => break,
            Err(e) => panic!("SSE stream error: {e:?}"),
        }
    }

    assert!(start_received, "Should receive start event");
    assert!(complete_received, "Should receive complete event");
    // We expect either success or error for each datapoint
    assert!(
        success_count + error_count > 0,
        "Should receive at least one success or error event"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_evaluation_streaming_missing_variant() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Request without variant_name or internal_dynamic_variant_config
    let payload = json!({
        "evaluation_config": {
            "type": "inference",
            "function_name": "basic_test",
            "evaluators": {}
        },
        "function_config": { "type": "chat" },
        "evaluation_name": "test_evaluation",
        "dataset_name": "some_dataset",
        // Missing variant_name and internal_dynamic_variant_config
        "concurrency": 1,
        "inference_cache": "off",
    });

    let resp = http_client
        .post(get_gateway_endpoint("/internal/evaluations/run"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Should return 400 for missing variant"
    );

    let body: Value = resp.json().await.unwrap();
    assert!(
        body.get("error").is_some(),
        "Response should contain an error"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_evaluation_streaming_nonexistent_dataset() {
    skip_for_postgres!();
    let http_client = Client::new();

    let payload = json!({
        "evaluation_config": {
            "type": "inference",
            "function_name": "basic_test",
            "evaluators": {}
        },
        "function_config": { "type": "chat" },
        "evaluation_name": "test_evaluation",
        "dataset_name": format!("nonexistent-dataset-{}", Uuid::now_v7()),
        "variant_name": "test",
        "concurrency": 1,
        "inference_cache": "off",
    });

    // The request should start streaming. We should get a start event with 0 datapoints,
    // then a complete event.
    let mut event_stream = http_client
        .post(get_gateway_endpoint("/internal/evaluations/run"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut found_error_or_empty = false;

    while let Some(event_result) = event_stream.next().await {
        match event_result {
            Ok(Event::Open) => continue,
            Ok(Event::Message(message)) => {
                if message.data == "[DONE]" {
                    break;
                }
                let event: Value = serde_json::from_str(&message.data).unwrap();
                let event_type = event.get("type").and_then(|t| t.as_str());

                match event_type {
                    Some("start") => {
                        let num_datapoints = event.get("num_datapoints").and_then(|n| n.as_u64());
                        if num_datapoints == Some(0) {
                            found_error_or_empty = true;
                        }
                    }
                    Some("fatal_error") => {
                        found_error_or_empty = true;
                        break;
                    }
                    Some("complete") => {
                        found_error_or_empty = true;
                        break;
                    }
                    _ => {}
                }
            }
            Err(reqwest_eventsource::Error::StreamEnded) => break,
            Err(_) => {
                found_error_or_empty = true;
                break;
            }
        }
    }

    assert!(
        found_error_or_empty,
        "Should report error or empty dataset for nonexistent dataset"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_evaluation_streaming_with_specific_datapoint_ids() {
    skip_for_postgres!();
    let http_client = Client::new();
    let _clickhouse = get_clickhouse().await;

    // Create a unique dataset with test datapoints
    let dataset_name = format!("test-eval-ids-dataset-{}", Uuid::now_v7());
    let datapoint_id1 = Uuid::now_v7();
    let datapoint_id2 = Uuid::now_v7();
    let datapoint_id3 = Uuid::now_v7();

    // Create test datapoints
    create_test_chat_datapoint(&http_client, &dataset_name, datapoint_id1).await;
    create_test_chat_datapoint(&http_client, &dataset_name, datapoint_id2).await;
    create_test_chat_datapoint(&http_client, &dataset_name, datapoint_id3).await;

    // Wait for data to be available
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Only evaluate specific datapoint IDs (2 out of 3)
    let payload = json!({
        "evaluation_config": {
            "type": "inference",
            "function_name": "basic_test",
            "evaluators": {
                "exact_match_eval": {
                    "type": "exact_match",
                }
            }
        },
        "function_config": { "type": "chat" },
        "evaluation_name": "test_evaluation_datapoint_ids",
        "datapoint_ids": [datapoint_id1.to_string(), datapoint_id2.to_string()],
        "variant_name": "test",
        "concurrency": 1,
        "inference_cache": "off",
    });

    let mut event_stream = http_client
        .post(get_gateway_endpoint("/internal/evaluations/run"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut num_datapoints_reported = None;
    let mut start_received = false;

    while let Some(event_result) = event_stream.next().await {
        match event_result {
            Ok(Event::Open) => continue,
            Ok(Event::Message(message)) => {
                if message.data == "[DONE]" {
                    break;
                }

                let event: Value = serde_json::from_str(&message.data).unwrap();
                let event_type = event.get("type").and_then(|t| t.as_str());

                match event_type {
                    Some("start") => {
                        start_received = true;
                        num_datapoints_reported =
                            event.get("num_datapoints").and_then(|n| n.as_u64());
                    }
                    Some("complete") => break,
                    Some("fatal_error") => {
                        panic!("Received fatal_error event: {:?}", event.get("message"));
                    }
                    _ => {}
                }
            }
            Err(reqwest_eventsource::Error::StreamEnded) => break,
            Err(e) => panic!("SSE stream error: {e:?}"),
        }
    }

    assert!(start_received, "Should receive start event");
    assert_eq!(
        num_datapoints_reported,
        Some(2),
        "Should report exactly 2 datapoints for the 2 specific IDs"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_evaluation_streaming_conflicting_variant_config() {
    skip_for_postgres!();
    let http_client = Client::new();

    // Provide both variant_name AND internal_dynamic_variant_config (should fail)
    let payload = json!({
        "evaluation_config": {
            "type": "inference",
            "function_name": "basic_test",
            "evaluators": {}
        },
        "function_config": { "type": "chat" },
        "evaluation_name": "test_evaluation",
        "dataset_name": "some_dataset",
        "variant_name": "test",
        "internal_dynamic_variant_config": {
            "type": "chat_completion",
            "model": "test",
            "active": true,
        },
        "concurrency": 1,
        "inference_cache": "off",
    });

    let resp = http_client
        .post(get_gateway_endpoint("/internal/evaluations/run"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Should return 400 when both variant_name and internal_dynamic_variant_config are provided"
    );

    let body: Value = resp.json().await.unwrap();
    assert!(
        body.get("error").is_some(),
        "Response should contain an error"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_evaluation_streaming_invalid_inference_cache() {
    skip_for_postgres!();
    let http_client = Client::new();

    let payload = json!({
        "evaluation_config": {
            "type": "inference",
            "function_name": "basic_test",
            "evaluators": {}
        },
        "function_config": { "type": "chat" },
        "evaluation_name": "test_evaluation",
        "dataset_name": "some_dataset",
        "variant_name": "test",
        "concurrency": 1,
        "inference_cache": "invalid_cache_mode",  // Invalid value
    });

    let resp = http_client
        .post(get_gateway_endpoint("/internal/evaluations/run"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Should return 400 for invalid inference_cache setting"
    );

    let body: Value = resp.json().await.unwrap();
    assert!(
        body.get("error").is_some(),
        "Response should contain an error"
    );
}

// ============================================================================
// get_human_feedback endpoint tests
// ============================================================================

/// Test that get_human_feedback returns feedback when it exists.
/// This test creates an inference, submits human feedback with the required tags,
/// and then verifies the endpoint returns the correct feedback.
#[tokio::test(flavor = "multi_thread")]
async fn test_get_human_feedback_returns_feedback_when_exists() {
    skip_for_postgres!();
    let http_client = Client::new();

    // First, run an inference to get an inference_id
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = http_client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "inference request failed: {:?}",
        response.status()
    );
    let response_json: Value = response.json().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let serialized_inference_output =
        serde_json::to_string(response_json.get("output").unwrap()).unwrap();

    // Create datapoint_id and evaluator_inference_id
    let datapoint_id = Uuid::now_v7();
    let evaluator_inference_id = Uuid::now_v7();

    // Submit human feedback with required tags
    let feedback_payload = json!({
        "inference_id": inference_id,
        "metric_name": "brevity_score",
        "value": 0.85,
        "internal": true,
        "tags": {
            "tensorzero::human_feedback": "true",
            "tensorzero::datapoint_id": datapoint_id.to_string(),
            "tensorzero::evaluator_inference_id": evaluator_inference_id.to_string()
        }
    });

    let response = http_client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "feedback request failed: {:?}",
        response.status()
    );

    // Wait for ClickHouse to process the data
    sleep(Duration::from_secs(1)).await;

    // Now call the get_human_feedback endpoint
    let resp = http_client
        .post(get_gateway_endpoint(&format!(
            "/internal/evaluations/datapoints/{datapoint_id}/get_human_feedback"
        )))
        .json(&json!({
            "metric_name": "brevity_score",
            "output": serialized_inference_output
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "get_human_feedback request failed: status={:?}",
        resp.status()
    );

    let response: serde_json::Value = resp.json().await.unwrap();

    assert!(
        response.get("feedback").is_some(),
        "Expected feedback to be present"
    );

    let feedback = response.get("feedback").unwrap();
    assert_eq!(feedback.get("value").unwrap(), &json!(0.85));
    assert_eq!(
        feedback
            .get("evaluator_inference_id")
            .unwrap()
            .as_str()
            .unwrap(),
        evaluator_inference_id.to_string()
    );
}

/// Test that get_human_feedback returns None when no feedback exists.
#[tokio::test(flavor = "multi_thread")]
async fn test_get_human_feedback_returns_none_when_not_exists() {
    skip_for_postgres!();
    let http_client = Client::new();

    let nonexistent_datapoint_id = Uuid::now_v7();

    let resp = http_client
        .post(get_gateway_endpoint(&format!(
            "/internal/evaluations/datapoints/{nonexistent_datapoint_id}/get_human_feedback"
        )))
        .json(&json!({
            "metric_name": "task_success",
            "output": "some_output"
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "get_human_feedback request failed: status={:?}",
        resp.status()
    );

    let response: serde_json::Value = resp.json().await.unwrap();

    // When feedback is None, the field is omitted from the response
    assert!(
        response.get("feedback").is_none() || response.get("feedback").unwrap().is_null(),
        "Expected feedback to be None for nonexistent metric"
    );
}

/// Test that get_human_feedback works with boolean feedback values.
#[tokio::test(flavor = "multi_thread")]
async fn test_get_human_feedback_with_boolean_value() {
    skip_for_postgres!();
    let http_client = Client::new();

    // First, run an inference to get an inference_id
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = http_client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let response_json: Value = response.json().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let serialized_inference_output =
        serde_json::to_string(response_json.get("output").unwrap()).unwrap();

    // Create datapoint_id and evaluator_inference_id
    let datapoint_id = Uuid::now_v7();
    let evaluator_inference_id = Uuid::now_v7();

    // Submit boolean human feedback with required tags
    let feedback_payload = json!({
        "inference_id": inference_id,
        "metric_name": "task_success",
        "value": true,
        "internal": true,
        "tags": {
            "tensorzero::human_feedback": "true",
            "tensorzero::datapoint_id": datapoint_id.to_string(),
            "tensorzero::evaluator_inference_id": evaluator_inference_id.to_string()
        }
    });

    let response = http_client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Wait for ClickHouse to process the data
    sleep(Duration::from_secs(1)).await;

    // Now call the get_human_feedback endpoint
    let resp = http_client
        .post(get_gateway_endpoint(&format!(
            "/internal/evaluations/datapoints/{datapoint_id}/get_human_feedback"
        )))
        .json(&json!({
            "metric_name": "task_success",
            "output": serialized_inference_output
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let response: serde_json::Value = resp.json().await.unwrap();

    assert!(response.get("feedback").is_some());

    let feedback = response.get("feedback").unwrap();
    assert_eq!(feedback.get("value").unwrap(), &json!(true));
    assert_eq!(
        feedback
            .get("evaluator_inference_id")
            .unwrap()
            .as_str()
            .unwrap(),
        evaluator_inference_id.to_string()
    );
}

/// Test that get_human_feedback returns the correct feedback when output doesn't match.
#[tokio::test(flavor = "multi_thread")]
async fn test_get_human_feedback_output_mismatch() {
    skip_for_postgres!();
    let http_client = Client::new();

    // First, run an inference to get an inference_id
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = http_client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let response_json: Value = response.json().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let serialized_inference_output =
        serde_json::to_string(response_json.get("output").unwrap()).unwrap();

    // Create datapoint_id and evaluator_inference_id
    let datapoint_id = Uuid::now_v7();
    let evaluator_inference_id = Uuid::now_v7();

    // Submit human feedback
    let feedback_payload = json!({
        "inference_id": inference_id,
        "metric_name": "brevity_score",
        "value": 0.5,
        "internal": true,
        "tags": {
            "tensorzero::human_feedback": "true",
            "tensorzero::datapoint_id": datapoint_id.to_string(),
            "tensorzero::evaluator_inference_id": evaluator_inference_id.to_string()
        }
    });

    let response = http_client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Wait for ClickHouse to process the data
    sleep(Duration::from_secs(1)).await;

    // Query with correct output - should find feedback
    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/get_human_feedback"
    ));

    let resp = http_client
        .post(url.clone())
        .json(&json!({
            "metric_name": "brevity_score",
            "output": serialized_inference_output
        }))
        .send()
        .await
        .unwrap();
    let response: serde_json::Value = resp.json().await.unwrap();
    assert!(
        response.get("feedback").is_some() && !response.get("feedback").unwrap().is_null(),
        "Should find feedback with matching output"
    );

    // Query with different output - should not find feedback
    let resp = http_client
        .post(url)
        .json(&json!({
            "metric_name": "brevity_score",
            "output": "different_output"
        }))
        .send()
        .await
        .unwrap();
    let response: serde_json::Value = resp.json().await.unwrap();
    // When feedback is None, the field is omitted from the response
    assert!(
        response.get("feedback").is_none() || response.get("feedback").unwrap().is_null(),
        "Should not find feedback with different output"
    );
}

/// Test that get_human_feedback handles invalid UUID in datapoint_id.
#[tokio::test(flavor = "multi_thread")]
async fn test_get_human_feedback_invalid_uuid() {
    skip_for_postgres!();
    let http_client = Client::new();

    let resp = http_client
        .post(get_gateway_endpoint(
            "/internal/evaluations/datapoints/not-a-uuid/get_human_feedback",
        ))
        .json(&json!({
            "metric_name": "test_metric",
            "output": "test_output"
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected client error for invalid UUID, got status={:?}",
        resp.status()
    );
}

/// Test that get_human_feedback requires all parameters.
#[tokio::test(flavor = "multi_thread")]
async fn test_get_human_feedback_missing_parameters() {
    skip_for_postgres!();
    let http_client = Client::new();
    let datapoint_id = Uuid::now_v7();

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/get_human_feedback"
    ));

    // Missing metric_name
    let resp = http_client
        .post(url.clone())
        .json(&json!({
            "output": "test_output"
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected client error for missing metric_name, got status={:?}",
        resp.status()
    );

    // Missing output
    let resp = http_client
        .post(url.clone())
        .json(&json!({
            "metric_name": "test_metric"
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected client error for missing output, got status={:?}",
        resp.status()
    );

    // Empty body
    let resp = http_client.post(url).json(&json!({})).send().await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected client error for empty body, got status={:?}",
        resp.status()
    );
}
