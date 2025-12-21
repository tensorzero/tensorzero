//! E2E tests for the evaluation endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::internal::evaluations::types::GetEvaluationStatisticsResponse;
use tensorzero_core::endpoints::internal::evaluations::{
    GetEvaluationResultsResponse, GetEvaluationRunInfosResponse,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_endpoint() {
    let http_client = Client::new();

    // Use evaluation run IDs from the test fixture data
    let evaluation_run_id1 = "0196368f-19bd-7082-a677-1c0bf346ff24";
    let evaluation_run_id2 = "0196368e-53a8-7e82-a88d-db7086926d81";

    let url = get_gateway_endpoint("/internal/evaluations/run-infos").to_string()
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
    let http_client = Client::new();

    let evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";

    let url = get_gateway_endpoint("/internal/evaluations/run-infos").to_string()
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
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/evaluations/run-infos").to_string()
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
    let http_client = Client::new();

    // Use a valid evaluation run ID but with wrong function name
    let evaluation_run_id = "0196368f-19bd-7082-a677-1c0bf346ff24";

    let url = get_gateway_endpoint("/internal/evaluations/run-infos").to_string()
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
    let http_client = Client::new();

    // Use datapoint ID from the test fixture data for extract_entities function
    let datapoint_id = "0196368e-0b64-7321-ab5b-c32eefbf3e9f";

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/run-infos"
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
    let http_client = Client::new();

    // Use datapoint ID from the test fixture data for write_haiku function
    let datapoint_id = "0196374a-d03f-7420-9da5-1561cba71ddb";

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/run-infos"
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
    assert_eq!(run_info.variant_name, "better_prompt_haiku_3_5");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_run_infos_for_datapoint_nonexistent() {
    let http_client = Client::new();

    let datapoint_id = "00000000-0000-0000-0000-000000000000";

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/run-infos"
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
    let http_client = Client::new();

    // Use a valid datapoint ID but with wrong function name - this will return an error since
    // the function doesn't exist in the config
    let datapoint_id = "0196368e-0b64-7321-ab5b-c32eefbf3e9f";

    let url = get_gateway_endpoint(&format!(
        "/internal/evaluations/datapoints/{datapoint_id}/run-infos"
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
    use tensorzero_core::db::evaluation_queries::EvaluationResultRow;
    let expected_run_id = Uuid::parse_str(evaluation_run_id).unwrap();
    for result in &response.results {
        match result {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.evaluation_run_id, expected_run_id);
                assert_eq!(row.variant_name, "better_prompt_haiku_3_5");
            }
            _ => panic!("Expected Chat result"),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_entity_extraction() {
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
    use tensorzero_core::db::evaluation_queries::EvaluationResultRow;
    for result in &response.results {
        match result {
            EvaluationResultRow::Json(_) => {}
            _ => panic!("Expected Json result"),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_multiple_runs() {
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
    use tensorzero_core::db::evaluation_queries::EvaluationResultRow;
    let eval_run_ids: std::collections::HashSet<_> = response
        .results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.evaluation_run_id,
            EvaluationResultRow::Json(row) => row.evaluation_run_id,
        })
        .collect();
    assert_eq!(
        eval_run_ids.len(),
        2,
        "Expected results from 2 evaluation runs"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_pagination() {
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
    use tensorzero_core::db::evaluation_queries::EvaluationResultRow;
    let page1_datapoints: std::collections::HashSet<_> = page1
        .results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.datapoint_id,
            EvaluationResultRow::Json(row) => row.datapoint_id,
        })
        .collect();
    let page2_datapoints: std::collections::HashSet<_> = page2
        .results
        .iter()
        .map(|r| match r {
            EvaluationResultRow::Chat(row) => row.datapoint_id,
            EvaluationResultRow::Json(row) => row.datapoint_id,
        })
        .collect();

    let overlap: Vec<_> = page1_datapoints.intersection(&page2_datapoints).collect();
    assert!(
        overlap.is_empty(),
        "Pages should not have overlapping datapoints"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_evaluation_results_evaluation_not_found() {
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
