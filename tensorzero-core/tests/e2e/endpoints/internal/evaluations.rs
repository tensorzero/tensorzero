//! E2E tests for the evaluation endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::internal::evaluations::GetEvaluationRunInfosResponse;
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
