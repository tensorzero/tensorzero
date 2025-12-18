//! E2E tests for the workflow evaluation endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::workflow_evaluations::internal::{
    CountWorkflowEvaluationRunsResponse, GetWorkflowEvaluationProjectCountResponse,
    GetWorkflowEvaluationProjectsResponse, ListWorkflowEvaluationRunsResponse,
    SearchWorkflowEvaluationRunsResponse,
};

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_projects_endpoint() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/projects");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_projects request failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationProjectsResponse = resp.json().await.unwrap();

    // The test database should have at least one project from the workflow evaluation tests
    assert!(
        !response.projects.is_empty(),
        "Expected at least one workflow evaluation project in the database"
    );

    // Check that the first project has the expected fields
    let first_project = &response.projects[0];
    assert!(
        !first_project.name.is_empty(),
        "Project name should not be empty"
    );
    assert!(
        first_project.count > 0,
        "Project count should be greater than 0"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_projects_with_pagination() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/projects?limit=1&offset=0");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_projects request with pagination failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationProjectsResponse = resp.json().await.unwrap();

    // With limit=1, we should get at most 1 project
    assert!(
        response.projects.len() <= 1,
        "Expected at most 1 project with limit=1, got {}",
        response.projects.len()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_project_count_endpoint() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/projects/count");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_project_count request failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationProjectCountResponse = resp.json().await.unwrap();

    assert!(
        response.count > 0,
        "Expected workflow evaluation project count to be greater than 0"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_workflow_evaluation_runs_endpoint() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/list-runs");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_workflow_evaluation_runs request failed: status={:?}",
        resp.status()
    );

    let response: ListWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // The test database should have at least one run from the workflow evaluation tests
    assert!(
        !response.runs.is_empty(),
        "Expected at least one workflow evaluation run in the database"
    );

    // Check that the first run has the expected fields
    let first_run = &response.runs[0];
    assert!(!first_run.id.is_nil(), "Run ID should not be nil");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_workflow_evaluation_runs_with_pagination() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/list-runs?limit=1&offset=0");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_workflow_evaluation_runs request with pagination failed: status={:?}",
        resp.status()
    );

    let response: ListWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // With limit=1, we should get at most 1 run
    assert!(
        response.runs.len() <= 1,
        "Expected at most 1 run with limit=1, got {}",
        response.runs.len()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_workflow_evaluation_runs_with_project_filter() {
    let http_client = Client::new();
    let url =
        get_gateway_endpoint("/internal/workflow-evaluations/list-runs?project_name=21_questions");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_workflow_evaluation_runs request with project filter failed: status={:?}",
        resp.status()
    );

    let response: ListWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // All returned runs should have the specified project name
    for run in &response.runs {
        assert_eq!(
            run.project_name.as_deref(),
            Some("21_questions"),
            "Expected all runs to have project_name '21_questions'"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_workflow_evaluation_runs_endpoint() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/runs/count");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "count_workflow_evaluation_runs request failed: status={:?}",
        resp.status()
    );

    let response: CountWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    assert!(
        response.count > 0,
        "Expected workflow evaluation run count to be greater than 0"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_workflow_evaluation_runs_endpoint() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/runs/search");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "search_workflow_evaluation_runs request failed: status={:?}",
        resp.status()
    );

    let response: SearchWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // The test database should have at least one run from the workflow evaluation tests
    assert!(
        !response.runs.is_empty(),
        "Expected at least one workflow evaluation run in the database"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_workflow_evaluation_runs_with_project_filter() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/workflow-evaluations/runs/search?project_name=21_questions",
    );

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "search_workflow_evaluation_runs request with project filter failed: status={:?}",
        resp.status()
    );

    let response: SearchWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // All returned runs should have the specified project name
    for run in &response.runs {
        assert_eq!(
            run.project_name.as_deref(),
            Some("21_questions"),
            "Expected all runs to have project_name '21_questions'"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_workflow_evaluation_runs_with_search_query() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/runs/search?q=baseline");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "search_workflow_evaluation_runs request with search query failed: status={:?}",
        resp.status()
    );

    let response: SearchWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // All returned runs should have names or IDs containing "baseline"
    for run in &response.runs {
        let matches_name = run.name.as_ref().is_some_and(|n| n.contains("baseline"));
        let matches_id = run.id.to_string().contains("baseline");
        assert!(
            matches_name || matches_id,
            "Expected run to match search query 'baseline', got name={:?}, id={}",
            run.name,
            run.id
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_workflow_evaluation_runs_with_pagination() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow-evaluations/runs/search?limit=1&offset=0");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "search_workflow_evaluation_runs request with pagination failed: status={:?}",
        resp.status()
    );

    let response: SearchWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // With limit=1, we should get at most 1 run
    assert!(
        response.runs.len() <= 1,
        "Expected at most 1 run with limit=1, got {}",
        response.runs.len()
    );
}
