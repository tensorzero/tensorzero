//! E2E tests for the workflow evaluation endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::workflow_evaluations::internal::{
    GetWorkflowEvaluationProjectCountResponse, GetWorkflowEvaluationProjectsResponse,
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
