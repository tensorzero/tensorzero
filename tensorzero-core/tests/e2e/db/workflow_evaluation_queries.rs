//! E2E tests for workflow evaluation ClickHouse queries.

use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::workflow_evaluation_queries::WorkflowEvaluationQueries;

/// Test using the shared test database with pre-existing fixture data.
/// This test relies on the fixture data loaded from ui/fixtures/dynamic_evaluation_run_examples.jsonl.
#[tokio::test]
async fn test_list_workflow_evaluation_projects_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .list_workflow_evaluation_projects(100, 0)
        .await
        .unwrap();

    // The fixture data has projects: "beerqa-agentic-rag", "21_questions", and one with NULL project_name
    // Results should be ordered by last_updated DESC
    // Based on the fixture data, both projects have runs with updated_at around 2025-05-05 and 2025-05-01

    // We should have at least 2 projects from the fixture data
    assert!(
        result.len() >= 2,
        "Expected at least 2 projects from fixture data, got {}",
        result.len()
    );

    // Verify the project names are present
    let project_names: Vec<&str> = result.iter().map(|r| r.name.as_str()).collect();

    // These projects exist in the fixture data
    // Note: Other tests may have added more projects, so we just check these exist
    let has_beerqa = project_names.contains(&"beerqa-agentic-rag");
    let has_21_questions = project_names.contains(&"21_questions");

    assert!(
        has_beerqa || has_21_questions,
        "Expected at least one of the fixture projects to be present. Found: {project_names:?}",
    );
}

/// Ensures workflow evaluation project counts are returned from ClickHouse.
#[tokio::test]
async fn test_count_workflow_evaluation_projects_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .count_workflow_evaluation_projects()
        .await
        .unwrap();

    assert!(
        result >= 2,
        "Expected at least 2 workflow evaluation projects from fixtures, got {result}",
    );
}
