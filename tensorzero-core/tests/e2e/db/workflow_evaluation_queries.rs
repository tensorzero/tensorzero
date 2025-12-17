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

/// Test listing workflow evaluation runs with default parameters.
#[tokio::test]
async fn test_list_workflow_evaluation_runs_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .list_workflow_evaluation_runs(100, 0, None, None)
        .await
        .unwrap();

    // The fixture data has multiple runs
    assert!(
        !result.is_empty(),
        "Expected at least one workflow evaluation run from fixture data"
    );

    // Verify runs have the expected fields populated
    let first_run = &result[0];
    assert!(!first_run.id.is_nil(), "Run ID should not be nil");
}

/// Test listing workflow evaluation runs with project_name filter.
#[tokio::test]
async fn test_list_workflow_evaluation_runs_with_project_filter() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .list_workflow_evaluation_runs(100, 0, None, Some("21_questions"))
        .await
        .unwrap();

    // All returned runs should have the specified project name
    for run in &result {
        assert_eq!(
            run.project_name.as_deref(),
            Some("21_questions"),
            "Expected all runs to have project_name '21_questions'"
        );
    }
}

/// Test listing workflow evaluation runs with pagination.
#[tokio::test]
async fn test_list_workflow_evaluation_runs_with_pagination() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .list_workflow_evaluation_runs(1, 0, None, None)
        .await
        .unwrap();

    // With limit=1, we should get at most 1 run
    assert!(
        result.len() <= 1,
        "Expected at most 1 run with limit=1, got {}",
        result.len()
    );
}

/// Test counting workflow evaluation runs.
#[tokio::test]
async fn test_count_workflow_evaluation_runs_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse.count_workflow_evaluation_runs().await.unwrap();

    assert!(
        result >= 1,
        "Expected at least 1 workflow evaluation run from fixtures, got {result}",
    );
}

/// Test searching workflow evaluation runs with default parameters.
#[tokio::test]
async fn test_search_workflow_evaluation_runs_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .search_workflow_evaluation_runs(100, 0, None, None)
        .await
        .unwrap();

    // The fixture data has multiple runs
    assert!(
        !result.is_empty(),
        "Expected at least one workflow evaluation run from fixture data"
    );

    // Verify runs have the expected fields populated
    let first_run = &result[0];
    assert!(!first_run.id.is_nil(), "Run ID should not be nil");
}

/// Test searching workflow evaluation runs with project_name filter.
#[tokio::test]
async fn test_search_workflow_evaluation_runs_with_project_filter() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .search_workflow_evaluation_runs(100, 0, Some("21_questions"), None)
        .await
        .unwrap();

    // All returned runs should have the specified project name
    for run in &result {
        assert_eq!(
            run.project_name.as_deref(),
            Some("21_questions"),
            "Expected all runs to have project_name '21_questions'"
        );
    }
}

/// Test searching workflow evaluation runs with search query.
#[tokio::test]
async fn test_search_workflow_evaluation_runs_with_search_query() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .search_workflow_evaluation_runs(100, 0, None, Some("baseline"))
        .await
        .unwrap();

    // All returned runs should have names containing "baseline"
    for run in &result {
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

/// Test searching workflow evaluation runs with pagination.
#[tokio::test]
async fn test_search_workflow_evaluation_runs_with_pagination() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .search_workflow_evaluation_runs(1, 0, None, None)
        .await
        .unwrap();

    // With limit=1, we should get at most 1 run
    assert!(
        result.len() <= 1,
        "Expected at most 1 run with limit=1, got {}",
        result.len()
    );
}

/// Test getting workflow evaluation runs by IDs.
#[tokio::test]
async fn test_get_workflow_evaluation_runs_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    // Use a known run ID from the fixture data
    let run_id = uuid::Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

    let result = clickhouse
        .get_workflow_evaluation_runs(&[run_id], None)
        .await
        .unwrap();

    // Should return exactly 1 run
    assert_eq!(
        result.len(),
        1,
        "Expected exactly 1 run, got {}",
        result.len()
    );
    assert_eq!(result[0].id, run_id, "Expected run ID to match");
}

/// Test getting workflow evaluation runs with multiple IDs.
#[tokio::test]
async fn test_get_workflow_evaluation_runs_multiple_ids() {
    let clickhouse = get_clickhouse().await;

    // Use known run IDs from the fixture data
    let run_id1 = uuid::Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();
    let run_id2 = uuid::Uuid::parse_str("01968d05-d734-7751-ab33-75dd8b3fb4a3").unwrap();

    let result = clickhouse
        .get_workflow_evaluation_runs(&[run_id1, run_id2], None)
        .await
        .unwrap();

    // Should return 2 runs
    assert_eq!(result.len(), 2, "Expected 2 runs, got {}", result.len());
}

/// Test getting workflow evaluation runs with project_name filter.
#[tokio::test]
async fn test_get_workflow_evaluation_runs_with_project_filter() {
    let clickhouse = get_clickhouse().await;

    let run_id = uuid::Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

    let result = clickhouse
        .get_workflow_evaluation_runs(&[run_id], Some("21_questions"))
        .await
        .unwrap();

    // Should return the run since it belongs to 21_questions project
    assert_eq!(result.len(), 1, "Expected 1 run, got {}", result.len());
    assert_eq!(
        result[0].project_name.as_deref(),
        Some("21_questions"),
        "Expected project_name to be '21_questions'"
    );
}

/// Test getting workflow evaluation runs with empty IDs.
#[tokio::test]
async fn test_get_workflow_evaluation_runs_empty_ids() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .get_workflow_evaluation_runs(&[], None)
        .await
        .unwrap();

    // Should return empty list
    assert_eq!(
        result.len(),
        0,
        "Expected 0 runs for empty IDs, got {}",
        result.len()
    );
}
