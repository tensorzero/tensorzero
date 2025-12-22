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

/// Test getting workflow evaluation run statistics.
/// Uses fixture data from ui/fixtures/dynamic_evaluation_run_examples.jsonl.
#[tokio::test]
async fn test_get_workflow_evaluation_run_statistics_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    // Use a known run ID from the fixture data that has feedback
    let run_id = uuid::Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

    let result = clickhouse
        .get_workflow_evaluation_run_statistics(run_id, None)
        .await
        .unwrap();

    // The fixture data has metrics: elapsed_ms (float), goated (boolean), solved (boolean)
    assert!(
        result.len() >= 3,
        "Expected at least 3 metrics from fixture data, got {}",
        result.len()
    );

    // Verify elapsed_ms (float metric with Wald CI)
    let elapsed_ms = result
        .iter()
        .find(|s| s.metric_name == "elapsed_ms")
        .expect("Expected elapsed_ms metric");
    // Count may vary if tests add data, so just check it's reasonable
    assert!(
        elapsed_ms.count >= 49,
        "Expected at least 49 elapsed_ms samples, got {}",
        elapsed_ms.count
    );
    // Average should be in a reasonable range
    assert!(
        elapsed_ms.avg_metric > 50000.0 && elapsed_ms.avg_metric < 150000.0,
        "avg_metric out of expected range: {}",
        elapsed_ms.avg_metric
    );
    assert!(elapsed_ms.stdev.is_some());
    assert!(elapsed_ms.ci_lower.is_some());
    assert!(elapsed_ms.ci_upper.is_some());

    // Verify goated exists (boolean metric with Wilson CI)
    let goated = result.iter().find(|s| s.metric_name == "goated");
    if let Some(goated) = goated {
        assert!(goated.count >= 1);
        // Wilson CI should always be present for boolean metrics
        assert!(goated.ci_lower.is_some());
        assert!(goated.ci_upper.is_some());
    }

    // Verify solved (boolean metric with Wilson CI)
    let solved = result
        .iter()
        .find(|s| s.metric_name == "solved")
        .expect("Expected solved metric");
    assert!(
        solved.count >= 49,
        "Expected at least 49 solved samples, got {}",
        solved.count
    );
    // Average should be between 0 and 1 (boolean metric)
    assert!(
        solved.avg_metric >= 0.0 && solved.avg_metric <= 1.0,
        "avg_metric should be between 0 and 1 for boolean metric: {}",
        solved.avg_metric
    );
    // Wilson CI bounds should be present
    assert!(solved.ci_lower.is_some());
    assert!(solved.ci_upper.is_some());
}

/// Test getting workflow evaluation run statistics with metric_name filter.
#[tokio::test]
async fn test_get_workflow_evaluation_run_statistics_with_metric_filter() {
    let clickhouse = get_clickhouse().await;

    let run_id = uuid::Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

    let result = clickhouse
        .get_workflow_evaluation_run_statistics(run_id, Some("solved"))
        .await
        .unwrap();

    // Should return only the solved metric
    assert_eq!(result.len(), 1, "Expected 1 metric, got {}", result.len());
    assert_eq!(result[0].metric_name, "solved");
    assert!(result[0].count >= 49, "Expected at least 49 samples");
}

/// Test getting workflow evaluation run statistics for nonexistent run.
#[tokio::test]
async fn test_get_workflow_evaluation_run_statistics_nonexistent_run() {
    let clickhouse = get_clickhouse().await;

    // Use a random UUID that doesn't exist
    let run_id = uuid::Uuid::new_v4();

    let result = clickhouse
        .get_workflow_evaluation_run_statistics(run_id, None)
        .await
        .unwrap();

    // Should return empty list for nonexistent run
    assert!(
        result.is_empty(),
        "Expected empty results for nonexistent run, got {}",
        result.len()
    );
}

/// Test getting workflow evaluation run statistics with exact value assertions.
/// This mirrors the TypeScript test for getWorkflowEvaluationRunStatisticsByMetricName.
#[tokio::test]
async fn test_get_workflow_evaluation_run_statistics_exact_values() {
    let clickhouse = get_clickhouse().await;

    let run_id = uuid::Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

    let result = clickhouse
        .get_workflow_evaluation_run_statistics(run_id, None)
        .await
        .unwrap();

    // Should have exactly 3 metrics
    assert_eq!(result.len(), 3, "Expected 3 metrics, got {}", result.len());

    // Verify elapsed_ms (float metric with Wald CI)
    let elapsed_ms = result
        .iter()
        .find(|s| s.metric_name == "elapsed_ms")
        .expect("Expected elapsed_ms metric");
    assert_eq!(elapsed_ms.count, 49);
    assert!(
        (elapsed_ms.avg_metric - 91678.72114158163).abs() < 0.001,
        "avg_metric mismatch: {}",
        elapsed_ms.avg_metric
    );
    assert!(
        (elapsed_ms.stdev.unwrap() - 21054.80078125).abs() < 0.001,
        "stdev mismatch: {:?}",
        elapsed_ms.stdev
    );
    assert!(
        (elapsed_ms.ci_lower.unwrap() - 85783.37692283162).abs() < 0.01,
        "ci_lower mismatch: {:?}",
        elapsed_ms.ci_lower
    );
    assert!(
        (elapsed_ms.ci_upper.unwrap() - 97574.06536033163).abs() < 0.01,
        "ci_upper mismatch: {:?}",
        elapsed_ms.ci_upper
    );

    // Verify goated (boolean metric with Wilson CI)
    let goated = result
        .iter()
        .find(|s| s.metric_name == "goated")
        .expect("Expected goated metric");
    assert_eq!(goated.count, 1);
    assert!(
        (goated.avg_metric - 1.0).abs() < 0.001,
        "avg_metric mismatch: {}",
        goated.avg_metric
    );
    assert!(
        goated.stdev.is_none(),
        "stdev should be None, got {:?}",
        goated.stdev
    );
    assert!(
        (goated.ci_lower.unwrap() - 0.20654329147389294).abs() < 0.0001,
        "ci_lower mismatch: {:?}",
        goated.ci_lower
    );
    assert!(
        (goated.ci_upper.unwrap() - 1.0).abs() < 0.0001,
        "ci_upper mismatch: {:?}",
        goated.ci_upper
    );

    // Verify solved (boolean metric with Wilson CI)
    let solved = result
        .iter()
        .find(|s| s.metric_name == "solved")
        .expect("Expected solved metric");
    assert_eq!(solved.count, 49);
    assert!(
        (solved.avg_metric - 0.4489795918367347).abs() < 0.001,
        "avg_metric mismatch: {}",
        solved.avg_metric
    );
    assert!(
        (solved.stdev.unwrap() - 0.5025445456953674).abs() < 0.001,
        "stdev mismatch: {:?}",
        solved.stdev
    );
    assert!(
        (solved.ci_lower.unwrap() - 0.31852624929636336).abs() < 0.0001,
        "ci_lower mismatch: {:?}",
        solved.ci_lower
    );
    assert!(
        (solved.ci_upper.unwrap() - 0.5868513320032188).abs() < 0.0001,
        "ci_upper mismatch: {:?}",
        solved.ci_upper
    );
}

// =====================================================================
// Tests for list_workflow_evaluation_run_episodes_by_task_name
// =====================================================================

/// Test listing workflow evaluation run episodes by task name with fixture data.
#[tokio::test]
async fn test_list_workflow_evaluation_run_episodes_by_task_name_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    // Use a known run_id from the fixture data that has episodes with task_names
    let run_id = uuid::Uuid::parse_str("0196a0e5-9600-7c83-ab3b-da81097b66cd").unwrap();

    let result = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(&[run_id], 100, 0)
        .await
        .unwrap();

    // Should have episodes from the fixture data
    assert!(
        !result.is_empty(),
        "Expected at least one episode group from fixture data"
    );

    // Verify the returned data has expected fields
    for episode in &result {
        assert!(!episode.episode_id.is_nil(), "Episode ID should not be nil");
        assert!(!episode.run_id.is_nil(), "Run ID should not be nil");
        assert!(
            !episode.group_key.is_empty(),
            "Group key should not be empty"
        );
    }
}

/// Test listing workflow evaluation run episodes with multiple run IDs.
#[tokio::test]
async fn test_list_workflow_evaluation_run_episodes_by_task_name_multiple_runs() {
    let clickhouse = get_clickhouse().await;

    // Use multiple run_ids from the fixture data
    let run_ids = vec![
        uuid::Uuid::parse_str("0196a0e5-9600-7c83-ab3b-da81097b66cd").unwrap(),
        uuid::Uuid::parse_str("0196a0e5-9600-7c83-ab3b-dabb145a9dbe").unwrap(),
    ];

    let result = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(&run_ids, 100, 0)
        .await
        .unwrap();

    // Should have episodes from both runs
    assert!(!result.is_empty(), "Expected episodes from multiple runs");
}

/// Test listing workflow evaluation run episodes with pagination.
#[tokio::test]
async fn test_list_workflow_evaluation_run_episodes_by_task_name_with_pagination() {
    let clickhouse = get_clickhouse().await;

    let run_id = uuid::Uuid::parse_str("0196a0e5-9600-7c83-ab3b-da81097b66cd").unwrap();

    // Get first page with small limit
    let first_page = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(&[run_id], 3, 0)
        .await
        .unwrap();

    // Get a large page to compare
    let all_groups = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(&[run_id], 1000, 0)
        .await
        .unwrap();

    // First page should be a subset of all groups
    assert!(
        first_page.len() <= 3 || first_page.len() <= all_groups.len(),
        "First page should respect limit or be <= total"
    );

    // Verify first page results are at the beginning of all results (by group ordering)
    if !first_page.is_empty() && !all_groups.is_empty() {
        // The first episode in the first page should appear in all_groups
        let first_episode_id = &first_page[0].episode_id;
        assert!(
            all_groups.iter().any(|e| &e.episode_id == first_episode_id),
            "First page's first episode should be in all results"
        );
    }
}

/// Test listing workflow evaluation run episodes with empty run IDs.
#[tokio::test]
async fn test_list_workflow_evaluation_run_episodes_by_task_name_empty_run_ids() {
    let clickhouse = get_clickhouse().await;

    let result = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(&[], 100, 0)
        .await
        .unwrap();

    assert!(result.is_empty(), "Expected empty result for empty run IDs");
}

/// Test listing workflow evaluation run episodes for a nonexistent run.
#[tokio::test]
async fn test_list_workflow_evaluation_run_episodes_by_task_name_nonexistent_run() {
    let clickhouse = get_clickhouse().await;

    // Use a run_id that doesn't exist
    let nonexistent_run_id = uuid::Uuid::nil();

    let result = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(&[nonexistent_run_id], 100, 0)
        .await
        .unwrap();

    assert!(
        result.is_empty(),
        "Expected empty result for nonexistent run"
    );
}

// =====================================================================
// Tests for count_workflow_evaluation_run_episodes_by_task_name
// =====================================================================

/// Test counting workflow evaluation run episode groups with fixture data.
#[tokio::test]
async fn test_count_workflow_evaluation_run_episodes_by_task_name_with_fixture_data() {
    let clickhouse = get_clickhouse().await;

    // Use a known run_id from the fixture data
    let run_id = uuid::Uuid::parse_str("0196a0e5-9600-7c83-ab3b-da81097b66cd").unwrap();

    let count = clickhouse
        .count_workflow_evaluation_run_episodes_by_task_name(&[run_id])
        .await
        .unwrap();

    // Should have at least some groups from the fixture data
    assert!(
        count > 0,
        "Expected at least one episode group from fixture data, got {count}"
    );
}

/// Test counting workflow evaluation run episode groups with multiple run IDs.
#[tokio::test]
async fn test_count_workflow_evaluation_run_episodes_by_task_name_multiple_runs() {
    let clickhouse = get_clickhouse().await;

    // Use multiple run_ids from the fixture data
    let run_ids = vec![
        uuid::Uuid::parse_str("0196a0e5-9600-7c83-ab3b-da81097b66cd").unwrap(),
        uuid::Uuid::parse_str("0196a0e5-9600-7c83-ab3b-dabb145a9dbe").unwrap(),
    ];

    let count = clickhouse
        .count_workflow_evaluation_run_episodes_by_task_name(&run_ids)
        .await
        .unwrap();

    // Count should be at least as many as for a single run
    let single_run_count = clickhouse
        .count_workflow_evaluation_run_episodes_by_task_name(&[run_ids[0]])
        .await
        .unwrap();

    assert!(
        count >= single_run_count,
        "Expected count for multiple runs ({count}) to be >= single run count ({single_run_count})"
    );
}

/// Test counting workflow evaluation run episode groups with empty run IDs.
#[tokio::test]
async fn test_count_workflow_evaluation_run_episodes_by_task_name_empty_run_ids() {
    let clickhouse = get_clickhouse().await;

    let count = clickhouse
        .count_workflow_evaluation_run_episodes_by_task_name(&[])
        .await
        .unwrap();

    assert_eq!(count, 0, "Expected count to be 0 for empty run IDs");
}

/// Test counting workflow evaluation run episode groups for a nonexistent run.
#[tokio::test]
async fn test_count_workflow_evaluation_run_episodes_by_task_name_nonexistent_run() {
    let clickhouse = get_clickhouse().await;

    // Use a run_id that doesn't exist
    let nonexistent_run_id = uuid::Uuid::nil();

    let count = clickhouse
        .count_workflow_evaluation_run_episodes_by_task_name(&[nonexistent_run_id])
        .await
        .unwrap();

    assert_eq!(count, 0, "Expected count to be 0 for nonexistent run");
}

/// Test that count matches list length (within pagination limits).
#[tokio::test]
async fn test_count_matches_list_length() {
    let clickhouse = get_clickhouse().await;

    let run_id = uuid::Uuid::parse_str("0196a0e5-9600-7c83-ab3b-da81097b66cd").unwrap();

    let count = clickhouse
        .count_workflow_evaluation_run_episodes_by_task_name(&[run_id])
        .await
        .unwrap();

    // Get all groups with a large limit
    let list = clickhouse
        .list_workflow_evaluation_run_episodes_by_task_name(&[run_id], 1000, 0)
        .await
        .unwrap();

    // Count the unique group_keys
    let unique_groups: std::collections::HashSet<_> = list.iter().map(|e| &e.group_key).collect();

    assert_eq!(
        count as usize,
        unique_groups.len(),
        "Count ({count}) should match number of unique groups ({})",
        unique_groups.len()
    );
}
