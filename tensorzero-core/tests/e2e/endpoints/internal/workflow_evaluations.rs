//! E2E tests for the workflow evaluation endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::workflow_evaluations::internal::{
    CountWorkflowEvaluationRunEpisodesByTaskNameResponse,
    CountWorkflowEvaluationRunEpisodesResponse, CountWorkflowEvaluationRunsResponse,
    GetWorkflowEvaluationProjectCountResponse, GetWorkflowEvaluationProjectsResponse,
    GetWorkflowEvaluationRunEpisodesWithFeedbackResponse, GetWorkflowEvaluationRunsResponse,
    ListWorkflowEvaluationRunEpisodesByTaskNameResponse, ListWorkflowEvaluationRunsResponse,
    SearchWorkflowEvaluationRunsResponse,
};

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_projects_endpoint() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow_evaluations/projects");

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
    let url = get_gateway_endpoint("/internal/workflow_evaluations/projects?limit=1&offset=0");

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
    let url = get_gateway_endpoint("/internal/workflow_evaluations/projects/count");

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
    let url = get_gateway_endpoint("/internal/workflow_evaluations/list_runs");

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
    let url = get_gateway_endpoint("/internal/workflow_evaluations/list_runs?limit=1&offset=0");

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
        get_gateway_endpoint("/internal/workflow_evaluations/list_runs?project_name=21_questions");

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
    let url = get_gateway_endpoint("/internal/workflow_evaluations/runs/count");

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
    let url = get_gateway_endpoint("/internal/workflow_evaluations/runs/search");

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
        "/internal/workflow_evaluations/runs/search?project_name=21_questions",
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
    let url = get_gateway_endpoint("/internal/workflow_evaluations/runs/search?q=baseline");

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
    let url = get_gateway_endpoint("/internal/workflow_evaluations/runs/search?limit=1&offset=0");

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

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_runs_endpoint() {
    let http_client = Client::new();
    // Use a known run ID from the fixture data
    let run_id = "01968d04-142c-7e53-8ea7-3a3255b518dc";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/get_runs?run_ids={run_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_runs request failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // Should return exactly 1 run with the specified ID
    assert_eq!(
        response.runs.len(),
        1,
        "Expected exactly 1 run, got {}",
        response.runs.len()
    );
    assert_eq!(
        response.runs[0].id.to_string(),
        run_id,
        "Expected run ID to match"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_runs_multiple_ids() {
    let http_client = Client::new();
    // Use known run IDs from the fixture data
    let run_id1 = "01968d04-142c-7e53-8ea7-3a3255b518dc";
    let run_id2 = "01968d05-d734-7751-ab33-75dd8b3fb4a3";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/get_runs?run_ids={run_id1},{run_id2}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_runs request with multiple IDs failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // Should return 2 runs
    assert_eq!(
        response.runs.len(),
        2,
        "Expected 2 runs, got {}",
        response.runs.len()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_runs_with_project_filter() {
    let http_client = Client::new();
    let run_id = "01968d04-142c-7e53-8ea7-3a3255b518dc";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/get_runs?run_ids={run_id}&project_name=21_questions"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_runs request with project filter failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // Should return the run since it belongs to 21_questions project
    assert_eq!(
        response.runs.len(),
        1,
        "Expected 1 run, got {}",
        response.runs.len()
    );
    assert_eq!(
        response.runs[0].project_name.as_deref(),
        Some("21_questions")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_runs_empty_ids() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow_evaluations/get_runs?run_ids=");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_runs request with empty IDs failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationRunsResponse = resp.json().await.unwrap();

    // Should return empty list
    assert_eq!(
        response.runs.len(),
        0,
        "Expected 0 runs for empty IDs, got {}",
        response.runs.len()
    );
}

// =====================================================================
// Episodes By Task Name Endpoints
// =====================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episodes_by_task_name_endpoint() {
    let http_client = Client::new();
    // Use a known run_id from the fixture data
    let run_id = "0196a0e5-9600-7c83-ab3b-da81097b66cd";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/episodes_by_task_name?run_ids={run_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_episodes_by_task_name request failed: status={:?}",
        resp.status()
    );

    let response: ListWorkflowEvaluationRunEpisodesByTaskNameResponse = resp.json().await.unwrap();

    // Should have at least one group of episodes
    assert!(
        !response.episodes.is_empty(),
        "Expected at least one episode group from fixture data"
    );

    // Each group should have at least one episode
    for group in &response.episodes {
        assert!(
            !group.is_empty(),
            "Each episode group should have at least one episode"
        );
        // All episodes in a group should have the same group_key
        let group_key = &group[0].group_key;
        for episode in group {
            assert_eq!(
                &episode.group_key, group_key,
                "All episodes in a group should have the same group_key"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episodes_by_task_name_with_multiple_run_ids() {
    let http_client = Client::new();
    // Use multiple run_ids from the fixture data
    let run_ids = "0196a0e5-9600-7c83-ab3b-da81097b66cd,0196a0e5-9600-7c83-ab3b-dabb145a9dbe";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/episodes_by_task_name?run_ids={run_ids}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_episodes_by_task_name request with multiple run_ids failed: status={:?}",
        resp.status()
    );

    let response: ListWorkflowEvaluationRunEpisodesByTaskNameResponse = resp.json().await.unwrap();

    // Should have episodes from multiple runs
    assert!(
        !response.episodes.is_empty(),
        "Expected episodes from multiple runs"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episodes_by_task_name_with_pagination() {
    let http_client = Client::new();
    let run_id = "0196a0e5-9600-7c83-ab3b-da81097b66cd";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/episodes_by_task_name?run_ids={run_id}&limit=2&offset=0"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_episodes_by_task_name request with pagination failed: status={:?}",
        resp.status()
    );

    let response: ListWorkflowEvaluationRunEpisodesByTaskNameResponse = resp.json().await.unwrap();

    // With limit=2, we should get at most 2 groups
    assert!(
        response.episodes.len() <= 2,
        "Expected at most 2 episode groups with limit=2, got {}",
        response.episodes.len()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_list_episodes_by_task_name_empty_run_ids() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow_evaluations/episodes_by_task_name");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "list_episodes_by_task_name request with no run_ids failed: status={:?}",
        resp.status()
    );

    let response: ListWorkflowEvaluationRunEpisodesByTaskNameResponse = resp.json().await.unwrap();

    // With no run_ids, we should get empty result
    assert!(
        response.episodes.is_empty(),
        "Expected empty result for no run_ids"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_episode_groups_endpoint() {
    let http_client = Client::new();
    // Use a known run_id from the fixture data
    let run_id = "0196a0e5-9600-7c83-ab3b-da81097b66cd";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/episodes_by_task_name/count?run_ids={run_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "count_episode_groups request failed: status={:?}",
        resp.status()
    );

    let response: CountWorkflowEvaluationRunEpisodesByTaskNameResponse = resp.json().await.unwrap();

    // Should have at least one group
    assert!(
        response.count > 0,
        "Expected at least one episode group from fixture data, got {}",
        response.count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_episode_groups_with_multiple_run_ids() {
    let http_client = Client::new();
    // Use multiple run_ids from the fixture data
    let run_ids = "0196a0e5-9600-7c83-ab3b-da81097b66cd,0196a0e5-9600-7c83-ab3b-dabb145a9dbe";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/episodes_by_task_name/count?run_ids={run_ids}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "count_episode_groups request with multiple run_ids failed: status={:?}",
        resp.status()
    );

    let response: CountWorkflowEvaluationRunEpisodesByTaskNameResponse = resp.json().await.unwrap();

    // Should have at least as many groups as for a single run
    assert!(
        response.count > 0,
        "Expected at least one episode group from multiple runs, got {}",
        response.count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_episode_groups_empty_run_ids() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow_evaluations/episodes_by_task_name/count");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "count_episode_groups request with no run_ids failed: status={:?}",
        resp.status()
    );

    let response: CountWorkflowEvaluationRunEpisodesByTaskNameResponse = resp.json().await.unwrap();

    // With no run_ids, we should get 0
    assert_eq!(response.count, 0, "Expected count 0 for no run_ids");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_matches_list_length_endpoint() {
    let http_client = Client::new();
    let run_id = "0196a0e5-9600-7c83-ab3b-da81097b66cd";

    // Get count
    let count_url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/episodes_by_task_name/count?run_ids={run_id}"
    ));
    let count_resp = http_client.get(count_url).send().await.unwrap();
    let count_response: CountWorkflowEvaluationRunEpisodesByTaskNameResponse =
        count_resp.json().await.unwrap();

    // Get list with large limit
    let list_url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/episodes_by_task_name?run_ids={run_id}&limit=1000"
    ));
    let list_resp = http_client.get(list_url).send().await.unwrap();
    let list_response: ListWorkflowEvaluationRunEpisodesByTaskNameResponse =
        list_resp.json().await.unwrap();

    // Count should match number of groups
    assert_eq!(
        count_response.count as usize,
        list_response.episodes.len(),
        "Count ({}) should match number of groups ({})",
        count_response.count,
        list_response.episodes.len()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_run_episodes_endpoint() {
    let http_client = Client::new();
    // Use a known run ID from the fixture data
    let run_id = "01968d04-142c-7e53-8ea7-3a3255b518dc";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/run_episodes?run_id={run_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_run_episodes request failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationRunEpisodesWithFeedbackResponse = resp.json().await.unwrap();

    // Should return episodes for the run
    assert!(
        !response.episodes.is_empty(),
        "Expected at least one episode for the run"
    );

    // Check that the first episode has the expected fields
    let first_episode = &response.episodes[0];
    assert!(
        !first_episode.episode_id.is_nil(),
        "Episode ID should not be nil"
    );
    assert_eq!(
        first_episode.run_id.to_string(),
        run_id,
        "Expected episode to belong to the specified run"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_run_episodes_with_pagination() {
    let http_client = Client::new();
    let run_id = "01968d04-142c-7e53-8ea7-3a3255b518dc";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/run_episodes?run_id={run_id}&limit=1&offset=0"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_run_episodes request with pagination failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationRunEpisodesWithFeedbackResponse = resp.json().await.unwrap();

    // With limit=1, we should get at most 1 episode
    assert!(
        response.episodes.len() <= 1,
        "Expected at most 1 episode with limit=1, got {}",
        response.episodes.len()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_workflow_evaluation_run_episodes_beyond_offset() {
    let http_client = Client::new();
    let run_id = "01968d04-142c-7e53-8ea7-3a3255b518dc";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/run_episodes?run_id={run_id}&limit=10&offset=10000"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "get_workflow_evaluation_run_episodes request with large offset failed: status={:?}",
        resp.status()
    );

    let response: GetWorkflowEvaluationRunEpisodesWithFeedbackResponse = resp.json().await.unwrap();

    // Should return empty list when offset is beyond data
    assert_eq!(
        response.episodes.len(),
        0,
        "Expected 0 episodes with large offset, got {}",
        response.episodes.len()
    );
}

// =====================================================================
// Count Workflow Evaluation Run Episodes Endpoint Tests
// =====================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_count_workflow_evaluation_run_episodes_endpoint() {
    let http_client = Client::new();
    let run_id = "01968d04-142c-7e53-8ea7-3a3255b518dc";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/run_episodes/count?run_id={run_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "count_workflow_evaluation_run_episodes request failed: status={:?}",
        resp.status()
    );

    let response: CountWorkflowEvaluationRunEpisodesResponse = resp.json().await.unwrap();

    // The fixture data should have episodes for this run
    assert!(
        response.count > 0,
        "Expected episode count > 0 for this run, got {}",
        response.count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_workflow_evaluation_run_episodes_nonexistent_run() {
    let http_client = Client::new();
    // Use a valid but non-existent UUIDv7
    let run_id = "01942e26-4693-7e80-8591-47b98e25d999";
    let url = get_gateway_endpoint(&format!(
        "/internal/workflow_evaluations/run_episodes/count?run_id={run_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "count_workflow_evaluation_run_episodes request for non-existent run failed: status={:?}",
        resp.status()
    );

    let response: CountWorkflowEvaluationRunEpisodesResponse = resp.json().await.unwrap();

    // Should return 0 for non-existent run
    assert_eq!(
        response.count, 0,
        "Expected 0 episodes for non-existent run, got {}",
        response.count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_workflow_evaluation_run_episodes_missing_run_id() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/workflow_evaluations/run_episodes/count");

    let resp = http_client.get(url).send().await.unwrap();
    // Should fail with 400 Bad Request when run_id is missing
    assert_eq!(
        resp.status().as_u16(),
        400,
        "Expected 400 status for missing run_id, got {:?}",
        resp.status()
    );
}
