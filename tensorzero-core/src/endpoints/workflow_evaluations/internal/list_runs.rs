//! Handler for listing workflow evaluation runs.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use super::types::{ListWorkflowEvaluationRunsResponse, WorkflowEvaluationRunWithEpisodeCount};
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for listing workflow evaluation runs.
#[derive(Debug, Deserialize)]
pub struct ListWorkflowEvaluationRunsParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub run_id: Option<Uuid>,
    pub project_name: Option<String>,
}

const DEFAULT_LIST_WORKFLOW_EVALUATION_RUNS_LIMIT: u32 = 100;
const DEFAULT_LIST_WORKFLOW_EVALUATION_RUNS_OFFSET: u32 = 0;

/// Handler for `GET /internal/workflow_evaluations/runs`
///
/// Returns a paginated list of workflow evaluation runs with episode counts.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.list_runs", skip_all)]
pub async fn list_workflow_evaluation_runs_handler(
    State(app_state): AppState,
    Query(params): Query<ListWorkflowEvaluationRunsParams>,
) -> Result<Json<ListWorkflowEvaluationRunsResponse>, Error> {
    let response = list_workflow_evaluation_runs(
        &app_state.clickhouse_connection_info,
        params
            .limit
            .unwrap_or(DEFAULT_LIST_WORKFLOW_EVALUATION_RUNS_LIMIT),
        params
            .offset
            .unwrap_or(DEFAULT_LIST_WORKFLOW_EVALUATION_RUNS_OFFSET),
        params.run_id,
        params.project_name.as_deref(),
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for listing workflow evaluation runs
pub async fn list_workflow_evaluation_runs(
    clickhouse: &impl WorkflowEvaluationQueries,
    limit: u32,
    offset: u32,
    run_id: Option<Uuid>,
    project_name: Option<&str>,
) -> Result<ListWorkflowEvaluationRunsResponse, Error> {
    let runs_database = clickhouse
        .list_workflow_evaluation_runs(limit, offset, run_id, project_name)
        .await?;
    let runs = runs_database
        .into_iter()
        .map(|run| WorkflowEvaluationRunWithEpisodeCount {
            name: run.name,
            id: run.id,
            variant_pins: run.variant_pins,
            tags: run.tags,
            project_name: run.project_name,
            num_episodes: run.num_episodes,
            timestamp: run.timestamp,
        })
        .collect();
    Ok(ListWorkflowEvaluationRunsResponse { runs })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;
    use uuid::Uuid;

    use super::*;
    use crate::db::workflow_evaluation_queries::{
        MockWorkflowEvaluationQueries, WorkflowEvaluationRunWithEpisodeCountRow,
    };

    fn create_test_run_with_episode_count(
        name: Option<&str>,
        project: Option<&str>,
        num_episodes: u32,
    ) -> WorkflowEvaluationRunWithEpisodeCountRow {
        WorkflowEvaluationRunWithEpisodeCountRow {
            name: name.map(|s| s.to_string()),
            id: Uuid::now_v7(),
            variant_pins: HashMap::new(),
            tags: HashMap::new(),
            project_name: project.map(|s| s.to_string()),
            num_episodes,
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_runs_with_defaults() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_runs()
            .withf(|limit, offset, run_id, project_name| {
                assert_eq!(*limit, 100);
                assert_eq!(*offset, 0);
                assert!(run_id.is_none());
                assert!(project_name.is_none());
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                let run =
                    create_test_run_with_episode_count(Some("test_run"), Some("test_project"), 5);
                Box::pin(async move { Ok(vec![run]) })
            });

        let result = list_workflow_evaluation_runs(&mock_clickhouse, 100, 0, None, None)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 1);
        assert_eq!(result.runs[0].name, Some("test_run".to_string()));
        assert_eq!(
            result.runs[0].project_name,
            Some("test_project".to_string())
        );
        assert_eq!(result.runs[0].num_episodes, 5);
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_runs_with_run_id() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_runs()
            .withf(move |_limit, _offset, rid, project_name| {
                assert_eq!(*rid, Some(run_id));
                assert!(project_name.is_none());
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                let run = create_test_run_with_episode_count(Some("specific_run"), None, 10);
                Box::pin(async move { Ok(vec![run]) })
            });

        let result = list_workflow_evaluation_runs(&mock_clickhouse, 100, 0, Some(run_id), None)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 1);
        assert_eq!(result.runs[0].num_episodes, 10);
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_runs_with_project_name() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_runs()
            .withf(|limit, offset, run_id, project_name| {
                assert_eq!(*limit, 50);
                assert_eq!(*offset, 10);
                assert!(run_id.is_none());
                assert_eq!(*project_name, Some("my_project"));
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                let run =
                    create_test_run_with_episode_count(Some("project_run"), Some("my_project"), 3);
                Box::pin(async move { Ok(vec![run]) })
            });

        let result =
            list_workflow_evaluation_runs(&mock_clickhouse, 50, 10, None, Some("my_project"))
                .await
                .unwrap();

        assert_eq!(result.runs.len(), 1);
        assert_eq!(result.runs[0].project_name, Some("my_project".to_string()));
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_runs_empty_results() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_runs()
            .times(1)
            .returning(|_, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = list_workflow_evaluation_runs(&mock_clickhouse, 100, 0, None, None)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 0);
    }

    #[tokio::test]
    async fn test_list_workflow_evaluation_runs_multiple_results() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_runs()
            .times(1)
            .returning(move |_, _, _, _| {
                Box::pin(async move {
                    Ok(vec![
                        create_test_run_with_episode_count(Some("run1"), Some("project1"), 5),
                        create_test_run_with_episode_count(Some("run2"), Some("project1"), 10),
                        create_test_run_with_episode_count(Some("run3"), Some("project2"), 0),
                    ])
                })
            });

        let result = list_workflow_evaluation_runs(&mock_clickhouse, 100, 0, None, None)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 3);
        assert_eq!(result.runs[0].name, Some("run1".to_string()));
        assert_eq!(result.runs[1].name, Some("run2".to_string()));
        assert_eq!(result.runs[2].name, Some("run3".to_string()));
    }
}
