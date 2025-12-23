//! Handler for counting workflow evaluation run episode groups by task name.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use super::types::CountWorkflowEvaluationRunEpisodesByTaskNameResponse;
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for counting workflow evaluation run episode groups.
#[derive(Debug, Deserialize)]
pub struct CountWorkflowEvaluationRunEpisodesByTaskNameParams {
    /// Comma-separated list of run IDs to filter episodes by.
    pub run_ids: Option<String>,
}

/// Handler for `GET /internal/workflow_evaluations/episodes_by_task_name/count`
///
/// Returns the count of distinct episode groups (by task_name) for the given run IDs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.count_episodes_by_task_name", skip_all)]
pub async fn count_workflow_evaluation_run_episodes_handler(
    State(app_state): AppState,
    Query(params): Query<CountWorkflowEvaluationRunEpisodesByTaskNameParams>,
) -> Result<Json<CountWorkflowEvaluationRunEpisodesByTaskNameResponse>, Error> {
    // Parse run_ids from comma-separated string
    let run_ids = params
        .run_ids
        .map(|s| {
            s.split(',')
                .filter(|s| !s.is_empty())
                .filter_map(|s| Uuid::parse_str(s.trim()).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let response =
        count_workflow_evaluation_run_episodes(&app_state.clickhouse_connection_info, &run_ids)
            .await?;

    Ok(Json(response))
}

/// Core business logic for counting workflow evaluation run episode groups
pub async fn count_workflow_evaluation_run_episodes(
    clickhouse: &impl WorkflowEvaluationQueries,
    run_ids: &[Uuid],
) -> Result<CountWorkflowEvaluationRunEpisodesByTaskNameResponse, Error> {
    let count = clickhouse
        .count_workflow_evaluation_run_episodes_by_task_name(run_ids)
        .await?;

    Ok(CountWorkflowEvaluationRunEpisodesByTaskNameResponse { count })
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::db::workflow_evaluation_queries::MockWorkflowEvaluationQueries;

    #[tokio::test]
    async fn test_count_episode_groups_empty_run_ids() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_count_workflow_evaluation_run_episodes_by_task_name()
            .withf(|run_ids| run_ids.is_empty())
            .times(1)
            .returning(|_| Box::pin(async move { Ok(0) }));

        let result = count_workflow_evaluation_run_episodes(&mock_clickhouse, &[])
            .await
            .unwrap();

        assert_eq!(result.count, 0);
    }

    #[tokio::test]
    async fn test_count_episode_groups_with_run_ids() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();

        mock_clickhouse
            .expect_count_workflow_evaluation_run_episodes_by_task_name()
            .withf(move |run_ids| {
                assert_eq!(run_ids.len(), 1);
                assert_eq!(run_ids[0], run_id);
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(5) }));

        let result = count_workflow_evaluation_run_episodes(&mock_clickhouse, &[run_id])
            .await
            .unwrap();

        assert_eq!(result.count, 5);
    }

    #[tokio::test]
    async fn test_count_episode_groups_multiple_run_ids() {
        let run_id1 = Uuid::now_v7();
        let run_id2 = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();

        mock_clickhouse
            .expect_count_workflow_evaluation_run_episodes_by_task_name()
            .withf(move |run_ids| {
                assert_eq!(run_ids.len(), 2);
                true
            })
            .times(1)
            .returning(|_| Box::pin(async move { Ok(10) }));

        let result = count_workflow_evaluation_run_episodes(&mock_clickhouse, &[run_id1, run_id2])
            .await
            .unwrap();

        assert_eq!(result.count, 10);
    }
}
