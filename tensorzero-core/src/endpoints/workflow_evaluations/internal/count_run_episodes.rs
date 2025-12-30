//! Handler for counting workflow evaluation run episodes.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use super::types::CountWorkflowEvaluationRunEpisodesResponse;
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for counting workflow evaluation run episodes.
#[derive(Debug, Deserialize)]
pub struct CountWorkflowEvaluationRunEpisodesParams {
    /// The run ID to count episodes for
    pub run_id: Uuid,
}

/// Handler for `GET /internal/workflow_evaluations/run_episodes/count`
///
/// Returns the total count of episodes for a given workflow evaluation run.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.count_run_episodes", skip_all)]
pub async fn count_workflow_evaluation_run_episodes_total_handler(
    State(app_state): AppState,
    Query(params): Query<CountWorkflowEvaluationRunEpisodesParams>,
) -> Result<Json<CountWorkflowEvaluationRunEpisodesResponse>, Error> {
    let response = count_workflow_evaluation_run_episodes_total(
        &app_state.clickhouse_connection_info,
        params.run_id,
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for counting workflow evaluation run episodes
pub async fn count_workflow_evaluation_run_episodes_total(
    clickhouse: &impl WorkflowEvaluationQueries,
    run_id: Uuid,
) -> Result<CountWorkflowEvaluationRunEpisodesResponse, Error> {
    let count = clickhouse
        .count_workflow_evaluation_run_episodes(run_id)
        .await?;

    Ok(CountWorkflowEvaluationRunEpisodesResponse { count })
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::db::workflow_evaluation_queries::MockWorkflowEvaluationQueries;

    #[tokio::test]
    async fn test_count_run_episodes_empty() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_count_workflow_evaluation_run_episodes()
            .withf(move |id| *id == run_id)
            .times(1)
            .returning(|_| Box::pin(async move { Ok(0) }));

        let result = count_workflow_evaluation_run_episodes_total(&mock_clickhouse, run_id)
            .await
            .unwrap();

        assert_eq!(result.count, 0);
    }

    #[tokio::test]
    async fn test_count_run_episodes_with_data() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();

        mock_clickhouse
            .expect_count_workflow_evaluation_run_episodes()
            .withf(move |id| *id == run_id)
            .times(1)
            .returning(|_| Box::pin(async move { Ok(50) }));

        let result = count_workflow_evaluation_run_episodes_total(&mock_clickhouse, run_id)
            .await
            .unwrap();

        assert_eq!(result.count, 50);
    }
}
