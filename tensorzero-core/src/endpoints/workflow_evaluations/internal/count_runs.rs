//! Handler for counting workflow evaluation runs.

use axum::Json;
use axum::extract::State;
use tracing::instrument;

use super::types::CountWorkflowEvaluationRunsResponse;
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/workflow_evaluations/runs/count`
///
/// Returns the total count of workflow evaluation runs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.count_runs", skip_all)]
pub async fn count_workflow_evaluation_runs_handler(
    State(app_state): AppState,
) -> Result<Json<CountWorkflowEvaluationRunsResponse>, Error> {
    let response = count_workflow_evaluation_runs(&app_state.clickhouse_connection_info).await?;

    Ok(Json(response))
}

/// Core business logic for counting workflow evaluation runs
pub async fn count_workflow_evaluation_runs(
    clickhouse: &impl WorkflowEvaluationQueries,
) -> Result<CountWorkflowEvaluationRunsResponse, Error> {
    let count = clickhouse.count_workflow_evaluation_runs().await?;
    Ok(CountWorkflowEvaluationRunsResponse { count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::workflow_evaluation_queries::MockWorkflowEvaluationQueries;

    #[tokio::test]
    async fn test_count_workflow_evaluation_runs() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_count_workflow_evaluation_runs()
            .times(1)
            .returning(|| Box::pin(async move { Ok(42) }));

        let response = count_workflow_evaluation_runs(&mock_clickhouse)
            .await
            .unwrap();

        assert_eq!(response.count, 42);
    }

    #[tokio::test]
    async fn test_count_workflow_evaluation_runs_zero() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_count_workflow_evaluation_runs()
            .times(1)
            .returning(|| Box::pin(async move { Ok(0) }));

        let response = count_workflow_evaluation_runs(&mock_clickhouse)
            .await
            .unwrap();

        assert_eq!(response.count, 0);
    }
}
