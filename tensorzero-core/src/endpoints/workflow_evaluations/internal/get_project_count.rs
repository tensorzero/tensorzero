//! Handler for getting the workflow evaluation project count.

use axum::Json;
use axum::extract::State;
use tracing::instrument;

use super::types::GetWorkflowEvaluationProjectCountResponse;
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/workflow_evaluations/projects/count`
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.count_projects", skip_all)]
pub async fn get_workflow_evaluation_project_count_handler(
    State(app_state): AppState,
) -> Result<Json<GetWorkflowEvaluationProjectCountResponse>, Error> {
    let response =
        get_workflow_evaluation_project_count(&app_state.clickhouse_connection_info).await?;

    Ok(Json(response))
}

/// Core business logic for retrieving the workflow evaluation project count.
pub async fn get_workflow_evaluation_project_count(
    clickhouse: &impl WorkflowEvaluationQueries,
) -> Result<GetWorkflowEvaluationProjectCountResponse, Error> {
    let count = clickhouse.count_workflow_evaluation_projects().await?;
    Ok(GetWorkflowEvaluationProjectCountResponse { count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::workflow_evaluation_queries::MockWorkflowEvaluationQueries;

    #[tokio::test]
    async fn test_get_workflow_evaluation_project_count() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_count_workflow_evaluation_projects()
            .times(1)
            .returning(|| Box::pin(async move { Ok(7) }));

        let response = get_workflow_evaluation_project_count(&mock_clickhouse)
            .await
            .unwrap();

        assert_eq!(response.count, 7);
    }
}
