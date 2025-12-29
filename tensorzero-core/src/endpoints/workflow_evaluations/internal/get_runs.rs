//! Handler for getting workflow evaluation runs by IDs.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use super::types::{GetWorkflowEvaluationRunsResponse, WorkflowEvaluationRun};
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for getting workflow evaluation runs by IDs.
#[derive(Debug, Deserialize)]
pub struct GetWorkflowEvaluationRunsParams {
    /// Comma-separated list of run IDs
    pub run_ids: String,
    pub project_name: Option<String>,
}

/// Handler for `GET /internal/workflow_evaluations/get_runs`
///
/// Gets workflow evaluation runs by their IDs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.get_runs", skip_all)]
pub async fn get_workflow_evaluation_runs_handler(
    State(app_state): AppState,
    Query(params): Query<GetWorkflowEvaluationRunsParams>,
) -> Result<Json<GetWorkflowEvaluationRunsResponse>, Error> {
    // Parse comma-separated UUIDs
    let run_ids: Vec<Uuid> = params
        .run_ids
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            Uuid::parse_str(s.trim()).map_err(|e| {
                Error::new(crate::error::ErrorDetails::InvalidRequest {
                    message: format!("Invalid UUID '{s}': {e}"),
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let response = get_workflow_evaluation_runs(
        &app_state.clickhouse_connection_info,
        &run_ids,
        params.project_name.as_deref(),
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for getting workflow evaluation runs by IDs
pub async fn get_workflow_evaluation_runs(
    clickhouse: &impl WorkflowEvaluationQueries,
    run_ids: &[Uuid],
    project_name: Option<&str>,
) -> Result<GetWorkflowEvaluationRunsResponse, Error> {
    let runs_database = clickhouse
        .get_workflow_evaluation_runs(run_ids, project_name)
        .await?;
    let runs = runs_database
        .into_iter()
        .map(|run| WorkflowEvaluationRun {
            name: run.name,
            id: run.id,
            variant_pins: run.variant_pins,
            tags: run.tags,
            project_name: run.project_name,
            timestamp: run.timestamp,
        })
        .collect();
    Ok(GetWorkflowEvaluationRunsResponse { runs })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;
    use uuid::Uuid;

    use super::*;
    use crate::db::workflow_evaluation_queries::{
        MockWorkflowEvaluationQueries, WorkflowEvaluationRunRow,
    };

    fn create_test_run(
        id: Uuid,
        name: Option<&str>,
        project: Option<&str>,
    ) -> WorkflowEvaluationRunRow {
        WorkflowEvaluationRunRow {
            name: name.map(|s| s.to_string()),
            id,
            variant_pins: HashMap::new(),
            tags: HashMap::new(),
            project_name: project.map(|s| s.to_string()),
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_empty_ids() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_runs()
            .withf(|run_ids, project_name| {
                assert!(run_ids.is_empty());
                assert!(project_name.is_none());
                true
            })
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_workflow_evaluation_runs(&mock_clickhouse, &[], None)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 0);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_with_ids() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_runs()
            .withf(move |run_ids, project_name| {
                assert_eq!(run_ids.len(), 1);
                assert_eq!(run_ids[0], run_id);
                assert!(project_name.is_none());
                true
            })
            .times(1)
            .returning(move |_, _| {
                let run = create_test_run(run_id, Some("test_run"), Some("test_project"));
                Box::pin(async move { Ok(vec![run]) })
            });

        let result = get_workflow_evaluation_runs(&mock_clickhouse, &[run_id], None)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 1);
        assert_eq!(result.runs[0].name, Some("test_run".to_string()));
        assert_eq!(
            result.runs[0].project_name,
            Some("test_project".to_string())
        );
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_with_project_name() {
        let run_id = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_runs()
            .withf(move |run_ids, project_name| {
                assert_eq!(run_ids.len(), 1);
                assert_eq!(*project_name, Some("my_project"));
                true
            })
            .times(1)
            .returning(move |_, _| {
                let run = create_test_run(run_id, Some("filtered_run"), Some("my_project"));
                Box::pin(async move { Ok(vec![run]) })
            });

        let result = get_workflow_evaluation_runs(&mock_clickhouse, &[run_id], Some("my_project"))
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 1);
        assert_eq!(result.runs[0].project_name, Some("my_project".to_string()));
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_runs_multiple_ids() {
        let run_id1 = Uuid::now_v7();
        let run_id2 = Uuid::now_v7();
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_get_workflow_evaluation_runs()
            .withf(move |run_ids, _project_name| {
                assert_eq!(run_ids.len(), 2);
                true
            })
            .times(1)
            .returning(move |_, _| {
                let run1 = create_test_run(run_id1, Some("run1"), Some("project"));
                let run2 = create_test_run(run_id2, Some("run2"), Some("project"));
                Box::pin(async move { Ok(vec![run1, run2]) })
            });

        let result = get_workflow_evaluation_runs(&mock_clickhouse, &[run_id1, run_id2], None)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 2);
    }
}
