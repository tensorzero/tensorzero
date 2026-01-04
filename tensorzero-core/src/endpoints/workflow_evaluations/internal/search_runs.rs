//! Handler for searching workflow evaluation runs.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;

use super::types::{SearchWorkflowEvaluationRunsResponse, WorkflowEvaluationRun};
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for searching workflow evaluation runs.
#[derive(Debug, Deserialize)]
pub struct SearchWorkflowEvaluationRunsParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub project_name: Option<String>,
    #[serde(rename = "q")]
    pub search_query: Option<String>,
}

const DEFAULT_SEARCH_WORKFLOW_EVALUATION_RUNS_LIMIT: u32 = 100;
const DEFAULT_SEARCH_WORKFLOW_EVALUATION_RUNS_OFFSET: u32 = 0;

/// Handler for `GET /internal/workflow_evaluations/runs/search`
///
/// Searches workflow evaluation runs by project_name and/or search_query.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.search_runs", skip_all)]
pub async fn search_workflow_evaluation_runs_handler(
    State(app_state): AppState,
    Query(params): Query<SearchWorkflowEvaluationRunsParams>,
) -> Result<Json<SearchWorkflowEvaluationRunsResponse>, Error> {
    let response = search_workflow_evaluation_runs(
        &app_state.clickhouse_connection_info,
        params
            .limit
            .unwrap_or(DEFAULT_SEARCH_WORKFLOW_EVALUATION_RUNS_LIMIT),
        params
            .offset
            .unwrap_or(DEFAULT_SEARCH_WORKFLOW_EVALUATION_RUNS_OFFSET),
        params.project_name.as_deref(),
        params.search_query.as_deref(),
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for searching workflow evaluation runs
pub async fn search_workflow_evaluation_runs(
    clickhouse: &impl WorkflowEvaluationQueries,
    limit: u32,
    offset: u32,
    project_name: Option<&str>,
    search_query: Option<&str>,
) -> Result<SearchWorkflowEvaluationRunsResponse, Error> {
    let runs_database = clickhouse
        .search_workflow_evaluation_runs(limit, offset, project_name, search_query)
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
    Ok(SearchWorkflowEvaluationRunsResponse { runs })
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

    fn create_test_run(name: Option<&str>, project: Option<&str>) -> WorkflowEvaluationRunRow {
        WorkflowEvaluationRunRow {
            name: name.map(|s| s.to_string()),
            id: Uuid::now_v7(),
            variant_pins: HashMap::new(),
            tags: HashMap::new(),
            project_name: project.map(|s| s.to_string()),
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_search_workflow_evaluation_runs_with_defaults() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_search_workflow_evaluation_runs()
            .withf(|limit, offset, project_name, search_query| {
                assert_eq!(*limit, 100);
                assert_eq!(*offset, 0);
                assert!(project_name.is_none());
                assert!(search_query.is_none());
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                let run = create_test_run(Some("test_run"), Some("test_project"));
                Box::pin(async move { Ok(vec![run]) })
            });

        let result = search_workflow_evaluation_runs(&mock_clickhouse, 100, 0, None, None)
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
    async fn test_search_workflow_evaluation_runs_with_project_name() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_search_workflow_evaluation_runs()
            .withf(|limit, offset, project_name, search_query| {
                assert_eq!(*limit, 50);
                assert_eq!(*offset, 10);
                assert_eq!(*project_name, Some("my_project"));
                assert!(search_query.is_none());
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                let run = create_test_run(Some("filtered_run"), Some("my_project"));
                Box::pin(async move { Ok(vec![run]) })
            });

        let result =
            search_workflow_evaluation_runs(&mock_clickhouse, 50, 10, Some("my_project"), None)
                .await
                .unwrap();

        assert_eq!(result.runs.len(), 1);
        assert_eq!(result.runs[0].project_name, Some("my_project".to_string()));
    }

    #[tokio::test]
    async fn test_search_workflow_evaluation_runs_with_search_query() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_search_workflow_evaluation_runs()
            .withf(|_limit, _offset, project_name, search_query| {
                assert!(project_name.is_none());
                assert_eq!(*search_query, Some("test"));
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                let run = create_test_run(Some("test_run_matching"), None);
                Box::pin(async move { Ok(vec![run]) })
            });

        let result = search_workflow_evaluation_runs(&mock_clickhouse, 100, 0, None, Some("test"))
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 1);
        assert_eq!(result.runs[0].name, Some("test_run_matching".to_string()));
    }

    #[tokio::test]
    async fn test_search_workflow_evaluation_runs_empty_results() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_search_workflow_evaluation_runs()
            .times(1)
            .returning(|_, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = search_workflow_evaluation_runs(&mock_clickhouse, 100, 0, None, None)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 0);
    }
}
