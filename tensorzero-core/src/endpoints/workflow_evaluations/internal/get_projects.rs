//! Handler for getting workflow evaluation projects.

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use tracing::instrument;

use super::types::{GetWorkflowEvaluationProjectsResponse, WorkflowEvaluationProject};
use crate::db::workflow_evaluation_queries::WorkflowEvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for getting workflow evaluation projects.
#[derive(Debug, Deserialize)]
pub struct GetWorkflowEvaluationProjectsParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

const DEFAULT_GET_WORKFLOW_EVALUATION_PROJECTS_LIMIT: u32 = 100;
const DEFAULT_GET_WORKFLOW_EVALUATION_PROJECTS_OFFSET: u32 = 0;

/// Handler for `GET /internal/workflow_evaluations/projects`
///
/// Returns a paginated list of workflow evaluation projects.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "workflow_evaluations.get_projects", skip_all)]
pub async fn get_workflow_evaluation_projects_handler(
    State(app_state): AppState,
    Query(params): Query<GetWorkflowEvaluationProjectsParams>,
) -> Result<Json<GetWorkflowEvaluationProjectsResponse>, Error> {
    let response = get_workflow_evaluation_projects(
        &app_state.clickhouse_connection_info,
        params
            .limit
            .unwrap_or(DEFAULT_GET_WORKFLOW_EVALUATION_PROJECTS_LIMIT),
        params
            .offset
            .unwrap_or(DEFAULT_GET_WORKFLOW_EVALUATION_PROJECTS_OFFSET),
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for getting workflow evaluation projects
pub async fn get_workflow_evaluation_projects(
    clickhouse: &impl WorkflowEvaluationQueries,
    limit: u32,
    offset: u32,
) -> Result<GetWorkflowEvaluationProjectsResponse, Error> {
    let projects_database = clickhouse
        .list_workflow_evaluation_projects(limit, offset)
        .await?;
    let projects = projects_database
        .into_iter()
        .map(|project| WorkflowEvaluationProject {
            name: project.name,
            count: project.count,
            last_updated: project.last_updated,
        })
        .collect();
    Ok(GetWorkflowEvaluationProjectsResponse { projects })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::workflow_evaluation_queries::MockWorkflowEvaluationQueries;
    use crate::db::workflow_evaluation_queries::WorkflowEvaluationProjectRow;
    use chrono::Utc;

    /// Helper to create a test workflow evaluation project row.
    fn create_test_project(name: &str, count: u32) -> WorkflowEvaluationProjectRow {
        WorkflowEvaluationProjectRow {
            name: name.to_string(),
            count,
            last_updated: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_projects_with_defaults() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_projects()
            .withf(|limit, offset| {
                assert_eq!(*limit, 100, "Should use default limit of 100");
                assert_eq!(*offset, 0, "Should use default offset of 0");
                true
            })
            .times(1)
            .returning(move |_, _| {
                let project = create_test_project("test_project", 5);
                Box::pin(async move { Ok(vec![project]) })
            });

        let result = get_workflow_evaluation_projects(&mock_clickhouse, 100, 0)
            .await
            .unwrap();

        assert_eq!(result.projects.len(), 1);
        assert_eq!(result.projects[0].name, "test_project");
        assert_eq!(result.projects[0].count, 5);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_projects_with_custom_pagination() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_projects()
            .withf(|limit, offset| {
                assert_eq!(*limit, 50, "Should use custom limit");
                assert_eq!(*offset, 100, "Should use custom offset");
                true
            })
            .times(1)
            .returning(move |_, _| {
                let project = create_test_project("test_project", 5);
                Box::pin(async move { Ok(vec![project]) })
            });

        let result = get_workflow_evaluation_projects(&mock_clickhouse, 50, 100)
            .await
            .unwrap();

        assert_eq!(result.projects.len(), 1);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_projects_empty_results() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_projects()
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_workflow_evaluation_projects(&mock_clickhouse, 100, 0)
            .await
            .unwrap();

        assert_eq!(result.projects.len(), 0);
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_projects_multiple_results() {
        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_projects()
            .times(1)
            .returning(move |_, _| {
                Box::pin(async move {
                    Ok(vec![
                        create_test_project("project1", 5),
                        create_test_project("project2", 10),
                        create_test_project("project3", 3),
                    ])
                })
            });

        let result = get_workflow_evaluation_projects(&mock_clickhouse, 100, 0)
            .await
            .unwrap();

        assert_eq!(result.projects.len(), 3);
        assert_eq!(result.projects[0].name, "project1");
        assert_eq!(result.projects[1].name, "project2");
        assert_eq!(result.projects[2].name, "project3");
    }

    #[tokio::test]
    async fn test_get_workflow_evaluation_projects_returns_all_fields() {
        let timestamp = Utc::now();

        let mut mock_clickhouse = MockWorkflowEvaluationQueries::new();
        mock_clickhouse
            .expect_list_workflow_evaluation_projects()
            .times(1)
            .returning(move |_, _| {
                let project = WorkflowEvaluationProjectRow {
                    name: "my_project".to_string(),
                    count: 42,
                    last_updated: timestamp,
                };
                Box::pin(async move { Ok(vec![project]) })
            });

        let result = get_workflow_evaluation_projects(&mock_clickhouse, 100, 0)
            .await
            .unwrap();

        assert_eq!(result.projects.len(), 1);
        let project = &result.projects[0];
        assert_eq!(project.name, "my_project");
        assert_eq!(project.count, 42);
        assert_eq!(project.last_updated, timestamp);
    }
}
