//! Handler for listing evaluation runs.

use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;

use super::types::{ListEvaluationRunsParams, ListEvaluationRunsResponse};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::endpoints::internal::evaluations::types::EvaluationRunInfo;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/evaluations/runs`
///
/// Returns a paginated list of evaluation runs across all functions.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.list_runs", skip_all)]
pub async fn list_evaluation_runs_handler(
    State(app_state): AppState,
    Query(params): Query<ListEvaluationRunsParams>,
) -> Result<Json<ListEvaluationRunsResponse>, Error> {
    let list_evaluation_runs_response = list_evaluation_runs(
        &app_state.clickhouse_connection_info,
        params.limit,
        params.offset,
    )
    .await?;

    Ok(Json(list_evaluation_runs_response))
}

/// Core business logic for listing evaluation runs
pub async fn list_evaluation_runs(
    clickhouse: &impl EvaluationQueries,
    limit: u32,
    offset: u32,
) -> Result<ListEvaluationRunsResponse, Error> {
    let runs_database = clickhouse.list_evaluation_runs(limit, offset).await?;
    let runs = runs_database
        .into_iter()
        .map(|run| EvaluationRunInfo {
            evaluation_run_id: run.evaluation_run_id,
            evaluation_name: run.evaluation_name,
            dataset_name: run.dataset_name,
            function_name: run.function_name,
            variant_name: run.variant_name,
            last_inference_timestamp: run.last_inference_timestamp,
        })
        .collect();
    Ok(ListEvaluationRunsResponse { runs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::EvaluationRunInfoRow;
    use crate::db::evaluation_queries::MockEvaluationQueries;
    use chrono::Utc;
    use uuid::Uuid;

    /// Helper to create a test evaluation run info row.
    fn create_test_evaluation_run_info(id: Uuid) -> EvaluationRunInfoRow {
        EvaluationRunInfoRow {
            evaluation_run_id: id,
            evaluation_name: "test_evaluation".to_string(),
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            last_inference_timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_with_defaults() {
        let id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_list_evaluation_runs()
            .withf(|limit, offset| {
                // Verify default pagination values
                assert_eq!(*limit, 100, "Should use default limit of 100");
                assert_eq!(*offset, 0, "Should use default offset of 0");
                true
            })
            .times(1)
            .returning(move |_, _| {
                let info = create_test_evaluation_run_info(id);
                Box::pin(async move { Ok(vec![info]) })
            });

        let result = list_evaluation_runs(&mock_clickhouse, 100, 0)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 1);
        assert_eq!(result.runs[0].evaluation_run_id, id);
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_with_custom_pagination() {
        let id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_list_evaluation_runs()
            .withf(|limit, offset| {
                // Verify custom pagination values
                assert_eq!(*limit, 50, "Should use custom limit");
                assert_eq!(*offset, 100, "Should use custom offset");
                true
            })
            .times(1)
            .returning(move |_, _| {
                let info = create_test_evaluation_run_info(id);
                Box::pin(async move { Ok(vec![info]) })
            });

        let result = list_evaluation_runs(&mock_clickhouse, 50, 100)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 1);
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_empty_results() {
        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_list_evaluation_runs()
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(vec![]) }));

        let result = list_evaluation_runs(&mock_clickhouse, 100, 0)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 0);
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_multiple_results() {
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_list_evaluation_runs()
            .times(1)
            .returning(move |_, _| {
                Box::pin(async move {
                    Ok(vec![
                        create_test_evaluation_run_info(id1),
                        create_test_evaluation_run_info(id2),
                        create_test_evaluation_run_info(id3),
                    ])
                })
            });

        let result = list_evaluation_runs(&mock_clickhouse, 100, 0)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 3);
        assert_eq!(result.runs[0].evaluation_run_id, id1);
        assert_eq!(result.runs[1].evaluation_run_id, id2);
        assert_eq!(result.runs[2].evaluation_run_id, id3);
    }

    #[tokio::test]
    async fn test_list_evaluation_runs_returns_all_fields() {
        let id = Uuid::now_v7();
        let timestamp = Utc::now();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_list_evaluation_runs()
            .times(1)
            .returning(move |_, _| {
                let run_info = EvaluationRunInfoRow {
                    evaluation_run_id: id,
                    evaluation_name: "my_evaluation".to_string(),
                    dataset_name: "my_dataset".to_string(),
                    function_name: "my_function".to_string(),
                    variant_name: "my_variant".to_string(),
                    last_inference_timestamp: timestamp,
                };
                Box::pin(async move { Ok(vec![run_info]) })
            });

        let result = list_evaluation_runs(&mock_clickhouse, 100, 0)
            .await
            .unwrap();

        assert_eq!(result.runs.len(), 1);
        let run = &result.runs[0];
        assert_eq!(run.evaluation_run_id, id);
        assert_eq!(run.evaluation_name, "my_evaluation");
        assert_eq!(run.dataset_name, "my_dataset");
        assert_eq!(run.function_name, "my_function");
        assert_eq!(run.variant_name, "my_variant");
        assert_eq!(run.last_inference_timestamp, timestamp);
    }
}
