use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;

use super::types::{
    SearchEvaluationRunResult, SearchEvaluationRunsParams, SearchEvaluationRunsResponse,
};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/evaluations/runs/search`
///
/// Searches evaluation runs by ID or variant name.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.search_runs", skip_all)]
pub async fn search_evaluation_runs_handler(
    State(app_state): AppState,
    Query(params): Query<SearchEvaluationRunsParams>,
) -> Result<Json<SearchEvaluationRunsResponse>, Error> {
    let response = search_evaluation_runs_internal(
        &app_state.clickhouse_connection_info,
        params.evaluation_name,
        params.function_name,
        params.query,
        params.limit,
        params.offset,
    )
    .await?;
    Ok(Json(response))
}

/// Internal function for searching evaluation runs, testable with mock ClickHouse.
async fn search_evaluation_runs_internal(
    clickhouse: &impl EvaluationQueries,
    evaluation_name: String,
    function_name: String,
    query: String,
    limit: u32,
    offset: u32,
) -> Result<SearchEvaluationRunsResponse, Error> {
    let db_results = clickhouse
        .search_evaluation_runs(&evaluation_name, &function_name, &query, limit, offset)
        .await?;

    // Convert database results to API response format
    let results = db_results
        .into_iter()
        .map(|row| SearchEvaluationRunResult {
            evaluation_run_id: row.evaluation_run_id,
            variant_name: row.variant_name,
        })
        .collect();

    Ok(SearchEvaluationRunsResponse { results })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::{EvaluationRunSearchResult, MockEvaluationQueries};
    use uuid::Uuid;

    #[tokio::test]
    async fn test_search_evaluation_runs_basic() {
        let mut mock_clickhouse = MockEvaluationQueries::new();
        let run_id = Uuid::now_v7();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .withf(|eval_name, fn_name, query, limit, offset| {
                assert_eq!(eval_name, "test_eval");
                assert_eq!(fn_name, "test_function");
                assert_eq!(query, "test-query");
                assert_eq!(*limit, 100);
                assert_eq!(*offset, 0);
                true
            })
            .times(1)
            .returning(move |_, _, _, _, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationRunSearchResult {
                        evaluation_run_id: run_id,
                        variant_name: "variant_1".to_string(),
                    }])
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            "test_eval".to_string(),
            "test_function".to_string(),
            "test-query".to_string(),
            100,
            0,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].evaluation_run_id, run_id);
        assert_eq!(result.results[0].variant_name, "variant_1");
    }

    #[tokio::test]
    async fn test_search_evaluation_runs_empty() {
        let mut mock_clickhouse = MockEvaluationQueries::new();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .returning(|_, _, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            "test_eval".to_string(),
            "test_function".to_string(),
            "no-results".to_string(),
            100,
            0,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 0);
    }

    #[tokio::test]
    async fn test_search_evaluation_runs_multiple_results() {
        let mut mock_clickhouse = MockEvaluationQueries::new();
        let run_id1 = Uuid::now_v7();
        let run_id2 = Uuid::now_v7();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .returning(move |_, _, _, _, _| {
                Box::pin(async move {
                    Ok(vec![
                        EvaluationRunSearchResult {
                            evaluation_run_id: run_id1,
                            variant_name: "variant_1".to_string(),
                        },
                        EvaluationRunSearchResult {
                            evaluation_run_id: run_id2,
                            variant_name: "variant_2".to_string(),
                        },
                    ])
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            "test_eval".to_string(),
            "test_function".to_string(),
            "variant".to_string(),
            100,
            0,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].evaluation_run_id, run_id1);
        assert_eq!(result.results[0].variant_name, "variant_1");
        assert_eq!(result.results[1].evaluation_run_id, run_id2);
        assert_eq!(result.results[1].variant_name, "variant_2");
    }

    #[tokio::test]
    async fn test_search_evaluation_runs_with_pagination() {
        let mut mock_clickhouse = MockEvaluationQueries::new();
        let run_id = Uuid::now_v7();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .withf(|_, _, _, limit, offset| {
                assert_eq!(*limit, 50);
                assert_eq!(*offset, 100);
                true
            })
            .times(1)
            .returning(move |_, _, _, _, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationRunSearchResult {
                        evaluation_run_id: run_id,
                        variant_name: "variant_3".to_string(),
                    }])
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            "test_eval".to_string(),
            "test_function".to_string(),
            "query".to_string(),
            50,
            100,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_evaluation_runs_error_handling() {
        let mut mock_clickhouse = MockEvaluationQueries::new();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .returning(|_, _, _, _, _| {
                Box::pin(async move {
                    Err(Error::new(crate::error::ErrorDetails::InvalidRequest {
                        message: "Database error".to_string(),
                    }))
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            "test_eval".to_string(),
            "test_function".to_string(),
            "query".to_string(),
            100,
            0,
        )
        .await;

        assert!(result.is_err());
    }
}
