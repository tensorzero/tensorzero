use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;

use super::types::{
    SearchEvaluationRunResult, SearchEvaluationRunsParams, SearchEvaluationRunsResponse,
};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::AppState;

/// Handler for `GET /internal/evaluations/runs/search`
///
/// Searches evaluation runs by ID or variant name.
#[instrument(name = "evaluations.search_runs", skip_all)]
pub async fn search_evaluation_runs_handler(
    State(app_state): AppState,
    Query(params): Query<SearchEvaluationRunsParams>,
) -> Result<Json<SearchEvaluationRunsResponse>, Error> {
    let database = app_state.get_delegating_database();
    let response = search_evaluation_runs_internal(
        &database,
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
    evaluation_name: Option<String>,
    function_name: Option<String>,
    query: String,
    limit: u32,
    offset: u32,
) -> Result<SearchEvaluationRunsResponse, Error> {
    if evaluation_name.is_none() && function_name.is_none() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "At least one of `evaluation_name` or `function_name` must be provided"
                .to_string(),
        }));
    }

    let db_results = clickhouse
        .search_evaluation_runs(
            evaluation_name.as_deref(),
            function_name.as_deref(),
            &query,
            limit,
            offset,
        )
        .await?;

    // Convert database results to API response format
    let results = db_results
        .into_iter()
        .map(|row| SearchEvaluationRunResult {
            evaluation_run_id: row.evaluation_run_id,
            evaluation_name: row.evaluation_name,
            dataset_name: row.dataset_name,
            variant_name: row.variant_name,
        })
        .collect();

    Ok(SearchEvaluationRunsResponse { results })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::{EvaluationRunSearchResult, MockEvaluationQueries};
    use googletest::prelude::*;
    use uuid::Uuid;

    #[gtest]
    #[tokio::test]
    async fn test_search_evaluation_runs_basic() {
        let mut mock_clickhouse = MockEvaluationQueries::new();
        let run_id = Uuid::now_v7();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .withf(|eval_name, fn_name, query, limit, offset| {
                assert_eq!(eval_name, &Some("test_eval"));
                assert_eq!(
                    fn_name,
                    &Some("test_function"),
                    "function_name should match"
                );
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
                        evaluation_name: "test_eval".to_string(),
                        dataset_name: "test_dataset".to_string(),
                        variant_name: "variant_1".to_string(),
                    }])
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            Some("test_eval".to_string()),
            Some("test_function".to_string()),
            "test-query".to_string(),
            100,
            0,
        )
        .await
        .expect("search should succeed");

        expect_that!(
            &result.results,
            elements_are![matches_pattern!(SearchEvaluationRunResult {
                evaluation_run_id: eq(&run_id),
                evaluation_name: eq("test_eval"),
                dataset_name: eq("test_dataset"),
                variant_name: eq("variant_1"),
            })]
        );
    }

    #[gtest]
    #[tokio::test]
    async fn test_search_evaluation_runs_empty() {
        let mut mock_clickhouse = MockEvaluationQueries::new();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .returning(|_, _, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            Some("test_eval".to_string()),
            Some("test_function".to_string()),
            "no-results".to_string(),
            100,
            0,
        )
        .await
        .expect("search should succeed");

        expect_that!(result.results, is_empty());
    }

    #[gtest]
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
                            evaluation_name: "test_eval".to_string(),
                            dataset_name: "dataset_1".to_string(),
                            variant_name: "variant_1".to_string(),
                        },
                        EvaluationRunSearchResult {
                            evaluation_run_id: run_id2,
                            evaluation_name: "test_eval".to_string(),
                            dataset_name: "dataset_2".to_string(),
                            variant_name: "variant_2".to_string(),
                        },
                    ])
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            Some("test_eval".to_string()),
            Some("test_function".to_string()),
            "variant".to_string(),
            100,
            0,
        )
        .await
        .expect("search should succeed");

        expect_that!(
            &result.results,
            elements_are![
                matches_pattern!(SearchEvaluationRunResult {
                    evaluation_run_id: eq(&run_id1),
                    evaluation_name: eq("test_eval"),
                    dataset_name: eq("dataset_1"),
                    variant_name: eq("variant_1"),
                }),
                matches_pattern!(SearchEvaluationRunResult {
                    evaluation_run_id: eq(&run_id2),
                    evaluation_name: eq("test_eval"),
                    dataset_name: eq("dataset_2"),
                    variant_name: eq("variant_2"),
                })
            ]
        );
    }

    #[gtest]
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
                        evaluation_name: "test_eval".to_string(),
                        dataset_name: "test_dataset".to_string(),
                        variant_name: "variant_3".to_string(),
                    }])
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            Some("test_eval".to_string()),
            Some("test_function".to_string()),
            "query".to_string(),
            50,
            100,
        )
        .await
        .expect("search should succeed");

        expect_that!(
            &result.results,
            elements_are![matches_pattern!(SearchEvaluationRunResult {
                evaluation_run_id: eq(&run_id),
                evaluation_name: eq("test_eval"),
                dataset_name: eq("test_dataset"),
                variant_name: eq("variant_3"),
            })]
        );
    }

    #[gtest]
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
            Some("test_eval".to_string()),
            Some("test_function".to_string()),
            "query".to_string(),
            100,
            0,
        )
        .await;

        expect_that!(result, err(anything()));
    }

    #[gtest]
    #[tokio::test]
    async fn test_search_evaluation_runs_no_function_name() {
        let mut mock_clickhouse = MockEvaluationQueries::new();
        let run_id = Uuid::now_v7();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .withf(|eval_name, fn_name, query, limit, offset| {
                assert_eq!(eval_name, &Some("test_eval"));
                assert_eq!(fn_name, &None, "function_name should be None");
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
                        evaluation_name: "test_eval".to_string(),
                        dataset_name: "test_dataset".to_string(),
                        variant_name: "variant_1".to_string(),
                    }])
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            Some("test_eval".to_string()),
            None,
            "test-query".to_string(),
            100,
            0,
        )
        .await
        .expect("search should succeed");

        expect_that!(
            &result.results,
            elements_are![matches_pattern!(SearchEvaluationRunResult {
                evaluation_run_id: eq(&run_id),
                evaluation_name: eq("test_eval"),
                dataset_name: eq("test_dataset"),
                variant_name: eq("variant_1"),
            })]
        );
    }

    #[gtest]
    #[tokio::test]
    async fn test_search_evaluation_runs_function_only() {
        let mut mock_clickhouse = MockEvaluationQueries::new();
        let run_id = Uuid::now_v7();

        mock_clickhouse
            .expect_search_evaluation_runs()
            .withf(|eval_name, fn_name, query, limit, offset| {
                assert_that!(eval_name, eq(&None));
                assert_that!(fn_name, eq(&Some("test_function")));
                assert_that!(query, eq("dataset"));
                assert_that!(*limit, eq(100));
                assert_that!(*offset, eq(0));
                true
            })
            .times(1)
            .returning(move |_, _, _, _, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationRunSearchResult {
                        evaluation_run_id: run_id,
                        evaluation_name: "generated_eval".to_string(),
                        dataset_name: "test_dataset".to_string(),
                        variant_name: "variant_1".to_string(),
                    }])
                })
            });

        let result = search_evaluation_runs_internal(
            &mock_clickhouse,
            None,
            Some("test_function".to_string()),
            "dataset".to_string(),
            100,
            0,
        )
        .await
        .expect("search should succeed");

        expect_that!(
            &result.results,
            elements_are![matches_pattern!(SearchEvaluationRunResult {
                evaluation_run_id: eq(&run_id),
                evaluation_name: eq("generated_eval"),
                dataset_name: eq("test_dataset"),
                variant_name: eq("variant_1"),
            })]
        );
    }

    #[gtest]
    #[tokio::test]
    async fn test_search_evaluation_runs_requires_scope() {
        let mock_clickhouse = MockEvaluationQueries::new();

        let result =
            search_evaluation_runs_internal(&mock_clickhouse, None, None, String::new(), 100, 0)
                .await;

        expect_that!(
            result,
            err(anything()),
            "Expected missing scope to return an error"
        );
    }
}
