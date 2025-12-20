//! Handler for getting evaluation statistics across evaluation runs.

use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;
use uuid::Uuid;

use super::types::{
    EvaluationStatistics, GetEvaluationStatisticsParams, GetEvaluationStatisticsResponse,
};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::Error;
use crate::function::FunctionConfigType;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/evaluations/statistics`
///
/// Returns aggregated statistics (mean, confidence intervals) for the specified
/// evaluation runs and metrics.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.get_statistics", skip_all)]
pub async fn get_evaluation_statistics_handler(
    State(app_state): AppState,
    Query(params): Query<GetEvaluationStatisticsParams>,
) -> Result<Json<GetEvaluationStatisticsResponse>, Error> {
    let response = get_evaluation_statistics_internal(
        &app_state.clickhouse_connection_info,
        params.function_name,
        params.function_type,
        params.metric_names,
        params.evaluation_run_ids,
    )
    .await?;
    Ok(Json(response))
}

/// Internal function for getting evaluation statistics, testable with mock ClickHouse.
#[cfg(test)]
pub async fn get_evaluation_statistics(
    clickhouse: &impl EvaluationQueries,
    function_name: String,
    function_type: String,
    metric_names: String,
    evaluation_run_ids_str: String,
) -> Result<GetEvaluationStatisticsResponse, Error> {
    get_evaluation_statistics_internal(
        clickhouse,
        function_name,
        function_type,
        metric_names,
        evaluation_run_ids_str,
    )
    .await
}

async fn get_evaluation_statistics_internal(
    clickhouse: &impl EvaluationQueries,
    function_name: String,
    function_type: String,
    metric_names_str: String,
    evaluation_run_ids_str: String,
) -> Result<GetEvaluationStatisticsResponse, Error> {
    // Parse function type
    let function_type = match function_type.as_str() {
        "chat" => FunctionConfigType::Chat,
        "json" => FunctionConfigType::Json,
        _ => {
            return Err(Error::new(crate::error::ErrorDetails::InvalidRequest {
                message: format!(
                    "Invalid function_type: {function_type}. Must be 'chat' or 'json'"
                ),
            }));
        }
    };

    // Parse comma-separated metric names
    let metric_names: Vec<String> = metric_names_str
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();

    // Parse comma-separated UUIDs
    let evaluation_run_ids: Vec<Uuid> = evaluation_run_ids_str
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<Uuid>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            Error::new(crate::error::ErrorDetails::InvalidRequest {
                message: format!("Invalid UUID in evaluation_run_ids: {e}"),
            })
        })?;

    if evaluation_run_ids.is_empty() {
        return Ok(GetEvaluationStatisticsResponse {
            statistics: Vec::new(),
        });
    }

    if metric_names.is_empty() {
        return Ok(GetEvaluationStatisticsResponse {
            statistics: Vec::new(),
        });
    }

    let rows = clickhouse
        .get_evaluation_statistics(
            &function_name,
            function_type,
            &metric_names,
            &evaluation_run_ids,
        )
        .await?;

    let statistics = rows
        .into_iter()
        .map(|row| EvaluationStatistics {
            evaluation_run_id: row.evaluation_run_id,
            metric_name: row.metric_name,
            datapoint_count: row.datapoint_count,
            mean_metric: row.mean_metric,
            ci_lower: row.ci_lower,
            ci_upper: row.ci_upper,
        })
        .collect();

    Ok(GetEvaluationStatisticsResponse { statistics })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::{EvaluationStatisticsRow, MockEvaluationQueries};
    use uuid::Uuid;

    #[tokio::test]
    async fn test_get_evaluation_statistics_single_run() {
        let run_id = Uuid::now_v7();
        let run_id_str = run_id.to_string();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_statistics()
            .withf(move |fn_name, fn_type, metric_names, run_ids| {
                assert_eq!(fn_name, "test_function");
                assert_eq!(*fn_type, FunctionConfigType::Chat);
                assert_eq!(metric_names.len(), 1);
                assert_eq!(metric_names[0], "metric1");
                assert_eq!(run_ids.len(), 1);
                assert_eq!(run_ids[0], run_id);
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationStatisticsRow {
                        evaluation_run_id: run_id,
                        metric_name: "metric1".to_string(),
                        datapoint_count: 10,
                        mean_metric: 0.75,
                        ci_lower: Some(0.65),
                        ci_upper: Some(0.85),
                    }])
                })
            });

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "chat".to_string(),
            "metric1".to_string(),
            run_id_str,
        )
        .await
        .unwrap();

        assert_eq!(result.statistics.len(), 1);
        assert_eq!(result.statistics[0].datapoint_count, 10);
        assert!((result.statistics[0].mean_metric - 0.75).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_json_function() {
        let run_id = Uuid::now_v7();
        let run_id_str = run_id.to_string();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_statistics()
            .withf(|_fn_name, fn_type, _metric_names, _run_ids| {
                assert_eq!(*fn_type, FunctionConfigType::Json);
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationStatisticsRow {
                        evaluation_run_id: run_id,
                        metric_name: "accuracy".to_string(),
                        datapoint_count: 5,
                        mean_metric: 0.9,
                        ci_lower: Some(0.8),
                        ci_upper: Some(1.0),
                    }])
                })
            });

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "json".to_string(),
            "accuracy".to_string(),
            run_id_str,
        )
        .await
        .unwrap();

        assert_eq!(result.statistics.len(), 1);
        assert_eq!(result.statistics[0].metric_name, "accuracy");
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_multiple_runs() {
        let run_id1 = Uuid::now_v7();
        let run_id2 = Uuid::now_v7();
        let run_ids_str = format!("{run_id1},{run_id2}");

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_statistics()
            .withf(move |_fn_name, _fn_type, _metric_names, run_ids| {
                assert_eq!(run_ids.len(), 2);
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                Box::pin(async move {
                    Ok(vec![
                        EvaluationStatisticsRow {
                            evaluation_run_id: run_id1,
                            metric_name: "metric1".to_string(),
                            datapoint_count: 10,
                            mean_metric: 0.75,
                            ci_lower: Some(0.65),
                            ci_upper: Some(0.85),
                        },
                        EvaluationStatisticsRow {
                            evaluation_run_id: run_id2,
                            metric_name: "metric1".to_string(),
                            datapoint_count: 15,
                            mean_metric: 0.80,
                            ci_lower: Some(0.70),
                            ci_upper: Some(0.90),
                        },
                    ])
                })
            });

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "chat".to_string(),
            "metric1".to_string(),
            run_ids_str,
        )
        .await
        .unwrap();

        assert_eq!(result.statistics.len(), 2);
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_multiple_metrics() {
        let run_id = Uuid::now_v7();
        let run_id_str = run_id.to_string();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_statistics()
            .withf(move |_fn_name, _fn_type, metric_names, _run_ids| {
                assert_eq!(metric_names.len(), 2);
                assert_eq!(metric_names[0], "metric1");
                assert_eq!(metric_names[1], "metric2");
                true
            })
            .times(1)
            .returning(move |_, _, _, _| {
                Box::pin(async move {
                    Ok(vec![
                        EvaluationStatisticsRow {
                            evaluation_run_id: run_id,
                            metric_name: "metric1".to_string(),
                            datapoint_count: 10,
                            mean_metric: 0.75,
                            ci_lower: Some(0.65),
                            ci_upper: Some(0.85),
                        },
                        EvaluationStatisticsRow {
                            evaluation_run_id: run_id,
                            metric_name: "metric2".to_string(),
                            datapoint_count: 10,
                            mean_metric: 0.80,
                            ci_lower: Some(0.70),
                            ci_upper: Some(0.90),
                        },
                    ])
                })
            });

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "chat".to_string(),
            "metric1,metric2".to_string(),
            run_id_str,
        )
        .await
        .unwrap();

        assert_eq!(result.statistics.len(), 2);
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_empty_run_ids() {
        let mock_clickhouse = MockEvaluationQueries::new();

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "chat".to_string(),
            "metric1".to_string(),
            String::new(),
        )
        .await
        .unwrap();

        assert_eq!(result.statistics.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_empty_metric_names() {
        let run_id = Uuid::now_v7();
        let mock_clickhouse = MockEvaluationQueries::new();

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "chat".to_string(),
            String::new(),
            run_id.to_string(),
        )
        .await
        .unwrap();

        assert_eq!(result.statistics.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_invalid_uuid() {
        let mock_clickhouse = MockEvaluationQueries::new();

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "chat".to_string(),
            "metric1".to_string(),
            "not-a-uuid".to_string(),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_invalid_function_type() {
        let run_id = Uuid::now_v7();
        let mock_clickhouse = MockEvaluationQueries::new();

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "invalid".to_string(),
            "metric1".to_string(),
            run_id.to_string(),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_evaluation_statistics_with_spaces() {
        let run_id1 = Uuid::now_v7();
        let run_id2 = Uuid::now_v7();
        let run_ids_str = format!("{run_id1} , {run_id2}");

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_statistics()
            .withf(move |_fn_name, _fn_type, metric_names, run_ids| {
                assert_eq!(run_ids.len(), 2);
                assert_eq!(metric_names.len(), 2);
                assert_eq!(metric_names[0], "metric1");
                assert_eq!(metric_names[1], "metric2");
                true
            })
            .times(1)
            .returning(|_, _, _, _| Box::pin(async move { Ok(Vec::new()) }));

        let result = get_evaluation_statistics(
            &mock_clickhouse,
            "test_function".to_string(),
            "chat".to_string(),
            "metric1 , metric2".to_string(),
            run_ids_str,
        )
        .await
        .unwrap();

        assert_eq!(result.statistics.len(), 0);
    }
}
