//! Handler for counting datapoints across evaluation runs.

use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;
use uuid::Uuid;

use super::types::{CountDatapointsParams, DatapointStatsResponse};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/evaluations/datapoint_count`
///
/// Returns the count of unique datapoints across the specified evaluation runs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.count_datapoints", skip_all)]
pub async fn count_datapoints_handler(
    State(app_state): AppState,
    Query(params): Query<CountDatapointsParams>,
) -> Result<Json<DatapointStatsResponse>, Error> {
    let response = count_datapoints_internal(
        &app_state.clickhouse_connection_info,
        params.function_name,
        params.evaluation_run_ids,
    )
    .await?;
    Ok(Json(response))
}

/// Internal function for counting datapoints, testable with mock ClickHouse.
#[cfg(test)]
pub async fn count_datapoints(
    clickhouse: &impl EvaluationQueries,
    function_name: String,
    evaluation_run_ids_str: String,
) -> Result<DatapointStatsResponse, Error> {
    count_datapoints_internal(clickhouse, function_name, evaluation_run_ids_str).await
}

async fn count_datapoints_internal(
    clickhouse: &impl EvaluationQueries,
    function_name: String,
    evaluation_run_ids_str: String,
) -> Result<DatapointStatsResponse, Error> {
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
        return Ok(DatapointStatsResponse { count: 0 });
    }

    let count = clickhouse
        .count_datapoints_for_evaluation(&function_name, &evaluation_run_ids)
        .await?;

    Ok(DatapointStatsResponse { count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::MockEvaluationQueries;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_count_datapoints_single_run() {
        let run_id = Uuid::now_v7();
        let run_id_str = run_id.to_string();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_count_datapoints_for_evaluation()
            .withf(move |fn_name, run_ids| {
                assert_eq!(fn_name, "test_function");
                assert_eq!(run_ids.len(), 1);
                assert_eq!(run_ids[0], run_id);
                true
            })
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(42) }));

        let result = count_datapoints(&mock_clickhouse, "test_function".to_string(), run_id_str)
            .await
            .unwrap();

        assert_eq!(result.count, 42);
    }

    #[tokio::test]
    async fn test_count_datapoints_multiple_runs() {
        let run_id1 = Uuid::now_v7();
        let run_id2 = Uuid::now_v7();
        let run_ids_str = format!("{run_id1},{run_id2}");

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_count_datapoints_for_evaluation()
            .withf(move |fn_name, run_ids| {
                assert_eq!(fn_name, "test_function");
                assert_eq!(run_ids.len(), 2);
                true
            })
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(100) }));

        let result = count_datapoints(&mock_clickhouse, "test_function".to_string(), run_ids_str)
            .await
            .unwrap();

        assert_eq!(result.count, 100);
    }

    #[tokio::test]
    async fn test_count_datapoints_empty_string() {
        let mock_clickhouse = MockEvaluationQueries::new();

        let result = count_datapoints(&mock_clickhouse, "test_function".to_string(), String::new())
            .await
            .unwrap();

        assert_eq!(result.count, 0);
    }

    #[tokio::test]
    async fn test_count_datapoints_invalid_uuid() {
        let mock_clickhouse = MockEvaluationQueries::new();

        let result = count_datapoints(
            &mock_clickhouse,
            "test_function".to_string(),
            "not-a-uuid".to_string(),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_count_datapoints_with_spaces() {
        let run_id1 = Uuid::now_v7();
        let run_id2 = Uuid::now_v7();
        let run_ids_str = format!("{run_id1} , {run_id2}");

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_count_datapoints_for_evaluation()
            .withf(move |_, run_ids| {
                assert_eq!(run_ids.len(), 2);
                true
            })
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(75) }));

        let result = count_datapoints(&mock_clickhouse, "test_function".to_string(), run_ids_str)
            .await
            .unwrap();

        assert_eq!(result.count, 75);
    }
}
