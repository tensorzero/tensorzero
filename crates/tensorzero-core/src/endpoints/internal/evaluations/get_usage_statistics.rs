//! Handler for getting evaluation usage statistics across evaluation runs.

use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;
use uuid::Uuid;

use super::types::{
    EvaluationUsageStatistics, GetEvaluationUsageStatisticsParams,
    GetEvaluationUsageStatisticsResponse,
};
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::Error;
use crate::function::FunctionConfigType;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/evaluations/usage_statistics`
///
/// Returns aggregated usage statistics (avg tokens, cost, processing time) for the specified
/// evaluation runs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.get_usage_statistics", skip_all)]
pub async fn get_evaluation_usage_statistics_handler(
    State(app_state): AppState,
    Query(params): Query<GetEvaluationUsageStatisticsParams>,
) -> Result<Json<GetEvaluationUsageStatisticsResponse>, Error> {
    let database = app_state.get_delegating_database();
    let response = get_evaluation_usage_statistics_internal(
        &database,
        params.function_name,
        params.function_type,
        params.evaluation_run_ids,
    )
    .await?;
    Ok(Json(response))
}

async fn get_evaluation_usage_statistics_internal(
    database: &impl EvaluationQueries,
    function_name: String,
    function_type: String,
    evaluation_run_ids_str: String,
) -> Result<GetEvaluationUsageStatisticsResponse, Error> {
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
        return Ok(GetEvaluationUsageStatisticsResponse {
            usage_statistics: Vec::new(),
        });
    }

    let rows = database
        .get_evaluation_usage_statistics(&function_name, function_type, &evaluation_run_ids)
        .await?;

    let usage_statistics = rows
        .into_iter()
        .map(|row| EvaluationUsageStatistics {
            evaluation_run_id: row.evaluation_run_id,
            inference_count: row.inference_count,
            avg_input_tokens: row.avg_input_tokens,
            avg_output_tokens: row.avg_output_tokens,
            avg_cost: row.avg_cost,
            avg_processing_time_ms: row.avg_processing_time_ms,
        })
        .collect();

    Ok(GetEvaluationUsageStatisticsResponse { usage_statistics })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::{EvaluationUsageStatisticsRow, MockEvaluationQueries};

    #[tokio::test]
    async fn test_get_evaluation_usage_statistics_single_run() {
        let run_id = Uuid::now_v7();
        let run_id_str = run_id.to_string();

        let mut mock = MockEvaluationQueries::new();
        mock.expect_get_evaluation_usage_statistics()
            .withf(move |fn_name, fn_type, run_ids| {
                assert_eq!(fn_name, "test_function");
                assert_eq!(*fn_type, FunctionConfigType::Chat);
                assert_eq!(run_ids.len(), 1);
                assert_eq!(run_ids[0], run_id);
                true
            })
            .times(1)
            .returning(move |_, _, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationUsageStatisticsRow {
                        evaluation_run_id: run_id,
                        inference_count: 10,
                        avg_input_tokens: Some(100.0),
                        avg_output_tokens: Some(50.0),
                        avg_cost: Some(0.001),
                        avg_processing_time_ms: Some(500.0),
                    }])
                })
            });

        let result = get_evaluation_usage_statistics_internal(
            &mock,
            "test_function".to_string(),
            "chat".to_string(),
            run_id_str,
        )
        .await
        .unwrap();

        assert_eq!(result.usage_statistics.len(), 1);
        assert_eq!(result.usage_statistics[0].inference_count, 10);
        assert!((result.usage_statistics[0].avg_input_tokens.unwrap() - 100.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_get_evaluation_usage_statistics_empty_run_ids() {
        let mock = MockEvaluationQueries::new();

        let result = get_evaluation_usage_statistics_internal(
            &mock,
            "test_function".to_string(),
            "chat".to_string(),
            String::new(),
        )
        .await
        .unwrap();

        assert_eq!(result.usage_statistics.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_usage_statistics_invalid_function_type() {
        let run_id = Uuid::now_v7();
        let mock = MockEvaluationQueries::new();

        let result = get_evaluation_usage_statistics_internal(
            &mock,
            "test_function".to_string(),
            "invalid".to_string(),
            run_id.to_string(),
        )
        .await;

        assert!(result.is_err());
    }
}
