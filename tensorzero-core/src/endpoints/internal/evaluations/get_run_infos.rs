//! Handlers for getting evaluation run infos.

use axum::Json;
use axum::extract::{Path, Query, State};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::{Error, ErrorDetails};
use crate::function::{FunctionConfigType, get_function};
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for getting evaluation run infos by IDs.
#[derive(Debug, Deserialize)]
pub struct GetEvaluationRunInfosParams {
    /// Comma-separated list of evaluation run UUIDs
    pub evaluation_run_ids: String,
    pub function_name: String,
}

/// Query parameters for getting evaluation run infos for a datapoint.
#[derive(Debug, Deserialize)]
pub struct GetEvaluationRunInfosForDatapointParams {
    pub function_name: String,
}

/// Information about a single evaluation run (returned by get_evaluation_run_infos).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EvaluationRunInfoById {
    pub evaluation_run_id: Uuid,
    pub variant_name: String,
    pub most_recent_inference_date: String,
}

/// Response containing evaluation run infos.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetEvaluationRunInfosResponse {
    pub run_infos: Vec<EvaluationRunInfoById>,
}

/// Handler for `GET /internal/evaluations/run_infos`
///
/// Returns information about specific evaluation runs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.get_run_infos", skip_all)]
pub async fn get_evaluation_run_infos_handler(
    State(app_state): AppState,
    Query(params): Query<GetEvaluationRunInfosParams>,
) -> Result<Json<GetEvaluationRunInfosResponse>, Error> {
    // Parse comma-separated UUIDs
    let evaluation_run_ids: Vec<Uuid> = params
        .evaluation_run_ids
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            Uuid::parse_str(s.trim()).map_err(|_| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Invalid UUID: {s}"),
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let response = get_evaluation_run_infos(
        &app_state.clickhouse_connection_info,
        &evaluation_run_ids,
        &params.function_name,
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for getting evaluation run infos.
pub async fn get_evaluation_run_infos(
    clickhouse: &impl EvaluationQueries,
    evaluation_run_ids: &[Uuid],
    function_name: &str,
) -> Result<GetEvaluationRunInfosResponse, Error> {
    let run_infos_database = clickhouse
        .get_evaluation_run_infos(evaluation_run_ids, function_name)
        .await?;

    let run_infos = run_infos_database
        .into_iter()
        .map(|row| EvaluationRunInfoById {
            evaluation_run_id: row.evaluation_run_id,
            variant_name: row.variant_name,
            most_recent_inference_date: row.most_recent_inference_date.to_rfc3339(),
        })
        .collect();

    Ok(GetEvaluationRunInfosResponse { run_infos })
}

/// Handler for `GET /internal/evaluations/datapoints/{datapoint_id}/run_infos`
///
/// Returns information about evaluation runs associated with a specific datapoint.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.get_run_infos_for_datapoint", skip_all)]
pub async fn get_evaluation_run_infos_for_datapoint_handler(
    State(app_state): AppState,
    Path(datapoint_id): Path<Uuid>,
    Query(params): Query<GetEvaluationRunInfosForDatapointParams>,
) -> Result<Json<GetEvaluationRunInfosResponse>, Error> {
    // Look up the function config to determine the function type
    let function_config = get_function(&app_state.config.functions, &params.function_name)?;
    let function_type = function_config.config_type();

    let response = get_evaluation_run_infos_for_datapoint(
        &app_state.clickhouse_connection_info,
        &datapoint_id,
        &params.function_name,
        function_type,
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for getting evaluation run infos for a datapoint.
pub async fn get_evaluation_run_infos_for_datapoint(
    clickhouse: &impl EvaluationQueries,
    datapoint_id: &Uuid,
    function_name: &str,
    function_type: FunctionConfigType,
) -> Result<GetEvaluationRunInfosResponse, Error> {
    let run_infos_database = clickhouse
        .get_evaluation_run_infos_for_datapoint(datapoint_id, function_name, function_type)
        .await?;

    let run_infos = run_infos_database
        .into_iter()
        .map(|row| EvaluationRunInfoById {
            evaluation_run_id: row.evaluation_run_id,
            variant_name: row.variant_name,
            most_recent_inference_date: row.most_recent_inference_date.to_rfc3339(),
        })
        .collect();

    Ok(GetEvaluationRunInfosResponse { run_infos })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::EvaluationRunInfoByIdRow;
    use crate::db::evaluation_queries::MockEvaluationQueries;
    use chrono::Utc;

    #[tokio::test]
    async fn test_get_evaluation_run_infos_returns_results() {
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let timestamp = Utc::now();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_run_infos()
            .withf(move |run_ids, fn_name| run_ids.len() == 2 && fn_name == "test_function")
            .times(1)
            .returning(move |_, _| {
                Box::pin(async move {
                    Ok(vec![
                        EvaluationRunInfoByIdRow {
                            evaluation_run_id: id1,
                            variant_name: "variant1".to_string(),
                            most_recent_inference_date: timestamp,
                        },
                        EvaluationRunInfoByIdRow {
                            evaluation_run_id: id2,
                            variant_name: "variant2".to_string(),
                            most_recent_inference_date: timestamp,
                        },
                    ])
                })
            });

        let result = get_evaluation_run_infos(&mock_clickhouse, &[id1, id2], "test_function")
            .await
            .unwrap();

        assert_eq!(result.run_infos.len(), 2);
        assert_eq!(result.run_infos[0].evaluation_run_id, id1);
        assert_eq!(result.run_infos[0].variant_name, "variant1");
        assert_eq!(result.run_infos[1].evaluation_run_id, id2);
        assert_eq!(result.run_infos[1].variant_name, "variant2");
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_empty_results() {
        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_run_infos()
            .times(1)
            .returning(|_, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_evaluation_run_infos(&mock_clickhouse, &[Uuid::now_v7()], "nonexistent")
            .await
            .unwrap();

        assert_eq!(result.run_infos.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_single_run() {
        let id = Uuid::now_v7();
        let timestamp = Utc::now();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_run_infos()
            .withf(move |run_ids, fn_name| {
                run_ids.len() == 1 && run_ids[0] == id && fn_name == "my_function"
            })
            .times(1)
            .returning(move |_, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationRunInfoByIdRow {
                        evaluation_run_id: id,
                        variant_name: "my_variant".to_string(),
                        most_recent_inference_date: timestamp,
                    }])
                })
            });

        let result = get_evaluation_run_infos(&mock_clickhouse, &[id], "my_function")
            .await
            .unwrap();

        assert_eq!(result.run_infos.len(), 1);
        assert_eq!(result.run_infos[0].evaluation_run_id, id);
        assert_eq!(result.run_infos[0].variant_name, "my_variant");
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_for_datapoint_returns_results() {
        let datapoint_id = Uuid::now_v7();
        let eval_run_id1 = Uuid::now_v7();
        let eval_run_id2 = Uuid::now_v7();
        let timestamp = Utc::now();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_run_infos_for_datapoint()
            .withf(move |dp_id, fn_name, fn_type| {
                *dp_id == datapoint_id
                    && fn_name == "test_function"
                    && *fn_type == FunctionConfigType::Chat
            })
            .times(1)
            .returning(move |_, _, _| {
                Box::pin(async move {
                    Ok(vec![
                        EvaluationRunInfoByIdRow {
                            evaluation_run_id: eval_run_id1,
                            variant_name: "variant1".to_string(),
                            most_recent_inference_date: timestamp,
                        },
                        EvaluationRunInfoByIdRow {
                            evaluation_run_id: eval_run_id2,
                            variant_name: "variant2".to_string(),
                            most_recent_inference_date: timestamp,
                        },
                    ])
                })
            });

        let result = get_evaluation_run_infos_for_datapoint(
            &mock_clickhouse,
            &datapoint_id,
            "test_function",
            FunctionConfigType::Chat,
        )
        .await
        .unwrap();

        assert_eq!(result.run_infos.len(), 2);
        assert_eq!(result.run_infos[0].evaluation_run_id, eval_run_id1);
        assert_eq!(result.run_infos[0].variant_name, "variant1");
        assert_eq!(result.run_infos[1].evaluation_run_id, eval_run_id2);
        assert_eq!(result.run_infos[1].variant_name, "variant2");
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_for_datapoint_empty_results() {
        let datapoint_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_run_infos_for_datapoint()
            .times(1)
            .returning(|_, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_evaluation_run_infos_for_datapoint(
            &mock_clickhouse,
            &datapoint_id,
            "nonexistent",
            FunctionConfigType::Json,
        )
        .await
        .unwrap();

        assert_eq!(result.run_infos.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_run_infos_for_datapoint_single_run() {
        let datapoint_id = Uuid::now_v7();
        let eval_run_id = Uuid::now_v7();
        let timestamp = Utc::now();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_run_infos_for_datapoint()
            .withf(move |dp_id, fn_name, fn_type| {
                *dp_id == datapoint_id
                    && fn_name == "my_function"
                    && *fn_type == FunctionConfigType::Chat
            })
            .times(1)
            .returning(move |_, _, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationRunInfoByIdRow {
                        evaluation_run_id: eval_run_id,
                        variant_name: "my_variant".to_string(),
                        most_recent_inference_date: timestamp,
                    }])
                })
            });

        let result = get_evaluation_run_infos_for_datapoint(
            &mock_clickhouse,
            &datapoint_id,
            "my_function",
            FunctionConfigType::Chat,
        )
        .await
        .unwrap();

        assert_eq!(result.run_infos.len(), 1);
        assert_eq!(result.run_infos[0].evaluation_run_id, eval_run_id);
        assert_eq!(result.run_infos[0].variant_name, "my_variant");
    }
}
