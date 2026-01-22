//! Handler for getting paginated evaluation results.

use axum::Json;
use axum::extract::{Query, State};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::config::Config;
use crate::db::evaluation_queries::{EvaluationQueries, EvaluationResultRow};
use crate::error::{Error, ErrorDetails};
use crate::evaluations::EvaluationConfig;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for getting evaluation results.
#[derive(Debug, Deserialize)]
pub struct GetEvaluationResultsParams {
    /// The name of the evaluation (e.g., "haiku", "entity_extraction")
    pub evaluation_name: String,
    /// Comma-separated list of evaluation run UUIDs
    pub evaluation_run_ids: String,
    /// Optional datapoint ID to filter results to a specific datapoint
    pub datapoint_id: Option<Uuid>,
    /// Maximum number of datapoints to return (default: 100)
    pub limit: Option<u32>,
    /// Number of datapoints to skip (default: 0)
    pub offset: Option<u32>,
}

/// Response containing paginated evaluation results.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetEvaluationResultsResponse {
    pub results: Vec<EvaluationResultRow>,
}

/// Handler for `GET /internal/evaluations/results`
///
/// Returns paginated evaluation results across one or more evaluation runs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.get_results", skip_all)]
pub async fn get_evaluation_results_handler(
    State(app_state): AppState,
    Query(params): Query<GetEvaluationResultsParams>,
) -> Result<Json<GetEvaluationResultsResponse>, Error> {
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

    let limit = params.limit.unwrap_or(100);
    let offset = params.offset.unwrap_or(0);

    let response = get_evaluation_results(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        &params.evaluation_name,
        &evaluation_run_ids,
        params.datapoint_id.as_ref(),
        limit,
        offset,
    )
    .await?;

    Ok(Json(response))
}

/// Core business logic for getting paginated evaluation results.
pub async fn get_evaluation_results(
    config: &Config,
    clickhouse: &impl EvaluationQueries,
    evaluation_name: &str,
    evaluation_run_ids: &[Uuid],
    datapoint_id: Option<&Uuid>,
    limit: u32,
    offset: u32,
) -> Result<GetEvaluationResultsResponse, Error> {
    // Look up the evaluation config
    let evaluation_config = config.evaluations.get(evaluation_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Evaluation '{evaluation_name}' not found in config"),
        })
    })?;

    // Extract function_name and evaluators from the config
    let (function_name, evaluators) = match evaluation_config.as_ref() {
        EvaluationConfig::Inference(inference_config) => (
            &inference_config.function_name,
            &inference_config.evaluators,
        ),
    };

    // Get the function config to determine the function type
    let function_config = config.functions.get(function_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Function '{function_name}' not found in config"),
        })
    })?;

    // Get function type
    let function_type = function_config.config_type();

    // Build metric names from evaluator names
    // Format: tensorzero::evaluation_name::{evaluation_name}::evaluator_name::{evaluator_name}
    let metric_names: Vec<String> = evaluators
        .keys()
        .map(|evaluator_name| {
            format!(
                "tensorzero::evaluation_name::{evaluation_name}::evaluator_name::{evaluator_name}"
            )
        })
        .collect();

    // Query the database
    let results = clickhouse
        .get_evaluation_results(
            function_name,
            evaluation_run_ids,
            function_type,
            &metric_names,
            datapoint_id,
            limit,
            offset,
        )
        .await?;

    Ok(GetEvaluationResultsResponse { results })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::evaluation_queries::{ChatEvaluationResultRow, MockEvaluationQueries};
    use crate::evaluations::{EvaluatorConfig, ExactMatchConfig, InferenceEvaluationConfig};
    use crate::function::{
        FunctionConfig, FunctionConfigChat, FunctionConfigJson, FunctionConfigType,
    };
    use crate::inference::types::Input;
    use crate::jsonschema_util::JSONSchema;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_test_config_with_chat_function() -> Config {
        let mut evaluations = HashMap::new();
        let mut evaluators = HashMap::new();
        evaluators.insert(
            "exact_match".to_string(),
            EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
        );
        evaluations.insert(
            "test_eval".to_string(),
            Arc::new(EvaluationConfig::Inference(InferenceEvaluationConfig {
                function_name: "test_function".to_string(),
                evaluators,
                description: None,
            })),
        );

        let mut functions = HashMap::new();
        functions.insert(
            "test_function".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat::default())),
        );

        Config {
            functions,
            evaluations,
            ..Default::default()
        }
    }

    fn create_test_config_with_json_function() -> Config {
        let mut evaluations = HashMap::new();
        let mut evaluators = HashMap::new();
        evaluators.insert(
            "exact_match".to_string(),
            EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
        );
        evaluations.insert(
            "test_eval".to_string(),
            Arc::new(EvaluationConfig::Inference(InferenceEvaluationConfig {
                function_name: "test_function".to_string(),
                evaluators,
                description: None,
            })),
        );

        let mut functions = HashMap::new();
        functions.insert(
            "test_function".to_string(),
            Arc::new(FunctionConfig::Json(FunctionConfigJson {
                output_schema: JSONSchema::from_value(serde_json::json!({"type": "object"}))
                    .unwrap(),
                ..Default::default()
            })),
        );

        Config {
            functions,
            evaluations,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_get_evaluation_results_chat_function() {
        let config = create_test_config_with_chat_function();
        let evaluation_run_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_results()
            .withf(
                move |fn_name, run_ids, fn_type, metrics, dp_id, limit, offset| {
                    fn_name == "test_function"
                        && run_ids.len() == 1
                        && run_ids[0] == evaluation_run_id
                        && *fn_type == FunctionConfigType::Chat
                        && metrics.len() == 1
                        && metrics[0]
                            == "tensorzero::evaluation_name::test_eval::evaluator_name::exact_match"
                        && dp_id.is_none()
                        && *limit == 100
                        && *offset == 0
                },
            )
            .times(1)
            .returning(move |_, _, _, _, _, _, _| {
                let datapoint_id = Uuid::now_v7();
                Box::pin(async move {
                    Ok(vec![EvaluationResultRow::Chat(ChatEvaluationResultRow {
                        inference_id: Uuid::now_v7(),
                        episode_id: Uuid::now_v7(),
                        datapoint_id,
                        evaluation_run_id,
                        evaluator_inference_id: None,
                        input: Input::default(),
                        generated_output: vec![],
                        reference_output: Some(vec![]),
                        dataset_name: "test_dataset".to_string(),
                        metric_name: Some(
                            "tensorzero::evaluation_name::test_eval::evaluator_name::exact_match"
                                .to_string(),
                        ),
                        metric_value: Some("true".to_string()),
                        feedback_id: Some(Uuid::now_v7()),
                        is_human_feedback: false,
                        variant_name: "test_variant".to_string(),
                        name: None,
                        staled_at: None,
                    })])
                })
            });

        let result = get_evaluation_results(
            &config,
            &mock_clickhouse,
            "test_eval",
            &[evaluation_run_id],
            None,
            100,
            0,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 1);
        match &result.results[0] {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.evaluation_run_id, evaluation_run_id);
            }
            EvaluationResultRow::Json(_) => panic!("Expected Chat result"),
        }
    }

    #[tokio::test]
    async fn test_get_evaluation_results_json_function() {
        let config = create_test_config_with_json_function();
        let evaluation_run_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_results()
            .withf(
                |_fn_name, _run_ids, fn_type, _metrics, _dp_id, _limit, _offset| {
                    *fn_type == FunctionConfigType::Json
                },
            )
            .times(1)
            .returning(|_, _, _, _, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_evaluation_results(
            &config,
            &mock_clickhouse,
            "test_eval",
            &[evaluation_run_id],
            None,
            100,
            0,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_results_evaluation_not_found() {
        let config = create_test_config_with_chat_function();
        let mock_clickhouse = MockEvaluationQueries::new();

        let result = get_evaluation_results(
            &config,
            &mock_clickhouse,
            "nonexistent_eval",
            &[Uuid::now_v7()],
            None,
            100,
            0,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not found in config"));
    }

    #[tokio::test]
    async fn test_get_evaluation_results_function_not_found() {
        // Create config with evaluation but missing function
        let mut evaluations = HashMap::new();
        let mut evaluators = HashMap::new();
        evaluators.insert(
            "exact_match".to_string(),
            EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
        );
        evaluations.insert(
            "test_eval".to_string(),
            Arc::new(EvaluationConfig::Inference(InferenceEvaluationConfig {
                function_name: "missing_function".to_string(),
                evaluators,
                description: None,
            })),
        );

        let config = Config {
            evaluations,
            functions: HashMap::new(),
            ..Default::default()
        };

        let mock_clickhouse = MockEvaluationQueries::new();

        let result = get_evaluation_results(
            &config,
            &mock_clickhouse,
            "test_eval",
            &[Uuid::now_v7()],
            None,
            100,
            0,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Function"));
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_get_evaluation_results_with_pagination() {
        let config = create_test_config_with_chat_function();
        let evaluation_run_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_results()
            .withf(
                |_fn_name, _run_ids, _fn_type, _metrics, _dp_id, limit, offset| {
                    *limit == 50 && *offset == 100
                },
            )
            .times(1)
            .returning(|_, _, _, _, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_evaluation_results(
            &config,
            &mock_clickhouse,
            "test_eval",
            &[evaluation_run_id],
            None,
            50,
            100,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_results_multiple_evaluators() {
        let mut evaluators = HashMap::new();
        evaluators.insert(
            "exact_match".to_string(),
            EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
        );
        evaluators.insert(
            "llm_judge".to_string(),
            EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }), // Just using ExactMatch for simplicity
        );

        let mut evaluations = HashMap::new();
        evaluations.insert(
            "test_eval".to_string(),
            Arc::new(EvaluationConfig::Inference(InferenceEvaluationConfig {
                function_name: "test_function".to_string(),
                evaluators,
                description: None,
            })),
        );

        let mut functions = HashMap::new();
        functions.insert(
            "test_function".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat::default())),
        );

        let config = Config {
            functions,
            evaluations,
            ..Default::default()
        };

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_results()
            .withf(
                |_fn_name, _run_ids, _fn_type, metrics, _dp_id, _limit, _offset| {
                    // Should have 2 metric names, one for each evaluator
                    metrics.len() == 2
                },
            )
            .times(1)
            .returning(|_, _, _, _, _, _, _| Box::pin(async move { Ok(vec![]) }));

        let result = get_evaluation_results(
            &config,
            &mock_clickhouse,
            "test_eval",
            &[Uuid::now_v7()],
            None,
            100,
            0,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_results_with_datapoint_id() {
        let config = create_test_config_with_chat_function();
        let evaluation_run_id = Uuid::now_v7();
        let datapoint_id = Uuid::now_v7();

        let mut mock_clickhouse = MockEvaluationQueries::new();
        mock_clickhouse
            .expect_get_evaluation_results()
            .withf(
                move |_fn_name, _run_ids, _fn_type, _metrics, dp_id, _limit, _offset| {
                    dp_id.is_some() && *dp_id.unwrap() == datapoint_id
                },
            )
            .times(1)
            .returning(move |_, _, _, _, _, _, _| {
                Box::pin(async move {
                    Ok(vec![EvaluationResultRow::Chat(ChatEvaluationResultRow {
                        inference_id: Uuid::now_v7(),
                        episode_id: Uuid::now_v7(),
                        datapoint_id,
                        evaluation_run_id,
                        evaluator_inference_id: None,
                        input: Input::default(),
                        generated_output: vec![],
                        reference_output: Some(vec![]),
                        dataset_name: "test_dataset".to_string(),
                        metric_name: Some(
                            "tensorzero::evaluation_name::test_eval::evaluator_name::exact_match"
                                .to_string(),
                        ),
                        metric_value: Some("true".to_string()),
                        feedback_id: Some(Uuid::now_v7()),
                        is_human_feedback: false,
                        variant_name: "test_variant".to_string(),
                        name: None,
                        staled_at: None,
                    })])
                })
            });

        let result = get_evaluation_results(
            &config,
            &mock_clickhouse,
            "test_eval",
            &[evaluation_run_id],
            Some(&datapoint_id),
            u32::MAX,
            0,
        )
        .await
        .unwrap();

        assert_eq!(result.results.len(), 1);
        match &result.results[0] {
            EvaluationResultRow::Chat(row) => {
                assert_eq!(row.datapoint_id, datapoint_id);
            }
            EvaluationResultRow::Json(_) => panic!("Expected Chat result"),
        }
    }
}
