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
use crate::function::FunctionConfigType;
use crate::utils::gateway::{AppState, SwappableAppStateData};

/// Query parameters for getting evaluation results.
#[derive(Debug, Deserialize)]
pub struct GetEvaluationResultsParams {
    /// The name of the evaluation (e.g., "haiku", "entity_extraction")
    pub evaluation_name: Option<String>,
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
#[derive(ts_rs::TS)]
#[derive(Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct GetEvaluationResultsResponse {
    pub results: Vec<EvaluationResultRow>,
}

/// Handler for `GET /internal/evaluations/results`
///
/// Returns paginated evaluation results across one or more evaluation runs.
#[axum::debug_handler(state = SwappableAppStateData)]
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

    let database = app_state.get_delegating_database();
    let response = get_evaluation_results(
        &app_state.config,
        &database,
        params.evaluation_name.as_deref(),
        &evaluation_run_ids,
        params.datapoint_id.as_ref(),
        limit,
        offset,
    )
    .await?;

    Ok(Json(response))
}

/// Resolved metadata needed to query evaluation results.
struct ResolvedEvaluationMetadata {
    function_name: String,
    function_type: FunctionConfigType,
    metric_names: Vec<String>,
}

/// Resolves evaluation metadata by first checking the `inference_evaluation_runs` table,
/// then falling back to the evaluation config.
async fn resolve_evaluation_metadata(
    config: &Config,
    db: &impl EvaluationQueries,
    evaluation_name: Option<&str>,
    evaluation_run_ids: &[Uuid],
) -> Result<ResolvedEvaluationMetadata, Error> {
    // Try to read metadata from the database for all requested runs
    if !evaluation_run_ids.is_empty() {
        let results = db
            .get_inference_evaluation_run_metadata(evaluation_run_ids)
            .await?;
        if let Some((_, first_metadata)) = results.first() {
            let function_name = first_metadata.function_name.clone();
            let function_type = first_metadata.function_type;

            // Collect metric names from all runs (deduplicated, preserving insertion order)
            let mut seen = std::collections::HashSet::new();
            let mut metric_names = Vec::new();
            for (_, metadata) in &results {
                for m in &metadata.metrics {
                    if seen.insert(&m.name) {
                        metric_names.push(m.name.clone());
                    }
                }
            }

            return Ok(ResolvedEvaluationMetadata {
                function_name,
                function_type,
                metric_names,
            });
        }
    }

    // Fall back to reading from config (legacy path)
    let Some(evaluation_name) = evaluation_name else {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Did not provide an evaluation name when resolving evaluation metadata, and run metadata is not stored."
                .to_string(),
        }));
    };

    let evaluation_config = config.evaluations.get(evaluation_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Evaluation `{evaluation_name}` not found in config"),
        })
    })?;

    let (function_name, evaluators) = match evaluation_config.as_ref() {
        EvaluationConfig::Inference(inference_config) => (
            &inference_config.function_name,
            &inference_config.evaluators,
        ),
    };

    let function_config = config.functions.get(function_name).ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!("Function `{function_name}` not found in config"),
        })
    })?;

    let metric_names = evaluators
        .keys()
        .map(|evaluator_name| {
            format!(
                "tensorzero::evaluation_name::{evaluation_name}::evaluator_name::{evaluator_name}"
            )
        })
        .collect();

    Ok(ResolvedEvaluationMetadata {
        function_name: function_name.clone(),
        function_type: function_config.config_type(),
        metric_names,
    })
}

/// Core business logic for getting paginated evaluation results.
///
/// Tries to read function_name and metric_names from the `inference_evaluation_runs` table first.
/// Falls back to the evaluation config if the run metadata is not found in the database.
pub async fn get_evaluation_results(
    config: &Config,
    db: &impl EvaluationQueries,
    evaluation_name: Option<&str>,
    evaluation_run_ids: &[Uuid],
    datapoint_id: Option<&Uuid>,
    limit: u32,
    offset: u32,
) -> Result<GetEvaluationResultsResponse, Error> {
    let metadata =
        resolve_evaluation_metadata(config, db, evaluation_name, evaluation_run_ids).await?;

    let results = db
        .get_evaluation_results(
            &metadata.function_name,
            evaluation_run_ids,
            metadata.function_type,
            &metadata.metric_names,
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
    use crate::db::evaluation_queries::{
        ChatEvaluationResultRow, InferenceEvaluationRunMetadata,
        InferenceEvaluationRunMetricMetadata, MockEvaluationQueries,
    };
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
        #[expect(deprecated)]
        let exact_match = EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None });
        evaluators.insert("exact_match".to_string(), exact_match);
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
        #[expect(deprecated)]
        let exact_match = EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None });
        evaluators.insert("exact_match".to_string(), exact_match);
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

    /// Helper to set up the mock to return empty metadata, triggering the config fallback.
    fn mock_no_db_metadata(mock: &mut MockEvaluationQueries) {
        mock.expect_get_inference_evaluation_run_metadata()
            .returning(|_| Box::pin(async { Ok(Vec::new()) }));
    }

    #[tokio::test]
    async fn test_get_evaluation_results_from_db_metadata() {
        let config = Config::default();
        let evaluation_run_id = Uuid::now_v7();

        let mut mock_db = MockEvaluationQueries::new();
        mock_db
            .expect_get_inference_evaluation_run_metadata()
            .withf(move |ids| ids == [evaluation_run_id])
            .times(1)
            .returning(move |ids| {
                let run_id = ids[0];
                Box::pin(async move {
                    Ok(vec![(
                        run_id,
                        InferenceEvaluationRunMetadata {
                            evaluation_name: "db_eval".to_string(),
                            function_name: "db_function".to_string(),
                            function_type: FunctionConfigType::Chat,
                            metrics: vec![InferenceEvaluationRunMetricMetadata {
                                name: "metric_from_db".to_string(),
                                evaluator_name: Some("my_evaluator".to_string()),
                                value_type: "boolean".to_string(),
                                optimize: None,
                            }],
                        },
                    )])
                })
            });
        mock_db
            .expect_get_evaluation_results()
            .withf(
                move |fn_name, _run_ids, fn_type, metrics, _dp_id, _limit, _offset| {
                    fn_name == "db_function"
                        && *fn_type == FunctionConfigType::Chat
                        && metrics == ["metric_from_db"]
                },
            )
            .times(1)
            .returning(|_, _, _, _, _, _, _| Box::pin(async { Ok(vec![]) }));

        let result =
            get_evaluation_results(&config, &mock_db, None, &[evaluation_run_id], None, 100, 0)
                .await
                .unwrap();

        assert_eq!(result.results.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_results_merges_metrics_across_runs() {
        let config = Config::default();
        let run_id_1 = Uuid::now_v7();
        let run_id_2 = Uuid::now_v7();

        let mut mock_db = MockEvaluationQueries::new();
        mock_db
            .expect_get_inference_evaluation_run_metadata()
            .withf(move |ids| ids.len() == 2 && ids.contains(&run_id_1) && ids.contains(&run_id_2))
            .times(1)
            .returning(move |_| {
                Box::pin(async move {
                    Ok(vec![
                        (
                            run_id_1,
                            InferenceEvaluationRunMetadata {
                                evaluation_name: "eval".to_string(),
                                function_name: "my_function".to_string(),
                                function_type: FunctionConfigType::Chat,
                                metrics: vec![
                                    InferenceEvaluationRunMetricMetadata {
                                        name: "metric_a".to_string(),
                                        evaluator_name: Some("eval_a".to_string()),
                                        value_type: "boolean".to_string(),
                                        optimize: None,
                                    },
                                    InferenceEvaluationRunMetricMetadata {
                                        name: "metric_shared".to_string(),
                                        evaluator_name: Some("eval_shared".to_string()),
                                        value_type: "float".to_string(),
                                        optimize: None,
                                    },
                                ],
                            },
                        ),
                        (
                            run_id_2,
                            InferenceEvaluationRunMetadata {
                                evaluation_name: "eval".to_string(),
                                function_name: "my_function".to_string(),
                                function_type: FunctionConfigType::Chat,
                                metrics: vec![
                                    InferenceEvaluationRunMetricMetadata {
                                        name: "metric_shared".to_string(),
                                        evaluator_name: Some("eval_shared".to_string()),
                                        value_type: "float".to_string(),
                                        optimize: None,
                                    },
                                    InferenceEvaluationRunMetricMetadata {
                                        name: "metric_b".to_string(),
                                        evaluator_name: Some("eval_b".to_string()),
                                        value_type: "boolean".to_string(),
                                        optimize: None,
                                    },
                                ],
                            },
                        ),
                    ])
                })
            });
        mock_db
            .expect_get_evaluation_results()
            .withf(
                move |fn_name, run_ids, fn_type, metrics, _dp_id, _limit, _offset| {
                    fn_name == "my_function"
                    && run_ids.len() == 2
                    && *fn_type == FunctionConfigType::Chat
                    // All three unique metrics should be present (deduplicated)
                    && metrics.len() == 3
                    && metrics.contains(&"metric_a".to_string())
                    && metrics.contains(&"metric_shared".to_string())
                    && metrics.contains(&"metric_b".to_string())
                },
            )
            .times(1)
            .returning(|_, _, _, _, _, _, _| Box::pin(async { Ok(vec![]) }));

        let result =
            get_evaluation_results(&config, &mock_db, None, &[run_id_1, run_id_2], None, 100, 0)
                .await
                .unwrap();

        assert_eq!(result.results.len(), 0);
    }

    #[tokio::test]
    async fn test_get_evaluation_results_chat_function() {
        let config = create_test_config_with_chat_function();
        let evaluation_run_id = Uuid::now_v7();

        let mut mock_db = MockEvaluationQueries::new();
        mock_no_db_metadata(&mut mock_db);
        mock_db
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
                        input: Some(Input::default()),
                        generated_output: Some(vec![]),
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
                        input_tokens: Some(100),
                        output_tokens: Some(50),
                        cost: Some(0.001),
                        processing_time_ms: Some(200),
                    })])
                })
            });

        let result = get_evaluation_results(
            &config,
            &mock_db,
            Some("test_eval"),
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

        let mut mock_db = MockEvaluationQueries::new();
        mock_no_db_metadata(&mut mock_db);
        mock_db
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
            &mock_db,
            Some("test_eval"),
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
        let mut mock_db = MockEvaluationQueries::new();
        mock_no_db_metadata(&mut mock_db);

        let result = get_evaluation_results(
            &config,
            &mock_db,
            Some("nonexistent_eval"),
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
        #[expect(deprecated)]
        let exact_match = EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None });
        evaluators.insert("exact_match".to_string(), exact_match);
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

        let mut mock_db = MockEvaluationQueries::new();
        mock_no_db_metadata(&mut mock_db);

        let result = get_evaluation_results(
            &config,
            &mock_db,
            Some("test_eval"),
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

        let mut mock_db = MockEvaluationQueries::new();
        mock_no_db_metadata(&mut mock_db);
        mock_db
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
            &mock_db,
            Some("test_eval"),
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
        #[expect(deprecated)]
        let exact_match = EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None });
        evaluators.insert("exact_match".to_string(), exact_match);
        #[expect(deprecated)]
        let llm_judge = EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None });
        evaluators.insert("llm_judge".to_string(), llm_judge);

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

        let mut mock_db = MockEvaluationQueries::new();
        mock_no_db_metadata(&mut mock_db);
        mock_db
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
            &mock_db,
            Some("test_eval"),
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

        let mut mock_db = MockEvaluationQueries::new();
        mock_no_db_metadata(&mut mock_db);
        mock_db
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
                        input: Some(Input::default()),
                        generated_output: Some(vec![]),
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
                        input_tokens: Some(100),
                        output_tokens: Some(50),
                        cost: Some(0.001),
                        processing_time_ms: Some(200),
                    })])
                })
            });

        let result = get_evaluation_results(
            &config,
            &mock_db,
            Some("test_eval"),
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
