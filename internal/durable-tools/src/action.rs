//! Action execution logic for the TensorZero action endpoint.
//!
//! This module provides the core action dispatch logic used by both the gateway
//! HTTP handler and embedded clients.

use std::collections::HashMap;

use evaluations::{
    EvaluationVariant, RunEvaluationWithAppStateParams, run_evaluation_with_app_state,
    stats::{EvaluationStats, EvaluationUpdate},
};
use tensorzero_core::endpoints::feedback::feedback;
use tensorzero_core::endpoints::inference::{InferenceOutput, inference};
use tensorzero_core::endpoints::internal::action::{
    ActionInput, ActionInputInfo, ActionResponse, DatapointResult, EvaluatorStatsResponse,
    RunEvaluationActionResponse, get_or_load_config,
};
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::evaluations::{EvaluationConfig, EvaluationFunctionConfig};
use tensorzero_core::utils::gateway::AppStateData;

/// Executes an inference or feedback action using a historical config snapshot.
pub async fn action(
    app_state: &AppStateData,
    params: ActionInputInfo,
) -> Result<ActionResponse, Error> {
    let config = get_or_load_config(app_state, &params.snapshot_hash).await?;

    match params.input {
        ActionInput::Inference(inference_params) => {
            // Reject streaming requests
            if inference_params.stream.unwrap_or(false) {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Streaming is not supported for the action endpoint".to_string(),
                }));
            }

            let data = Box::pin(inference(
                config,
                &app_state.http_client,
                app_state.clickhouse_connection_info.clone(),
                app_state.postgres_connection_info.clone(),
                app_state.deferred_tasks.clone(),
                app_state.rate_limiting_manager.clone(),
                (*inference_params).try_into()?,
                None, // No API key for internal endpoint
            ))
            .await?;

            match data.output {
                InferenceOutput::NonStreaming(response) => Ok(ActionResponse::Inference(response)),
                InferenceOutput::Streaming(_) => {
                    // Should not happen since we checked stream=false above
                    Err(Error::new(ErrorDetails::InternalError {
                        message: "Unexpected streaming response".to_string(),
                    }))
                }
            }
        }
        ActionInput::Feedback(feedback_params) => {
            // Build AppStateData with snapshot config
            let snapshot_app_state = AppStateData::new_for_snapshot(
                config,
                app_state.http_client.clone(),
                app_state.clickhouse_connection_info.clone(),
                app_state.postgres_connection_info.clone(),
                app_state.deferred_tasks.clone(),
            );

            let response = feedback(snapshot_app_state, *feedback_params, None).await?;
            Ok(ActionResponse::Feedback(response))
        }
        ActionInput::RunEvaluation(eval_params) => {
            // Validate concurrency
            if eval_params.concurrency == 0 {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Concurrency must be greater than 0".to_string(),
                }));
            }

            // Validate: exactly one of dataset_name or datapoint_ids must be provided
            let has_dataset = eval_params.dataset_name.is_some();
            let has_datapoints = eval_params
                .datapoint_ids
                .as_ref()
                .is_some_and(|ids| !ids.is_empty());
            if has_dataset == has_datapoints {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: if has_dataset {
                        "Cannot provide both dataset_name and datapoint_ids".to_string()
                    } else {
                        "Must provide either dataset_name or datapoint_ids".to_string()
                    },
                }));
            }

            // Look up evaluation config from snapshot
            let evaluation_config = config
                .evaluations
                .get(&eval_params.evaluation_name)
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Evaluation `{}` not found in config",
                            eval_params.evaluation_name
                        ),
                    })
                })?
                .clone();

            // Get function config
            let EvaluationConfig::Inference(ref inference_eval_config) = *evaluation_config;
            let function_config = config
                .functions
                .get(&inference_eval_config.function_name)
                .map(|func| EvaluationFunctionConfig::from(func.as_ref()))
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!(
                            "Function `{}` not found in config",
                            inference_eval_config.function_name
                        ),
                    })
                })?;

            // Build AppStateData with snapshot config
            let snapshot_app_state = AppStateData::new_for_snapshot(
                config,
                app_state.http_client.clone(),
                app_state.clickhouse_connection_info.clone(),
                app_state.postgres_connection_info.clone(),
                app_state.deferred_tasks.clone(),
            );

            // Build params for run_evaluation_with_app_state
            let run_params = RunEvaluationWithAppStateParams {
                evaluation_config: (*evaluation_config).clone(),
                function_config,
                evaluation_name: eval_params.evaluation_name.clone(),
                dataset_name: eval_params.dataset_name.clone(),
                datapoint_ids: eval_params.datapoint_ids.clone(),
                variant: EvaluationVariant::Name(eval_params.variant_name.clone()),
                concurrency: eval_params.concurrency,
                cache_mode: eval_params.inference_cache,
                max_datapoints: eval_params.max_datapoints,
                precision_targets: eval_params.precision_targets.clone(),
            };

            // Run the evaluation
            let result = run_evaluation_with_app_state(snapshot_app_state, run_params)
                .await
                .map_err(|e| {
                    Error::new(ErrorDetails::EvaluationRun {
                        message: format!("Evaluation failed: {e}"),
                    })
                })?;

            // Collect results from stream
            let response = collect_evaluation_results(
                result,
                &inference_eval_config.evaluators,
                eval_params.include_datapoint_results,
            )
            .await?;

            Ok(ActionResponse::RunEvaluation(response))
        }
    }
}

/// Collects evaluation results from the stream and computes statistics.
async fn collect_evaluation_results(
    result: evaluations::EvaluationStreamResult,
    evaluators: &HashMap<String, tensorzero_core::evaluations::EvaluatorConfig>,
    include_datapoint_results: bool,
) -> Result<RunEvaluationActionResponse, Error> {
    let mut receiver = result.receiver;
    let num_datapoints = result.run_info.num_datapoints;
    let evaluation_run_id = result.run_info.evaluation_run_id;

    // Collect results using EvaluationStats (with Jsonl output format to skip progress bar)
    let mut evaluation_stats =
        EvaluationStats::new(evaluations::OutputFormat::Jsonl, num_datapoints);
    let mut dummy_writer = std::io::sink();

    while let Some(update) = receiver.recv().await {
        match update {
            EvaluationUpdate::RunInfo(_) => continue,
            update => {
                let _ = evaluation_stats.push(update, &mut dummy_writer);
            }
        }
    }

    // Compute statistics
    let stats = evaluation_stats.compute_stats(evaluators);

    // Convert to response format
    let stats_response: HashMap<String, EvaluatorStatsResponse> = stats
        .into_iter()
        .map(|(name, s)| {
            (
                name,
                EvaluatorStatsResponse {
                    mean: s.mean,
                    stderr: s.stderr,
                    count: s.count,
                },
            )
        })
        .collect();

    // Build per-datapoint results if requested
    let datapoint_results = if include_datapoint_results {
        let mut results = Vec::with_capacity(
            evaluation_stats.evaluation_infos.len() + evaluation_stats.evaluation_errors.len(),
        );

        for info in &evaluation_stats.evaluation_infos {
            let evaluations: HashMap<String, Option<f64>> = info
                .evaluations
                .iter()
                .map(|(name, value)| {
                    let score = value.as_ref().and_then(|v| {
                        v.as_f64()
                            .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                    });
                    (name.clone(), score)
                })
                .collect();

            results.push(DatapointResult {
                datapoint_id: info.datapoint.id(),
                success: true,
                evaluations,
                evaluator_errors: info.evaluator_errors.clone(),
                error: None,
            });
        }

        for error in &evaluation_stats.evaluation_errors {
            results.push(DatapointResult {
                datapoint_id: error.datapoint_id,
                success: false,
                evaluations: HashMap::new(),
                evaluator_errors: HashMap::new(),
                error: Some(error.message.clone()),
            });
        }

        Some(results)
    } else {
        None
    };

    // Wait for ClickHouse batch writer
    if let Some(handle) = result.batcher_join_handle {
        handle.await.map_err(|e| {
            Error::new(ErrorDetails::EvaluationRun {
                message: format!("Error waiting for ClickHouse batch writer: {e}"),
            })
        })?;
    }

    Ok(RunEvaluationActionResponse {
        evaluation_run_id,
        num_datapoints,
        num_successes: evaluation_stats.evaluation_infos.len(),
        num_errors: evaluation_stats.evaluation_errors.len(),
        stats: stats_response,
        datapoint_results,
    })
}
