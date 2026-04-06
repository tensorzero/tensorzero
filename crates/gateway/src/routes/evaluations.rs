//! Evaluation streaming endpoint.
//!
//! This module provides an SSE endpoint for running evaluations and streaming results.

use std::collections::HashMap;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use serde::Deserialize;
use tokio::sync::mpsc;
use uuid::Uuid;

use evaluations::sse_events::{
    EvaluationRunCompleteEvent, EvaluationRunErrorEvent, EvaluationRunEvent,
    EvaluationRunFatalErrorEvent, EvaluationRunStartEvent, EvaluationRunSuccessEvent,
    EvaluationRunUsageSummary,
};
use evaluations::stats::{EvaluationError, EvaluationInfo, EvaluationUpdate};
use evaluations::{
    EvaluationVariant, RunEvaluationWithAppStateParams, run_evaluation_with_app_state,
};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::config::UninitializedVariantInfo;
use tensorzero_core::db::BatchWriterHandle;
use tensorzero_core::error::{Error, ErrorDetails};
use tensorzero_core::evaluations::{EvaluationConfig, EvaluationFunctionConfig, EvaluatorConfig};
use tensorzero_core::utils::gateway::{
    AppState, ResolvedAppStateData, StructuredJson, SwappableAppStateData,
};

// =============================================================================
// Request/Response Types
// =============================================================================

/// Identifies the evaluation to run.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(untagged)]
pub enum EvaluationIdentifier {
    /// Legacy evaluation config, with evaluators configured as part of a named evaluation.
    LegacyNamedEvaluation {
        evaluation_config: EvaluationConfig,
        evaluation_name: String,
        function_config: EvaluationFunctionConfig,
    },
    /// Named evaluators resolved from gateway config.
    Evaluators {
        function_name: String,
        evaluator_names: Vec<String>,
    },
    /// Named evaluation resolved from gateway config.
    /// The gateway looks up the evaluation by name and resolves its function and evaluators.
    NamedEvaluation { evaluation_name: String },
}

/// Request body for running an evaluation.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct RunEvaluationRequest {
    /// How the evaluation is configured: either explicit config or named evaluators.
    #[serde(flatten)]
    pub source: EvaluationIdentifier,
    /// Name of the dataset to evaluate (optional, either this or datapoint_ids must be provided)
    #[serde(default)]
    pub dataset_name: Option<String>,
    /// Specific datapoint IDs to evaluate (optional)
    #[serde(default)]
    pub datapoint_ids: Option<Vec<Uuid>>,
    /// Variant to use: either a variant name or a dynamic variant configuration
    #[serde(default)]
    pub variant_name: Option<String>,
    /// Dynamic variant configuration (alternative to variant_name)
    #[serde(default)]
    pub internal_dynamic_variant_config: Option<UninitializedVariantInfo>,
    /// Number of concurrent requests
    #[serde(default = "default_concurrency")]
    pub concurrency: u32,
    /// Cache mode for inference requests
    #[serde(default = "default_inference_cache")]
    pub inference_cache: String,
    /// Maximum number of datapoints to evaluate
    #[serde(default)]
    pub max_datapoints: Option<u32>,
    /// Per-evaluator precision targets for adaptive stopping
    #[serde(default)]
    pub precision_targets: Option<HashMap<String, f32>>,
}

fn default_concurrency() -> u32 {
    5
}

fn default_inference_cache() -> String {
    "on".to_string()
}

fn success_event_from_info(
    evaluation_run_id: Uuid,
    info: EvaluationInfo,
) -> Result<EvaluationRunSuccessEvent, Error> {
    let datapoint = serde_json::to_value(info.datapoint).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize datapoint: {e}"),
        })
    })?;
    let response = serde_json::to_value(info.response).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize inference response: {e}"),
        })
    })?;

    Ok(EvaluationRunSuccessEvent {
        evaluation_run_id,
        datapoint,
        response,
        evaluations: info.evaluations,
        evaluator_errors: info.evaluator_errors,
        processing_time_ms: info.processing_time_ms,
    })
}

// =============================================================================
// Stream Creation
// =============================================================================

struct EvaluationStreamParams {
    evaluation_run_id: Uuid,
    num_datapoints: usize,
    evaluation_name: String,
    dataset_name: Option<String>,
    variant_name: Option<String>,
    evaluation_config: Option<EvaluationConfig>,
    receiver: mpsc::Receiver<EvaluationUpdate>,
    batcher_join_handles: Vec<BatchWriterHandle>,
}

fn create_evaluation_stream(
    params: EvaluationStreamParams,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let EvaluationStreamParams {
        evaluation_run_id,
        num_datapoints,
        evaluation_name,
        dataset_name,
        variant_name,
        evaluation_config,
        mut receiver,
        batcher_join_handles,
    } = params;

    async_stream::stream! {
        // Send start event
        let start_event = EvaluationRunEvent::Start(EvaluationRunStartEvent {
            evaluation_run_id,
            num_datapoints,
            evaluation_name,
            dataset_name,
            variant_name,
            evaluation_config,
        });

        match serde_json::to_string(&start_event) {
            Ok(data) => yield Ok(Event::default().event("event").data(data)),
            Err(e) => {
                let error_event = EvaluationRunEvent::FatalError(EvaluationRunFatalErrorEvent {
                    evaluation_run_id: Some(evaluation_run_id),
                    message: format!("Failed to serialize start event: {e}"),
                });
                if let Ok(data) = serde_json::to_string(&error_event) {
                    yield Ok(Event::default().event("event").data(data));
                }
                return;
            }
        }

        // Accumulate usage statistics across all successful inferences
        let mut usage_summary = EvaluationRunUsageSummary::default();

        // Stream evaluation updates
        while let Some(update) = receiver.recv().await {
            let event = match update {
                EvaluationUpdate::RunInfo(_) => continue, // Already sent start event
                EvaluationUpdate::Success(info) => {
                    usage_summary.record_success(&info);
                    match success_event_from_info(evaluation_run_id, info) {
                        Ok(success_event) => EvaluationRunEvent::Success(success_event),
                        Err(e) => EvaluationRunEvent::Error(EvaluationRunErrorEvent {
                            evaluation_run_id,
                            datapoint_id: Uuid::nil(),
                            message: format!("Failed to serialize success event: {e}"),
                        }),
                    }
                }
                EvaluationUpdate::Error(EvaluationError { datapoint_id, message }) => {
                    usage_summary.record_error();
                    EvaluationRunEvent::Error(EvaluationRunErrorEvent {
                        evaluation_run_id,
                        datapoint_id,
                        message,
                    })
                }
                EvaluationUpdate::FatalError(message) => {
                    EvaluationRunEvent::FatalError(EvaluationRunFatalErrorEvent {
                        evaluation_run_id: Some(evaluation_run_id),
                        message,
                    })
                }
            };

            match serde_json::to_string(&event) {
                Ok(data) => yield Ok(Event::default().event("event").data(data)),
                Err(e) => {
                    tracing::error!("Failed to serialize evaluation event: {e}");
                    continue;
                }
            }
        }

        // Wait for batch writers to finish
        for handle in batcher_join_handles {
            if let Err(e) = handle.await {
                let error_event = EvaluationRunEvent::FatalError(EvaluationRunFatalErrorEvent {
                    evaluation_run_id: Some(evaluation_run_id),
                    message: format!("Error waiting for batch writer: {e}"),
                });
                if let Ok(data) = serde_json::to_string(&error_event) {
                    yield Ok(Event::default().event("event").data(data));
                }
                return;
            }
        }

        // Send complete event with aggregated usage
        let complete_event = EvaluationRunEvent::Complete(EvaluationRunCompleteEvent {
            evaluation_run_id,
            usage: usage_summary,
        });
        if let Ok(data) = serde_json::to_string(&complete_event) {
            yield Ok(Event::default().event("event").data(data));
        }
    }
}

// =============================================================================
// HTTP Handler
// =============================================================================

/// Resolved evaluation configuration from either the old or new request path.
struct ResolvedEvaluationConfig {
    function_name: String,
    evaluators: HashMap<String, EvaluatorConfig>,
    function_config: EvaluationFunctionConfig,
    /// Evaluation name for metric naming. `None` for standalone evaluators (top-level naming).
    evaluation_name: Option<String>,
}

/// Resolves evaluation and function configs from the request.
fn resolve_evaluation_config(
    request: &RunEvaluationRequest,
    app_state: &ResolvedAppStateData,
) -> Result<ResolvedEvaluationConfig, Error> {
    match &request.source {
        EvaluationIdentifier::LegacyNamedEvaluation {
            evaluation_config,
            evaluation_name,
            function_config,
        } => {
            let EvaluationConfig::Inference(ref inference_config) = *evaluation_config;
            Ok(ResolvedEvaluationConfig {
                function_name: inference_config.function_name.clone(),
                function_config: function_config.clone(),
                evaluators: inference_config.evaluators.clone(),
                evaluation_name: Some(evaluation_name.clone()),
            })
        }
        EvaluationIdentifier::Evaluators {
            function_name,
            evaluator_names,
        } => {
            let function_config =
                app_state
                    .config
                    .functions
                    .get(function_name)
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidRequest {
                            message: format!("Function `{function_name}` not found in config"),
                        })
                    })?;
            let evaluators = resolve_evaluators(function_name, evaluator_names, app_state)?;
            Ok(ResolvedEvaluationConfig {
                function_name: function_name.clone(),
                function_config: EvaluationFunctionConfig::from(function_config.as_ref()),
                evaluators,
                evaluation_name: None,
            })
        }
        EvaluationIdentifier::NamedEvaluation { evaluation_name } => {
            let evaluation_config = app_state
                .config
                .evaluations
                .get(evaluation_name)
                .ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidRequest {
                        message: format!("Evaluation `{evaluation_name}` not found in config"),
                    })
                })?;
            let EvaluationConfig::Inference(ref inference_config) = **evaluation_config;
            let function_name = &inference_config.function_name;
            let function_config =
                app_state
                    .config
                    .functions
                    .get(function_name)
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidRequest {
                            message: format!(
                                "Function `{function_name}` (referenced by evaluation `{evaluation_name}`) not found in config"
                            ),
                        })
                    })?;
            Ok(ResolvedEvaluationConfig {
                function_name: function_name.clone(),
                function_config: EvaluationFunctionConfig::from(function_config.as_ref()),
                evaluators: inference_config.evaluators.clone(),
                evaluation_name: Some(evaluation_name.clone()),
            })
        }
    }
}

/// Resolves evaluator configs by name from the function's evaluators.
///
/// TODO(#6983): allow evaluators defined under evaluations to be referenced here
/// for (function, evaluators) style evaluation runs.
fn resolve_evaluators(
    function_name: &str,
    evaluator_names: &[String],
    app_state: &ResolvedAppStateData,
) -> Result<HashMap<String, EvaluatorConfig>, Error> {
    let function_config = app_state
        .config
        .functions
        .get(function_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: format!("Function `{function_name}` not found in config"),
            })
        })?;
    let function_evaluators = function_config.evaluators();
    let mut evaluators = HashMap::new();
    for name in evaluator_names {
        let evaluator = function_evaluators.get(name).ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: format!("Evaluator `{name}` not found on function `{function_name}`"),
            })
        })?;
        evaluators.insert(name.clone(), evaluator.clone());
    }
    Ok(evaluators)
}

/// Handler for `POST /internal/evaluations/run`
///
/// Runs an evaluation and streams results via SSE.
#[axum::debug_handler(state = SwappableAppStateData)]
pub async fn run_evaluation_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<RunEvaluationRequest>,
) -> Result<impl IntoResponse, Error> {
    // Resolve evaluation config from either old or new request path
    let resolved = resolve_evaluation_config(&request, &app_state)?;

    tracing::info!(
        evaluation_name = ?resolved.evaluation_name,
        dataset_name = ?request.dataset_name,
        variant_name = ?request.variant_name,
        concurrency = %request.concurrency,
        "Starting evaluation run"
    );

    // Parse cache mode
    let cache_mode = match request.inference_cache.as_str() {
        "on" => CacheEnabledMode::On,
        "off" => CacheEnabledMode::Off,
        "read_only" => CacheEnabledMode::ReadOnly,
        "write_only" => CacheEnabledMode::WriteOnly,
        other => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: format!("Invalid inference cache setting '{other}'"),
            }));
        }
    };

    // Parse concurrency
    let concurrency = usize::try_from(request.concurrency).map_err(|_| {
        Error::new(ErrorDetails::InvalidRequest {
            message: format!(
                "Concurrency {} is larger than supported on this platform",
                request.concurrency
            ),
        })
    })?;

    // Determine the variant
    let variant = match (
        request.variant_name.clone(),
        request.internal_dynamic_variant_config,
    ) {
        (Some(name), None) => EvaluationVariant::Name(name),
        (None, Some(config)) => EvaluationVariant::Info(Box::new(config)),
        (Some(_), Some(_)) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot provide both variant_name and internal_dynamic_variant_config"
                    .into(),
            }));
        }
        (None, None) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message:
                    "Exactly one of variant_name or internal_dynamic_variant_config must be provided"
                        .into(),
            }));
        }
    };

    let precision_targets: HashMap<String, f32> = request.precision_targets.unwrap_or_default();

    // Get variant name for the start event
    let variant_name_for_event = match &variant {
        EvaluationVariant::Name(name) => Some(name.clone()),
        EvaluationVariant::Info(_) => None,
    };

    let dataset_name_for_event = request.dataset_name.clone();

    // Always include evaluation_config in the start event so HTTP clients
    // can compute summary_stats without having the config locally.
    let evaluation_config_for_event = Some(EvaluationConfig::Inference(
        tensorzero_core::evaluations::InferenceEvaluationConfig {
            evaluators: resolved.evaluators.clone(),
            function_name: resolved.function_name.clone(),
            description: None,
        },
    ));

    // Build the params for run_evaluation_with_app_state
    let params = RunEvaluationWithAppStateParams {
        function_name: resolved.function_name,
        function_config: resolved.function_config,
        evaluators: resolved.evaluators,
        evaluation_name: resolved.evaluation_name,
        dataset_name: request.dataset_name,
        datapoint_ids: request.datapoint_ids,
        variant,
        concurrency,
        cache_mode,
        max_datapoints: request.max_datapoints,
        precision_targets,
        tags: HashMap::new(), // No external tags for HTTP-initiated evaluations
    };

    // Run the evaluation
    let result = run_evaluation_with_app_state(app_state, params)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::EvaluationRun {
                message: format!("Failed to start evaluation run: {e}"),
            })
        })?;

    let evaluation_run_id = result.run_info.evaluation_run_id;
    let evaluation_name = result.run_info.evaluation_name;
    let num_datapoints = result.run_info.num_datapoints;

    // Create the SSE stream
    let sse_stream = create_evaluation_stream(EvaluationStreamParams {
        evaluation_run_id,
        num_datapoints,
        evaluation_name,
        dataset_name: dataset_name_for_event,
        variant_name: variant_name_for_event,
        evaluation_config: evaluation_config_for_event,
        receiver: result.receiver,
        batcher_join_handles: result.batcher_join_handles,
    });

    Ok(Sse::new(sse_stream).keep_alive(KeepAlive::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_format_deserialization() {
        let json = serde_json::json!({
            "evaluation_config": {
                "type": "inference",
                "evaluators": {
                    "em": {
                        "type": "exact_match"
                    }
                },
                "function_name": "my_func"
            },
            "function_config": {
                "type": "chat"
            },
            "evaluation_name": "my_eval",
            "variant_name": "v1"
        });
        let request: RunEvaluationRequest =
            serde_json::from_value(json).expect("explicit format should deserialize");
        assert!(
            matches!(&request.source, EvaluationIdentifier::LegacyNamedEvaluation { evaluation_name, .. } if evaluation_name == "my_eval"),
            "should deserialize as Explicit with evaluation_name"
        );
    }

    #[test]
    fn test_named_format_deserialization() {
        let json = serde_json::json!({
            "function_name": "my_func",
            "evaluator_names": ["em_evaluator", "judge"],
            "dataset_name": "test_ds",
            "variant_name": "v1"
        });
        let request: RunEvaluationRequest =
            serde_json::from_value(json).expect("named format should deserialize");
        assert!(
            matches!(&request.source, EvaluationIdentifier::Evaluators { function_name, evaluator_names } if function_name == "my_func" && evaluator_names == &["em_evaluator", "judge"]),
            "should deserialize as Named with function_name and evaluator_names"
        );
    }

    #[test]
    fn test_named_evaluation_deserialization() {
        let json = serde_json::json!({
            "evaluation_name": "my_eval",
            "dataset_name": "test_ds",
            "variant_name": "v1"
        });
        let request: RunEvaluationRequest =
            serde_json::from_value(json).expect("named evaluation format should deserialize");
        assert!(
            matches!(&request.source, EvaluationIdentifier::NamedEvaluation { evaluation_name } if evaluation_name == "my_eval"),
            "should deserialize as NamedEvaluation with evaluation_name"
        );
    }
}
