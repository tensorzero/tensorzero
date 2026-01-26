use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::{FeedbackParams, InferenceResponse, Input};
use tensorzero_core::endpoints::datasets::Datapoint;
use tensorzero_core::error::IMPOSSIBLE_ERROR_MESSAGE;
use tensorzero_core::evaluations::{EvaluationConfig, EvaluatorConfig, get_evaluator_metric_name};

mod exact_match;
use exact_match::run_exact_match_evaluator;
pub mod llm_judge;
use futures::stream::{FuturesUnordered, StreamExt};
use llm_judge::{LLMJudgeEvaluationResult, RunLLMJudgeEvaluatorParams, run_llm_judge_evaluator};
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

use crate::Clients;
use crate::stopping::CancellationTokens;

pub type EvaluationResult = HashMap<String, Result<Option<Value>>>;

pub struct EvaluateInferenceParams {
    pub inference_response: Arc<InferenceResponse>,
    pub datapoint: Arc<Datapoint>,
    pub input: Arc<Input>,
    pub evaluation_config: Arc<EvaluationConfig>,
    pub evaluation_name: Arc<String>,
    pub clients: Arc<Clients>,
    pub evaluation_run_id: Uuid,
    pub inference_cache: CacheEnabledMode,
    pub external_tags: Arc<HashMap<String, String>>,
    pub send_feedback: bool,
}

/// Evaluates the inference response for the given datapoint using the evaluators specified in the evaluation config.
///
/// ## Adaptive Stopping
///
/// The `cancellation_tokens` parameter controls which evaluators run:
/// - If the token map is empty: runs all evaluators in the config (no adaptive stopping)
/// - If the token map is non-empty: skips evaluators in the map with cancelled tokens
///
/// ## Return Value
///
/// Returns a map from evaluator name to Result<Option<Value>>.
/// The semantics of the Result<Option<Value>> are as follows:
/// - Ok(Some(value)): The evaluator was run successfully and the result was a valid value.
/// - Ok(None): The evaluator was run successfully but the result was None (e.g., if the evaluator requires a reference output but none is present).
/// - Err(e): The evaluator failed to run due to some error (e.g., the LLM Judge failed to infer).
#[instrument(skip_all, fields(datapoint_id = %params.datapoint.id(), evaluation_name = %params.evaluation_name))]
pub(crate) async fn evaluate_inference(
    params: EvaluateInferenceParams,
    cancellation_tokens: &CancellationTokens,
) -> Result<EvaluationResult> {
    let EvaluateInferenceParams {
        inference_response,
        datapoint,
        input,
        evaluation_config,
        evaluation_name,
        clients,
        evaluation_run_id,
        inference_cache,
        external_tags,
        send_feedback,
    } = params;
    let EvaluationConfig::Inference(inference_evaluation_config) = &*evaluation_config;

    // Filter evaluators based on cancellation tokens
    let evaluators_to_run = inference_evaluation_config
        .evaluators
        .keys()
        .filter(|name| {
            // Only run evaluators whose tokens are not in the map or not cancelled
            // Empty map means no adaptive stopping, so all evaluators run
            cancellation_tokens
                .get(*name)
                .is_none_or(|token| !token.is_cancelled())
        });

    info!(
        evaluators = ?evaluators_to_run,
        "Starting evaluation with evaluators"
    );

    let results: EvaluationResult =
        FuturesUnordered::from_iter(evaluators_to_run.into_iter().map(
            |evaluator_name| async {
                let inference_response = inference_response.clone();
                let evaluation_config = evaluation_config.clone();
                let evaluator_name = evaluator_name.clone();
                debug!(evaluator_name = %evaluator_name, "Starting evaluator");
                let datapoint = datapoint.clone();
                let input = input.clone();
                let evaluation_name = evaluation_name.clone();
                let clients = clients.clone();
                let evaluator_name_clone = evaluator_name.clone();
                let external_tags = external_tags.clone();
                let external_tags_for_feedback = external_tags.clone();

                let result = run_evaluator(RunEvaluatorParams {
                    evaluation_config: &evaluation_config,
                    evaluator_name: evaluator_name_clone,
                    inference_response: &inference_response,
                    clients: &clients,
                    datapoint: &datapoint,
                    evaluation_name: &evaluation_name,
                    evaluation_run_id,
                    input: &input,
                    inference_cache,
                    external_tags,
                })
                .await;

                debug!(
                    evaluator_name = %evaluator_name,
                    success = result.is_ok(),
                    "Evaluator completed"
                );

                let evaluation_result = match result {
                    Ok(result) => {
                        if let Some(value) = result.value() {
                            debug!(evaluator_name = %evaluator_name, value = ?value, "Evaluator produced value, sending feedback");
                            // If there is a valid result, send feedback to TensorZero
                            // Start with external tags, then apply internal tags last so they always win
                            let mut tags: HashMap<String, String> = external_tags_for_feedback
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect();
                            tags.extend(result.tags());
                            tags.extend([
                                (
                                    "tensorzero::evaluation_run_id".to_string(),
                                    evaluation_run_id.to_string(),
                                ),
                                (
                                    "tensorzero::datapoint_id".to_string(),
                                    datapoint.id().to_string(),
                                ),
                                (
                                    "tensorzero::evaluation_name".to_string(),
                                    evaluation_name.to_string(),
                                ),
                                (
                                    "tensorzero::evaluator_name".to_string(),
                                    evaluator_name.to_string(),
                                ),
                            ]);
                            if let Some(evaluator_inference_id) = result.evaluator_inference_id() {
                                tags.insert(
                                    "tensorzero::evaluator_inference_id".to_string(),
                                    evaluator_inference_id.to_string(),
                                );
                            }
                            // Only send feedback when send_feedback is true
                            // Dynamic variants have send_feedback=false and skip feedback persistence
                            if send_feedback {
                                match clients
                                    .inference_executor
                                    .feedback(FeedbackParams {
                                        metric_name: get_evaluator_metric_name(
                                            &evaluation_name,
                                            &evaluator_name,
                                        ),
                                        value: value.clone(),
                                        inference_id: Some(inference_response.inference_id()),
                                        dryrun: Some(false),
                                        episode_id: None,
                                        internal: true,
                                        tags,
                                    })
                                    .await
                                {
                                    Ok(_) => {
                                        debug!(evaluator_name = %evaluator_name, "Feedback sent successfully");
                                    },
                                    Err(e) => {
                                        error!(evaluator_name = %evaluator_name, error = %e, "Failed to send feedback");
                                        return (evaluator_name, Err(e.into()));
                                    }
                                }
                            }
                        } else {
                            debug!(evaluator_name = %evaluator_name, "Evaluator produced no value");
                        }
                        Ok(result.value_owned())
                    }
                    Err(e) => {
                        tracing::warn!(evaluator_name = %evaluator_name, error = %e, "Evaluator failed");
                        Err(e)
                    }
                };

                (evaluator_name, evaluation_result)
            },
        ))
        .collect()
        .await;
    Ok(results)
}

struct RunEvaluatorParams<'a> {
    evaluation_config: &'a EvaluationConfig,
    evaluator_name: String,
    inference_response: &'a InferenceResponse,
    clients: &'a Clients,
    datapoint: &'a Datapoint,
    evaluation_name: &'a str,
    evaluation_run_id: Uuid,
    input: &'a Input,
    inference_cache: CacheEnabledMode,
    external_tags: Arc<HashMap<String, String>>,
}

/// Runs the evaluator specified by evaluator_name on the given inference response and datapoint.
/// Returns Result<Option<Value>>.
///
/// The semantics of the Result<Option<Value>> are as follows:
/// - Ok(Some(value)): The evaluator was run successfully and the result was a valid value.
/// - Ok(None): The evaluator was run successfully but the result was None (if for example the evaluator requires a reference output but none is present).
/// - Err(e): The evaluator failed to run due to some error (like the LLM Judge failed to infer).
///
/// NOTE: Each evaluator we implement in the match statement below should follow this contract.
#[instrument(skip_all, fields(evaluator_name = %params.evaluator_name, datapoint_id = %params.datapoint.id()))]
async fn run_evaluator(params: RunEvaluatorParams<'_>) -> Result<EvaluatorResult> {
    let RunEvaluatorParams {
        evaluation_config,
        evaluator_name,
        inference_response,
        clients,
        datapoint,
        evaluation_name,
        evaluation_run_id,
        input,
        inference_cache,
        external_tags,
    } = params;
    let EvaluationConfig::Inference(inference_evaluation_config) = evaluation_config;
    let evaluator_config = match inference_evaluation_config.evaluators.get(&evaluator_name) {
        Some(evaluator_config) => {
            debug!("Evaluator config found");
            evaluator_config
        }
        None => {
            error!("Evaluator config not found");
            return Err(anyhow::anyhow!(
                "Evaluator config not found for {evaluator_name}.  {IMPOSSIBLE_ERROR_MESSAGE}"
            ));
        }
    };
    Ok(match evaluator_config {
        EvaluatorConfig::ExactMatch(_exact_match_config) => {
            debug!("Running exact match evaluator");
            let result = run_exact_match_evaluator(inference_response, datapoint)?;
            debug!(result = ?result, "Exact match evaluator completed");
            EvaluatorResult::ExactMatch(result)
        }
        EvaluatorConfig::LLMJudge(llm_judge_config) => {
            debug!("Running LLM judge evaluator");
            let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
                inference_response,
                datapoint,
                clients,
                llm_judge_config,
                evaluation_name,
                evaluator_name: &evaluator_name,
                evaluation_run_id,
                input,
                inference_cache,
                external_tags: external_tags.as_ref(),
            })
            .await?;
            debug!(result = ?result, "LLM judge evaluator completed");
            EvaluatorResult::LLMJudge(result)
        }
    })
}

#[derive(Debug)]
pub enum EvaluatorResult {
    ExactMatch(Option<Value>),
    LLMJudge(Option<LLMJudgeEvaluationResult>),
}

impl<'a> EvaluatorResult {
    pub fn value(&'a self) -> Option<&'a Value> {
        match self {
            EvaluatorResult::ExactMatch(value) => value.as_ref(),
            EvaluatorResult::LLMJudge(value) => value.as_ref().map(|v| &v.value),
        }
    }

    pub fn evaluator_inference_id(&'a self) -> Option<&'a Uuid> {
        match self {
            EvaluatorResult::ExactMatch(_) => None,
            EvaluatorResult::LLMJudge(value) => value.as_ref().map(|v| &v.evaluator_inference_id),
        }
    }
    pub fn value_owned(self) -> Option<Value> {
        match self {
            EvaluatorResult::ExactMatch(value) => value,
            EvaluatorResult::LLMJudge(value) => value.map(|v| v.value),
        }
    }
    pub fn tags(&'a self) -> HashMap<String, String> {
        match self {
            EvaluatorResult::ExactMatch(_) => HashMap::new(),
            EvaluatorResult::LLMJudge(value) => value
                .as_ref()
                .map(LLMJudgeEvaluationResult::tags)
                .unwrap_or_default(),
        }
    }
}
