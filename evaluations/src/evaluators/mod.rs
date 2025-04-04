use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;
use tensorzero::{FeedbackParams, InferenceResponse, Input};
use tensorzero_internal::endpoints::datasets::Datapoint;
use tensorzero_internal::evaluations::{
    get_evaluator_metric_name, EvaluationConfig, EvaluatorConfig,
};

mod exact_match;
use exact_match::run_exact_match_evaluator;
pub mod llm_judge;
use futures::stream::{FuturesUnordered, StreamExt};
use llm_judge::{run_llm_judge_evaluator, LLMJudgeEvaluationResult};
use uuid::Uuid;

use crate::ThrottledTensorZeroClient;

pub type EvaluationResult = HashMap<String, Result<Option<Value>>>;

/// Evaluates the inference response for the given datapoint using all the evaluators specified in the evaluation config.
/// Returns a map from evaluator name to Result<Option<Value>>.
/// The semantics of the Result<Option<Value>> are as follows:
/// - Ok(Some(value)): The evaluator was run successfully and the result was a valid value.
/// - Ok(None): The evaluator was run successfully but the result was None (if for example the evaluator requires a reference output but none is present).
/// - Err(e): The evaluator failed to run due to some error (like the LLM Judge failed to infer).
pub(crate) async fn evaluate_inference(
    inference_response: Arc<InferenceResponse>,
    datapoint: Arc<Datapoint>,
    input: Arc<Input>,
    evaluation_config: Arc<EvaluationConfig>,
    evaluation_name: Arc<String>,
    tensorzero_client: Arc<ThrottledTensorZeroClient>,
    evaluation_run_id: Uuid,
) -> Result<EvaluationResult> {
    let EvaluationConfig::Static(static_evaluation_config) = &*evaluation_config;
    let results: EvaluationResult =
        FuturesUnordered::from_iter(static_evaluation_config.evaluators.keys().map(
            |evaluator_name| async {
                let inference_response = inference_response.clone();
                let evaluation_config = evaluation_config.clone();
                let evaluator_name = evaluator_name.clone();
                let datapoint = datapoint.clone();
                let input = input.clone();
                let evaluation_name = evaluation_name.clone();
                let tensorzero_client = tensorzero_client.clone();
                let evaluator_name_clone = evaluator_name.clone();

                let result = run_evaluator(
                    &evaluation_config,
                    evaluator_name_clone.clone(),
                    &inference_response,
                    &tensorzero_client,
                    &datapoint,
                    &evaluation_name,
                    evaluation_run_id,
                    &input,
                )
                .await;

                let evaluation_result = match result {
                    Ok(result) => {
                        if let Some(value) = result.value() {
                            // If there is a valid result, send feedback to TensorZero
                            let mut tags = HashMap::from([
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
                            if let Some(inference_id) = result.inference_id() {
                                tags.insert(
                                    "tensorzero::evaluator_inference_id".to_string(),
                                    inference_id.to_string(),
                                );
                            }
                            match tensorzero_client
                                .feedback(FeedbackParams {
                                    metric_name: get_evaluator_metric_name(
                                        &evaluation_name,
                                        &evaluator_name_clone,
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
                                Ok(_) => (),
                                Err(e) => return (evaluator_name, Err(e)),
                            }
                        }
                        Ok(result.value_owned())
                    }
                    Err(e) => Err(e),
                };

                (evaluator_name, evaluation_result)
            },
        ))
        .collect()
        .await;
    Ok(results)
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
async fn run_evaluator(
    evaluation_config: &EvaluationConfig,
    evaluator_name: String,
    inference_response: &InferenceResponse,
    tensorzero_client: &ThrottledTensorZeroClient,
    datapoint: &Datapoint,
    evaluation_name: &str,
    evaluation_run_id: Uuid,
    input: &Input,
) -> Result<EvaluatorResult> {
    let EvaluationConfig::Static(static_evaluation_config) = evaluation_config;
    let evaluator_config = match static_evaluation_config.evaluators.get(&evaluator_name) {
        Some(evaluator_config) => evaluator_config,
        None => {
            return Err(anyhow::anyhow!("Evaluator config not found for {}. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports.", evaluator_name));
        }
    };
    Ok(match evaluator_config {
        EvaluatorConfig::ExactMatch(_exact_match_config) => {
            EvaluatorResult::ExactMatch(run_exact_match_evaluator(inference_response, datapoint)?)
        }
        EvaluatorConfig::LLMJudge(llm_judge_config) => EvaluatorResult::LLMJudge(
            run_llm_judge_evaluator(
                inference_response,
                datapoint,
                tensorzero_client,
                llm_judge_config,
                evaluation_name,
                &evaluator_name,
                evaluation_run_id,
                input,
            )
            .await?,
        ),
    })
}

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

    pub fn inference_id(&'a self) -> Option<&'a Uuid> {
        match self {
            EvaluatorResult::ExactMatch(_) => None,
            EvaluatorResult::LLMJudge(value) => value.as_ref().map(|v| &v.inference_id),
        }
    }
    pub fn value_owned(self) -> Option<Value> {
        match self {
            EvaluatorResult::ExactMatch(value) => value,
            EvaluatorResult::LLMJudge(value) => value.map(|v| v.value),
        }
    }
}
