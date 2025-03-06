use std::collections::HashMap;

use anyhow::{bail, Result};
use serde_json::Value;
use tensorzero::Client;
use tensorzero::InferenceResponse;
use tensorzero_internal::endpoints::datasets::Datapoint;
use tensorzero_internal::endpoints::feedback::Params;
use tensorzero_internal::evals::get_evaluator_metric_name;
use tensorzero_internal::evals::{EvalConfig, EvaluatorConfig};

mod llm_judge;
use llm_judge::run_llm_judge_evaluator;
use uuid::Uuid;

pub async fn evaluate_inference(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
    eval_config: &EvalConfig,
    eval_name: &str,
    tensorzero_client: &Client,
    eval_run_id: Uuid,
) -> Result<HashMap<String, Value>> {
    let mut results = HashMap::new();
    for (evaluator_name, evaluator_config) in &eval_config.evaluators {
        if let Some(value) = run_evaluator(
            evaluator_config,
            inference_response,
            tensorzero_client,
            datapoint,
            eval_name,
            evaluator_name,
            eval_run_id,
        )
        .await?
        {
            results.insert(evaluator_name.clone(), value.clone());
            tensorzero_client
                .feedback(Params {
                    metric_name: get_evaluator_metric_name(eval_name, evaluator_name),
                    value,
                    inference_id: Some(inference_response.inference_id()),
                    dryrun: Some(false),
                    episode_id: None,
                    tags: HashMap::from([(
                        "tensorzero::eval_run_id".to_string(),
                        eval_run_id.to_string(),
                    )]),
                })
                .await?;
        }
    }
    Ok(results)
}

async fn run_evaluator(
    evaluator_config: &EvaluatorConfig,
    inference_response: &InferenceResponse,
    tensorzero_client: &Client,
    datapoint: &Datapoint,
    eval_name: &str,
    evaluator_name: &str,
    eval_run_id: Uuid,
) -> Result<Option<Value>> {
    match evaluator_config {
        EvaluatorConfig::ExactMatch => run_exact_match_evaluator(inference_response, datapoint),
        EvaluatorConfig::LLMJudge(llm_judge_config) => {
            run_llm_judge_evaluator(
                inference_response,
                datapoint,
                tensorzero_client,
                llm_judge_config,
                eval_name,
                evaluator_name,
                eval_run_id,
            )
            .await
        }
    }
}

fn run_exact_match_evaluator(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
) -> Result<Option<Value>> {
    match (inference_response, datapoint) {
        (InferenceResponse::Chat(response), Datapoint::ChatInference(datapoint)) => {
            match &datapoint.output {
                Some(output) => Ok(Some(Value::Bool(output == &response.content))),
                None => Ok(None),
            }
        }
        (InferenceResponse::Json(json_completion), Datapoint::JsonInference(json_inference)) => {
            match &json_inference.output {
                Some(output) => {
                    // `output.parsed` is an Option<Value> but it should always be Some here
                    if output.parsed.is_none() {
                        tracing::warn!("Datapoint {} has no parsed output", json_inference.id);
                        return Ok(None);
                    }
                    Ok(Some(Value::Bool(
                        output.parsed == json_completion.output.parsed,
                    )))
                }
                None => Ok(None),
            }
        }
        _ => bail!("Datapoint and inference response types do not match"),
    }
}
