use anyhow::Result;
use serde_json::Value;
use tensorzero::InferenceResponse;
use tensorzero_internal::endpoints::datasets::Datapoint;
use tensorzero_internal::evals::LLMJudgeConfig;

pub async fn run_llm_judge_evaluator(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
    llm_judge_config: &LLMJudgeConfig,
) -> Result<Option<Value>> {
    todo!()
}
