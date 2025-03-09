use anyhow::{bail, Result};
use serde_json::Value;
use tensorzero::InferenceResponse;
use tensorzero_internal::endpoints::datasets::Datapoint;

pub(super) fn run_exact_match_evaluator(
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
