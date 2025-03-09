use std::collections::HashMap;

use anyhow::{bail, Result};
use serde_json::{json, Value};
use tensorzero::{
    ClientInferenceParams, DynamicToolParams, InferenceOutput, InferenceParams, InferenceResponse,
    Input, InputMessage, InputMessageContent, Role,
};
use tensorzero_internal::cache::CacheEnabledMode;
use tensorzero_internal::endpoints::datasets::Datapoint;
use tensorzero_internal::evals::{get_llm_judge_function_name, LLMJudgeConfig, LLMJudgeOutputType};
use tensorzero_internal::inference::types::{
    ContentBlockChatOutput, JsonInferenceOutput, ResolvedInput, ResolvedInputMessageContent,
    TextKind,
};
use uuid::Uuid;

use crate::ThrottledTensorZeroClient;

pub(crate) async fn run_llm_judge_evaluator(
    inference_response: &InferenceResponse,
    datapoint: &Datapoint,
    tensorzero_client: &ThrottledTensorZeroClient,
    llm_judge_config: &LLMJudgeConfig,
    eval_name: &str,
    evaluator_name: &str,
    eval_run_id: Uuid,
) -> Result<Option<Value>> {
    let function_name = get_llm_judge_function_name(eval_name, evaluator_name);
    let resolved_input = datapoint.input();
    let serialized_datapoint_input = prepare_serialized_input(resolved_input)?;
    let generated_output = match &inference_response {
        InferenceResponse::Chat(chat_response) => {
            prepare_serialized_chat_output(&chat_response.content)?
        }
        InferenceResponse::Json(json_response) => {
            prepare_serialized_json_output(&json_response.output)?
        }
    };
    let reference_output = match handle_reference_output(llm_judge_config, datapoint) {
        Ok(reference_output) => reference_output,
        // Reference output is optional so if it's needed but not present, we can just return None
        // TODO (Viraj): should we log an error here?
        Err(_e) => return Ok(None),
    };
    let input = Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(TextKind::Arguments{
                arguments: json!({"input": serialized_datapoint_input, "generated_output": generated_output, "reference_output": reference_output})
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert LLM judge arguments to Map. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports."))?
                    .clone()
            })],
        }],
    };

    let params = ClientInferenceParams {
        function_name: Some(function_name),
        model_name: None,
        episode_id: None,
        input,
        stream: Some(false),
        params: InferenceParams::default(),
        variant_name: None,
        dryrun: Some(false),
        tags: HashMap::from([(
            "tensorzero::eval_run_id".to_string(),
            eval_run_id.to_string(),
        )]),
        dynamic_tool_params: DynamicToolParams::default(),
        output_schema: None,
        credentials: HashMap::new(),
        cache_options: tensorzero::CacheParamsOptions {
            max_age_s: None,
            enabled: CacheEnabledMode::On,
        },
    };
    let result = tensorzero_client.inference(params).await?;
    let response = match result {
        InferenceOutput::NonStreaming(response) => response,
        InferenceOutput::Streaming(..) => {
            bail!("Streaming not supported for LLM judge evals. This is a bug, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.")
        }
    };
    let output = match response {
        InferenceResponse::Chat(..) => {
            bail!("Chat output not supported for LLM judge evals. This is a bug, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.")
        }
        InferenceResponse::Json(json_response) => json_response
            .output
            .parsed
            .ok_or_else(|| anyhow::anyhow!("JSON output does not contain a `parsed` field"))?,
    };
    match llm_judge_config.output_type {
        LLMJudgeOutputType::Float => Ok(Some(
            output
                .get("score")
                .ok_or_else(|| anyhow::anyhow!("JSON output does not contain a `score` field"))?
                .clone(),
        )),
        LLMJudgeOutputType::Boolean => Ok(Some(
            output
                .get("score")
                .ok_or_else(|| anyhow::anyhow!("JSON output does not contain a `score` field"))?
                .clone(),
        )),
    }
}

pub fn prepare_serialized_input(resolved_input: &ResolvedInput) -> Result<String> {
    for message in &resolved_input.messages {
        for content in &message.content {
            match content {
                ResolvedInputMessageContent::Image(..) => {
                    bail!("Image content not supported for LLM judge evals")
                }
                ResolvedInputMessageContent::Unknown { .. } => {
                    bail!("Unknown content not supported for LLM judge evals")
                }
                _ => {}
            }
        }
    }
    Ok(serde_json::to_string(resolved_input)?)
}

/// We prepare the serialized output by converting the content blocks to a string.
/// The only reason this doesn't directly use serde_json::to_string is because we want to
/// strip out the Unknown content blocks, which we don't want to include in the output.
fn prepare_serialized_chat_output(content: &Vec<ContentBlockChatOutput>) -> Result<String> {
    // TODO (Viraj): test this
    let mut blocks_to_serialized = Vec::new();
    for block in content {
        if let ContentBlockChatOutput::Unknown { .. } = block {
            continue;
        }
        blocks_to_serialized.push(block);
    }
    if blocks_to_serialized.is_empty() {
        bail!("No valid content blocks to serialize");
    }
    Ok(serde_json::to_string(&blocks_to_serialized)?)
}

fn prepare_serialized_json_output(output: &JsonInferenceOutput) -> Result<String> {
    if output.parsed.is_none() {
        bail!("JSON output does not contain a `parsed` field");
    }
    Ok(output.raw.clone())
}

/// Handles the reference output for the LLM judge evaluator.
/// If the reference output is not needed, we return None.
/// If the reference output is needed, we return the serialized output of the datapoint.
/// If the reference output is needed but not present, we throw an error. (this could be mapped to None above this call)
fn handle_reference_output(
    llm_judge_config: &LLMJudgeConfig,
    datapoint: &Datapoint,
) -> Result<Option<String>> {
    if !llm_judge_config.include.reference_output {
        return Ok(None);
    }
    match datapoint {
        Datapoint::ChatInference(chat_datapoint) => match &chat_datapoint.output {
            Some(output) => prepare_serialized_chat_output(output).map(Some),
            None => bail!("Datapoint does not contain an output when this is expected"),
        },
        Datapoint::JsonInference(json_datapoint) => match &json_datapoint.output {
            Some(output) => prepare_serialized_json_output(output).map(Some),
            None => bail!("Datapoint does not contain an output when this is expected"),
        },
    }
}
