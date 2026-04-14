//! RLM-specific LLM query behavior used by JS ops and the core RLM loop.

use std::sync::Arc;

use crate::ExtraInferenceTags;
use crate::RlmConfig;
use crate::TsError;
use crate::inference::run_inference;
use crate::runtime::{
    ExposedTools, RlmDataSource, RlmLoopParams, RlmPermit, RlmRuntimeInput, RlmRuntimeState,
    RuntimeMode, RuntimeParams, spawn_child_runtime,
};
use crate::state::OomSnapshotConfig;
use crate::tensorzero_client::{PoolInferenceParams, TensorZeroClient};
use crate::ts_checker::TsCheckerPool;
use durable::ControlFlow;
use tensorzero_core::endpoints::inference::{ChatInferenceResponse, InferenceResponse};
use tensorzero_core::inference::types::ContentBlockChatOutput;
use tensorzero_types::{Input, InputMessage, InputMessageContent, Role, Text};

/// Shared implementation for `op_llm_query` and `op_llm_query_batched`.
///
/// At `depth < max_depth`, spawns a child RLM loop with the prompt as context.
/// At `depth >= max_depth`, makes a single-shot inference call to
/// `rlm_text_analysis`.
#[expect(clippy::too_many_arguments, reason = "threading OOM snapshot config")]
pub async fn llm_query_with_timeout(
    t0_client: Arc<dyn TensorZeroClient>,
    rlm_state: &RlmRuntimeState,
    extra_inference_tags: &ExtraInferenceTags,
    prompt: &str,
    rlm_permit: &RlmPermit,
    ts_checker: Arc<TsCheckerPool>,
    exposed_tools: Option<ExposedTools>,
    oom_snapshot_config: Option<OomSnapshotConfig>,
) -> Result<Result<String, ControlFlow>, TsError> {
    let timeout = std::time::Duration::from_secs(rlm_state.execution_timeout_secs);
    Box::pin(tokio::time::timeout(
        timeout,
        llm_query_inner(
            t0_client,
            rlm_state,
            extra_inference_tags,
            prompt,
            rlm_permit,
            ts_checker,
            exposed_tools,
            oom_snapshot_config,
        ),
    ))
    .await
    .map_err(|_| TsError::JsRuntime {
        message: "llm_query timed out".to_string(),
    })?
}

/// Inner implementation without the timeout wrapper.
#[expect(clippy::too_many_arguments, reason = "threading OOM snapshot config")]
async fn llm_query_inner(
    t0_client: Arc<dyn TensorZeroClient>,
    rlm_state: &RlmRuntimeState,
    extra_inference_tags: &ExtraInferenceTags,
    prompt: &str,
    rlm_permit: &RlmPermit,
    ts_checker: Arc<TsCheckerPool>,
    exposed_tools: Option<ExposedTools>,
    oom_snapshot_config: Option<OomSnapshotConfig>,
) -> Result<Result<String, ControlFlow>, TsError> {
    let rlm_query = if rlm_state.depth < rlm_state.max_depth {
        rlm_state.rlm_query.clone()
    } else {
        None
    };
    if let Some(rlm_query) = rlm_query {
        // Recursive: spawn a child runtime on its own blocking thread and
        // run the loop through the RlmQuery trait. The parent's RlmPermit
        // propagates so cancellation applies to the entire recursion tree.
        let config = RlmConfig {
            max_iterations: rlm_state.max_iterations,
            max_depth: rlm_state.max_depth,
            char_limit: 0, // not used within the loop
            execution_timeout_secs: rlm_state.execution_timeout_secs,
            // Child loops process LLM output, not typed tool output
            output_ts_type: None,
        };
        let child_rlm_state = RlmRuntimeState {
            depth: rlm_state.depth + 1,
            ..rlm_state.clone()
        };
        let child_permit = rlm_permit.child_permit();
        let handle = spawn_child_runtime(
            RuntimeParams {
                t0_client: t0_client.clone(),
                extra_inference_tags: extra_inference_tags.clone(),
                mode: RuntimeMode::Rlm {
                    input: RlmRuntimeInput::Context(prompt.to_string()),
                    rlm_state: child_rlm_state,
                },
                ts_checker: ts_checker.clone(),
                exposed_tools: exposed_tools.clone(),
                oom_snapshot_config,
            },
            child_permit,
        )
        .await?;
        Box::pin(
            rlm_permit.run_with_cancellation(rlm_query.run(RlmLoopParams {
                handle: &handle,
                data_source: RlmDataSource::Context(prompt),
                t0_client,
                extra_inference_tags,
                episode_id: rlm_state.episode_id,
                config: &config,
                instructions: &rlm_state.instructions,
                ts_checker: &ts_checker,
                exposed_tools: exposed_tools.as_ref(),
                function_name: "rlm_recursive_query",
            })),
        )
        .await
        .and_then(|r| r)
    } else {
        // Leaf: single-shot inference via rlm_text_analysis.
        let user_text = format!(
            "Instructions: {}\n\n---\n\n{}",
            rlm_state.instructions, prompt
        );
        let messages = vec![make_user_message(vec![user_text])];
        let response = rlm_permit
            .run_with_cancellation(run_inference(
                PoolInferenceParams {
                    function_name: Some("rlm_text_analysis".to_string()),
                    episode_id: Some(rlm_state.episode_id),
                    input: Input {
                        system: None,
                        messages,
                    },
                    ..Default::default()
                },
                extra_inference_tags,
                t0_client.as_ref(),
            ))
            .await
            .map_err(|e| TsError::Execution {
                message: format!("rlm_text_analysis inference failed: {e}"),
            })??;
        let extracted = extract_text_from_response(&response)?;
        Ok(Ok(extracted))
    }
}

/// Extract text content from an `InferenceResponse`.
pub fn extract_text_from_response(response: &InferenceResponse) -> Result<String, TsError> {
    match response {
        InferenceResponse::Chat(ChatInferenceResponse { content, .. }) => {
            let text: String = content
                .iter()
                .filter_map(|block| match block {
                    ContentBlockChatOutput::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            if text.is_empty() {
                return Err(TsError::Execution {
                    message: "Inference response contained no text blocks".to_string(),
                });
            }
            Ok(text)
        }
        InferenceResponse::Json(_) => Err(TsError::Execution {
            message: "Expected Chat response, got Json".to_string(),
        }),
    }
}

/// Create a user message with text content.
pub fn make_user_message(text: impl IntoIterator<Item = String>) -> InputMessage {
    InputMessage {
        role: Role::User,
        content: text
            .into_iter()
            .map(|t| InputMessageContent::Text(Text { text: t }))
            .collect(),
    }
}

/// Create an assistant message with text content.
pub fn make_assistant_message(text: impl Into<String>) -> InputMessage {
    InputMessage {
        role: Role::Assistant,
        content: vec![InputMessageContent::Text(Text { text: text.into() })],
    }
}
