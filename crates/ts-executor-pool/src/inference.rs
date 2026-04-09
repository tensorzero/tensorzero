//! Centralized inference functions with automatic organization/workspace tagging
//! and UUID substitution.
//!
//! This module provides two entry points for calling TensorZero inference:
//! - [`run_inference`] — direct (non-checkpointed) call for use outside durable execution
//! - [`run_inference_checkpointed`] — checkpointed call via [`ToolContext`] for durable execution
//!
//! Both share [`build_inference_params`] to ensure consistent parameter construction and tagging,
//! and both apply UUID-to-BIP39-triple substitution to improve LLM reliability with identifiers.

use crate::ExtraInferenceTags;
use crate::runtime::ToolContextHelper;
use crate::tensorzero_client::TensorZeroClient;
use bip39_uuid_substitution::{
    UuidSubstituter, UuidSubstitutionError, postprocess_response, preprocess_message,
};
use tensorzero_core::client::ClientInferenceParams;
use tensorzero_core::endpoints::inference::InferenceResponse;
use tensorzero_types::ToolError;
use tensorzero_types::tool_error::ToolResult;
use tensorzero_types::tool_failure::NonControlToolError;

use tracing::debug;

/// Convert a UUID-substitution error into our standard `ToolError`. The
/// `From` impl cannot live on `UuidSubstitutionError` because that would
/// force `bip39-uuid-substitution` to depend on `durable-tools`/`tensorzero-types`,
/// recreating the cycle we want to avoid.
fn uuid_err_to_tool_err(err: UuidSubstitutionError) -> ToolError {
    ToolError::NonControl(NonControlToolError::Internal {
        message: err.to_string(),
    })
}

async fn run_inference_inner<F: Future<Output = ToolResult<InferenceResponse>>>(
    extra_inference_tags: &ExtraInferenceTags,
    mut params: ClientInferenceParams,
    run_inference_fn: impl FnOnce(ClientInferenceParams) -> F,
) -> ToolResult<InferenceResponse> {
    let mut substituter = UuidSubstituter::new();
    params.input.messages = params
        .input
        .messages
        .into_iter()
        .map(|message| preprocess_message(&mut substituter, message))
        .collect::<Result<Vec<_>, _>>()
        .map_err(uuid_err_to_tool_err)?;

    debug!(
        uuid_count = substituter.len(),
        "UUID substitution preprocessing complete"
    );

    params.tags.extend(extra_inference_tags.0.clone());

    let response = run_inference_fn(params).await?;

    let result = postprocess_response(&substituter, response).map_err(uuid_err_to_tool_err)?;

    debug!(
        uuid_count = substituter.len(),
        "UUID substitution postprocessing complete"
    );

    Ok(result)
}

/// Run inference directly (non-checkpointed) with automatic organization/workspace tagging
/// and UUID substitution.
///
/// This calls the TensorZero client directly without durable execution checkpointing.
/// Use this for contexts that don't require durability (e.g., RLM strategy).
///
/// # Errors
///
/// Returns an error if the inference call fails.
pub async fn run_inference(
    params: ClientInferenceParams,
    extra_inference_tags: &ExtraInferenceTags,
    client: &dyn TensorZeroClient,
) -> ToolResult<InferenceResponse> {
    run_inference_inner(extra_inference_tags, params, async |params| {
        client.inference(params).await.map_err(|e| {
            ToolError::NonControl(NonControlToolError::Internal {
                message: e.to_string(),
            })
        })
    })
    .await
}

/// Run inference via [`ToolContext`] (checkpointed) with automatic organization/workspace tagging
/// and UUID substitution.
///
/// This uses the durable execution context for checkpointed inference calls.
/// The `episode_id` is automatically set from the context.
///
/// # Errors
///
/// Returns an error if the inference call fails or if the output is empty.
pub async fn run_inference_checkpointed(
    mut params: ClientInferenceParams,
    extra_inference_tags: &ExtraInferenceTags,
    ctx: &mut dyn ToolContextHelper,
) -> ToolResult<InferenceResponse> {
    if params.episode_id.is_some() {
        return Err(ToolError::NonControl(NonControlToolError::Internal {
            message: "episode_id must be None when using run_inference_checkpointed".to_string(),
        }));
    }
    params.episode_id = Some(ctx.episode_id());
    run_inference_inner(extra_inference_tags, params, |params| ctx.inference(params)).await
}
