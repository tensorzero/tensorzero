//! Centralized inference function with automatic organization/workspace tagging
//! and UUID substitution.
//!
//! [`run_inference`] is a direct (non-checkpointed) call for use outside durable
//! execution. For checkpointed inference via a
//! [`ToolContextHelper`](tensorzero_core::client::ToolContextHelper), see
//! [`tensorzero_core::client::checkpointed_inference`].

use crate::ExtraInferenceTags;
use crate::tensorzero_client::{PoolInferenceParams, TensorZeroClient};
use bip39_uuid_substitution::{
    UuidSubstituter, UuidSubstitutionError, postprocess_response, preprocess_message,
};
use tensorzero_types::InferenceResponse;
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
    mut params: PoolInferenceParams,
    run_inference_fn: impl FnOnce(PoolInferenceParams) -> F,
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
    params: PoolInferenceParams,
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
