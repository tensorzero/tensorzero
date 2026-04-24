//! Minimal TensorZero client trait for inference.
//!
//! This provides the subset of the full `durable_tools::TensorZeroClient` that
//! `ts-executor-pool` actually requires: just the `inference` method.
//!
//! [`PoolInferenceParams`] is a lightweight version of the full inference
//! parameters containing only the fields that this crate reads or writes.
//! Implementations convert to the full parameter type at the call-site
//! boundary.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;
use tensorzero_inference_types::tool::DynamicToolParams;
use tensorzero_types::inference_params::InferenceParams;
use tensorzero_types::{InferenceResponse, Input};
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

/// Minimal inference parameters used within `ts-executor-pool`.
///
/// This contains only the fields that this crate actually accesses.
/// Implementations of [`TensorZeroClient`] are responsible for converting
/// this into the full `ClientInferenceParams` before calling the gateway.
#[derive(Debug, Clone, Default)]
pub struct PoolInferenceParams {
    pub function_name: Option<String>,
    pub episode_id: Option<Uuid>,
    pub input: Input,
    pub tags: HashMap<String, String>,
    pub params: InferenceParams,
    pub dynamic_tool_params: DynamicToolParams,
    pub output_schema: Option<Value>,
}

/// Minimal client trait for running TensorZero inference.
///
/// Callers provide an implementation that wraps the full `TensorZeroClient`
/// (or any other inference backend). Only `inference` is required.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait TensorZeroClient: Send + Sync + 'static {
    /// Run inference with the given parameters.
    async fn inference(
        &self,
        params: PoolInferenceParams,
    ) -> Result<InferenceResponse, Box<dyn std::error::Error + Send + Sync>>;
}
