//! SSE event types for evaluation streaming.
//!
//! These types are used both by the gateway (for serializing SSE events) and by
//! the client (for deserializing SSE events when consuming the stream over HTTP).

use std::collections::HashMap;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_core::evaluations::EvaluationConfig;
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

/// SSE event types for evaluation streaming.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, TensorZeroDeserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum EvaluationRunEvent {
    Start(EvaluationRunStartEvent),
    Success(EvaluationRunSuccessEvent),
    Error(EvaluationRunErrorEvent),
    FatalError(EvaluationRunFatalErrorEvent),
    Complete(EvaluationRunCompleteEvent),
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct EvaluationRunStartEvent {
    pub evaluation_run_id: Uuid,
    pub num_datapoints: usize,
    pub evaluation_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_name: Option<String>,
    /// The evaluation configuration, included when the server resolves it
    /// (i.e. when the client didn't provide it in the request).
    /// This allows HTTP clients to compute summary_stats() without having
    /// the config locally.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    pub evaluation_config: Option<EvaluationConfig>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EvaluationRunSuccessEvent {
    pub evaluation_run_id: Uuid,
    pub datapoint: Value,
    pub response: Value,
    pub evaluations: HashMap<String, Option<Value>>,
    pub evaluator_errors: HashMap<String, String>,
    /// Wall-clock time for the inference call, in milliseconds
    pub inference_time_ms: f64,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EvaluationRunErrorEvent {
    pub evaluation_run_id: Uuid,
    pub datapoint_id: Uuid,
    pub message: String,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct EvaluationRunFatalErrorEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_run_id: Option<Uuid>,
    pub message: String,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EvaluationRunCompleteEvent {
    pub evaluation_run_id: Uuid,
    pub usage: EvaluationRunUsageSummary,
}

/// Aggregated usage statistics for a completed evaluation run.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct EvaluationRunUsageSummary {
    /// Number of successful inferences
    pub count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_output_tokens: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "rust_decimal::serde::float_option"
    )]
    #[cfg_attr(feature = "ts-bindings", ts(type = "number | null"))]
    pub total_cost: Option<Decimal>,
    /// Total wall-clock inference time in milliseconds
    pub total_inference_time_ms: f64,
}

impl EvaluationRunUsageSummary {
    pub fn accumulate(&mut self, info: &super::stats::EvaluationInfo) {
        self.count += 1;
        let usage = info.response.usage();
        if let Some(input_tokens) = usage.input_tokens {
            *self.total_input_tokens.get_or_insert(0) += input_tokens as u64;
        }
        if let Some(output_tokens) = usage.output_tokens {
            *self.total_output_tokens.get_or_insert(0) += output_tokens as u64;
        }
        if let Some(cost) = usage.cost {
            *self.total_cost.get_or_insert(Decimal::ZERO) += cost;
        }
        self.total_inference_time_ms += info.inference_time_ms;
    }
}
