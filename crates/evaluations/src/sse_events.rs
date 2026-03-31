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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::str::FromStr;

    use googletest::prelude::*;
    use rust_decimal::Decimal;
    use tensorzero_core::endpoints::inference::{ChatInferenceResponse, InferenceResponse};
    use tensorzero_core::inference::types::{ContentBlockChatOutput, Text};
    use uuid::Uuid;

    use super::*;
    use crate::stats::EvaluationInfo;

    /// Build a minimal `EvaluationInfo` with the given usage and timing.
    /// We construct a real `InferenceResponse` and a fake datapoint via JSON round-trip.
    fn make_evaluation_info(
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
        cost: Option<Decimal>,
        inference_time_ms: f64,
    ) -> EvaluationInfo {
        let usage = tensorzero_provider_types::Usage {
            input_tokens,
            output_tokens,
            cost,
            ..Default::default()
        };
        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: "hello".to_string(),
            })],
            usage,
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        });

        // Build a minimal datapoint via JSON deserialization
        let datapoint: tensorzero_core::endpoints::datasets::Datapoint =
            serde_json::from_value(serde_json::json!({
                "type": "chat",
                "id": Uuid::now_v7(),
                "dataset_name": "test_ds",
                "function_name": "test_fn",
                "input": {"messages": []},
                "auxiliary": "",
                "tags": {},
                "is_deleted": false,
                "is_custom": false,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z"
            }))
            .expect("valid datapoint JSON");

        EvaluationInfo {
            datapoint,
            response,
            evaluations: HashMap::new(),
            evaluator_errors: HashMap::new(),
            inference_time_ms,
        }
    }

    #[gtest]
    fn test_accumulate_empty() {
        let summary = EvaluationRunUsageSummary::default();
        expect_that!(summary.count, eq(0));
        expect_that!(summary.total_input_tokens, none());
        expect_that!(summary.total_output_tokens, none());
        expect_that!(summary.total_cost, none());
        expect_that!(summary.total_inference_time_ms, eq(0.0));
    }

    #[gtest]
    fn test_accumulate_single_with_full_usage() {
        let mut summary = EvaluationRunUsageSummary::default();
        let info = make_evaluation_info(
            Some(100),
            Some(50),
            Some(Decimal::from_str("0.005").expect("valid decimal")),
            150.0,
        );
        summary.accumulate(&info);

        expect_that!(summary.count, eq(1));
        expect_that!(summary.total_input_tokens, some(eq(100)));
        expect_that!(summary.total_output_tokens, some(eq(50)));
        expect_that!(
            summary.total_cost,
            some(eq(Decimal::from_str("0.005").expect("valid decimal")))
        );
        expect_that!(summary.total_inference_time_ms, eq(150.0));
    }

    #[gtest]
    fn test_accumulate_multiple() {
        let mut summary = EvaluationRunUsageSummary::default();

        let info1 = make_evaluation_info(
            Some(100),
            Some(50),
            Some(Decimal::from_str("0.005").expect("valid decimal")),
            100.0,
        );
        let info2 = make_evaluation_info(
            Some(200),
            Some(80),
            Some(Decimal::from_str("0.010").expect("valid decimal")),
            200.0,
        );

        summary.accumulate(&info1);
        summary.accumulate(&info2);

        expect_that!(summary.count, eq(2));
        expect_that!(summary.total_input_tokens, some(eq(300)));
        expect_that!(summary.total_output_tokens, some(eq(130)));
        expect_that!(
            summary.total_cost,
            some(eq(Decimal::from_str("0.015").expect("valid decimal")))
        );
        expect_that!(summary.total_inference_time_ms, eq(300.0));
    }

    #[gtest]
    fn test_accumulate_with_none_usage() {
        let mut summary = EvaluationRunUsageSummary::default();

        // First inference has no usage at all
        let info1 = make_evaluation_info(None, None, None, 50.0);
        // Second has partial usage
        let info2 = make_evaluation_info(Some(100), None, None, 75.0);

        summary.accumulate(&info1);
        summary.accumulate(&info2);

        expect_that!(summary.count, eq(2));
        expect_that!(
            summary.total_input_tokens,
            some(eq(100)),
            "only the second inference had input tokens"
        );
        expect_that!(
            summary.total_output_tokens,
            none(),
            "neither inference had output tokens"
        );
        expect_that!(summary.total_cost, none(), "neither inference had cost");
        expect_that!(summary.total_inference_time_ms, eq(125.0));
    }

    #[gtest]
    fn test_complete_event_serialization_round_trip() {
        let run_id = Uuid::now_v7();
        let event = EvaluationRunEvent::Complete(EvaluationRunCompleteEvent {
            evaluation_run_id: run_id,
            usage: EvaluationRunUsageSummary {
                count: 5,
                total_input_tokens: Some(1000),
                total_output_tokens: Some(500),
                total_cost: Some(Decimal::from_str("0.05").expect("valid decimal")),
                total_inference_time_ms: 750.0,
            },
        });

        let json = serde_json::to_string(&event).expect("serialization should succeed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("should be valid JSON");

        expect_that!(parsed["type"], eq("complete"));
        expect_that!(parsed["evaluation_run_id"], eq(run_id.to_string().as_str()));
        expect_that!(parsed["usage"]["count"], eq(5));
        expect_that!(parsed["usage"]["total_input_tokens"], eq(1000));
        expect_that!(parsed["usage"]["total_output_tokens"], eq(500));
        expect_that!(parsed["usage"]["total_cost"], eq(0.05));
        expect_that!(parsed["usage"]["total_inference_time_ms"], eq(750.0));
    }

    #[gtest]
    fn test_complete_event_omits_none_fields() {
        let run_id = Uuid::now_v7();
        let event = EvaluationRunEvent::Complete(EvaluationRunCompleteEvent {
            evaluation_run_id: run_id,
            usage: EvaluationRunUsageSummary::default(),
        });

        let json = serde_json::to_string(&event).expect("serialization should succeed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("should be valid JSON");

        expect_that!(parsed["usage"]["count"], eq(0));
        // None fields should be omitted (skip_serializing_if)
        expect_that!(parsed["usage"].get("total_input_tokens"), none());
        expect_that!(parsed["usage"].get("total_output_tokens"), none());
        expect_that!(parsed["usage"].get("total_cost"), none());
    }

    #[gtest]
    fn test_success_event_includes_inference_time() {
        let run_id = Uuid::now_v7();
        let event = EvaluationRunEvent::Success(EvaluationRunSuccessEvent {
            evaluation_run_id: run_id,
            datapoint: serde_json::json!({}),
            response: serde_json::json!({}),
            evaluations: HashMap::new(),
            evaluator_errors: HashMap::new(),
            inference_time_ms: 123.456,
        });

        let json = serde_json::to_string(&event).expect("serialization should succeed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("should be valid JSON");

        expect_that!(parsed["type"], eq("success"));
        expect_that!(parsed["inference_time_ms"], eq(123.456));
    }

    #[gtest]
    fn test_complete_event_deserialization() {
        let json = serde_json::json!({
            "type": "complete",
            "evaluation_run_id": "01963691-9d3c-7793-a8be-3937ebb849c1",
            "usage": {
                "count": 10,
                "total_input_tokens": 5000,
                "total_output_tokens": 2500,
                "total_cost": 0.123,
                "total_inference_time_ms": 1500.5
            }
        });

        let event: EvaluationRunEvent =
            serde_json::from_value(json).expect("deserialization should succeed");
        match event {
            EvaluationRunEvent::Complete(complete) => {
                expect_that!(complete.usage.count, eq(10));
                expect_that!(complete.usage.total_input_tokens, some(eq(5000)));
                expect_that!(complete.usage.total_output_tokens, some(eq(2500)));
                expect_that!(complete.usage.total_cost, some(anything()));
                expect_that!(complete.usage.total_inference_time_ms, eq(1500.5));
            }
            other => panic!("Expected Complete event, got {other:?}"),
        }
    }
}
