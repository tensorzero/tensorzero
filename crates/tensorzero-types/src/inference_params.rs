//! Inference parameter types shared across crates.
//!
//! These types were originally in `tensorzero-core::endpoints::inference` but are
//! needed by providers, variants, inference, and db modules — creating circular
//! dependencies. Moving them here breaks those cycles.

use schemars::JsonSchema;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Credentials for inference requests, keyed by credential name.
pub type InferenceCredentials = HashMap<String, SecretString>;

/// Identifiers for an inference request.
#[derive(Copy, Clone, Debug)]
pub struct InferenceIds {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
}

/// Top-level struct for inference parameters.
/// We backfill these from the configs given in the variants used and ultimately write them to the database.
#[derive(ts_rs::TS)]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct InferenceParams {
    pub chat_completion: ChatCompletionInferenceParams,
}

#[derive(ts_rs::TS)]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionInferenceParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_mode: Option<JsonMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

impl ChatCompletionInferenceParams {
    #[expect(clippy::too_many_arguments)]
    pub fn backfill_with_variant_params(
        &mut self,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        seed: Option<u32>,
        top_p: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        stop_sequences: Option<Vec<String>>,
        inference_params_v2: ChatCompletionInferenceParamsV2,
    ) {
        if self.temperature.is_none() {
            self.temperature = temperature;
        }
        if self.max_tokens.is_none() {
            self.max_tokens = max_tokens;
        }
        if self.seed.is_none() {
            self.seed = seed;
        }
        if self.top_p.is_none() {
            self.top_p = top_p;
        }
        if self.presence_penalty.is_none() {
            self.presence_penalty = presence_penalty;
        }
        if self.frequency_penalty.is_none() {
            self.frequency_penalty = frequency_penalty;
        }
        if self.stop_sequences.is_none() {
            self.stop_sequences = stop_sequences;
        }
        let ChatCompletionInferenceParamsV2 {
            reasoning_effort,
            service_tier,
            thinking_budget_tokens,
            verbosity,
        } = inference_params_v2;

        if self.reasoning_effort.is_none() {
            self.reasoning_effort = reasoning_effort;
        }
        if self.service_tier.is_none() {
            self.service_tier = service_tier;
        }
        if self.thinking_budget_tokens.is_none() {
            self.thinking_budget_tokens = thinking_budget_tokens;
        }
        if self.verbosity.is_none() {
            self.verbosity = verbosity;
        }
    }
}

/// The V2 inference parameters — a transitional struct for parameters being
/// migrated to explicit per-provider handling.
#[derive(ts_rs::TS)]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[ts(export, optional_fields)]
pub struct ChatCompletionInferenceParamsV2 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

/// JSON mode for inference requests.
#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum JsonMode {
    Off,
    On,
    Strict,
    #[serde(alias = "implicit_tool")] // Legacy name (stored in CH --> permanent alias)
    Tool,
}

/// Service tier for inference requests.
///
/// Controls the priority and latency characteristics of the request.
/// Different providers map these values differently to their own service tiers.
#[derive(ts_rs::TS)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[ts(export)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    #[default]
    Auto,
    Default,
    Priority,
    Flex,
}

impl std::fmt::Display for ServiceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ServiceTier::Auto => "auto",
            ServiceTier::Default => "default",
            ServiceTier::Priority => "priority",
            ServiceTier::Flex => "flex",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_tier_serialization() {
        assert_eq!(
            serde_json::to_string(&ServiceTier::Auto).unwrap(),
            r#""auto""#
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Default).unwrap(),
            r#""default""#
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Priority).unwrap(),
            r#""priority""#
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Flex).unwrap(),
            r#""flex""#
        );
    }

    #[test]
    fn test_service_tier_deserialization() {
        assert_eq!(
            serde_json::from_str::<ServiceTier>(r#""auto""#).unwrap(),
            ServiceTier::Auto
        );
        assert_eq!(
            serde_json::from_str::<ServiceTier>(r#""default""#).unwrap(),
            ServiceTier::Default
        );
        assert_eq!(
            serde_json::from_str::<ServiceTier>(r#""priority""#).unwrap(),
            ServiceTier::Priority
        );
        assert_eq!(
            serde_json::from_str::<ServiceTier>(r#""flex""#).unwrap(),
            ServiceTier::Flex
        );
    }
}
