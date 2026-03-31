use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tensorzero_types::inference_params::{JsonMode, ServiceTier};
use uuid::Uuid;

use crate::{
    StoredExtraBodyConfig, StoredExtraHeadersConfig, StoredPromptRef, StoredRetryConfig,
    StoredTimeoutsConfig,
};

/// Reference to a `variant_versions_config` row.
/// Replaces inline variant configs in stored function config.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredVariantRef {
    pub variant_version_id: Uuid,
}

/// Wrapper stored in `variant_versions_config.config`.
/// Uses an explicit `variant` field instead of `#[serde(flatten)]`.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredVariantVersionConfig {
    pub variant: StoredVariantConfig,
    pub timeouts: Option<StoredTimeoutsConfig>,
    pub namespace: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config")]
pub enum StoredVariantConfig {
    #[serde(rename = "chat_completion")]
    ChatCompletion(StoredChatCompletionVariantConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(StoredBestOfNVariantConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfN(StoredMixtureOfNVariantConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(StoredDiclVariantConfig),
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(StoredChatCompletionVariantConfig),
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredChatCompletionVariantConfig {
    pub weight: Option<f64>,
    pub model: Arc<str>,
    pub system_template: Option<StoredPromptRef>,
    pub user_template: Option<StoredPromptRef>,
    pub assistant_template: Option<StoredPromptRef>,
    pub input_wrappers: Option<StoredInputWrappers>,
    pub templates: Option<HashMap<String, StoredPromptRef>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub json_mode: Option<JsonMode>,
    pub stop_sequences: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<ServiceTier>,
    pub thinking_budget_tokens: Option<i32>,
    pub verbosity: Option<String>,
    pub retries: Option<StoredRetryConfig>,
    pub extra_body: Option<StoredExtraBodyConfig>,
    pub extra_headers: Option<StoredExtraHeadersConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredInputWrappers {
    pub user: Option<StoredPromptRef>,
    pub assistant: Option<StoredPromptRef>,
    pub system: Option<StoredPromptRef>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredBestOfNVariantConfig {
    pub weight: Option<f64>,
    pub timeout_s: Option<f64>,
    pub candidates: Option<Vec<String>>,
    pub evaluator: StoredChatCompletionVariantConfig,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredMixtureOfNVariantConfig {
    pub weight: Option<f64>,
    pub timeout_s: Option<f64>,
    pub candidates: Option<Vec<String>>,
    pub fuser: StoredChatCompletionVariantConfig,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredDiclVariantConfig {
    pub weight: Option<f64>,
    pub embedding_model: String,
    pub k: u32,
    pub model: String,
    pub system_instructions: Option<StoredPromptRef>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub json_mode: Option<JsonMode>,
    pub stop_sequences: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub thinking_budget_tokens: Option<i32>,
    pub verbosity: Option<String>,
    pub max_distance: Option<f32>,
    pub retries: Option<StoredRetryConfig>,
    pub extra_body: Option<StoredExtraBodyConfig>,
    pub extra_headers: Option<StoredExtraHeadersConfig>,
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unnecessary_wraps,
        reason = "`#[gtest]` tests return `googletest::Result<()>`"
    )]

    use googletest::prelude::*;
    use uuid::Uuid;

    use super::*;
    use crate::{StoredNonStreamingTimeouts, StoredStreamingTimeouts};

    fn make_prompt_ref() -> StoredPromptRef {
        StoredPromptRef {
            prompt_template_version_id: Uuid::now_v7(),
            template_key: "tpl".to_string(),
        }
    }

    fn make_chat_completion() -> StoredChatCompletionVariantConfig {
        StoredChatCompletionVariantConfig {
            weight: Some(1.0),
            model: "gpt-4".into(),
            system_template: Some(make_prompt_ref()),
            user_template: Some(make_prompt_ref()),
            assistant_template: None,
            input_wrappers: None,
            templates: None,
            temperature: Some(0.7),
            top_p: None,
            max_tokens: Some(1024),
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            json_mode: Some(JsonMode::Off),
            stop_sequences: None,
            reasoning_effort: None,
            service_tier: None,
            thinking_budget_tokens: None,
            verbosity: None,
            retries: Some(StoredRetryConfig {
                num_retries: 3,
                max_delay_s: 5.0,
            }),
            extra_body: None,
            extra_headers: None,
        }
    }

    fn assert_round_trip(config: StoredVariantVersionConfig) {
        let serialized = serde_json::to_value(config.clone()).expect("serialize stored config");
        let round_tripped: StoredVariantVersionConfig =
            serde_json::from_value(serialized).expect("deserialize stored config");
        expect_that!(&round_tripped, eq(&config));
    }

    #[gtest]
    fn chat_completion_round_trip() -> Result<()> {
        let config = StoredVariantVersionConfig {
            variant: StoredVariantConfig::ChatCompletion(make_chat_completion()),
            timeouts: Some(StoredTimeoutsConfig {
                non_streaming: Some(StoredNonStreamingTimeouts {
                    total_ms: Some(5000),
                }),
                streaming: Some(StoredStreamingTimeouts {
                    ttft_ms: Some(1000),
                    total_ms: Some(10000),
                }),
            }),
            namespace: Some("default".to_string()),
        };
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn chat_completion_minimal_round_trip() -> Result<()> {
        let config = StoredVariantVersionConfig {
            variant: StoredVariantConfig::ChatCompletion(StoredChatCompletionVariantConfig {
                weight: None,
                model: "gpt-3.5-turbo".into(),
                system_template: None,
                user_template: None,
                assistant_template: None,
                input_wrappers: None,
                templates: None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                presence_penalty: None,
                frequency_penalty: None,
                seed: None,
                json_mode: None,
                stop_sequences: None,
                reasoning_effort: None,
                service_tier: None,
                thinking_budget_tokens: None,
                verbosity: None,
                retries: None,
                extra_body: None,
                extra_headers: None,
            }),
            timeouts: None,
            namespace: None,
        };
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn best_of_n_round_trip() -> Result<()> {
        let config = StoredVariantVersionConfig {
            variant: StoredVariantConfig::BestOfNSampling(StoredBestOfNVariantConfig {
                weight: Some(1.0),
                timeout_s: Some(30.0),
                candidates: Some(vec!["c1".to_string(), "c2".to_string()]),
                evaluator: make_chat_completion(),
            }),
            timeouts: None,
            namespace: None,
        };
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn mixture_of_n_round_trip() -> Result<()> {
        let config = StoredVariantVersionConfig {
            variant: StoredVariantConfig::MixtureOfN(StoredMixtureOfNVariantConfig {
                weight: Some(1.0),
                timeout_s: None,
                candidates: Some(vec!["c1".to_string()]),
                fuser: make_chat_completion(),
            }),
            timeouts: None,
            namespace: None,
        };
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn dicl_round_trip() -> Result<()> {
        let config = StoredVariantVersionConfig {
            variant: StoredVariantConfig::Dicl(StoredDiclVariantConfig {
                weight: Some(0.5),
                embedding_model: "text-embedding-ada-002".to_string(),
                k: 5,
                model: "gpt-4".to_string(),
                system_instructions: Some(make_prompt_ref()),
                temperature: Some(0.5),
                top_p: None,
                max_tokens: Some(512),
                presence_penalty: None,
                frequency_penalty: None,
                seed: None,
                json_mode: None,
                stop_sequences: None,
                reasoning_effort: None,
                thinking_budget_tokens: None,
                verbosity: None,
                max_distance: Some(0.8),
                retries: None,
                extra_body: None,
                extra_headers: None,
            }),
            timeouts: None,
            namespace: Some("experiment_ns".to_string()),
        };
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn chain_of_thought_round_trip() -> Result<()> {
        let config = StoredVariantVersionConfig {
            variant: StoredVariantConfig::ChainOfThought(make_chat_completion()),
            timeouts: None,
            namespace: None,
        };
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn chat_completion_with_templates_and_wrappers() -> Result<()> {
        let config = StoredVariantVersionConfig {
            variant: StoredVariantConfig::ChatCompletion(StoredChatCompletionVariantConfig {
                weight: Some(1.0),
                model: "gpt-4".into(),
                system_template: None,
                user_template: None,
                assistant_template: None,
                input_wrappers: Some(StoredInputWrappers {
                    user: Some(make_prompt_ref()),
                    assistant: None,
                    system: Some(make_prompt_ref()),
                }),
                templates: Some(HashMap::from([
                    ("custom_role".to_string(), make_prompt_ref()),
                    ("another_role".to_string(), make_prompt_ref()),
                ])),
                temperature: None,
                top_p: None,
                max_tokens: None,
                presence_penalty: None,
                frequency_penalty: None,
                seed: None,
                json_mode: Some(JsonMode::Strict),
                stop_sequences: Some(vec!["END".to_string()]),
                reasoning_effort: Some("high".to_string()),
                service_tier: Some(ServiceTier::Default),
                thinking_budget_tokens: Some(1000),
                verbosity: Some("verbose".to_string()),
                retries: None,
                extra_body: None,
                extra_headers: None,
            }),
            timeouts: None,
            namespace: None,
        };
        assert_round_trip(config);
        Ok(())
    }
}
