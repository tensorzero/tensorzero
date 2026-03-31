use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{StoredEvaluatorConfig, StoredPromptRef, StoredVariantRef};

/// Stored in `function_versions_config.config`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum StoredFunctionConfig {
    Chat(StoredChatFunctionConfig),
    Json(StoredJsonFunctionConfig),
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredChatFunctionConfig {
    pub variants: Option<HashMap<String, StoredVariantRef>>,
    pub system_schema: Option<StoredPromptRef>,
    pub user_schema: Option<StoredPromptRef>,
    pub assistant_schema: Option<StoredPromptRef>,
    pub schemas: Option<HashMap<String, StoredPromptRef>>,
    pub tools: Option<Vec<String>>,
    pub tool_choice: Option<StoredToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub description: Option<String>,
    pub experimentation: Option<StoredExperimentationConfigWithNamespaces>,
    pub evaluators: Option<HashMap<String, StoredEvaluatorConfig>>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredJsonFunctionConfig {
    pub variants: Option<HashMap<String, StoredVariantRef>>,
    pub system_schema: Option<StoredPromptRef>,
    pub user_schema: Option<StoredPromptRef>,
    pub assistant_schema: Option<StoredPromptRef>,
    pub schemas: Option<HashMap<String, StoredPromptRef>>,
    pub output_schema: Option<StoredPromptRef>,
    pub description: Option<String>,
    pub experimentation: Option<StoredExperimentationConfigWithNamespaces>,
    pub evaluators: Option<HashMap<String, StoredEvaluatorConfig>>,
}

// --- ToolChoice ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoredToolChoice {
    None,
    Auto,
    Required,
    Specific(String),
}

// --- Experimentation ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredExperimentationConfigWithNamespaces {
    pub base: StoredExperimentationConfig,
    pub namespaces: Option<HashMap<String, StoredExperimentationConfig>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredExperimentationConfig {
    Static(StoredStaticExperimentationConfig),
    Adaptive(StoredAdaptiveExperimentationConfig),
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredStaticExperimentationConfig {
    /// Always stored as a map of variant name → weight.
    pub candidate_variants: Option<HashMap<String, f64>>,
    pub fallback_variants: Option<Vec<String>>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredAdaptiveExperimentationConfig {
    pub algorithm: Option<StoredAdaptiveExperimentationAlgorithm>,
    pub metric: String,
    pub candidate_variants: Option<Vec<String>>,
    pub fallback_variants: Option<Vec<String>>,
    pub min_samples_per_variant: Option<u64>,
    pub delta: Option<f64>,
    pub epsilon: Option<f64>,
    pub update_period_s: Option<u64>,
    pub min_prob: Option<f64>,
    pub max_samples_per_variant: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredAdaptiveExperimentationAlgorithm {
    TrackAndStop,
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unnecessary_wraps,
        reason = "`#[gtest]` tests return `googletest::Result<()>`"
    )]

    use std::collections::HashMap;

    use googletest::prelude::*;
    use uuid::Uuid;

    use super::*;
    use crate::{StoredPromptRef, StoredVariantRef};

    fn make_prompt_ref() -> StoredPromptRef {
        StoredPromptRef {
            prompt_template_version_id: Uuid::now_v7(),
            template_key: "tpl".to_string(),
        }
    }

    fn make_variant_ref() -> StoredVariantRef {
        StoredVariantRef {
            variant_version_id: Uuid::now_v7(),
        }
    }

    fn assert_round_trip(config: StoredFunctionConfig) {
        let serialized = serde_json::to_value(config.clone()).expect("serialize stored config");
        let round_tripped: StoredFunctionConfig =
            serde_json::from_value(serialized).expect("deserialize stored config");
        expect_that!(&round_tripped, eq(&config));
    }

    #[gtest]
    fn chat_function_full_round_trip() -> Result<()> {
        let config = StoredFunctionConfig::Chat(StoredChatFunctionConfig {
            variants: Some(HashMap::from([
                ("v1".to_string(), make_variant_ref()),
                ("v2".to_string(), make_variant_ref()),
            ])),
            system_schema: Some(make_prompt_ref()),
            user_schema: Some(make_prompt_ref()),
            assistant_schema: None,
            schemas: Some(HashMap::from([("custom".to_string(), make_prompt_ref())])),
            tools: Some(vec!["search".to_string(), "calculate".to_string()]),
            tool_choice: Some(StoredToolChoice::Auto),
            parallel_tool_calls: Some(true),
            description: Some("A chat function".to_string()),
            experimentation: Some(StoredExperimentationConfigWithNamespaces {
                base: StoredExperimentationConfig::Static(StoredStaticExperimentationConfig {
                    candidate_variants: Some(HashMap::from([
                        ("v1".to_string(), 0.7),
                        ("v2".to_string(), 0.3),
                    ])),
                    fallback_variants: Some(vec!["v1".to_string()]),
                }),
                namespaces: None,
            }),
            evaluators: Some(HashMap::from([(
                "exact".to_string(),
                StoredEvaluatorConfig::ExactMatch(crate::StoredExactMatchConfig { cutoff: None }),
            )])),
        });
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn json_function_round_trip() -> Result<()> {
        let config = StoredFunctionConfig::Json(StoredJsonFunctionConfig {
            variants: Some(HashMap::from([("v1".to_string(), make_variant_ref())])),
            system_schema: None,
            user_schema: Some(make_prompt_ref()),
            assistant_schema: None,
            schemas: None,
            output_schema: Some(make_prompt_ref()),
            description: None,
            experimentation: None,
            evaluators: None,
        });
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn minimal_chat_function_round_trip() -> Result<()> {
        let config = StoredFunctionConfig::Chat(StoredChatFunctionConfig {
            variants: Some(HashMap::from([("v1".to_string(), make_variant_ref())])),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            schemas: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            description: None,
            experimentation: None,
            evaluators: None,
        });
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn function_with_adaptive_experimentation_round_trip() -> Result<()> {
        let config = StoredFunctionConfig::Chat(StoredChatFunctionConfig {
            variants: Some(HashMap::from([
                ("v1".to_string(), make_variant_ref()),
                ("v2".to_string(), make_variant_ref()),
            ])),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            schemas: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            description: None,
            experimentation: Some(StoredExperimentationConfigWithNamespaces {
                base: StoredExperimentationConfig::Adaptive(StoredAdaptiveExperimentationConfig {
                    algorithm: Some(StoredAdaptiveExperimentationAlgorithm::TrackAndStop),
                    metric: "quality".to_string(),
                    candidate_variants: Some(vec!["v1".to_string(), "v2".to_string()]),
                    fallback_variants: Some(vec!["v1".to_string()]),
                    min_samples_per_variant: Some(10),
                    delta: Some(0.05),
                    epsilon: Some(0.0),
                    update_period_s: Some(60),
                    min_prob: Some(0.01),
                    max_samples_per_variant: Some(1000),
                }),
                namespaces: Some(HashMap::from([(
                    "premium".to_string(),
                    StoredExperimentationConfig::Static(StoredStaticExperimentationConfig {
                        candidate_variants: Some(HashMap::from([("v1".to_string(), 1.0)])),
                        fallback_variants: None,
                    }),
                )])),
            }),
            evaluators: None,
        });
        assert_round_trip(config);
        Ok(())
    }

    #[gtest]
    fn function_with_tool_choice_specific_round_trip() -> Result<()> {
        let config = StoredFunctionConfig::Chat(StoredChatFunctionConfig {
            variants: Some(HashMap::from([("v1".to_string(), make_variant_ref())])),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            schemas: None,
            tools: Some(vec!["search".to_string()]),
            tool_choice: Some(StoredToolChoice::Specific("search".to_string())),
            parallel_tool_calls: Some(false),
            description: None,
            experimentation: None,
            evaluators: None,
        });
        assert_round_trip(config);
        Ok(())
    }
}
