use std::future::Future;
use std::sync::Arc;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::inference::types::{Latency, Usage};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Note: ModerationModelConfig and related types have been removed as moderation
// is now handled through the unified model system with endpoint capabilities

/// Represents the input for moderation requests
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModerationInput {
    Single(String),
    Batch(Vec<String>),
}

impl ModerationInput {
    /// Get all input strings as a vector
    pub fn as_vec(&self) -> Vec<&str> {
        match self {
            ModerationInput::Single(text) => vec![text],
            ModerationInput::Batch(texts) => texts.iter().map(|s| s.as_str()).collect(),
        }
    }

    /// Get the number of inputs
    pub fn len(&self) -> usize {
        match self {
            ModerationInput::Single(_) => 1,
            ModerationInput::Batch(texts) => texts.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Request structure for moderation API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationRequest {
    pub input: ModerationInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Categories that can be flagged by the moderation API
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModerationCategory {
    Hate,
    #[serde(rename = "hate/threatening")]
    HateThreatening,
    Harassment,
    #[serde(rename = "harassment/threatening")]
    HarassmentThreatening,
    SelfHarm,
    #[serde(rename = "self-harm/intent")]
    SelfHarmIntent,
    #[serde(rename = "self-harm/instructions")]
    SelfHarmInstructions,
    Sexual,
    #[serde(rename = "sexual/minors")]
    SexualMinors,
    Violence,
    #[serde(rename = "violence/graphic")]
    ViolenceGraphic,
}

impl ModerationCategory {
    /// Get all category names
    pub fn all() -> &'static [ModerationCategory] {
        &[
            ModerationCategory::Hate,
            ModerationCategory::HateThreatening,
            ModerationCategory::Harassment,
            ModerationCategory::HarassmentThreatening,
            ModerationCategory::SelfHarm,
            ModerationCategory::SelfHarmIntent,
            ModerationCategory::SelfHarmInstructions,
            ModerationCategory::Sexual,
            ModerationCategory::SexualMinors,
            ModerationCategory::Violence,
            ModerationCategory::ViolenceGraphic,
        ]
    }

    /// Get the string representation of the category
    pub fn as_str(&self) -> &'static str {
        match self {
            ModerationCategory::Hate => "hate",
            ModerationCategory::HateThreatening => "hate/threatening",
            ModerationCategory::Harassment => "harassment",
            ModerationCategory::HarassmentThreatening => "harassment/threatening",
            ModerationCategory::SelfHarm => "self-harm",
            ModerationCategory::SelfHarmIntent => "self-harm/intent",
            ModerationCategory::SelfHarmInstructions => "self-harm/instructions",
            ModerationCategory::Sexual => "sexual",
            ModerationCategory::SexualMinors => "sexual/minors",
            ModerationCategory::Violence => "violence",
            ModerationCategory::ViolenceGraphic => "violence/graphic",
        }
    }
}

/// Categories flagged by the moderation API
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModerationCategories {
    pub hate: bool,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    pub harassment: bool,
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: bool,
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: bool,
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: bool,
    pub sexual: bool,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
    pub violence: bool,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,
}

/// Confidence scores for each moderation category
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModerationCategoryScores {
    pub hate: f32,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f32,
    pub harassment: f32,
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: f32,
    #[serde(rename = "self-harm")]
    pub self_harm: f32,
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: f32,
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: f32,
    pub sexual: f32,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f32,
    pub violence: f32,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f32,
}

/// Result for a single text input
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModerationResult {
    pub flagged: bool,
    pub categories: ModerationCategories,
    pub category_scores: ModerationCategoryScores,
}

/// Provider-specific moderation response
#[derive(Debug, Serialize)]
pub struct ModerationProviderResponse {
    pub id: Uuid,
    pub input: ModerationInput,
    pub results: Vec<ModerationResult>,
    pub created: u64,
    pub model: String,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

/// Full moderation response
#[derive(Debug, Serialize)]
pub struct ModerationResponse {
    pub id: Uuid,
    pub input: ModerationInput,
    pub results: Vec<ModerationResult>,
    pub created: u64,
    pub model: String,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub moderation_provider_name: Arc<str>,
    pub cached: bool,
}

impl ModerationResponse {
    pub fn new(
        moderation_provider_response: ModerationProviderResponse,
        moderation_provider_name: Arc<str>,
    ) -> Self {
        Self {
            id: moderation_provider_response.id,
            input: moderation_provider_response.input,
            results: moderation_provider_response.results,
            created: moderation_provider_response.created,
            model: moderation_provider_response.model,
            raw_request: moderation_provider_response.raw_request,
            raw_response: moderation_provider_response.raw_response,
            usage: moderation_provider_response.usage,
            latency: moderation_provider_response.latency,
            moderation_provider_name,
            cached: false,
        }
    }
}

/// Trait for providers that support moderation
pub trait ModerationProvider {
    fn moderate(
        &self,
        request: &ModerationRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> impl Future<Output = Result<ModerationProviderResponse, Error>> + Send;
}

// Note: ModerationProviderConfig has been removed as moderation
// is now handled through the unified model system

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moderation_input_single() {
        let input = ModerationInput::Single("test text".to_string());
        assert_eq!(input.len(), 1);
        assert!(!input.is_empty());
        assert_eq!(input.as_vec(), vec!["test text"]);
    }

    #[test]
    fn test_moderation_input_batch() {
        let input = ModerationInput::Batch(vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
        ]);
        assert_eq!(input.len(), 3);
        assert!(!input.is_empty());
        assert_eq!(input.as_vec(), vec!["text1", "text2", "text3"]);
    }

    #[test]
    fn test_moderation_input_empty_batch() {
        let input = ModerationInput::Batch(vec![]);
        assert_eq!(input.len(), 0);
        assert!(input.is_empty());
        assert!(input.as_vec().is_empty());
    }

    #[test]
    fn test_moderation_category_str_conversion() {
        assert_eq!(ModerationCategory::Hate.as_str(), "hate");
        assert_eq!(
            ModerationCategory::HateThreatening.as_str(),
            "hate/threatening"
        );
        assert_eq!(ModerationCategory::Harassment.as_str(), "harassment");
        assert_eq!(
            ModerationCategory::HarassmentThreatening.as_str(),
            "harassment/threatening"
        );
        assert_eq!(ModerationCategory::SelfHarm.as_str(), "self-harm");
        assert_eq!(
            ModerationCategory::SelfHarmIntent.as_str(),
            "self-harm/intent"
        );
        assert_eq!(
            ModerationCategory::SelfHarmInstructions.as_str(),
            "self-harm/instructions"
        );
        assert_eq!(ModerationCategory::Sexual.as_str(), "sexual");
        assert_eq!(ModerationCategory::SexualMinors.as_str(), "sexual/minors");
        assert_eq!(ModerationCategory::Violence.as_str(), "violence");
        assert_eq!(
            ModerationCategory::ViolenceGraphic.as_str(),
            "violence/graphic"
        );
    }

    #[test]
    fn test_moderation_category_all() {
        let all_categories = ModerationCategory::all();
        assert_eq!(all_categories.len(), 11);
        assert!(all_categories.contains(&ModerationCategory::Hate));
        assert!(all_categories.contains(&ModerationCategory::HateThreatening));
        assert!(all_categories.contains(&ModerationCategory::Harassment));
        assert!(all_categories.contains(&ModerationCategory::HarassmentThreatening));
        assert!(all_categories.contains(&ModerationCategory::SelfHarm));
        assert!(all_categories.contains(&ModerationCategory::SelfHarmIntent));
        assert!(all_categories.contains(&ModerationCategory::SelfHarmInstructions));
        assert!(all_categories.contains(&ModerationCategory::Sexual));
        assert!(all_categories.contains(&ModerationCategory::SexualMinors));
        assert!(all_categories.contains(&ModerationCategory::Violence));
        assert!(all_categories.contains(&ModerationCategory::ViolenceGraphic));
    }

    #[test]
    fn test_moderation_categories_default() {
        let categories = ModerationCategories::default();
        assert!(!categories.hate);
        assert!(!categories.hate_threatening);
        assert!(!categories.harassment);
        assert!(!categories.harassment_threatening);
        assert!(!categories.self_harm);
        assert!(!categories.self_harm_intent);
        assert!(!categories.self_harm_instructions);
        assert!(!categories.sexual);
        assert!(!categories.sexual_minors);
        assert!(!categories.violence);
        assert!(!categories.violence_graphic);
    }

    #[test]
    fn test_moderation_category_scores_default() {
        let scores = ModerationCategoryScores::default();
        assert_eq!(scores.hate, 0.0);
        assert_eq!(scores.hate_threatening, 0.0);
        assert_eq!(scores.harassment, 0.0);
        assert_eq!(scores.harassment_threatening, 0.0);
        assert_eq!(scores.self_harm, 0.0);
        assert_eq!(scores.self_harm_intent, 0.0);
        assert_eq!(scores.self_harm_instructions, 0.0);
        assert_eq!(scores.sexual, 0.0);
        assert_eq!(scores.sexual_minors, 0.0);
        assert_eq!(scores.violence, 0.0);
        assert_eq!(scores.violence_graphic, 0.0);
    }

    // Tests for ModerationModelConfig have been removed as moderation
    // is now handled through the unified model system
}
