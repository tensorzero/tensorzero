use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use crate::cache::{
    moderation_cache_lookup, start_cache_write, CacheData, ModerationCacheData,
    ModerationModelProviderRequest,
};
use crate::config_parser::ProviderTypesConfig;
use crate::endpoints::inference::InferenceClients;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::openai::OpenAIProvider;
use crate::inference::types::{current_timestamp, Latency, Usage};
use crate::model::ProviderConfig;
use crate::model::UninitializedProviderConfig;
use crate::model_table::BaseModelTable;
use crate::model_table::ShorthandModelConfig;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyProvider;

pub type ModerationModelTable = BaseModelTable<ModerationModelConfig>;

impl ShorthandModelConfig for ModerationModelConfig {
    const SHORTHAND_MODEL_PREFIXES: &[&str] = &["openai::"];
    const MODEL_TYPE: &str = "Moderation model";

    async fn from_shorthand(provider_type: &str, model_name: &str) -> Result<Self, Error> {
        let model_name = model_name.to_string();
        let provider_config = match provider_type {
            "openai" => {
                ModerationProviderConfig::OpenAI(OpenAIProvider::new(model_name, None, None)?)
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            "dummy" => ModerationProviderConfig::Dummy(DummyProvider::new(model_name, None)?),
            _ => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!("Invalid provider type for moderation: {provider_type}"),
                }));
            }
        };
        Ok(ModerationModelConfig {
            routing: vec![provider_type.to_string().into()],
            providers: HashMap::from([(provider_type.to_string().into(), provider_config)]),
        })
    }

    fn validate(&self, _key: &str) -> Result<(), Error> {
        // Credentials are validated during deserialization
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedModerationModelConfig {
    pub routing: Vec<Arc<str>>,
    pub providers: HashMap<Arc<str>, UninitializedModerationProviderConfig>,
}

impl UninitializedModerationModelConfig {
    pub fn load(
        self,
        provider_types: &ProviderTypesConfig,
    ) -> Result<ModerationModelConfig, Error> {
        let providers = self
            .providers
            .into_iter()
            .map(|(name, config)| {
                let provider_config = config.load(provider_types)?;
                Ok((name, provider_config))
            })
            .collect::<Result<HashMap<_, _>, Error>>()?;
        Ok(ModerationModelConfig {
            routing: self.routing,
            providers,
        })
    }
}

#[derive(Debug)]
pub struct ModerationModelConfig {
    pub routing: Vec<Arc<str>>,
    pub providers: HashMap<Arc<str>, ModerationProviderConfig>,
}

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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn from_cache(
        cache_lookup: CacheData<ModerationCacheData>,
        request: &ModerationModelProviderRequest,
    ) -> Self {
        let cache_data = cache_lookup.output;
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            input: request.request.input.clone(),
            results: cache_data.results,
            model: "cached".to_string(), // We don't store the model in cache
            raw_request: cache_lookup.raw_request,
            raw_response: cache_lookup.raw_response,
            usage: Usage {
                input_tokens: cache_lookup.input_tokens,
                output_tokens: cache_lookup.output_tokens,
            },
            latency: Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(0),
            },
            moderation_provider_name: Arc::from(request.provider_name),
            cached: true,
        }
    }
}

/// Request context for moderation providers
#[derive(Debug, Clone, Copy)]
pub struct ModerationProviderRequest<'request> {
    pub request: &'request ModerationRequest,
    pub model_name: &'request str,
    pub provider_name: &'request str,
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

/// Moderation provider configuration
#[derive(Debug)]
pub enum ModerationProviderConfig {
    OpenAI(OpenAIProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

/// Uninitialized moderation provider configuration
#[derive(Debug, Deserialize)]
pub struct UninitializedModerationProviderConfig {
    #[serde(flatten)]
    config: UninitializedProviderConfig,
}

impl UninitializedModerationProviderConfig {
    pub fn load(
        self,
        provider_types: &ProviderTypesConfig,
    ) -> Result<ModerationProviderConfig, Error> {
        let provider_config = self.config.load(provider_types)?;
        Ok(match provider_config {
            ProviderConfig::OpenAI(provider) => ModerationProviderConfig::OpenAI(provider),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => ModerationProviderConfig::Dummy(provider),
            _ => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Unsupported provider config for moderation: {provider_config:?}"
                    ),
                }));
            }
        })
    }
}

impl ModerationProvider for ModerationProviderConfig {
    async fn moderate(
        &self,
        request: &ModerationRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<ModerationProviderResponse, Error> {
        match self {
            ModerationProviderConfig::OpenAI(provider) => {
                provider.moderate(request, client, dynamic_api_keys).await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ModerationProviderConfig::Dummy(provider) => {
                provider.moderate(request, client, dynamic_api_keys).await
            }
        }
    }
}

/// Handle moderation request with caching support
#[instrument(skip(request, clients, credentials, table_config))]
pub async fn handle_moderation_request(
    mut request: ModerationRequest,
    clients: &InferenceClients<'_>,
    credentials: &InferenceCredentials,
    model_name: &str,
    table_config: &ModerationModelTable,
) -> Result<ModerationResponse, Error> {
    // Get the model configuration
    let model_config = table_config.get(model_name).await?.ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("Moderation model '{model_name}' not found"),
        })
    })?;

    // Try each provider in the routing order
    let mut provider_errors = HashMap::new();

    for provider_name in &model_config.routing {
        let provider_config = match model_config.providers.get(provider_name) {
            Some(config) => config,
            None => {
                let error = Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                });
                provider_errors.insert(provider_name.to_string(), error);
                continue;
            }
        };

        // Set model if not specified
        if request.model.is_none() {
            request.model = Some(match provider_config {
                ModerationProviderConfig::OpenAI(_) => "text-moderation-latest".to_string(),
                #[cfg(any(test, feature = "e2e_tests"))]
                ModerationProviderConfig::Dummy(_) => "dummy-moderation".to_string(),
            });
        }

        let _provider_request = ModerationProviderRequest {
            request: &request,
            model_name,
            provider_name,
        };

        // Check cache first
        let cache_key = ModerationModelProviderRequest {
            request: &request,
            model_name,
            provider_name,
        };
        let cache_result = if clients.cache_options.enabled.read() {
            moderation_cache_lookup(
                clients.clickhouse_connection_info,
                &cache_key,
                clients.cache_options.max_age_s,
            )
            .await
            .ok()
            .flatten()
        } else {
            None
        };

        if let Some(cache_response) = cache_result {
            return Ok(cache_response);
        }

        // Make the moderation request
        match provider_config
            .moderate(&request, clients.http_client, credentials)
            .await
        {
            Ok(provider_response) => {
                let response = ModerationResponse::new(provider_response, provider_name.clone());

                // Cache the response
                if clients.cache_options.enabled.write() {
                    let cache_data = ModerationCacheData {
                        results: response.results.clone(),
                    };

                    let _ = start_cache_write(
                        clients.clickhouse_connection_info,
                        cache_key.get_cache_key()?,
                        cache_data,
                        &response.raw_request,
                        &response.raw_response,
                        &response.usage,
                        None,
                    );
                }

                return Ok(response);
            }
            Err(e) => {
                // Log error and save for reporting
                tracing::warn!(
                    "Moderation request failed for provider {}: {}",
                    provider_name,
                    e
                );
                provider_errors.insert(provider_name.to_string(), e);
                continue;
            }
        }
    }

    Err(Error::new(ErrorDetails::ModelProvidersExhausted {
        provider_errors,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
        assert_eq!(ModerationCategory::HateThreatening.as_str(), "hate/threatening");
        assert_eq!(ModerationCategory::Harassment.as_str(), "harassment");
        assert_eq!(ModerationCategory::HarassmentThreatening.as_str(), "harassment/threatening");
        assert_eq!(ModerationCategory::SelfHarm.as_str(), "self-harm");
        assert_eq!(ModerationCategory::SelfHarmIntent.as_str(), "self-harm/intent");
        assert_eq!(ModerationCategory::SelfHarmInstructions.as_str(), "self-harm/instructions");
        assert_eq!(ModerationCategory::Sexual.as_str(), "sexual");
        assert_eq!(ModerationCategory::SexualMinors.as_str(), "sexual/minors");
        assert_eq!(ModerationCategory::Violence.as_str(), "violence");
        assert_eq!(ModerationCategory::ViolenceGraphic.as_str(), "violence/graphic");
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

    #[test]
    fn test_moderation_response_from_cache() {
        let cache_lookup = CacheData {
            output: ModerationCacheData {
                results: vec![ModerationResult {
                    flagged: true,
                    categories: ModerationCategories {
                        hate: true,
                        ..Default::default()
                    },
                    category_scores: ModerationCategoryScores {
                        hate: 0.95,
                        ..Default::default()
                    },
                }],
            },
            raw_request: "test request".to_string(),
            raw_response: "test response".to_string(),
            input_tokens: Some(10),
            output_tokens: Some(5),
        };

        let request = ModerationRequest {
            input: ModerationInput::Single("test".to_string()),
            model: Some("test-model".to_string()),
        };

        let provider_request = ModerationModelProviderRequest {
            request: &request,
            model_name: "test-model",
            provider_name: "test-provider",
        };

        let response = ModerationResponse::from_cache(cache_lookup, &provider_request);

        assert!(response.cached);
        assert_eq!(response.model, "cached");
        assert_eq!(response.moderation_provider_name.as_ref(), "test-provider");
        assert_eq!(response.results.len(), 1);
        assert!(response.results[0].flagged);
        assert!(response.results[0].categories.hate);
        assert_eq!(response.results[0].category_scores.hate, 0.95);
    }

    #[tokio::test]
    async fn test_shorthand_model_config_openai() {
        let config = ModerationModelConfig::from_shorthand("openai", "text-moderation-latest")
            .await
            .expect("Failed to create OpenAI config");

        assert_eq!(config.routing.len(), 1);
        assert_eq!(config.routing[0].as_ref(), "openai");
        assert_eq!(config.providers.len(), 1);
        assert!(config.providers.contains_key("openai"));
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    #[tokio::test]
    async fn test_shorthand_model_config_dummy() {
        let config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
            .await
            .expect("Failed to create dummy config");

        assert_eq!(config.routing.len(), 1);
        assert_eq!(config.routing[0].as_ref(), "dummy");
        assert_eq!(config.providers.len(), 1);
        assert!(config.providers.contains_key("dummy"));
    }

    #[tokio::test]
    async fn test_shorthand_model_config_invalid_provider() {
        let result = ModerationModelConfig::from_shorthand("invalid", "model");
        assert!(result.await.is_err());
    }
}