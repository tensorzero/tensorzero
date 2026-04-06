use crate::model::{CredentialLocation, CredentialLocationWithFallback};
use serde::{Deserialize, Serialize};
use tensorzero_stored_config::{
    StoredApiKeyDefaults, StoredFireworksProviderSFTConfig, StoredFireworksProviderTypeConfig,
    StoredGCPBatchConfigCloudStorage, StoredGCPBatchConfigType, StoredGCPCredentialDefaults,
    StoredGCPCredentialProviderTypeConfig, StoredGCPProviderSFTConfig,
    StoredGCPVertexGeminiProviderTypeConfig, StoredProviderTypesConfig,
    StoredSimpleProviderTypeConfig, StoredTogetherProviderSFTConfig,
    StoredTogetherProviderTypeConfig,
};

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTypesConfig {
    pub anthropic: Option<AnthropicProviderTypeConfig>,
    pub azure: Option<AzureProviderTypeConfig>,
    pub deepseek: Option<DeepSeekProviderTypeConfig>,
    pub fireworks: Option<FireworksProviderTypeConfig>,
    pub gcp_vertex_gemini: Option<GCPVertexGeminiProviderTypeConfig>,
    pub gcp_vertex_anthropic: Option<GCPVertexAnthropicProviderTypeConfig>,
    pub google_ai_studio_gemini: Option<GoogleAIStudioGeminiProviderTypeConfig>,
    pub groq: Option<GroqProviderTypeConfig>,
    pub hyperbolic: Option<HyperbolicProviderTypeConfig>,
    pub mistral: Option<MistralProviderTypeConfig>,
    pub openai: Option<OpenAIProviderTypeConfig>,
    pub openrouter: Option<OpenRouterProviderTypeConfig>,
    pub sglang: Option<SGLangProviderTypeConfig>,
    pub tgi: Option<TGIProviderTypeConfig>,
    pub together: Option<TogetherProviderTypeConfig>,
    pub vllm: Option<VLLMProviderTypeConfig>,
    pub xai: Option<XAIProviderTypeConfig>,
}

#[cfg(test)]
fn convert_simple_provider_type_config(
    defaults: &impl ApiKeyDefaultsConfig,
) -> StoredSimpleProviderTypeConfig {
    StoredSimpleProviderTypeConfig {
        defaults: Some(StoredApiKeyDefaults::from(defaults.api_key_location())),
    }
}

impl From<&CredentialLocationWithFallback> for StoredApiKeyDefaults {
    fn from(api_key_location: &CredentialLocationWithFallback) -> Self {
        StoredApiKeyDefaults {
            api_key_location: Some(api_key_location.into()),
        }
    }
}

impl From<&CredentialLocationWithFallback> for StoredGCPCredentialDefaults {
    fn from(credential_location: &CredentialLocationWithFallback) -> Self {
        StoredGCPCredentialDefaults {
            credential_location: Some(credential_location.into()),
        }
    }
}

impl From<&GCPBatchConfigType> for StoredGCPBatchConfigType {
    fn from(batch: &GCPBatchConfigType) -> Self {
        match batch {
            GCPBatchConfigType::None => StoredGCPBatchConfigType::None,
            GCPBatchConfigType::CloudStorage(config) => {
                StoredGCPBatchConfigType::CloudStorage(StoredGCPBatchConfigCloudStorage {
                    input_uri_prefix: config.input_uri_prefix.clone(),
                    output_uri_prefix: config.output_uri_prefix.clone(),
                })
            }
        }
    }
}

#[cfg(test)]
trait ApiKeyDefaultsConfig {
    fn api_key_location(&self) -> &CredentialLocationWithFallback;
}

#[cfg(test)]
macro_rules! impl_api_key_defaults_config {
    ($($defaults:ty),* $(,)?) => {
        $(
            impl ApiKeyDefaultsConfig for $defaults {
                fn api_key_location(&self) -> &CredentialLocationWithFallback {
                    &self.api_key_location
                }
            }
        )*
    };
}

#[cfg(test)]
impl_api_key_defaults_config!(
    AnthropicDefaults,
    AzureDefaults,
    DeepSeekDefaults,
    FireworksDefaults,
    GoogleAIStudioGeminiDefaults,
    GroqDefaults,
    HyperbolicDefaults,
    MistralDefaults,
    OpenAIDefaults,
    OpenRouterDefaults,
    SGLangDefaults,
    TGIDefaults,
    TogetherDefaults,
    VLLMDefaults,
    XAIDefaults,
);

// Anthropic

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct AnthropicProviderTypeConfig {
    pub defaults: Option<AnthropicDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AnthropicDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for AnthropicDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "ANTHROPIC_API_KEY".to_string(),
            )),
        }
    }
}

// Azure

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct AzureProviderTypeConfig {
    pub defaults: Option<AzureDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AzureDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for AzureDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "AZURE_API_KEY".to_string(),
            )),
        }
    }
}

// DeepSeek

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DeepSeekProviderTypeConfig {
    pub defaults: Option<DeepSeekDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DeepSeekDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for DeepSeekDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "DEEPSEEK_API_KEY".to_string(),
            )),
        }
    }
}

// Fireworks

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct FireworksProviderTypeConfig {
    pub sft: Option<FireworksSFTConfig>,
    pub defaults: Option<FireworksDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct FireworksSFTConfig {
    pub account_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FireworksDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for FireworksDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "FIREWORKS_API_KEY".to_string(),
            )),
        }
    }
}

// GCP Vertex Gemini

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct GCPVertexGeminiProviderTypeConfig {
    pub batch: Option<GCPBatchConfigType>,
    pub sft: Option<GCPSFTConfig>,
    pub defaults: Option<GCPDefaults>,
}

// GCP Vertex Anthropic

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GCPVertexAnthropicProviderTypeConfig {
    pub defaults: Option<GCPDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "storage_type", rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum GCPBatchConfigType {
    // In the future, we'll want to allow explicitly setting 'none' at the model provider level,
    // to override the global provider-types batch config.
    None,
    CloudStorage(GCPBatchConfigCloudStorage),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct GCPBatchConfigCloudStorage {
    pub input_uri_prefix: String,
    pub output_uri_prefix: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct GCPSFTConfig {
    pub project_id: String,
    pub region: String,
    pub bucket_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bucket_path_prefix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_account: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kms_key_name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GCPDefaults {
    pub credential_location: CredentialLocationWithFallback,
}

impl Default for GCPDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocationWithFallback::Single(
                CredentialLocation::PathFromEnv("GCP_VERTEX_CREDENTIALS_PATH".to_string()),
            ),
        }
    }
}

// Google AI Studio

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct GoogleAIStudioGeminiProviderTypeConfig {
    pub defaults: Option<GoogleAIStudioGeminiDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GoogleAIStudioGeminiDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for GoogleAIStudioGeminiDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "GOOGLE_AI_STUDIO_API_KEY".to_string(),
            )),
        }
    }
}

// Groq

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct GroqProviderTypeConfig {
    pub defaults: Option<GroqDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GroqDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for GroqDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "GROQ_API_KEY".to_string(),
            )),
        }
    }
}

// Hyperbolic

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct HyperbolicProviderTypeConfig {
    pub defaults: Option<HyperbolicDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HyperbolicDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for HyperbolicDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "HYPERBOLIC_API_KEY".to_string(),
            )),
        }
    }
}

// Mistral

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct MistralProviderTypeConfig {
    pub defaults: Option<MistralDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MistralDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for MistralDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "MISTRAL_API_KEY".to_string(),
            )),
        }
    }
}

// OpenAI

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct OpenAIProviderTypeConfig {
    pub defaults: Option<OpenAIDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAIDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for OpenAIDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "OPENAI_API_KEY".to_string(),
            )),
        }
    }
}

// Openrouter

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct OpenRouterProviderTypeConfig {
    pub defaults: Option<OpenRouterDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenRouterDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for OpenRouterDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "OPENROUTER_API_KEY".to_string(),
            )),
        }
    }
}

// SGLang

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct SGLangProviderTypeConfig {
    pub defaults: Option<SGLangDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SGLangDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for SGLangDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "SGLANG_API_KEY".to_string(),
            )),
        }
    }
}

// TGI

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct TGIProviderTypeConfig {
    pub defaults: Option<TGIDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TGIDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for TGIDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "TGI_API_KEY".to_string(),
            )),
        }
    }
}

// Together

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct TogetherProviderTypeConfig {
    pub sft: Option<TogetherSFTConfig>,
    pub defaults: Option<TogetherDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct TogetherSFTConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_project_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_api_token: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TogetherDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for TogetherDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "TOGETHER_API_KEY".to_string(),
            )),
        }
    }
}

// vLLM

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct VLLMProviderTypeConfig {
    pub defaults: Option<VLLMDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VLLMDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for VLLMDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "VLLM_API_KEY".to_string(),
            )),
        }
    }
}

// xAI

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct XAIProviderTypeConfig {
    pub defaults: Option<XAIDefaults>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct XAIDefaults {
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for XAIDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "XAI_API_KEY".to_string(),
            )),
        }
    }
}

impl From<StoredProviderTypesConfig> for ProviderTypesConfig {
    fn from(stored: StoredProviderTypesConfig) -> Self {
        ProviderTypesConfig {
            anthropic: stored.anthropic.map(Into::into),
            azure: stored.azure.map(Into::into),
            deepseek: stored.deepseek.map(Into::into),
            fireworks: stored.fireworks.map(Into::into),
            gcp_vertex_gemini: stored.gcp_vertex_gemini.map(Into::into),
            gcp_vertex_anthropic: stored.gcp_vertex_anthropic.map(Into::into),
            google_ai_studio_gemini: stored.google_ai_studio_gemini.map(Into::into),
            groq: stored.groq.map(Into::into),
            hyperbolic: stored.hyperbolic.map(Into::into),
            mistral: stored.mistral.map(Into::into),
            openai: stored.openai.map(Into::into),
            openrouter: stored.openrouter.map(Into::into),
            sglang: stored.sglang.map(Into::into),
            tgi: stored.tgi.map(Into::into),
            together: stored.together.map(Into::into),
            vllm: stored.vllm.map(Into::into),
            xai: stored.xai.map(Into::into),
        }
    }
}

fn convert_stored_api_key_defaults(
    stored: Option<StoredApiKeyDefaults>,
) -> Option<CredentialLocationWithFallback> {
    stored.and_then(|d| d.api_key_location.map(Into::into))
}

fn convert_stored_gcp_credential_defaults(
    stored: Option<StoredGCPCredentialDefaults>,
) -> Option<CredentialLocationWithFallback> {
    stored.and_then(|d| d.credential_location.map(Into::into))
}

// --- Simple provider types (api_key_location only) ---

macro_rules! impl_from_simple_provider_type {
    ($stored:ty => $target:ty, $defaults:ident) => {
        impl From<$stored> for $target {
            fn from(stored: $stored) -> Self {
                Self {
                    defaults: convert_stored_api_key_defaults(stored.defaults)
                        .map(|api_key_location| $defaults { api_key_location }),
                }
            }
        }
    };
}

impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => AnthropicProviderTypeConfig, AnthropicDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => AzureProviderTypeConfig, AzureDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => DeepSeekProviderTypeConfig, DeepSeekDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => GoogleAIStudioGeminiProviderTypeConfig, GoogleAIStudioGeminiDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => GroqProviderTypeConfig, GroqDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => HyperbolicProviderTypeConfig, HyperbolicDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => MistralProviderTypeConfig, MistralDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => OpenAIProviderTypeConfig, OpenAIDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => OpenRouterProviderTypeConfig, OpenRouterDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => SGLangProviderTypeConfig, SGLangDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => TGIProviderTypeConfig, TGIDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => VLLMProviderTypeConfig, VLLMDefaults);
impl_from_simple_provider_type!(StoredSimpleProviderTypeConfig => XAIProviderTypeConfig, XAIDefaults);

// --- Fireworks ---

impl From<StoredFireworksProviderTypeConfig> for FireworksProviderTypeConfig {
    fn from(stored: StoredFireworksProviderTypeConfig) -> Self {
        Self {
            sft: stored.sft.map(Into::into),
            defaults: convert_stored_api_key_defaults(stored.defaults)
                .map(|api_key_location| FireworksDefaults { api_key_location }),
        }
    }
}

impl From<StoredFireworksProviderSFTConfig> for FireworksSFTConfig {
    fn from(stored: StoredFireworksProviderSFTConfig) -> Self {
        Self {
            account_id: stored.account_id,
        }
    }
}

// --- GCP Vertex Anthropic ---

impl From<StoredGCPCredentialProviderTypeConfig> for GCPVertexAnthropicProviderTypeConfig {
    fn from(stored: StoredGCPCredentialProviderTypeConfig) -> Self {
        Self {
            defaults: convert_stored_gcp_credential_defaults(stored.defaults).map(
                |credential_location| GCPDefaults {
                    credential_location,
                },
            ),
        }
    }
}

// --- GCP Vertex Gemini ---

impl From<StoredGCPVertexGeminiProviderTypeConfig> for GCPVertexGeminiProviderTypeConfig {
    fn from(stored: StoredGCPVertexGeminiProviderTypeConfig) -> Self {
        Self {
            batch: stored.batch.map(Into::into),
            sft: stored.sft.map(Into::into),
            defaults: convert_stored_gcp_credential_defaults(stored.defaults).map(
                |credential_location| GCPDefaults {
                    credential_location,
                },
            ),
        }
    }
}

impl From<StoredGCPBatchConfigType> for GCPBatchConfigType {
    fn from(stored: StoredGCPBatchConfigType) -> Self {
        match stored {
            StoredGCPBatchConfigType::None => Self::None,
            StoredGCPBatchConfigType::CloudStorage(cs) => Self::CloudStorage(cs.into()),
        }
    }
}

impl From<StoredGCPBatchConfigCloudStorage> for GCPBatchConfigCloudStorage {
    fn from(stored: StoredGCPBatchConfigCloudStorage) -> Self {
        Self {
            input_uri_prefix: stored.input_uri_prefix,
            output_uri_prefix: stored.output_uri_prefix,
        }
    }
}

impl From<StoredGCPProviderSFTConfig> for GCPSFTConfig {
    fn from(stored: StoredGCPProviderSFTConfig) -> Self {
        Self {
            project_id: stored.project_id,
            region: stored.region,
            bucket_name: stored.bucket_name,
            bucket_path_prefix: stored.bucket_path_prefix,
            service_account: stored.service_account,
            kms_key_name: stored.kms_key_name,
        }
    }
}

// --- Together ---

impl From<StoredTogetherProviderTypeConfig> for TogetherProviderTypeConfig {
    fn from(stored: StoredTogetherProviderTypeConfig) -> Self {
        Self {
            sft: stored.sft.map(Into::into),
            defaults: convert_stored_api_key_defaults(stored.defaults)
                .map(|api_key_location| TogetherDefaults { api_key_location }),
        }
    }
}

impl From<StoredTogetherProviderSFTConfig> for TogetherSFTConfig {
    fn from(stored: StoredTogetherProviderSFTConfig) -> Self {
        Self {
            wandb_api_key: stored.wandb_api_key,
            wandb_base_url: stored.wandb_base_url,
            wandb_project_name: stored.wandb_project_name,
            hf_api_token: stored.hf_api_token,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn test_simple_provider_type_config_round_trip() {
        let defaults = OpenAIDefaults {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "OPENAI_API_KEY".to_string(),
            )),
        };
        let stored = convert_simple_provider_type_config(&defaults);
        let restored: OpenAIProviderTypeConfig = stored.into();
        expect_that!(
            restored
                .defaults
                .as_ref()
                .expect("should have defaults")
                .api_key_location,
            eq(&defaults.api_key_location)
        );
    }

    // ── GCP credentials defaults ───────────────────────────────────────

    #[gtest]
    fn test_gcp_credential_defaults_round_trip() {
        let original = CredentialLocationWithFallback::Single(CredentialLocation::PathFromEnv(
            "GCP_VERTEX_CREDENTIALS_PATH".to_string(),
        ));
        let stored = StoredGCPCredentialDefaults::from(&original);
        let restored = stored
            .credential_location
            .map(CredentialLocationWithFallback::from)
            .expect("stored credential_location should be present");
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_gcp_credential_defaults_with_fallback_round_trip() {
        let original = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::PathFromEnv("GCP_VERTEX_CREDENTIALS_PATH".to_string()),
            fallback: CredentialLocation::Sdk,
        };
        let stored = StoredGCPCredentialDefaults::from(&original);
        let restored = stored
            .credential_location
            .map(CredentialLocationWithFallback::from)
            .expect("stored credential_location should be present");
        expect_that!(restored, eq(&original));
    }

    // ── GCP Vertex Anthropic provider type config ──────────────────────

    #[gtest]
    fn test_gcp_vertex_anthropic_provider_type_config_round_trip() {
        let defaults = GCPDefaults {
            credential_location: CredentialLocationWithFallback::Single(
                CredentialLocation::PathFromEnv("GCP_VERTEX_CREDENTIALS_PATH".to_string()),
            ),
        };
        let stored = StoredGCPCredentialProviderTypeConfig {
            defaults: Some(StoredGCPCredentialDefaults::from(
                &defaults.credential_location,
            )),
        };
        let restored: GCPVertexAnthropicProviderTypeConfig = stored.into();
        expect_that!(
            restored
                .defaults
                .as_ref()
                .expect("should have defaults")
                .credential_location,
            eq(&defaults.credential_location)
        );
    }

    // ── GCP batch configs ──────────────────────────────────────────────

    #[gtest]
    fn test_gcp_batch_config_type_none_round_trip() {
        let original = GCPBatchConfigType::None;
        let stored = StoredGCPBatchConfigType::from(&original);
        let restored: GCPBatchConfigType = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_gcp_batch_config_type_cloud_storage_round_trip() {
        let original = GCPBatchConfigType::CloudStorage(GCPBatchConfigCloudStorage {
            input_uri_prefix: "gs://my-bucket/inputs/".to_string(),
            output_uri_prefix: "gs://my-bucket/outputs/".to_string(),
        });
        let stored = StoredGCPBatchConfigType::from(&original);
        let restored: GCPBatchConfigType = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_gcp_batch_config_cloud_storage_round_trip() {
        let original = GCPBatchConfigCloudStorage {
            input_uri_prefix: "gs://my-bucket/in/".to_string(),
            output_uri_prefix: "gs://my-bucket/out/".to_string(),
        };
        let stored = StoredGCPBatchConfigCloudStorage {
            input_uri_prefix: original.input_uri_prefix.clone(),
            output_uri_prefix: original.output_uri_prefix.clone(),
        };
        let restored: GCPBatchConfigCloudStorage = stored.into();
        expect_that!(restored, eq(&original));
    }

    // ── GCP Vertex Gemini provider type config (full round trip) ───────

    #[gtest]
    fn test_gcp_vertex_gemini_provider_type_config_round_trip() {
        let original = GCPVertexGeminiProviderTypeConfig {
            batch: Some(GCPBatchConfigType::CloudStorage(
                GCPBatchConfigCloudStorage {
                    input_uri_prefix: "gs://b/in/".to_string(),
                    output_uri_prefix: "gs://b/out/".to_string(),
                },
            )),
            sft: Some(GCPSFTConfig {
                project_id: "proj".to_string(),
                region: "us-central1".to_string(),
                bucket_name: "bucket".to_string(),
                bucket_path_prefix: Some("prefix/".to_string()),
                service_account: Some("svc@proj.iam".to_string()),
                kms_key_name: Some("kms-key".to_string()),
            }),
            defaults: Some(GCPDefaults {
                credential_location: CredentialLocationWithFallback::Single(
                    CredentialLocation::PathFromEnv("GCP_VERTEX_CREDENTIALS_PATH".to_string()),
                ),
            }),
        };
        let stored = StoredGCPVertexGeminiProviderTypeConfig {
            batch: original.batch.as_ref().map(StoredGCPBatchConfigType::from),
            sft: original.sft.as_ref().map(|s| StoredGCPProviderSFTConfig {
                project_id: s.project_id.clone(),
                region: s.region.clone(),
                bucket_name: s.bucket_name.clone(),
                bucket_path_prefix: s.bucket_path_prefix.clone(),
                service_account: s.service_account.clone(),
                kms_key_name: s.kms_key_name.clone(),
            }),
            defaults: original
                .defaults
                .as_ref()
                .map(|d| StoredGCPCredentialDefaults::from(&d.credential_location)),
        };
        let restored: GCPVertexGeminiProviderTypeConfig = stored.into();
        expect_that!(restored, eq(&original));
    }
}
