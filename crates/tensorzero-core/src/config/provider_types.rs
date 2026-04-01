use crate::model::{CredentialLocation, CredentialLocationWithFallback};
use serde::{Deserialize, Serialize};

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
