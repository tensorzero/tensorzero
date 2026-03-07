use crate::model::{CredentialLocation, CredentialLocationWithFallback};
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[serde(deny_unknown_fields)]
pub struct ProviderTypesConfig {
    #[serde(default)]
    pub anthropic: AnthropicProviderTypeConfig,
    #[serde(default)]
    pub azure: AzureProviderTypeConfig,
    #[serde(default)]
    pub deepseek: DeepSeekProviderTypeConfig,
    #[serde(default)]
    pub fireworks: FireworksProviderTypeConfig,
    #[serde(default)]
    pub gcp_vertex_gemini: GCPVertexGeminiProviderTypeConfig,
    #[serde(default)]
    pub gcp_vertex_anthropic: GCPVertexAnthropicProviderTypeConfig,
    #[serde(default)]
    pub google_ai_studio_gemini: GoogleAIStudioGeminiProviderTypeConfig,
    #[serde(default)]
    pub groq: GroqProviderTypeConfig,
    #[serde(default)]
    pub hyperbolic: HyperbolicProviderTypeConfig,
    #[serde(default)]
    pub mistral: MistralProviderTypeConfig,
    #[serde(default)]
    pub openai: OpenAIProviderTypeConfig,
    #[serde(default)]
    pub openrouter: OpenRouterProviderTypeConfig,
    #[serde(default)]
    pub sglang: SGLangProviderTypeConfig,
    #[serde(default)]
    pub tgi: TGIProviderTypeConfig,
    #[serde(default)]
    pub together: TogetherProviderTypeConfig,
    #[serde(default)]
    pub vllm: VLLMProviderTypeConfig,
    #[serde(default)]
    pub xai: XAIProviderTypeConfig,
}

// Anthropic

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AnthropicProviderTypeConfig {
    #[serde(default)]
    pub defaults: AnthropicDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AzureProviderTypeConfig {
    #[serde(default)]
    pub defaults: AzureDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct DeepSeekProviderTypeConfig {
    #[serde(default)]
    pub defaults: DeepSeekDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct FireworksProviderTypeConfig {
    #[serde(default)]
    pub sft: Option<FireworksSFTConfig>,
    #[serde(default)]
    pub defaults: FireworksDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, rename = "FireworksSFTProviderConfig")
)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct FireworksSFTConfig {
    pub account_id: String,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct GCPVertexGeminiProviderTypeConfig {
    #[serde(default)]
    pub batch: Option<GCPBatchConfigType>,
    #[serde(default)]
    pub sft: Option<GCPSFTConfig>,
    #[serde(default)]
    pub defaults: GCPDefaults,
}

// GCP Vertex Anthropic

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
pub struct GCPVertexAnthropicProviderTypeConfig {
    #[serde(default)]
    pub defaults: GCPDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "storage_type", rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum GCPBatchConfigType {
    // In the future, we'll want to allow explicitly setting 'none' at the model provider level,
    // to override the global provider-types batch config.
    None,
    CloudStorage(GCPBatchConfigCloudStorage),
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct GCPBatchConfigCloudStorage {
    pub input_uri_prefix: String,
    pub output_uri_prefix: String,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GoogleAIStudioGeminiProviderTypeConfig {
    #[serde(default)]
    pub defaults: GoogleAIStudioGeminiDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GroqProviderTypeConfig {
    #[serde(default)]
    pub defaults: GroqDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct HyperbolicProviderTypeConfig {
    #[serde(default)]
    pub defaults: HyperbolicDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MistralProviderTypeConfig {
    #[serde(default)]
    pub defaults: MistralDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct OpenAIProviderTypeConfig {
    #[serde(default)]
    pub defaults: OpenAIDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct OpenRouterProviderTypeConfig {
    #[serde(default)]
    pub defaults: OpenRouterDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct SGLangProviderTypeConfig {
    #[serde(default)]
    pub defaults: SGLangDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct TGIProviderTypeConfig {
    #[serde(default)]
    pub defaults: TGIDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct TogetherProviderTypeConfig {
    #[serde(default)]
    pub sft: Option<TogetherSFTConfig>,
    #[serde(default)]
    pub defaults: TogetherDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(
    feature = "ts-bindings",
    ts(export, optional_fields, rename = "TogetherSFTProviderConfig")
)]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct VLLMProviderTypeConfig {
    #[serde(default)]
    pub defaults: VLLMDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct XAIProviderTypeConfig {
    #[serde(default)]
    pub defaults: XAIDefaults,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
