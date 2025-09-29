use serde::{Deserialize, Serialize};

use crate::model::CredentialLocation;

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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
    pub gcp_vertex_gemini: GCPProviderTypeConfig,
    #[serde(default)]
    pub gcp_vertex_anthropic: GCPProviderTypeConfig,
    #[serde(default)]
    pub google_ai_studio: GoogleAIStudioProviderTypeConfig,
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

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct AnthropicProviderTypeConfig {
    defaults: AnthropicDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AnthropicDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for AnthropicDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("ANTHROPIC_API_KEY".to_string()),
        }
    }
}

// Azure

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct AzureProviderTypeConfig {
    defaults: AzureDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AzureDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for AzureDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("AZURE_OPENAI_API_KEY".to_string()),
        }
    }
}

// DeepSeek

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct DeepSeekProviderTypeConfig {
    defaults: DeepSeekDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct DeepSeekDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for DeepSeekDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("DEEPSEEK_API_KEY".to_string()),
        }
    }
}

// Fireworks

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct FireworksProviderTypeConfig {
    defaults: FireworksDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct FireworksDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for FireworksDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("FIREWORKS_API_KEY".to_string()),
        }
    }
}

// GCP Vertex

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GCPProviderTypeConfig {
    #[serde(default)]
    pub batch: Option<GCPBatchConfigType>,
    pub defaults: GCPDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "storage_type", rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
#[serde(deny_unknown_fields)]
pub enum GCPBatchConfigType {
    // In the future, we'll want to allow explicitly setting 'none' at the model provider level,
    // to override the global provider-types batch config.
    None,
    CloudStorage(GCPBatchConfigCloudStorage),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GCPBatchConfigCloudStorage {
    pub input_uri_prefix: String,
    pub output_uri_prefix: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GCPDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for GCPDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::PathFromEnv(
                "GCP_VERTEX_CREDENTIALS_PATH".to_string(),
            ),
        }
    }
}

// Google AI Studio

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GoogleAIStudioProviderTypeConfig {
    defaults: GoogleAIStudioDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GoogleAIStudioDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for GoogleAIStudioDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("GOOGLE_AI_STUDIO_API_KEY".to_string()),
        }
    }
}

// Groq

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GroqProviderTypeConfig {
    defaults: GroqDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GroqDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for GroqDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("GROQ_API_KEY".to_string()),
        }
    }
}

// Hyperbolic

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct HyperbolicProviderTypeConfig {
    defaults: HyperbolicDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct HyperbolicDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for HyperbolicDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("HYPERBOLIC_API_KEY".to_string()),
        }
    }
}

// Mistral

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct MistralProviderTypeConfig {
    defaults: MistralDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct MistralDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for MistralDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("MISTRAL_API_KEY".to_string()),
        }
    }
}

// OpenAI

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct OpenAIProviderTypeConfig {
    defaults: OpenAIDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenAIDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for OpenAIDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("OPENAI_API_KEY".to_string()),
        }
    }
}

// Openrouter

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct OpenRouterProviderTypeConfig {
    defaults: OpenRouterDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenRouterDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for OpenRouterDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("OPENROUTER_API_KEY".to_string()),
        }
    }
}

// SGLang

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct SGLangProviderTypeConfig {
    defaults: SGLangDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct SGLangDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for SGLangDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("SGLANG_API_KEY".to_string()),
        }
    }
}

// TGI

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct TGIProviderTypeConfig {
    defaults: TGIDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TGIDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for TGIDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::None,
        }
    }
}

// Together

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct TogetherProviderTypeConfig {
    defaults: TogetherDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TogetherDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for TogetherDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::None,
        }
    }
}

// vLLM

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct VLLMProviderTypeConfig {
    defaults: VLLMDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct VLLMDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for VLLMDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("VLLM_API_KEY".to_string()),
        }
    }
}

// xAI

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct XAIProviderTypeConfig {
    defaults: XAIDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct XAIDefaults {
    #[ts(type = "string")]
    credential_location: CredentialLocation,
}

impl Default for XAIDefaults {
    fn default() -> Self {
        Self {
            credential_location: CredentialLocation::Env("XAI_API_KEY".to_string()),
        }
    }
}
