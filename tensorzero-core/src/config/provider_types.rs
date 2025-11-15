use crate::model::{CredentialLocation, CredentialLocationWithFallback};
use serde::{Deserialize, Serialize};
#[cfg(feature = "e2e_tests")]
use url::Url;

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export)]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AnthropicProviderTypeConfig {
    #[serde(default)]
    pub defaults: AnthropicDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AnthropicDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AzureProviderTypeConfig {
    #[serde(default)]
    pub defaults: AzureDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AzureDefaults {
    #[ts(type = "string")]
    pub api_key_location: CredentialLocationWithFallback,
}

impl Default for AzureDefaults {
    fn default() -> Self {
        Self {
            api_key_location: CredentialLocationWithFallback::Single(CredentialLocation::Env(
                "AZURE_OPENAI_API_KEY".to_string(),
            )),
        }
    }
}

// DeepSeek

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct DeepSeekProviderTypeConfig {
    #[serde(default)]
    pub defaults: DeepSeekDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct DeepSeekDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct FireworksProviderTypeConfig {
    #[serde(default)]
    pub defaults: FireworksDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct FireworksDefaults {
    #[ts(type = "string")]
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

// GCP Vertex

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
#[derive(ts_rs::TS)]
#[ts(export)]
pub struct GCPProviderTypeConfig {
    #[serde(default)]
    pub batch: Option<GCPBatchConfigType>,
    #[cfg(feature = "e2e_tests")]
    #[ts(skip)]
    pub batch_inference_api_base: Option<Url>,
    #[serde(default)]
    pub defaults: GCPDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "storage_type", rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[ts(export)]
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
#[derive(ts_rs::TS)]
#[ts(export)]
pub struct GCPBatchConfigCloudStorage {
    pub input_uri_prefix: String,
    pub output_uri_prefix: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GCPDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GoogleAIStudioGeminiProviderTypeConfig {
    #[serde(default)]
    pub defaults: GoogleAIStudioGeminiDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GoogleAIStudioGeminiDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GroqProviderTypeConfig {
    #[serde(default)]
    pub defaults: GroqDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GroqDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct HyperbolicProviderTypeConfig {
    #[serde(default)]
    pub defaults: HyperbolicDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct HyperbolicDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct MistralProviderTypeConfig {
    #[serde(default)]
    pub defaults: MistralDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct MistralDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenAIProviderTypeConfig {
    #[serde(default)]
    pub defaults: OpenAIDefaults,
    #[cfg(feature = "e2e_tests")]
    #[ts(skip)]
    pub batch_inference_api_base: Option<Url>,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenAIDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenRouterProviderTypeConfig {
    pub defaults: OpenRouterDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenRouterDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct SGLangProviderTypeConfig {
    #[serde(default)]
    pub defaults: SGLangDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct SGLangDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TGIProviderTypeConfig {
    pub defaults: TGIDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TGIDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TogetherProviderTypeConfig {
    pub defaults: TogetherDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TogetherDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct VLLMProviderTypeConfig {
    #[serde(default)]
    pub defaults: VLLMDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct VLLMDefaults {
    #[ts(type = "string")]
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

#[derive(Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct XAIProviderTypeConfig {
    #[serde(default)]
    pub defaults: XAIDefaults,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct XAIDefaults {
    #[ts(type = "string")]
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
