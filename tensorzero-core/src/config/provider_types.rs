use crate::model::{CredentialLocation, CredentialLocationWithFallback};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AnthropicProviderTypeConfig {
    #[serde(default)]
    pub defaults: AnthropicDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AzureProviderTypeConfig {
    #[serde(default)]
    pub defaults: AzureDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct DeepSeekProviderTypeConfig {
    #[serde(default)]
    pub defaults: DeepSeekDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FireworksProviderTypeConfig {
    #[serde(default)]
    pub sft: Option<FireworksSFTConfig>,
    #[serde(default)]
    pub defaults: FireworksDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub struct FireworksSFTConfig {
    pub account_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GCPVertexAnthropicProviderTypeConfig {
    #[serde(default)]
    pub defaults: GCPDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "storage_type", rename_all = "snake_case")]
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
pub struct GCPBatchConfigCloudStorage {
    pub input_uri_prefix: String,
    pub output_uri_prefix: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct GoogleAIStudioGeminiProviderTypeConfig {
    #[serde(default)]
    pub defaults: GoogleAIStudioGeminiDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct GroqProviderTypeConfig {
    #[serde(default)]
    pub defaults: GroqDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct HyperbolicProviderTypeConfig {
    #[serde(default)]
    pub defaults: HyperbolicDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct MistralProviderTypeConfig {
    #[serde(default)]
    pub defaults: MistralDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct OpenAIProviderTypeConfig {
    #[serde(default)]
    pub defaults: OpenAIDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct OpenRouterProviderTypeConfig {
    #[serde(default)]
    pub defaults: OpenRouterDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SGLangProviderTypeConfig {
    #[serde(default)]
    pub defaults: SGLangDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TGIProviderTypeConfig {
    #[serde(default)]
    pub defaults: TGIDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TogetherProviderTypeConfig {
    #[serde(default)]
    pub sft: Option<TogetherSFTConfig>,
    #[serde(default)]
    pub defaults: TogetherDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct VLLMProviderTypeConfig {
    #[serde(default)]
    pub defaults: VLLMDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct XAIProviderTypeConfig {
    #[serde(default)]
    pub defaults: XAIDefaults,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
