use serde::{Deserialize, Serialize};

use crate::stored_credential_location::StoredCredentialLocationWithFallback;

// --- Top-level ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredProviderTypesConfig {
    pub anthropic: Option<StoredSimpleProviderTypeConfig>,
    pub azure: Option<StoredSimpleProviderTypeConfig>,
    pub deepseek: Option<StoredSimpleProviderTypeConfig>,
    pub fireworks: Option<StoredFireworksProviderTypeConfig>,
    pub gcp_vertex_gemini: Option<StoredGCPVertexGeminiProviderTypeConfig>,
    pub gcp_vertex_anthropic: Option<StoredGCPCredentialProviderTypeConfig>,
    pub google_ai_studio_gemini: Option<StoredSimpleProviderTypeConfig>,
    pub groq: Option<StoredSimpleProviderTypeConfig>,
    pub hyperbolic: Option<StoredSimpleProviderTypeConfig>,
    pub mistral: Option<StoredSimpleProviderTypeConfig>,
    pub openai: Option<StoredSimpleProviderTypeConfig>,
    pub openrouter: Option<StoredSimpleProviderTypeConfig>,
    pub sglang: Option<StoredSimpleProviderTypeConfig>,
    pub tgi: Option<StoredSimpleProviderTypeConfig>,
    pub together: Option<StoredTogetherProviderTypeConfig>,
    pub vllm: Option<StoredSimpleProviderTypeConfig>,
    pub xai: Option<StoredSimpleProviderTypeConfig>,
}

// --- Common pattern: just api_key_location defaults ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredSimpleProviderTypeConfig {
    pub defaults: Option<StoredApiKeyDefaults>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredApiKeyDefaults {
    pub api_key_location: Option<StoredCredentialLocationWithFallback>,
}

// --- GCP uses credential_location instead of api_key_location ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGCPCredentialProviderTypeConfig {
    pub defaults: Option<StoredGCPCredentialDefaults>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGCPCredentialDefaults {
    pub credential_location: Option<StoredCredentialLocationWithFallback>,
}

// --- Fireworks ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredFireworksProviderTypeConfig {
    pub sft: Option<StoredFireworksProviderSFTConfig>,
    pub defaults: Option<StoredApiKeyDefaults>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredFireworksProviderSFTConfig {
    pub account_id: String,
}

// --- GCP Vertex Gemini ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGCPVertexGeminiProviderTypeConfig {
    pub batch: Option<StoredGCPBatchConfigType>,
    pub sft: Option<StoredGCPProviderSFTConfig>,
    pub defaults: Option<StoredGCPCredentialDefaults>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "storage_type", rename_all = "snake_case")]
pub enum StoredGCPBatchConfigType {
    None,
    CloudStorage(StoredGCPBatchConfigCloudStorage),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGCPBatchConfigCloudStorage {
    pub input_uri_prefix: String,
    pub output_uri_prefix: String,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGCPProviderSFTConfig {
    pub project_id: String,
    pub region: String,
    pub bucket_name: String,
    pub bucket_path_prefix: Option<String>,
    pub service_account: Option<String>,
    pub kms_key_name: Option<String>,
}

// --- Together ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredTogetherProviderTypeConfig {
    pub sft: Option<StoredTogetherProviderSFTConfig>,
    pub defaults: Option<StoredApiKeyDefaults>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredTogetherProviderSFTConfig {
    pub wandb_api_key: Option<String>,
    pub wandb_base_url: Option<String>,
    pub wandb_project_name: Option<String>,
    pub hf_api_token: Option<String>,
}
