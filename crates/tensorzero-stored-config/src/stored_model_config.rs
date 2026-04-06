use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::stored_cost::{StoredCostConfig, StoredUnifiedCostConfig};
use crate::stored_credential_location::{
    StoredCredentialLocation, StoredCredentialLocationOrHardcoded,
    StoredCredentialLocationWithFallback, StoredEndpointLocation,
};
use crate::{StoredExtraBodyConfig, StoredExtraHeadersConfig, StoredTimeoutsConfig};

// --- Top-level model config ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredModelConfig {
    pub routing: Vec<String>,
    pub providers: BTreeMap<String, StoredModelProvider>,
    pub timeouts: Option<StoredTimeoutsConfig>,
    pub skip_relay: Option<bool>,
    pub namespace: Option<String>,
}

// --- Model provider ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredModelProvider {
    pub provider: StoredProviderConfig,
    pub extra_body: Option<StoredExtraBodyConfig>,
    pub extra_headers: Option<StoredExtraHeadersConfig>,
    pub timeouts: Option<StoredTimeoutsConfig>,
    pub discard_unknown_chunks: Option<bool>,
    pub cost: Option<StoredCostConfig>,
    pub batch_cost: Option<StoredUnifiedCostConfig>,
}

// --- Provider config (large tagged enum) ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum StoredProviderConfig {
    Anthropic {
        model_name: String,
        api_base: Option<String>,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
        beta_structured_outputs: Option<bool>,
        provider_tools: Option<Vec<Value>>,
    },
    #[serde(rename = "aws_bedrock")]
    AWSBedrock {
        model_id: String,
        region: Option<StoredCredentialLocationOrHardcoded>,
        allow_auto_detect_region: Option<bool>,
        endpoint_url: Option<StoredCredentialLocationOrHardcoded>,
        api_key: Option<StoredCredentialLocation>,
        access_key_id: Option<StoredCredentialLocation>,
        secret_access_key: Option<StoredCredentialLocation>,
        session_token: Option<StoredCredentialLocation>,
    },
    #[serde(rename = "aws_sagemaker")]
    AWSSagemaker {
        endpoint_name: String,
        model_name: String,
        region: Option<StoredCredentialLocationOrHardcoded>,
        allow_auto_detect_region: Option<bool>,
        hosted_provider: StoredHostedProviderKind,
        endpoint_url: Option<StoredCredentialLocationOrHardcoded>,
        access_key_id: Option<StoredCredentialLocation>,
        secret_access_key: Option<StoredCredentialLocation>,
        session_token: Option<StoredCredentialLocation>,
    },
    Azure {
        deployment_id: String,
        endpoint: StoredEndpointLocation,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    #[serde(rename = "gcp_vertex_anthropic")]
    GCPVertexAnthropic {
        model_id: String,
        location: String,
        project_id: String,
        credential_location: Option<StoredCredentialLocationWithFallback>,
        provider_tools: Option<Vec<Value>>,
    },
    #[serde(rename = "gcp_vertex_gemini")]
    GCPVertexGemini {
        model_id: Option<String>,
        endpoint_id: Option<String>,
        location: String,
        project_id: String,
        credential_location: Option<StoredCredentialLocationWithFallback>,
    },
    #[serde(rename = "google_ai_studio_gemini")]
    GoogleAIStudioGemini {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    #[serde(rename = "groq")]
    Groq {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
        reasoning_format: Option<String>,
    },
    Hyperbolic {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    #[serde(rename = "fireworks")]
    Fireworks {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
        parse_think_blocks: Option<bool>,
    },
    Mistral {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
        prompt_mode: Option<String>,
    },
    OpenAI {
        model_name: String,
        api_base: Option<String>,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
        api_type: Option<StoredOpenAIAPIType>,
        include_encrypted_reasoning: Option<bool>,
        provider_tools: Option<Vec<Value>>,
        content_type_overrides: Option<BTreeMap<String, StoredContentBlockType>>,
    },
    OpenRouter {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    Together {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
        parse_think_blocks: Option<bool>,
    },
    VLLM {
        model_name: String,
        api_base: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    XAI {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    TGI {
        api_base: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    SGLang {
        model_name: String,
        api_base: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    DeepSeek {
        model_name: String,
        api_key_location: Option<StoredCredentialLocationWithFallback>,
    },
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy {
        model_name: String,
        api_key_location: Option<Value>,
    },
}

// --- Stored versions of core-private types ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoredHostedProviderKind {
    OpenAI,
    TGI,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredOpenAIAPIType {
    ChatCompletions,
    Responses,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredContentBlockType {
    ImageUrl,
    File,
    InputAudio,
}
