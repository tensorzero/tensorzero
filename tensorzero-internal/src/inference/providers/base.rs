use crate::cache::ModelProviderRequest;
use crate::embeddings::{EmbeddingProviderResponse, EmbeddingRequest};
use crate::endpoints::capability::EndpointCapability;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
};
use crate::inference::types::{
    ModelInferenceRequest, PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
};
use crate::model::ModelProvider;
use futures::Future;
use reqwest::Client;
use std::collections::HashSet;

/// Unified provider trait that supports multiple endpoint capabilities
pub trait UnifiedProvider: Send + Sync {
    /// Returns the set of capabilities this provider supports
    fn supported_capabilities(&self) -> HashSet<EndpointCapability>;

    /// Checks if this provider supports a specific capability
    fn supports_capability(&self, capability: EndpointCapability) -> bool {
        self.supported_capabilities().contains(&capability)
    }

    /// Non-streaming chat inference (required if Chat capability is supported)
    fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> impl Future<Output = Result<ProviderInferenceResponse, Error>> + Send;

    /// Streaming chat inference (required if Chat capability is supported)
    fn infer_stream<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> impl Future<Output = Result<(PeekableProviderInferenceResponseStream, String), Error>> + Send;

    /// Start batch inference (optional, even for Chat capability)
    fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest],
        _client: &'a Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<StartBatchProviderInferenceResponse, Error>> + Send {
        async {
            Err(Error::new(
                crate::error::ErrorDetails::CapabilityNotSupported {
                    capability: "batch inference".to_string(),
                    provider: std::any::type_name::<Self>().to_string(),
                },
            ))
        }
    }

    /// Poll batch inference status (optional, even for Chat capability)
    fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<PollBatchInferenceResponse, Error>> + Send {
        async {
            Err(Error::new(
                crate::error::ErrorDetails::CapabilityNotSupported {
                    capability: "batch inference polling".to_string(),
                    provider: std::any::type_name::<Self>().to_string(),
                },
            ))
        }
    }

    /// Embedding inference (required if Embeddings capability is supported)
    fn embed<'a>(
        &'a self,
        _request: &'a EmbeddingRequest,
        _client: &'a Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<EmbeddingProviderResponse, Error>> + Send {
        async {
            Err(Error::new(
                crate::error::ErrorDetails::CapabilityNotSupported {
                    capability: EndpointCapability::Embedding.as_str().to_string(),
                    provider: std::any::type_name::<Self>().to_string(),
                },
            ))
        }
    }
}

/// Helper struct to simplify unified provider implementations
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    pub chat: bool,
    pub embeddings: bool,
    pub batch_inference: bool,
}

impl ProviderCapabilities {
    pub fn to_hashset(&self) -> HashSet<EndpointCapability> {
        let mut capabilities = HashSet::new();
        if self.chat {
            capabilities.insert(EndpointCapability::Chat);
        }
        if self.embeddings {
            capabilities.insert(EndpointCapability::Embedding);
        }
        capabilities
    }
}
