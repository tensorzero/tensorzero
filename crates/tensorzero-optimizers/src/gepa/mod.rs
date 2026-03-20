//! GEPA optimizer implementation

use tensorzero_core::{
    client::ClientInferenceParams,
    endpoints::inference::InferenceResponse,
    error::{Error, ErrorDetails},
    variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluate::VariantName;

pub mod analyze;
pub mod durable;
pub mod evaluate;
pub mod mutate;
pub mod pareto;
pub(crate) mod sequential;
pub mod validate;

/// A GEPA variant with its name and configuration
#[derive(Debug)]
pub struct GEPAVariant {
    pub name: VariantName,
    pub config: UninitializedChatCompletionConfig,
}

/// Minimal client trait for GEPA inference calls (analyze + mutate).
///
/// Both the sequential path (gateway `Client`) and the durable path
/// (`dyn TensorZeroClient`) implement this, eliminating code duplication
/// in `analyze_inference` and `mutate_variant`.
pub trait GepaClient: Send + Sync {
    /// Run a non-streaming inference and return the response directly.
    fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> impl std::future::Future<Output = Result<InferenceResponse, Error>> + Send;
}

/// Adapter that implements [`GepaClient`] for the gateway SDK `Client`.
///
/// Unwraps `InferenceOutput::NonStreaming` → `InferenceResponse`.
pub struct GatewayGepaClient<'a>(pub &'a tensorzero_core::client::Client);

impl GepaClient for GatewayGepaClient<'_> {
    async fn inference(&self, params: ClientInferenceParams) -> Result<InferenceResponse, Error> {
        let output = self.0.inference(params).await.map_err(|e| {
            Error::new(ErrorDetails::Inference {
                message: format!("{e}"),
            })
        })?;
        match output {
            tensorzero_core::client::InferenceOutput::NonStreaming(response) => Ok(response),
            tensorzero_core::client::InferenceOutput::Streaming(_) => {
                Err(Error::new(ErrorDetails::Inference {
                    message: "Expected NonStreaming response but got Streaming".to_string(),
                }))
            }
        }
    }
}
