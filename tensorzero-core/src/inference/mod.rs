pub mod types;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::batch::StartBatchProviderInferenceResponse;
use crate::inference::types::Latency;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::PeekableProviderInferenceResponseStream;
use crate::inference::types::ProviderInferenceResponse;
use crate::inference::types::ProviderInferenceResponseStreamInner;
use crate::model::ModelProvider;
use async_trait::async_trait;
use futures::Future;
use futures::Stream;
use reqwest_eventsource::Event;
use std::borrow::Cow;
use std::fmt::Debug;
use std::pin::Pin;
use tokio::time::Instant;

/// A helper type for preserving custom errors when working with `reqwest_eventsource`
/// This is currently used by `stream_openai` to allow using it with a provider
/// that needs to do additional validation when streaming (e.g. Sagemaker)
pub enum TensorZeroEventError {
    TensorZero(Error),
    EventSource(reqwest_eventsource::Error),
}

pub trait InferenceProvider {
    fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> impl Future<Output = Result<ProviderInferenceResponse, Error>> + Send + 'a;

    fn infer_stream<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> impl Future<Output = Result<(PeekableProviderInferenceResponseStream, String), Error>> + Send + 'a;

    fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest],
        client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<StartBatchProviderInferenceResponse, Error>> + Send + 'a;

    fn poll_batch_inference<'a>(
        &'a self,
        batch_request: &'a BatchRequestRow<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<PollBatchInferenceResponse, Error>> + Send + 'a;
}

/// A trait implemented for providers which can be 'wrapped' by another provider.
/// The AWS Sagemaker provider takes in a 'WrappedProvider', and uses it to build the request
/// body (which gets wrapped in SigV4) and to deserialized the response body retrieved from the
/// AWS sdk.
///
/// Currently, we only implement `WrappedProvider` for `OpenAI`
#[async_trait]
pub trait WrappedProvider: Debug {
    fn thought_block_provider_type_suffix(&self) -> Cow<'static, str>;

    async fn make_body<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
    ) -> Result<serde_json::Value, Error>;

    fn parse_response(
        &self,
        request: &ModelInferenceRequest,
        raw_request: String,
        raw_response: String,
        latency: Latency,
        model_name: &str,
        provider_name: &str,
    ) -> Result<ProviderInferenceResponse, Error>;

    fn stream_events(
        &self,
        event_source: Pin<
            Box<dyn Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static>,
        >,
        start_time: Instant,
        raw_request: &str,
    ) -> ProviderInferenceResponseStreamInner;
}
