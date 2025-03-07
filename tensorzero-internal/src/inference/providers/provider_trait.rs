use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::batch::StartBatchProviderInferenceResponse;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::PeekableProviderInferenceResponseStream;
use crate::inference::types::ProviderInferenceResponse;
use crate::model::ModelProvider;
use futures::Future;
use reqwest::Client;

pub trait InferenceProvider {
    fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> impl Future<Output = Result<ProviderInferenceResponse, Error>> + Send + 'a;

    fn infer_stream<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> impl Future<Output = Result<(PeekableProviderInferenceResponseStream, String), Error>> + Send + 'a;

    fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest],
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<StartBatchProviderInferenceResponse, Error>> + Send + 'a;

    fn poll_batch_inference<'a>(
        &'a self,
        batch_request: &'a BatchRequestRow<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<PollBatchInferenceResponse, Error>> + Send + 'a;
}
