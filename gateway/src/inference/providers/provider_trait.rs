use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::ProviderBatchInferenceResponse;
use crate::inference::types::ProviderInferenceResponse;
use crate::inference::types::ProviderInferenceResponseChunk;
use crate::inference::types::ProviderInferenceResponseStream;
use futures::Future;
use reqwest::Client;

pub trait InferenceProvider {
    fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<ProviderInferenceResponse, Error>> + Send + 'a;

    fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<
        Output = Result<
            (
                ProviderInferenceResponseChunk,
                ProviderInferenceResponseStream,
                String,
            ),
            Error,
        >,
    > + Send
           + 'a;

    fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest],
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl Future<Output = Result<ProviderBatchInferenceResponse, Error>> + Send + 'a;
}
