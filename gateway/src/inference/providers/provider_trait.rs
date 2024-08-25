use crate::error::Error;
use crate::inference::types::ModelInferenceRequest;
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
    ) -> impl Future<Output = Result<ProviderInferenceResponse, Error>> + Send + 'a;

    fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        client: &'a Client,
    ) -> impl Future<
        Output = Result<
            (
                ProviderInferenceResponseChunk,
                ProviderInferenceResponseStream,
            ),
            Error,
        >,
    > + Send
           + 'a;
}
