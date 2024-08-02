use crate::error::Error;
use crate::inference::types::InferenceResponseStream;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::ModelInferenceResponse;
use crate::inference::types::ModelInferenceResponseChunk;
use crate::model::ProviderConfig;
use futures::Future;
use reqwest::Client;

pub trait InferenceProvider {
    // fn with_config<'a>(
    //     model: &'a ProviderConfig,
    // ) -> impl Future<Output = Result<Self, Error>> + Send + 'a
    // where
    //     Self: Sized;

    fn infer<'a>(
        request: &'a ModelInferenceRequest,
        config: &'a ProviderConfig,
        client: &'a Client,
    ) -> impl Future<Output = Result<ModelInferenceResponse, Error>> + Send + 'a;

    fn infer_stream<'a>(
        request: &'a ModelInferenceRequest,
        config: &'a ProviderConfig,
        client: &'a Client,
    ) -> impl Future<Output = Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error>>
           + Send
           + 'a;
}
