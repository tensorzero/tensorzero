use crate::error::Error;
use crate::inference::types::InferenceResponseStream;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::ModelInferenceResponse;
use crate::model::ProviderConfig;
use futures::Future;
use reqwest::Client;

pub trait InferenceProvider {
    fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        config: &'a ProviderConfig,
        client: &'a Client,
    ) -> impl Future<Output = Result<ModelInferenceResponse, Error>> + Send + 'a;

    fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        config: &'a ProviderConfig,
        client: &'a Client,
    ) -> impl Future<Output = Result<InferenceResponseStream, Error>> + Send + 'a;
}
