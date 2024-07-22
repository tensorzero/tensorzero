use crate::config_parser::ProviderConfig;
use crate::error::Error;
use crate::inference::types::InferenceResponseStream;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::ModelInferenceResponse;
use futures::Future;
use reqwest::Client;
use secrecy::SecretString;

pub trait InferenceProvider {
    fn infer(
        &self,
        request: &ModelInferenceRequest,
        config: &ProviderConfig,
        client: &Client,
        api_key: &SecretString,
    ) -> impl Future<Output = Result<ModelInferenceResponse, Error>> + Send;

    fn infer_stream(
        &self,
        request: &ModelInferenceRequest,
        config: &ProviderConfig,
        client: &Client,
        api_key: &SecretString,
    ) -> impl Future<Output = Result<InferenceResponseStream, Error>> + Send;
}
