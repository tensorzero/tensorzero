use crate::config_parser::ProviderConfig;
use crate::error::Error;
use crate::inference::types::InferenceResponseStream;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::ModelInferenceResponse;
use reqwest::Client;
use secrecy::SecretString;

#[allow(async_fn_in_trait)]
pub trait InferenceProvider {
    async fn infer(
        &self,
        request: &ModelInferenceRequest,
        config: &ProviderConfig,
        client: &Client,
        api_key: &SecretString,
    ) -> Result<ModelInferenceResponse, Error>;

    async fn infer_stream(
        &self,
        request: &ModelInferenceRequest,
        config: &ProviderConfig,
        client: &Client,
        api_key: &SecretString,
    ) -> Result<InferenceResponseStream, Error>;
}
