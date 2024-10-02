use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::ProviderInferenceResponse;
use crate::inference::types::ProviderInferenceResponseChunk;
use crate::inference::types::ProviderInferenceResponseStream;
use crate::model::ProviderCredentials;
use futures::Future;
use reqwest::Client;

pub trait HasCredentials {
    fn has_credentials(&self) -> bool;
    fn get_credentials<'a>(
        &'a self,
        api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderCredentials<'a>, Error>;
}

pub trait InferenceProvider: HasCredentials {
    fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        client: &'a Client,
        api_key: ProviderCredentials<'a>,
    ) -> impl Future<Output = Result<ProviderInferenceResponse, Error>> + Send + 'a;

    fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        client: &'a Client,
        api_key: ProviderCredentials<'a>,
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
}
