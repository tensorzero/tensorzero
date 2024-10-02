use crate::endpoints::inference::InferenceApiKeys;
use crate::error::Error;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::ProviderInferenceResponse;
use crate::inference::types::ProviderInferenceResponseChunk;
use crate::inference::types::ProviderInferenceResponseStream;
use futures::Future;
use reqwest::Client;
use secrecy::SecretString;
use std::borrow::Cow;

pub trait HasCredentials {
    fn has_credentials(&self) -> bool;
    fn get_api_key<'a>(
        &'a self,
        api_keys: &'a InferenceApiKeys,
    ) -> Result<Cow<'a, SecretString>, Error>;
}

pub trait InferenceProvider: HasCredentials {
    fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        client: &'a Client,
        api_key: Cow<'a, SecretString>,
    ) -> impl Future<Output = Result<ProviderInferenceResponse, Error>> + Send + 'a;

    fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest,
        client: &'a Client,
        api_key: Cow<'a, SecretString>,
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
