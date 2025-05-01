use crate::inference::providers::aws_common::build_interceptor;
use crate::inference::types::Latency;
use crate::{
    cache::ModelProviderRequest,
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        providers::aws_common::InterceptorAndRawBody,
        types::{
            batch::{
                BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
            },
            ModelInferenceRequest, PeekableProviderInferenceResponseStream,
            ProviderInferenceResponse,
        },
    },
    model::ModelProvider,
};
use aws_config::Region;
use aws_sdk_sagemakerruntime::types::ResponseStream;
use aws_smithy_types::error::display::DisplayErrorContext;
use eventsource_stream::{EventStreamError, Eventsource};
use futures::StreamExt;
use std::time::Instant;

use super::provider_trait::TensorZeroEventError;
use super::{
    aws_common,
    provider_trait::{InferenceProvider, WrappedProvider},
};

#[expect(unused)]
const PROVIDER_NAME: &str = "AWS Sagemaker";
const PROVIDER_TYPE: &str = "aws_sagemaker";

// NB: If you add `Clone` someday, you'll need to wrap client in Arc
#[derive(Debug)]
pub struct AWSSagemakerProvider {
    endpoint_name: String,
    client: aws_sdk_sagemakerruntime::Client,
    hosted_provider: Box<dyn WrappedProvider + Send + Sync>,
}

impl AWSSagemakerProvider {
    pub async fn new(
        endpoint_name: String,
        hosted_provider: Box<dyn WrappedProvider + Send + Sync>,
        region: Option<Region>,
    ) -> Result<Self, Error> {
        let config = aws_common::config_with_region(PROVIDER_TYPE, region).await?;
        let client = aws_sdk_sagemakerruntime::Client::new(&config);

        Ok(Self {
            endpoint_name,
            client,
            hosted_provider,
        })
    }
}

impl InferenceProvider for AWSSagemakerProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = self.hosted_provider.make_body(request)?;
        let InterceptorAndRawBody {
            interceptor,
            get_raw_request,
        } = build_interceptor(
            request.request,
            model_provider,
            request.model_name.to_string(),
        );

        let start_time = Instant::now();
        let res = self
            .client
            .invoke_endpoint()
            .endpoint_name(self.endpoint_name.clone())
            .body(request_body.to_string().into_bytes().into())
            .content_type("application/json")
            .customize()
            .interceptor(interceptor)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error sending request to AWS Sagemaker: {}",
                        DisplayErrorContext(&e)
                    ),
                    raw_request: get_raw_request().ok(),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };

        let raw_request = get_raw_request()?;
        let raw_response = res.body.ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "Missing response body".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let raw_response_string = String::from_utf8(raw_response.into_inner()).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error converting raw response to string: {e}"),
                raw_request: Some(raw_request.clone()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        self.hosted_provider.parse_response(
            request.request,
            raw_request,
            raw_response_string,
            latency,
        )
    }

    async fn infer_stream<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = self.hosted_provider.make_body(request)?;

        let InterceptorAndRawBody {
            interceptor,
            get_raw_request,
        } = build_interceptor(
            request.request,
            model_provider,
            request.model_name.to_string(),
        );

        let start_time = Instant::now();
        let res = self
            .client
            .invoke_endpoint_with_response_stream()
            .endpoint_name(self.endpoint_name.clone())
            .body(request_body.to_string().into_bytes().into())
            .content_type("application/json")
            .customize()
            .interceptor(interceptor)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error sending request to AWS Sagemaker: {}",
                        DisplayErrorContext(&e)
                    ),
                    raw_request: get_raw_request().ok(),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        let raw_request = get_raw_request()?;

        let mut sdk_stream = res.body;
        let raw_request_clone = raw_request.clone();

        // We process the stream in two steps.
        // First, we flatten the `aws_sdk_sagemakerruntime` stream into a stream of `Result<Vec<u8>, Error>`
        // representing the raw bytes returned by the underlying Sagemaker container (from the internal `/invocations`)
        let sagemaker_byte_stream = async_stream::stream! {
            let raw_request = raw_request_clone;
            loop {
                match sdk_stream.recv().await {
                    Ok(Some(event)) =>  {
                        match event {
                            ResponseStream::PayloadPart(part) => {
                                let bytes = part.bytes.ok_or_else(|| {
                                    TensorZeroEventError::TensorZero(Error::new(ErrorDetails::InferenceServer {
                                        message: "Sagemaker payload part is empty".to_string(),
                                        provider_type: PROVIDER_TYPE.to_string(),
                                        raw_request: Some(raw_request.clone()),
                                        raw_response: None,
                                    }))
                                })?;
                                yield Ok(bytes.into_inner());
                            }
                            _ => {
                                yield Err(TensorZeroEventError::TensorZero(Error::new(ErrorDetails::InferenceServer {
                                    message: "Unexpected event type from Sagemaker".to_string(),
                                    provider_type: PROVIDER_TYPE.to_string(),
                                    raw_request: Some(raw_request.clone()),
                                    raw_response: None,
                                })));
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        yield Err(TensorZeroEventError::TensorZero(Error::new(ErrorDetails::InferenceServer {
                            raw_request: Some(raw_request.clone()),
                            raw_response: None,
                            message: e.to_string(),
                            provider_type: PROVIDER_TYPE.to_string(),
                        })));
                    }
                }
            }
        };

        // Second, we convert this into a `reqwest_eventsource::Event` stream, using the underlying `eventsource_stream` crate
        // to parse the raw byte stream into SEE events. We need to manually construct an `Open` event ourselves.
        // While Sagemaker allows an arbitrary byte stream to be returned, we only support a few different 'wrapped' providers,
        // (currently only the OpenAI provider targeting an ollama-based container), all of which currently produce an SSE event stream.
        let event_stream = futures::stream::iter([Ok(reqwest_eventsource::Event::Open)]).chain(
            sagemaker_byte_stream.eventsource().map(|r| match r {
                Ok(msg) => Ok(reqwest_eventsource::Event::Message(msg)),
                Err(e) => match e {
                    EventStreamError::Utf8(err) => Err(TensorZeroEventError::EventSource(
                        reqwest_eventsource::Error::Utf8(err),
                    )),
                    EventStreamError::Parser(err) => Err(TensorZeroEventError::EventSource(
                        reqwest_eventsource::Error::Parser(err),
                    )),
                    EventStreamError::Transport(err) => Err(err),
                },
            }),
        );
        let stream = self
            .hosted_provider
            .stream_events(Box::pin(event_stream), start_time.into())
            .peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}
