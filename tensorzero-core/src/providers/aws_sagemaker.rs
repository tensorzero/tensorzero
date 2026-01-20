use crate::http::TensorzeroHttpClient;
use crate::inference::types::Latency;
use crate::inference::{InferenceProvider, WrappedProvider};
use crate::providers::aws_common::{
    AWSCredentials, AWSEndpointUrl, AWSRegion, InterceptorAndRawBody, build_interceptor,
    warn_if_credential_exfiltration_risk,
};
use crate::{
    cache::ModelProviderRequest,
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::types::{
        ModelInferenceRequest, PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
        batch::{BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse},
    },
    model::ModelProvider,
};
use aws_config::Region;
use aws_sdk_sagemakerruntime::types::ResponseStream;
use aws_smithy_types::error::display::DisplayErrorContext;
use eventsource_stream::{EventStreamError, Eventsource};
use futures::StreamExt;
use serde::Serialize;
use std::time::Instant;

use super::aws_common;
use crate::inference::TensorZeroEventError;

#[expect(unused)]
const PROVIDER_NAME: &str = "AWS Sagemaker";
const PROVIDER_TYPE: &str = "aws_sagemaker";

// NB: If you add `Clone` someday, you'll need to wrap client in Arc
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AWSSagemakerProvider {
    endpoint_name: String,
    #[serde(skip)]
    client: aws_sdk_sagemakerruntime::Client,
    #[serde(skip)] // TODO: add a way to Serialize the WrappedProvider
    pub hosted_provider: Box<dyn WrappedProvider + Send + Sync>,
    #[serde(skip)]
    base_config: aws_sdk_sagemakerruntime::config::Builder,
    #[serde(skip)]
    region: Option<AWSRegion>,
    #[serde(skip)]
    endpoint_url: Option<AWSEndpointUrl>,
    #[serde(skip)]
    credentials: AWSCredentials,
}

impl AWSSagemakerProvider {
    pub async fn new(
        endpoint_name: String,
        hosted_provider: Box<dyn WrappedProvider + Send + Sync>,
        static_region: Option<Region>,
        region: Option<AWSRegion>,
        endpoint_url: Option<AWSEndpointUrl>,
        credentials: AWSCredentials,
    ) -> Result<Self, Error> {
        let config = aws_common::config_with_region(PROVIDER_TYPE, static_region).await?;

        let mut config_builder = aws_sdk_sagemakerruntime::config::Builder::from(&config);

        // Apply static endpoint URL at construction time
        if let Some(ref ep) = endpoint_url
            && let Some(url) = ep.get_static_url()
        {
            config_builder = config_builder.endpoint_url(url.as_str());
        }

        // Apply static credentials at construction time
        if let Some(static_creds) = credentials.get_static_credentials() {
            config_builder = config_builder.credentials_provider(static_creds);
        }

        // Warn about potential credential exfiltration risk
        warn_if_credential_exfiltration_risk(&endpoint_url, &credentials, PROVIDER_TYPE);

        let client = aws_sdk_sagemakerruntime::Client::from_conf(config_builder.clone().build());

        Ok(Self {
            endpoint_name,
            client,
            hosted_provider,
            base_config: config_builder,
            region,
            endpoint_url,
            credentials,
        })
    }

    /// Apply dynamic region, endpoint URL, and/or credentials to a config builder.
    fn apply_dynamic_overrides(
        &self,
        mut config: aws_sdk_sagemakerruntime::config::Builder,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<aws_sdk_sagemakerruntime::config::Builder, Error> {
        if let Some(region) = &self.region
            && region.is_dynamic()
        {
            let resolved_region = region.resolve(dynamic_api_keys)?;
            config = config.region(resolved_region);
        }
        if let Some(endpoint_url) = &self.endpoint_url
            && matches!(endpoint_url, AWSEndpointUrl::Dynamic(_))
        {
            let url = endpoint_url.resolve(dynamic_api_keys)?;
            config = config.endpoint_url(url.as_str());
        }
        if self.credentials.is_dynamic() {
            let resolved_credentials = self.credentials.resolve(dynamic_api_keys)?;
            config = config.credentials_provider(resolved_credentials);
        }
        Ok(config)
    }
}

impl InferenceProvider for AWSSagemakerProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = self.hosted_provider.make_body(request).await?;
        let InterceptorAndRawBody {
            interceptor,
            get_raw_request,
            get_raw_response,
        } = build_interceptor(
            request.request,
            model_provider,
            request.model_name.to_string(),
        );

        // Use our custom `reqwest::Client` when making requests to Sagemaker.
        // This ensures that our HTTP proxy (TENSORZERO_E2E_PROXY) is used
        // here when it's enabled.
        let new_config = self
            .base_config
            .clone()
            .http_client(super::aws_http_client::Client::new(http_client.clone()));
        let new_config = self.apply_dynamic_overrides(new_config, dynamic_api_keys)?;

        let start_time = Instant::now();
        let res = self
            .client
            .invoke_endpoint()
            .endpoint_name(self.endpoint_name.clone())
            .body(request_body.to_string().into_bytes().into())
            .content_type("application/json")
            .customize()
            .config_override(new_config)
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
                    raw_response: get_raw_response().ok(),
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
            request.model_name,
            request.provider_name,
            request.model_inference_id,
        )
    }

    async fn infer_stream<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = self.hosted_provider.make_body(request).await?;

        let InterceptorAndRawBody {
            interceptor,
            get_raw_request,
            get_raw_response,
        } = build_interceptor(
            request.request,
            model_provider,
            request.model_name.to_string(),
        );

        // See `infer` for more details
        let new_config = self
            .base_config
            .clone()
            .http_client(super::aws_http_client::Client::new(http_client.clone()));
        let new_config = self.apply_dynamic_overrides(new_config, dynamic_api_keys)?;

        let start_time = Instant::now();
        let res = self
            .client
            .invoke_endpoint_with_response_stream()
            .endpoint_name(self.endpoint_name.clone())
            .body(request_body.to_string().into_bytes().into())
            .content_type("application/json")
            .customize()
            .config_override(new_config)
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
                    raw_response: get_raw_response().ok(),
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
                        Box::new(reqwest_eventsource::Error::Utf8(err)),
                    )),
                    EventStreamError::Parser(err) => Err(TensorZeroEventError::EventSource(
                        Box::new(reqwest_eventsource::Error::Parser(err)),
                    )),
                    EventStreamError::Transport(err) => Err(err),
                },
            }),
        );
        let stream = self
            .hosted_provider
            .stream_events(
                Box::pin(event_stream),
                start_time.into(),
                &raw_request,
                request.model_inference_id,
            )
            .peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
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
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}
