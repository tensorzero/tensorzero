//! AWS SageMaker model provider using direct HTTP calls.

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_types::region::Region;
use bytes::BytesMut;
use futures::StreamExt;
use reqwest_sse_stream::SseStream;
use serde::Serialize;
use std::time::Instant;

use super::aws_common::{
    AWSCredentials, AWSEndpointUrl, AWSProviderConfig, AWSRegion, check_eventstream_exception,
    send_aws_request, sign_request,
};
use super::helpers::inject_extra_request_data;
use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::{
    Latency, ModelInferenceRequest, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, batch::StartBatchProviderInferenceResponse,
};
use crate::inference::{InferenceProvider, TensorZeroEventError, WrappedProvider};
use crate::model::ModelProvider;

#[expect(unused)]
const PROVIDER_NAME: &str = "AWS Sagemaker";
const PROVIDER_TYPE: &str = "aws_sagemaker";

/// AWS SageMaker provider using direct HTTP calls.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AWSSagemakerProvider {
    endpoint_name: String,
    #[serde(skip)]
    config: AWSProviderConfig,
    #[serde(skip)] // TODO: add a way to Serialize the WrappedProvider
    pub hosted_provider: Box<dyn WrappedProvider + Send + Sync>,
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
        let config = AWSProviderConfig::new(
            static_region,
            region,
            endpoint_url,
            credentials,
            PROVIDER_TYPE,
        )
        .await?;

        Ok(Self {
            endpoint_name,
            config,
            hosted_provider,
        })
    }
}

/// Prepared request body ready for signing and sending
struct PreparedSagemakerRequest {
    raw_request: String,
    body_bytes: Vec<u8>,
    http_extra_headers: http::HeaderMap,
}

/// Prepare the request body: build request using hosted provider, serialize, inject extras
async fn prepare_sagemaker_request<'a>(
    request: ModelProviderRequest<'a>,
    model_provider: &'a ModelProvider,
    hosted_provider: &'a (dyn WrappedProvider + Send + Sync),
) -> Result<PreparedSagemakerRequest, Error> {
    // Build request body using WrappedProvider
    let request_body = hosted_provider.make_body(request).await?;

    // Inject extra body/headers
    let mut body_json = request_body;
    let http_extra_headers = inject_extra_request_data(
        &request.request.extra_body,
        &request.request.extra_headers,
        model_provider,
        request.model_name,
        &mut body_json,
    )?;

    // Sort for consistent ordering in tests
    if cfg!(feature = "e2e_tests") {
        body_json.sort_all_objects();
    }

    let raw_request = serde_json::to_string(&body_json).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize request: {e}"),
        })
    })?;
    let body_bytes = raw_request.as_bytes().to_vec();

    Ok(PreparedSagemakerRequest {
        raw_request,
        body_bytes,
        http_extra_headers,
    })
}

impl InferenceProvider for AWSSagemakerProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        // Save references we need later (these are all Copy or references)
        let inner_request = request.request;
        let model_name = request.model_name;
        let provider_name = request.provider_name;
        let model_inference_id = request.model_inference_id;

        // Prepare the request body
        let PreparedSagemakerRequest {
            raw_request,
            body_bytes,
            http_extra_headers,
        } = prepare_sagemaker_request(request, model_provider, &*self.hosted_provider).await?;

        // Build URL
        let base_url =
            self.config
                .get_base_url(dynamic_api_keys, "runtime.sagemaker", PROVIDER_TYPE)?;
        let url = format!(
            "{}/endpoints/{}/invocations",
            base_url,
            urlencoding::encode(&self.endpoint_name)
        );

        // Get credentials and region
        let credentials = self
            .config
            .get_request_credentials(dynamic_api_keys, PROVIDER_TYPE)
            .await?;
        let region = self.config.get_region(dynamic_api_keys, PROVIDER_TYPE)?;

        // Send signed request
        let aws_response = send_aws_request(
            http_client,
            &url,
            http_extra_headers,
            body_bytes,
            &credentials,
            region.as_ref(),
            "sagemaker",
            PROVIDER_TYPE,
            &raw_request,
        )
        .await?;

        let latency = Latency::NonStreaming {
            response_time: aws_response.response_time,
        };
        let raw_response = aws_response.raw_response;

        // Parse response using WrappedProvider
        self.hosted_provider.parse_response(
            inner_request,
            raw_request,
            raw_response,
            latency,
            model_name,
            provider_name,
            model_inference_id,
        )
    }

    async fn infer_stream<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        // Save references we need later
        let model_inference_id = request.model_inference_id;

        // Prepare the request body
        let PreparedSagemakerRequest {
            raw_request,
            body_bytes,
            http_extra_headers,
        } = prepare_sagemaker_request(request, model_provider, &*self.hosted_provider).await?;

        // Build URL for streaming endpoint
        let base_url =
            self.config
                .get_base_url(dynamic_api_keys, "runtime.sagemaker", PROVIDER_TYPE)?;
        let url = format!(
            "{}/endpoints/{}/invocations-response-stream",
            base_url,
            urlencoding::encode(&self.endpoint_name)
        );

        // Get credentials and region
        let credentials = self
            .config
            .get_request_credentials(dynamic_api_keys, PROVIDER_TYPE)
            .await?;
        let region = self.config.get_region(dynamic_api_keys, PROVIDER_TYPE)?;

        // Build headers
        let mut headers = http_extra_headers;
        headers.insert(
            http::header::CONTENT_TYPE,
            http::header::HeaderValue::from_static("application/json"),
        );

        // Sign the request
        let signed_headers = sign_request(
            "POST",
            &url,
            &headers,
            &body_bytes,
            &credentials,
            region.as_ref(),
            "sagemaker",
            PROVIDER_TYPE,
        )?;

        // Send request
        let start_time = Instant::now();
        let response = http_client
            .post(&url)
            .headers(signed_headers)
            .body(body_bytes)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error sending request to AWS Sagemaker: {e}"),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        let status = response.status();
        if !status.is_success() {
            let raw_response = response.text().await.unwrap_or_default();
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("AWS Sagemaker returned error status {status}: {raw_response}"),
                raw_request: Some(raw_request),
                raw_response: Some(raw_response),
                provider_type: PROVIDER_TYPE.to_string(),
            }));
        }

        // Create the stream - decode Smithy EventStream, extract SSE payloads
        let bytes_stream = response.bytes_stream();
        let raw_request_clone = raw_request.clone();

        // First, decode the outer Smithy EventStream and extract payload bytes
        let sagemaker_byte_stream = async_stream::stream! {
            let raw_request = raw_request_clone;
            let mut decoder = MessageFrameDecoder::new();
            let mut buffer = BytesMut::new();
            let mut bytes_stream = bytes_stream;

            while let Some(chunk_result) = bytes_stream.next().await {
                match chunk_result {
                    Err(e) => {
                        yield Err(TensorZeroEventError::TensorZero(Error::new(
                            ErrorDetails::InferenceServer {
                                message: format!("Error reading stream: {e}"),
                                raw_request: Some(raw_request.clone()),
                                raw_response: None,
                                provider_type: PROVIDER_TYPE.to_string(),
                            }
                        )));
                        return;
                    }
                    Ok(chunk) => {
                        buffer.extend_from_slice(&chunk);

                        // Try to decode frames from the buffer
                        loop {
                            match decoder.decode_frame(&mut buffer) {
                                Ok(DecodedFrame::Complete(message)) => {
                                    // Check for exception messages using shared helper
                                    if let Some((exception_type, error_message)) = check_eventstream_exception(&message) {
                                        yield Err(TensorZeroEventError::TensorZero(Error::new(
                                            ErrorDetails::InferenceServer {
                                                message: format!("AWS Sagemaker streaming exception: {exception_type}"),
                                                raw_request: Some(raw_request.clone()),
                                                raw_response: Some(error_message),
                                                provider_type: PROVIDER_TYPE.to_string(),
                                            }
                                        )));
                                        return;
                                    }

                                    // Extract event type - should be PayloadPart for data
                                    let event_type = message.headers().iter()
                                        .find(|h| h.name().as_str() == ":event-type")
                                        .and_then(|h| h.value().as_string().ok())
                                        .map(|s| s.as_str().to_owned());

                                    if event_type.as_deref() == Some("PayloadPart") {
                                        // The payload contains the raw bytes from the hosted model
                                        let payload = message.payload();
                                        if !payload.is_empty() {
                                            yield Ok(bytes::Bytes::copy_from_slice(payload));
                                        }
                                    }
                                }
                                Ok(DecodedFrame::Incomplete) => {
                                    // Need more data
                                    break;
                                }
                                Err(e) => {
                                    yield Err(TensorZeroEventError::TensorZero(Error::new(
                                        ErrorDetails::InferenceServer {
                                            message: format!("Error decoding event stream frame: {e}"),
                                            raw_request: Some(raw_request.clone()),
                                            raw_response: None,
                                            provider_type: PROVIDER_TYPE.to_string(),
                                        }
                                    )));
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        };

        // Second, convert the byte stream to SSE events using sse_stream
        // The payload bytes contain SSE text from the hosted model (OpenAI/TGI)
        let event_stream = futures::stream::iter([Ok(reqwest_sse_stream::Event::Open)]).chain(
            SseStream::from_byte_stream(sagemaker_byte_stream).filter_map(|r| async {
                match r {
                    Ok(sse) => {
                        // Only yield Message events when data is present
                        sse.data.map(|data| {
                            Ok(reqwest_sse_stream::Event::Message(
                                reqwest_sse_stream::MessageEvent {
                                    event: sse.event.unwrap_or_default(),
                                    data,
                                    id: sse.id.unwrap_or_default(),
                                },
                            ))
                        })
                    }
                    Err(e) => Some(Err(TensorZeroEventError::EventSource(Box::new(
                        reqwest_sse_stream::ReqwestSseStreamError::SseError(e),
                    )))),
                }
            }),
        );

        // Use WrappedProvider's stream_events to handle the inner SSE format
        let stream = self
            .hosted_provider
            .stream_events(
                Box::pin(event_stream),
                start_time.into(),
                &raw_request,
                model_inference_id,
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
