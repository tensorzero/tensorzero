//! AWS SageMaker model provider using direct HTTP calls.

use aws_credential_types::Credentials;
use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_types::SdkConfig;
use aws_types::region::Region;
use bytes::BytesMut;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::StatusCode;
use secrecy::ExposeSecret;
use serde::Serialize;
use std::time::Instant;

use super::aws_common::{
    self, AWSCredentials, AWSEndpointUrl, AWSRegion, get_credentials, sign_request,
    warn_if_credential_exfiltration_risk,
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
use eventsource_stream::EventStreamError;

#[expect(unused)]
const PROVIDER_NAME: &str = "AWS Sagemaker";
const PROVIDER_TYPE: &str = "aws_sagemaker";

/// AWS SageMaker provider using direct HTTP calls.
#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AWSSagemakerProvider {
    endpoint_name: String,
    #[serde(skip)]
    sdk_config: SdkConfig,
    #[serde(skip)] // TODO: add a way to Serialize the WrappedProvider
    pub hosted_provider: Box<dyn WrappedProvider + Send + Sync>,
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
        // Get the SDK config for credential loading
        let sdk_config = aws_common::config_with_region(PROVIDER_TYPE, static_region).await?;

        // Warn about potential credential exfiltration risk
        warn_if_credential_exfiltration_risk(&endpoint_url, &credentials, PROVIDER_TYPE);

        Ok(Self {
            endpoint_name,
            sdk_config,
            hosted_provider,
            region,
            endpoint_url,
            credentials,
        })
    }

    /// Get the base URL for the SageMaker Runtime API.
    fn get_base_url(&self, dynamic_api_keys: &InferenceCredentials) -> Result<String, Error> {
        if let Some(endpoint_url) = &self.endpoint_url {
            let url = endpoint_url.resolve(dynamic_api_keys)?;
            Ok(url.to_string().trim_end_matches('/').to_string())
        } else {
            let region = self.get_region(dynamic_api_keys)?;
            Ok(format!(
                "https://runtime.sagemaker.{}.amazonaws.com",
                region.as_ref()
            ))
        }
    }

    /// Get the region for this request.
    fn get_region(&self, dynamic_api_keys: &InferenceCredentials) -> Result<Region, Error> {
        if let Some(region) = &self.region {
            region.resolve(dynamic_api_keys)
        } else {
            // Use the region from the SDK config
            self.sdk_config.region().cloned().ok_or_else(|| {
                Error::new(ErrorDetails::InferenceClient {
                    raw_request: None,
                    raw_response: None,
                    status_code: Some(StatusCode::INTERNAL_SERVER_ERROR),
                    message: "No region configured".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })
        }
    }

    /// Get credentials for this request.
    async fn get_request_credentials(
        &self,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<Credentials, Error> {
        match &self.credentials {
            AWSCredentials::Static {
                access_key_id,
                secret_access_key,
                session_token,
            } => {
                // Use static credentials directly
                Ok(Credentials::new(
                    access_key_id.clone(),
                    secret_access_key.expose_secret().to_string(),
                    session_token
                        .as_ref()
                        .map(|st| st.expose_secret().to_string()),
                    None,
                    "tensorzero",
                ))
            }
            AWSCredentials::Dynamic {
                access_key_id_key,
                secret_access_key_key,
                session_token_key,
            } => {
                // Resolve dynamic credentials from the request
                let ak = dynamic_api_keys.get(access_key_id_key).ok_or_else(|| {
                    Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: "aws".to_string(),
                        message: format!(
                            "Dynamic `access_key_id` with key `{access_key_id_key}` is missing"
                        ),
                    })
                })?;
                let sk = dynamic_api_keys.get(secret_access_key_key).ok_or_else(|| {
                    Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: "aws".to_string(),
                        message: format!(
                            "Dynamic `secret_access_key` with key `{secret_access_key_key}` is missing"
                        ),
                    })
                })?;
                let st = session_token_key
                    .as_ref()
                    .map(|key| {
                        dynamic_api_keys.get(key).ok_or_else(|| {
                            Error::new(ErrorDetails::ApiKeyMissing {
                                provider_name: "aws".to_string(),
                                message: format!(
                                    "Dynamic `session_token` with key `{key}` is missing"
                                ),
                            })
                        })
                    })
                    .transpose()?;

                Ok(Credentials::new(
                    ak.expose_secret().to_string(),
                    sk.expose_secret().to_string(),
                    st.map(|s| s.expose_secret().to_string()),
                    None,
                    "tensorzero",
                ))
            }
            AWSCredentials::Sdk => get_credentials(&self.sdk_config, PROVIDER_TYPE).await,
        }
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
        // Build request body using WrappedProvider
        let request_body = self.hosted_provider.make_body(request).await?;

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

        // Build URL
        let base_url = self.get_base_url(dynamic_api_keys)?;
        let url = format!(
            "{}/endpoints/{}/invocations",
            base_url,
            urlencoding::encode(&self.endpoint_name)
        );

        // Get credentials and region
        let credentials = self.get_request_credentials(dynamic_api_keys).await?;
        let region = self.get_region(dynamic_api_keys)?;

        // Build headers
        let mut headers = http_extra_headers;
        headers.insert(
            http::header::CONTENT_TYPE,
            http::header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            http::header::ACCEPT,
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

        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };

        let status = response.status();
        let raw_response = response.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error reading response from AWS Sagemaker: {e}"),
                raw_request: Some(raw_request.clone()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        if !status.is_success() {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("AWS Sagemaker returned error status {status}: {raw_response}"),
                raw_request: Some(raw_request),
                raw_response: Some(raw_response),
                provider_type: PROVIDER_TYPE.to_string(),
            }));
        }

        // Parse response using WrappedProvider
        self.hosted_provider.parse_response(
            request.request,
            raw_request,
            raw_response,
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
        // Build request body using WrappedProvider
        let request_body = self.hosted_provider.make_body(request).await?;

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

        // Build URL for streaming endpoint
        let base_url = self.get_base_url(dynamic_api_keys)?;
        let url = format!(
            "{}/endpoints/{}/invocations-response-stream",
            base_url,
            urlencoding::encode(&self.endpoint_name)
        );

        // Get credentials and region
        let credentials = self.get_request_credentials(dynamic_api_keys).await?;
        let region = self.get_region(dynamic_api_keys)?;

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
                                    // Check for exception messages
                                    let message_type = message.headers().iter()
                                        .find(|h| h.name().as_str() == ":message-type")
                                        .and_then(|h| h.value().as_string().ok())
                                        .map(|s| s.as_str().to_owned());

                                    if message_type.as_deref() == Some("exception") {
                                        let exception_type = message.headers().iter()
                                            .find(|h| h.name().as_str() == ":exception-type")
                                            .and_then(|h| h.value().as_string().ok())
                                            .map(|s| s.as_str().to_owned())
                                            .unwrap_or_else(|| "unknown".to_string());

                                        let error_message = String::from_utf8_lossy(message.payload()).to_string();

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
                                            yield Ok(payload.to_vec());
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

        // Second, convert the byte stream to SSE events using eventsource_stream
        // The payload bytes contain SSE text from the hosted model (OpenAI/TGI)
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

        // Use WrappedProvider's stream_events to handle the inner SSE format
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
