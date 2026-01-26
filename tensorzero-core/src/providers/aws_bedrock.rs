//! AWS Bedrock model provider using direct HTTP calls to the Converse API.

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_types::region::Region;
use bytes::BytesMut;
use futures::StreamExt;
use futures::future::try_join_all;
use reqwest::StatusCode;
use serde::Serialize;
use std::time::Duration;
use tokio::time::Instant;

use super::anthropic::{prefill_json_chunk_response, prefill_json_response};
use super::aws_common::{
    AWSCredentials, AWSEndpointUrl, AWSProviderConfig, AWSRegion, check_eventstream_exception,
    send_aws_request, sign_request,
};
use super::helpers::{inject_extra_request_data, peek_first_chunk};
use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::InferenceProvider;
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
};
use crate::inference::types::file::mime_type_to_ext;
use crate::inference::types::usage::raw_usage_entries_from_value;
use crate::inference::types::{
    ApiType, ContentBlock, ContentBlockChunk, ContentBlockOutput, FunctionType, Latency,
    ModelInferenceRequest, ModelInferenceRequestJsonMode, ObjectStorageFile,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, ProviderInferenceResponseStreamInner, RequestMessage,
    Role as TensorZeroRole, Text, TextChunk, Usage, batch::StartBatchProviderInferenceResponse,
};
use crate::inference::types::{FinishReason, ProviderInferenceResponseArgs, Thought, ThoughtChunk};
use crate::model::ModelProvider;
use crate::tool::{
    FunctionToolConfig, ToolCall, ToolCallChunk, ToolChoice as TensorZeroToolChoice,
};
use tensorzero_types_providers::aws_bedrock::{
    self as types, AdditionalModelRequestFields, ContentBlock as BedrockContentBlock,
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
    ConverseRequest, ConverseResponse, InferenceConfig, Message, MessageStopEvent, MetadataEvent,
    ResponseContentBlock, ResponseReasoningContent, Role, StopReason, SystemContentBlock,
    ThinkingConfig, ThinkingType, Tool, ToolChoice, ToolConfig, ToolInputSchema, ToolResultContent,
    ToolSpec,
};
use uuid::Uuid;

const PROVIDER_NAME: &str = "AWS Bedrock";
pub const PROVIDER_TYPE: &str = "aws_bedrock";

/// AWS Bedrock provider using direct HTTP calls.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AWSBedrockProvider {
    model_id: String,
    #[serde(skip)]
    config: AWSProviderConfig,
}

impl AWSBedrockProvider {
    pub async fn new(
        model_id: String,
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

        Ok(Self { model_id, config })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl InferenceProvider for AWSBedrockProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
            model_inference_id,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        // Prepare the request body
        let PreparedRequestBody {
            raw_request,
            body_bytes,
            http_extra_headers,
        } = prepare_request_body(&self.model_id, request, model_provider, model_name).await?;

        // Build URL
        let base_url =
            self.config
                .get_base_url(dynamic_api_keys, "bedrock-runtime", PROVIDER_TYPE)?;
        let url = format!(
            "{}/model/{}/converse",
            base_url,
            urlencoding::encode(&self.model_id)
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
            "bedrock",
            PROVIDER_TYPE,
            &raw_request,
        )
        .await?;

        let latency = Latency::NonStreaming {
            response_time: aws_response.response_time,
        };
        let raw_response = aws_response.raw_response;

        // Parse response
        let response: ConverseResponse = serde_json::from_str(&raw_response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing response from AWS Bedrock: {e}"),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        // Convert response to ProviderInferenceResponse
        convert_converse_response(
            response,
            latency,
            raw_request,
            raw_response,
            ResponseContext {
                system: request.system.clone(),
                input_messages: request.messages.clone(),
                model_id: &self.model_id,
                function_type: &request.function_type,
                json_mode: request.json_mode,
            },
            model_inference_id,
        )
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
            model_inference_id,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        // Prepare the request body
        let PreparedRequestBody {
            raw_request,
            body_bytes,
            http_extra_headers,
        } = prepare_request_body(&self.model_id, request, model_provider, model_name).await?;

        // Build URL for streaming endpoint
        let base_url =
            self.config
                .get_base_url(dynamic_api_keys, "bedrock-runtime", PROVIDER_TYPE)?;
        let url = format!(
            "{}/model/{}/converse-stream",
            base_url,
            urlencoding::encode(&self.model_id)
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
            "bedrock",
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
                    message: format!("Error sending request to AWS Bedrock: {e}"),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        let status = response.status();
        if !status.is_success() {
            let raw_response = response.text().await.unwrap_or_default();
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("AWS Bedrock returned error status {status}: {raw_response}"),
                raw_request: Some(raw_request),
                raw_response: Some(raw_response),
                provider_type: PROVIDER_TYPE.to_string(),
            }));
        }

        // Create the stream
        let bytes_stream = response.bytes_stream();
        let mut stream = stream_bedrock(
            bytes_stream,
            start_time,
            model_inference_id,
            raw_request.clone(),
        )
        .peekable();

        // Peek first chunk
        let chunk = peek_first_chunk(&mut stream, &raw_request, PROVIDER_TYPE).await?;

        // Handle JSON prefill for streaming.
        if needs_json_prefill(&self.model_id, request) {
            prefill_json_chunk_response(chunk);
        }

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

// =============================================================================
// Request Building
// =============================================================================

/// Prepared request body ready for signing and sending
struct PreparedRequestBody {
    raw_request: String,
    body_bytes: Vec<u8>,
    http_extra_headers: http::HeaderMap,
}

/// Prepare the request body: build converse request, apply JSON prefill, serialize, inject extras
async fn prepare_request_body(
    model_id: &str,
    request: &ModelInferenceRequest<'_>,
    model_provider: &ModelProvider,
    model_name: &str,
) -> Result<PreparedRequestBody, Error> {
    // Build the request body
    let mut converse_request =
        build_converse_request(request, &request.inference_params_v2).await?;

    // Add JSON prefill for Claude models in JSON mode
    if needs_json_prefill(model_id, request) {
        warn_bedrock_strict_json_mode(request.json_mode);
        prefill_json_converse_request(&mut converse_request);
    }

    // Serialize to JSON
    let mut body_json = serde_json::to_value(&converse_request).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize request: {e}"),
        })
    })?;

    // Inject extra body/headers
    let http_extra_headers = inject_extra_request_data(
        &request.extra_body,
        &request.extra_headers,
        model_provider,
        model_name,
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

    Ok(PreparedRequestBody {
        raw_request,
        body_bytes,
        http_extra_headers,
    })
}

/// Build a ConverseRequest from a ModelInferenceRequest
async fn build_converse_request(
    request: &ModelInferenceRequest<'_>,
    inference_params: &ChatCompletionInferenceParamsV2,
) -> Result<ConverseRequest, Error> {
    // Convert messages
    let messages: Vec<Message> = try_join_all(request.messages.iter().map(convert_request_message))
        .await?
        .into_iter()
        .filter(|m| !m.content.is_empty())
        .collect();

    // Build inference config
    let inference_config = Some(InferenceConfig {
        max_tokens: request.max_tokens.map(|t| t as i32),
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences: request
            .stop_sequences
            .as_ref()
            .map(|s| s.iter().cloned().collect()),
    });

    // Build system prompt
    let system = request
        .system
        .as_ref()
        .filter(|s| !s.is_empty())
        .map(|s| vec![SystemContentBlock::Text { text: s.clone() }]);

    // Build tool config
    let tool_config = if let Some(tc) = &request.tool_config {
        if matches!(tc.tool_choice, TensorZeroToolChoice::None) {
            None
        } else {
            let tools: Vec<Tool> = tc.strict_tools_available()?.map(convert_tool).collect();

            let tool_choice = convert_tool_choice(tc.tool_choice.clone());

            Some(ToolConfig {
                tools,
                tool_choice: Some(tool_choice),
            })
        }
    } else {
        None
    };

    // Build additional model request fields (for thinking, etc.) and warn about unsupported params
    let additional_model_request_fields = apply_inference_params(inference_params);

    Ok(ConverseRequest {
        messages,
        system,
        inference_config,
        tool_config,
        additional_model_request_fields,
    })
}

/// Check if JSON prefill is needed for Claude models
fn needs_json_prefill(model_id: &str, request: &ModelInferenceRequest<'_>) -> bool {
    needs_json_prefill_raw(model_id, &request.function_type, request.json_mode)
}

fn needs_json_prefill_raw(
    model_id: &str,
    function_type: &FunctionType,
    json_mode: ModelInferenceRequestJsonMode,
) -> bool {
    model_id.contains("claude")
        && matches!(function_type, FunctionType::Json)
        && matches!(
            json_mode,
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
        )
}

/// Warn if json_mode=strict is used since Bedrock doesn't support Anthropic's output_format
fn warn_bedrock_strict_json_mode(json_mode: ModelInferenceRequestJsonMode) {
    if matches!(json_mode, ModelInferenceRequestJsonMode::Strict) {
        tracing::warn!(
            "AWS Bedrock does not support Anthropic's structured outputs feature. \
            `json_mode = \"strict\"` will use prefill fallback instead of guaranteed schema compliance. \
            For strict JSON schema enforcement, use direct Anthropic."
        );
    }
}

/// Apply inference params and build additional model request fields.
/// Uses destructuring to ensure all params are handled when new ones are added.
fn apply_inference_params(
    inference_params: &ChatCompletionInferenceParamsV2,
) -> Option<AdditionalModelRequestFields> {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        warn_inference_parameter_not_supported(
            PROVIDER_NAME,
            "reasoning_effort",
            Some("Tip: You might want to use `thinking` for this provider."),
        );
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }

    // Build additional model request fields for thinking
    thinking_budget_tokens.map(|budget_tokens| AdditionalModelRequestFields {
        thinking: Some(ThinkingConfig {
            thinking_type: ThinkingType::Enabled,
            budget_tokens,
        }),
    })
}

/// Add JSON prefill message to the request
fn prefill_json_converse_request(request: &mut ConverseRequest) {
    request.messages.push(Message {
        role: Role::Assistant,
        content: vec![BedrockContentBlock::Text(types::TextBlock {
            text: "Here is the JSON requested:\n{".to_string(),
        })],
    });
}

/// Convert a TensorZero RequestMessage to a Bedrock Message
async fn convert_request_message(message: &RequestMessage) -> Result<Message, Error> {
    let role = match message.role {
        TensorZeroRole::User => Role::User,
        TensorZeroRole::Assistant => Role::Assistant,
    };

    let content: Vec<BedrockContentBlock> =
        try_join_all(message.content.iter().map(convert_content_block_to_bedrock))
            .await?
            .into_iter()
            .flatten()
            .collect();

    Ok(Message { role, content })
}

/// Convert a TensorZero ContentBlock to a Bedrock ContentBlock
async fn convert_content_block_to_bedrock(
    block: &ContentBlock,
) -> Result<Option<BedrockContentBlock>, Error> {
    match block {
        ContentBlock::Text(Text { text }) => {
            Ok(Some(BedrockContentBlock::Text(types::TextBlock {
                text: text.clone(),
            })))
        }
        ContentBlock::ToolCall(tool_call) => {
            let input: serde_json::Value =
                serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        raw_request: None,
                        raw_response: Some(tool_call.arguments.clone()),
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!(
                            "Error parsing tool call arguments as JSON: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?;

            Ok(Some(BedrockContentBlock::ToolUse(types::ToolUseBlock {
                tool_use: types::ToolUseData {
                    tool_use_id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    input,
                },
            })))
        }
        ContentBlock::ToolResult(tool_result) => Ok(Some(BedrockContentBlock::ToolResult(
            types::ToolResultBlock {
                tool_result: types::ToolResultData {
                    tool_use_id: tool_result.id.clone(),
                    content: vec![ToolResultContent::Text {
                        text: tool_result.result.clone(),
                    }],
                },
            },
        ))),
        ContentBlock::File(file) => {
            let resolved_file = file.resolve().await?;
            let ObjectStorageFile { file, data } = &*resolved_file;
            if file.detail.is_some() {
                tracing::warn!(
                    "The image detail parameter is not supported by AWS Bedrock. The `detail` field will be ignored."
                );
            }

            if file.mime_type.type_() == mime::IMAGE {
                Ok(Some(BedrockContentBlock::Image(types::ImageBlock {
                    image: types::ImageSource {
                        format: file.mime_type.subtype().to_string(),
                        source: types::ImageSourceData {
                            bytes: data.clone(),
                        },
                    },
                })))
            } else {
                let suffix = mime_type_to_ext(&file.mime_type)?.ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidMessage {
                        message: format!("Mime type {} has no filetype suffix", file.mime_type),
                    })
                })?;
                Ok(Some(BedrockContentBlock::Document(types::DocumentBlock {
                    document: types::DocumentSource {
                        format: suffix.to_string(),
                        name: "input".to_string(),
                        source: types::DocumentSourceData {
                            bytes: data.clone(),
                        },
                    },
                })))
            }
        }
        ContentBlock::Thought(thought) => {
            if let Some(text) = &thought.text {
                Ok(Some(BedrockContentBlock::ReasoningContent(
                    types::ReasoningContentBlock {
                        reasoning_content: types::ReasoningContent::ReasoningText(
                            types::ReasoningText {
                                text: text.clone(),
                                signature: thought.signature.clone(),
                            },
                        ),
                    },
                )))
            } else if thought.signature.is_some() {
                tracing::warn!(
                    "The TensorZero Gateway doesn't support redacted thinking for AWS Bedrock yet, as none of the models available at the time of implementation supported this content block correctly. If you're seeing this warning, this means that something must have changed, so please reach out to our team and we'll quickly collaborate on a solution. For now, the gateway will discard such content blocks."
                );
                Ok(None)
            } else {
                tracing::warn!(
                    "The gateway received a reasoning content block with neither text nor signature. This is unsupported, so we'll drop it."
                );
                Ok(None)
            }
        }
        ContentBlock::Unknown(_) => Err(Error::new(ErrorDetails::UnsupportedContentBlockType {
            content_block_type: "unknown".to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
        })),
    }
}

/// Convert a FunctionToolConfig to a Bedrock Tool
fn convert_tool(tool_config: &FunctionToolConfig) -> Tool {
    Tool {
        tool_spec: ToolSpec {
            name: tool_config.name().to_string(),
            description: tool_config.description().to_string(),
            input_schema: ToolInputSchema {
                json: tool_config.parameters().clone(),
            },
        },
    }
}

/// Convert a TensorZero ToolChoice to a Bedrock ToolChoice.
/// Note: ToolChoice::None is filtered out in build_converse_request before calling this function.
fn convert_tool_choice(choice: TensorZeroToolChoice) -> ToolChoice {
    match choice {
        TensorZeroToolChoice::Auto | TensorZeroToolChoice::None => {
            ToolChoice::Auto(types::AutoToolChoice {})
        }
        TensorZeroToolChoice::Required => ToolChoice::Any(types::AnyToolChoice {}),
        TensorZeroToolChoice::Specific(name) => {
            ToolChoice::Tool(types::SpecificToolChoice { name })
        }
    }
}

// =============================================================================
// Response Conversion
// =============================================================================

/// Context needed for response conversion
struct ResponseContext<'a> {
    system: Option<String>,
    input_messages: Vec<RequestMessage>,
    model_id: &'a str,
    function_type: &'a FunctionType,
    json_mode: ModelInferenceRequestJsonMode,
}

/// Convert a ConverseResponse to a ProviderInferenceResponse
fn convert_converse_response(
    response: ConverseResponse,
    latency: Latency,
    raw_request: String,
    raw_response: String,
    ctx: ResponseContext<'_>,
    model_inference_id: Uuid,
) -> Result<ProviderInferenceResponse, Error> {
    let message = response.output.message.ok_or_else(|| {
        Error::new(ErrorDetails::InferenceServer {
            raw_request: None,
            raw_response: Some(raw_response.clone()),
            message: "AWS Bedrock returned an empty message.".to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;

    // Convert content blocks
    let mut content: Vec<ContentBlockOutput> = message
        .content
        .into_iter()
        .map(convert_response_content_block)
        .filter_map(Result::transpose)
        .collect::<Result<Vec<_>, _>>()?;

    // Apply JSON prefill adjustment
    if needs_json_prefill_raw(ctx.model_id, ctx.function_type, ctx.json_mode) {
        content = prefill_json_response(content)?;
    }

    // Extract usage - include cache tokens in input_tokens
    // AWS Bedrock reports cache tokens separately from input_tokens
    let total_input_tokens = response.usage.input_tokens as u32
        + response.usage.cache_read_input_tokens.unwrap_or(0) as u32
        + response.usage.cache_write_input_tokens.unwrap_or(0) as u32;
    let usage = Usage {
        input_tokens: Some(total_input_tokens),
        output_tokens: Some(response.usage.output_tokens as u32),
    };

    // Extract raw usage from response
    let raw_usage = extract_raw_usage_from_response(&raw_response).map(|value| {
        raw_usage_entries_from_value(
            model_inference_id,
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
            value,
        )
    });

    Ok(ProviderInferenceResponse::new(
        ProviderInferenceResponseArgs {
            output: content,
            system: ctx.system,
            input_messages: ctx.input_messages,
            raw_request,
            raw_response,
            usage,
            raw_usage,
            relay_raw_response: None,
            provider_latency: latency,
            finish_reason: Some(convert_stop_reason(response.stop_reason)),
            id: model_inference_id,
        },
    ))
}

/// Convert a Bedrock response content block to a TensorZero ContentBlockOutput
fn convert_response_content_block(
    block: ResponseContentBlock,
) -> Result<Option<ContentBlockOutput>, Error> {
    match block {
        ResponseContentBlock::Text(text) => Ok(Some(text.into())),
        ResponseContentBlock::ToolUse {
            tool_use_id,
            name,
            input,
        } => {
            let arguments = serde_json::to_string(&input).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    raw_request: None,
                    raw_response: None,
                    message: format!(
                        "Error serializing tool call arguments: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            Ok(Some(ContentBlockOutput::ToolCall(ToolCall {
                name,
                arguments,
                id: tool_use_id,
            })))
        }
        ResponseContentBlock::ReasoningContent(reasoning) => match reasoning {
            ResponseReasoningContent::ReasoningText { text, signature } => {
                Ok(Some(ContentBlockOutput::Thought(Thought {
                    text: Some(text),
                    summary: None,
                    signature,
                    provider_type: Some(PROVIDER_TYPE.to_string()),
                    extra_data: None,
                })))
            }
            ResponseReasoningContent::RedactedContent(_) => {
                tracing::warn!(
                    "The TensorZero Gateway doesn't support redacted thinking for AWS Bedrock yet."
                );
                Ok(None)
            }
        },
    }
}

/// Convert a Bedrock StopReason to a TensorZero FinishReason
fn convert_stop_reason(stop_reason: StopReason) -> FinishReason {
    match stop_reason {
        StopReason::EndTurn => FinishReason::Stop,
        StopReason::ToolUse => FinishReason::ToolCall,
        StopReason::MaxTokens => FinishReason::Length,
        StopReason::StopSequence => FinishReason::StopSequence,
        StopReason::ContentFiltered | StopReason::GuardrailIntervened => {
            FinishReason::ContentFilter
        }
        StopReason::Unknown => FinishReason::Unknown,
    }
}

/// Extract raw usage from response JSON
fn extract_raw_usage_from_response(raw_response: &str) -> Option<serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(raw_response)
        .ok()
        .and_then(|value| value.get("usage").filter(|v| !v.is_null()).cloned())
}

// =============================================================================
// Streaming
// =============================================================================

/// Create a stream that processes the Bedrock event stream
fn stream_bedrock<S>(
    bytes_stream: S,
    start_time: Instant,
    model_inference_id: Uuid,
    raw_request: String,
) -> ProviderInferenceResponseStreamInner
where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
{
    Box::pin(async_stream::stream! {
        let mut decoder = MessageFrameDecoder::new();
        let mut buffer = BytesMut::new();
        let mut current_tool_id: Option<String> = None;
        let mut bytes_stream = bytes_stream;

        while let Some(chunk_result) = bytes_stream.next().await {
            match chunk_result {
                Err(e) => {
                    yield Err(ErrorDetails::InferenceServer {
                        raw_request: Some(raw_request.clone()),
                        raw_response: None,
                        message: format!("Error reading stream: {e}"),
                        provider_type: PROVIDER_TYPE.to_string(),
                    }.into());
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
                                    yield Err(ErrorDetails::InferenceServer {
                                        raw_request: Some(raw_request.clone()),
                                        raw_response: Some(error_message),
                                        message: format!("AWS Bedrock streaming exception: {exception_type}"),
                                        provider_type: PROVIDER_TYPE.to_string(),
                                    }.into());
                                    return;
                                }

                                // Extract event type from headers for normal events
                                let event_type = message.headers().iter()
                                    .find(|h| h.name().as_str() == ":event-type")
                                    .and_then(|h| h.value().as_string().ok())
                                    .map(|s| s.as_str().to_owned());

                                // Parse the JSON payload
                                let payload = message.payload();
                                let message_latency = start_time.elapsed();

                                match process_stream_event(
                                    event_type.as_deref(),
                                    payload,
                                    message_latency,
                                    &mut current_tool_id,
                                    model_inference_id,
                                ) {
                                    Ok(None) => {},
                                    Ok(Some(chunk)) => yield Ok(chunk),
                                    Err(e) => yield Err(e),
                                }
                            }
                            Ok(DecodedFrame::Incomplete) => {
                                // Need more data
                                break;
                            }
                            Err(e) => {
                                yield Err(ErrorDetails::InferenceServer {
                                    raw_request: Some(raw_request.clone()),
                                    raw_response: None,
                                    message: format!("Error decoding event stream frame: {e}"),
                                    provider_type: PROVIDER_TYPE.to_string(),
                                }.into());
                                return;
                            }
                        }
                    }
                }
            }
        }
    })
}

/// Parse a stream event payload into a typed struct
fn parse_stream_event<T: serde::de::DeserializeOwned>(
    payload: &[u8],
    event_name: &str,
    raw_message: &str,
) -> Result<T, Error> {
    serde_json::from_slice(payload).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            raw_request: None,
            raw_response: Some(raw_message.to_string()),
            message: format!("Error parsing {event_name}: {e}"),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })
}

/// Process a single stream event
fn process_stream_event(
    event_type: Option<&str>,
    payload: &[u8],
    message_latency: Duration,
    current_tool_id: &mut Option<String>,
    model_inference_id: Uuid,
) -> Result<Option<ProviderInferenceResponseChunk>, Error> {
    let raw_message = String::from_utf8_lossy(payload).to_string();

    match event_type {
        Some("messageStart") => {
            // Just signals start of message, no content to yield
            Ok(None)
        }
        Some("contentBlockStart") => {
            let event: ContentBlockStartEvent =
                parse_stream_event(payload, "contentBlockStart", &raw_message)?;

            match event.start {
                Some(ContentBlockStart::ToolUse { tool_use_id, name }) => {
                    *current_tool_id = Some(tool_use_id.clone());
                    Ok(Some(ProviderInferenceResponseChunk::new(
                        vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                            id: tool_use_id,
                            raw_name: Some(name),
                            raw_arguments: String::new(),
                        })],
                        None,
                        raw_message,
                        message_latency,
                        None,
                    )))
                }
                None => Ok(None),
            }
        }
        Some("contentBlockDelta") => {
            let event: ContentBlockDeltaEvent =
                parse_stream_event(payload, "contentBlockDelta", &raw_message)?;

            match event.delta {
                Some(ContentBlockDelta::Text(text)) => {
                    Ok(Some(ProviderInferenceResponseChunk::new(
                        vec![ContentBlockChunk::Text(TextChunk {
                            text,
                            id: event.content_block_index.to_string(),
                        })],
                        None,
                        raw_message,
                        message_latency,
                        None,
                    )))
                }
                Some(ContentBlockDelta::ToolUse { input }) => {
                    let tool_id = current_tool_id.clone().ok_or_else(|| {
                        Error::new(ErrorDetails::InferenceServer {
                            message: "Got tool use delta without current tool id".to_string(),
                            provider_type: PROVIDER_TYPE.to_string(),
                            raw_request: None,
                            raw_response: None,
                        })
                    })?;
                    Ok(Some(ProviderInferenceResponseChunk::new(
                        vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                            id: tool_id,
                            raw_name: None,
                            raw_arguments: input,
                        })],
                        None,
                        raw_message,
                        message_latency,
                        None,
                    )))
                }
                Some(ContentBlockDelta::ReasoningContent(reasoning)) => match reasoning {
                    types::ReasoningDelta::Text(text) => {
                        Ok(Some(ProviderInferenceResponseChunk::new(
                            vec![ContentBlockChunk::Thought(ThoughtChunk {
                                id: event.content_block_index.to_string(),
                                text: Some(text),
                                summary_id: None,
                                summary_text: None,
                                signature: None,
                                provider_type: Some(PROVIDER_TYPE.to_string()),
                                extra_data: None,
                            })],
                            None,
                            raw_message,
                            message_latency,
                            None,
                        )))
                    }
                    types::ReasoningDelta::Signature(signature) => {
                        Ok(Some(ProviderInferenceResponseChunk::new(
                            vec![ContentBlockChunk::Thought(ThoughtChunk {
                                id: event.content_block_index.to_string(),
                                text: None,
                                summary_id: None,
                                summary_text: None,
                                signature: Some(signature),
                                provider_type: Some(PROVIDER_TYPE.to_string()),
                                extra_data: None,
                            })],
                            None,
                            raw_message,
                            message_latency,
                            None,
                        )))
                    }
                    types::ReasoningDelta::RedactedContent(_) => {
                        tracing::warn!(
                            "The TensorZero Gateway doesn't support redacted thinking for AWS Bedrock yet."
                        );
                        Ok(None)
                    }
                },
                None => Ok(None),
            }
        }
        Some("contentBlockStop") => Ok(None),
        Some("messageStop") => {
            let event: MessageStopEvent = parse_stream_event(payload, "messageStop", &raw_message)?;

            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![],
                None,
                raw_message,
                message_latency,
                Some(convert_stop_reason(event.stop_reason)),
            )))
        }
        Some("metadata") => {
            // Parse into typed struct for structured usage
            let event: MetadataEvent = parse_stream_event(payload, "metadata", &raw_message)?;

            // Extract raw usage directly from the JSON payload
            let raw_usage = serde_json::from_slice::<serde_json::Value>(payload)
                .ok()
                .and_then(|value| value.get("usage").filter(|v| !v.is_null()).cloned())
                .map(|usage_value| {
                    raw_usage_entries_from_value(
                        model_inference_id,
                        PROVIDER_TYPE,
                        ApiType::ChatCompletions,
                        usage_value,
                    )
                });

            // Include cache tokens in input_tokens
            // AWS Bedrock reports cache tokens separately from input_tokens
            let total_input_tokens = event.usage.input_tokens as u32
                + event.usage.cache_read_input_tokens.unwrap_or(0) as u32
                + event.usage.cache_write_input_tokens.unwrap_or(0) as u32;
            let usage = Some(Usage {
                input_tokens: Some(total_input_tokens),
                output_tokens: Some(event.usage.output_tokens as u32),
            });

            Ok(Some(ProviderInferenceResponseChunk::new_with_raw_usage(
                vec![],
                usage,
                raw_message,
                message_latency,
                None,
                raw_usage,
            )))
        }
        _ => {
            tracing::warn!("Unknown event type from AWS Bedrock: {:?}", event_type);
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::testing::reset_capture_logs;

    #[tokio::test]
    async fn test_get_aws_bedrock_client_no_aws_credentials() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Every call should trigger client creation since each provider has its own AWS Bedrock client
        AWSBedrockProvider::new(
            "test".to_string(),
            Some(Region::new("uk-hogwarts-1")),
            None,
            None,
            AWSCredentials::Sdk,
        )
        .await
        .unwrap();

        assert!(logs_contain(
            "Creating new AWS config for region: uk-hogwarts-1"
        ));

        reset_capture_logs();

        AWSBedrockProvider::new(
            "test".to_string(),
            Some(Region::new("uk-hogwarts-1")),
            None,
            None,
            AWSCredentials::Sdk,
        )
        .await
        .unwrap();

        assert!(logs_contain(
            "Creating new AWS config for region: uk-hogwarts-1"
        ));

        reset_capture_logs();

        // We want auto-detection to fail, so we clear this environment variable.
        // We use 'nextest' as our runner, so each test runs in its own process
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_REGION");
        tensorzero_unsafe_helpers::remove_env_var_tests_only("AWS_DEFAULT_REGION");
        let err =
            AWSBedrockProvider::new("test".to_string(), None, None, None, AWSCredentials::Sdk)
                .await
                .expect_err("AWS Bedrock provider should fail when it cannot detect region");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Failed to determine AWS region."),
            "Unexpected error message: {err_msg}"
        );

        assert!(logs_contain("Failed to determine AWS region."));

        reset_capture_logs();

        AWSBedrockProvider::new(
            "test".to_string(),
            Some(Region::new("me-shire-2")),
            None,
            None,
            AWSCredentials::Sdk,
        )
        .await
        .unwrap();

        assert!(logs_contain(
            "Creating new AWS config for region: me-shire-2"
        ));
    }
}
