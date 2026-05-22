//! AWS Bedrock's Anthropic Converse API via Bedrock runtime endpoint.
//!
//! See:
//! - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use bytes::BytesMut;
use futures::StreamExt;
use futures::future::try_join_all;
use reqwest::StatusCode;
use secrecy::ExposeSecret;
use std::time::Duration;
use tensorzero_inference_types::FunctionToolDef;
use tensorzero_types_providers::aws_bedrock::{
    self as types, AdditionalModelRequestFields, ContentBlock as BedrockContentBlock,
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
    ConverseRequest, ConverseResponse, InferenceConfig, Message, MessageStopEvent, MetadataEvent,
    ResponseContentBlock, ResponseReasoningContent, Role, StopReason, SystemContentBlock,
    ThinkingConfig, ThinkingType, Tool, ToolChoice, ToolConfig, ToolInputSchema, ToolResultContent,
    ToolSpec,
};
use tokio::time::Instant;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
};
use crate::inference::types::file::mime_type_to_ext;
use crate::inference::types::resolved_input::LazyFileExt;
use crate::inference::types::usage::raw_usage_entries_from_value;
use crate::inference::types::{
    ApiType, ContentBlock, ContentBlockChunk, ContentBlockOutput, FunctionType, Latency,
    ModelInferenceRequest, ModelInferenceRequestJsonMode, ObjectStorageFile,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, RequestMessage, Role as TensorZeroRole, Text, TextChunk,
    Usage,
};
use crate::inference::types::{FinishReason, Thought, ThoughtChunk};
use crate::model::{ModelProviderRequestInfo, ProviderInferenceRequest};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice as TensorZeroToolChoice};

use super::anthropic::{prefill_json_chunk_response, prefill_json_response};
use super::aws_bedrock::{AWSBedrockProvider, PROVIDER_TYPE};
use super::aws_common::{check_eventstream_exception, send_aws_request};
use super::helpers::{inject_extra_request_data, peek_first_chunk};

const PROVIDER_NAME: &str = "AWS Bedrock";

/// Build HTTP headers with bearer token authentication.
///
/// Adds Content-Type and Authorization headers to the provided header map.
fn add_request_headers_converse_api(
    mut headers: http::HeaderMap,
    api_key: Option<&secrecy::SecretString>,
) -> Result<http::HeaderMap, Error> {
    headers.insert(
        http::header::CONTENT_TYPE,
        http::header::HeaderValue::from_static("application/json"),
    );
    if let Some(api_key) = api_key {
        headers.insert(
            http::header::AUTHORIZATION,
            http::header::HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                .map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Invalid API key format: {e}"),
                    })
                })?,
        );
    }
    Ok(headers)
}

impl AWSBedrockProvider {
    pub(super) async fn infer_converse<'a>(
        &'a self,
        ProviderInferenceRequest {
            request,
            provider_name: _,
            model_name,
            model_inference_id,
        }: ProviderInferenceRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProviderRequestInfo,
    ) -> Result<ProviderInferenceResponse, Error> {
        // Prepare the request body
        let PreparedRequestBody {
            raw_request,
            body_bytes,
            http_extra_headers,
        } = prepare_request_body(self.model_id(), request, model_provider, model_name).await?;

        // Build URL
        let base_url = self.get_base_url(dynamic_api_keys, ApiType::ChatCompletions)?;
        let url = format!(
            "{base_url}/model/{}/converse",
            urlencoding::encode(self.model_id())
        );

        // Build headers based on auth type
        let request_headers = self
            .build_request_headers(
                dynamic_api_keys,
                &url,
                http_extra_headers,
                &body_bytes,
                PROVIDER_NAME,
                add_request_headers_converse_api,
            )
            .await?;

        // Send request with appropriate auth method
        let aws_response = send_aws_request(
            http_client,
            &url,
            request_headers,
            body_bytes,
            "bedrock-runtime",
            PROVIDER_TYPE,
            &raw_request,
            ApiType::ChatCompletions,
        )
        .await?;

        let latency = Latency::NonStreaming {
            response_time: aws_response.response_time,
        };
        let raw_response = aws_response.raw_response;

        // Parse response
        let response: ConverseResponse = serde_json::from_str(&raw_response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing response from {PROVIDER_NAME}: {e}"),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
                api_type: ApiType::ChatCompletions,
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
                model_id: self.model_id(),
                function_type: &request.function_type,
                json_mode: request.json_mode,
            },
            model_inference_id,
        )
    }

    pub(super) async fn infer_stream_converse<'a>(
        &'a self,
        ProviderInferenceRequest {
            request,
            provider_name: _,
            model_name,
            model_inference_id,
        }: ProviderInferenceRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProviderRequestInfo,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        // Prepare the request body
        let PreparedRequestBody {
            raw_request,
            body_bytes,
            http_extra_headers,
        } = prepare_request_body(self.model_id(), request, model_provider, model_name).await?;

        // Build URL for streaming endpoint
        let base_url = self.get_base_url(dynamic_api_keys, ApiType::ChatCompletions)?;
        let url = format!(
            "{base_url}/model/{}/converse-stream",
            urlencoding::encode(self.model_id())
        );

        // Build headers based on auth type
        let request_headers = self
            .build_request_headers(
                dynamic_api_keys,
                &url,
                http_extra_headers,
                &body_bytes,
                PROVIDER_NAME,
                add_request_headers_converse_api,
            )
            .await?;

        // Send request
        let start_time = Instant::now();
        let response = http_client
            .post(&url)
            .headers(request_headers)
            .body(body_bytes)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error sending request to {PROVIDER_NAME}: {e}"),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    api_type: ApiType::ChatCompletions,
                })
            })?;

        let status = response.status();
        if !status.is_success() {
            let raw_response = response.text().await.unwrap_or_default();
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!("{PROVIDER_NAME} returned error status {status}: {raw_response}"),
                raw_request: Some(raw_request),
                raw_response: Some(raw_response),
                provider_type: PROVIDER_TYPE.to_string(),
                api_type: ApiType::ChatCompletions,
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
        let chunk = peek_first_chunk(
            &mut stream,
            &raw_request,
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
        )
        .await?;

        // Handle JSON prefill for streaming.
        if needs_json_prefill(self.model_id(), &request.function_type, request.json_mode) {
            prefill_json_chunk_response(chunk);
        }

        Ok((stream, raw_request))
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
    model_provider: &ModelProviderRequestInfo,
    model_name: &str,
) -> Result<PreparedRequestBody, Error> {
    // Build the request body
    let mut converse_request = build_request(request, &request.inference_params_v2).await?;

    // Add JSON prefill for Claude models in JSON mode
    if needs_json_prefill(model_id, &request.function_type, request.json_mode) {
        warn_bedrock_strict_json_mode(request.json_mode);
        prefill_json_request(&mut converse_request);
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
async fn build_request(
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
fn needs_json_prefill(
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
            "{PROVIDER_NAME} does not support Anthropic's structured outputs feature. \
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
fn prefill_json_request(request: &mut ConverseRequest) {
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
                        api_type: ApiType::ChatCompletions,
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
                    "The image detail parameter is not supported by {PROVIDER_NAME}. The `detail` field will be ignored."
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
                    "The TensorZero Gateway doesn't support redacted thinking for {PROVIDER_NAME} yet, as none of the models available at the time of implementation supported this content block correctly. If you're seeing this warning, this means that something must have changed, so please reach out to our team and we'll quickly collaborate on a solution. For now, the gateway will discard such content blocks."
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

/// Convert a FunctionToolDef to a Bedrock Tool
fn convert_tool(tool_config: &FunctionToolDef) -> Tool {
    Tool {
        tool_spec: ToolSpec {
            name: tool_config.name.clone(),
            description: tool_config.description.clone(),
            input_schema: ToolInputSchema {
                json: tool_config.parameters.clone(),
            },
        },
    }
}

/// Convert a TensorZero ToolChoice to a Bedrock ToolChoice.
/// Note: ToolChoice::None is filtered out in build_request before calling this function.
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
            message: format!("{PROVIDER_NAME} returned an empty message."),
            provider_type: PROVIDER_TYPE.to_string(),
            api_type: ApiType::ChatCompletions,
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
    if needs_json_prefill(ctx.model_id, ctx.function_type, ctx.json_mode) {
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
        provider_cache_read_input_tokens: response.usage.cache_read_input_tokens.map(|v| v as u32),
        provider_cache_write_input_tokens: response
            .usage
            .cache_write_input_tokens
            .map(|v| v as u32),
        cost: None,
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
            id: model_inference_id,
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
                    api_type: ApiType::ChatCompletions,
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
                    "The TensorZero Gateway doesn't support redacted thinking for {PROVIDER_NAME} yet."
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
                        api_type: ApiType::ChatCompletions,
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
                                        message: format!("{PROVIDER_NAME} streaming exception: {exception_type}"),
                                        provider_type: PROVIDER_TYPE.to_string(),
                                        api_type: ApiType::ChatCompletions,
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
                                    api_type: ApiType::ChatCompletions,
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
            api_type: ApiType::ChatCompletions,
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
                            api_type: ApiType::ChatCompletions,
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
                            "The TensorZero Gateway doesn't support redacted thinking for {PROVIDER_NAME} yet."
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
                provider_cache_read_input_tokens: event
                    .usage
                    .cache_read_input_tokens
                    .map(|v| v as u32),
                provider_cache_write_input_tokens: event
                    .usage
                    .cache_write_input_tokens
                    .map(|v| v as u32),
                cost: None,
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
            tracing::warn!("Unknown event type from {PROVIDER_NAME}: {:?}", event_type);
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;

    use googletest::prelude::*;
    use uuid::Uuid;

    use super::*;
    use crate::error::ErrorDetails;
    use crate::inference::types::chat_completion_inference_params::{
        ChatCompletionInferenceParamsV2, ServiceTier,
    };
    use crate::inference::types::{
        ContentBlockOutput, FunctionType, ModelInferenceRequestJsonMode, RequestMessage,
        Role as TensorZeroRole,
    };
    use crate::model::ModelProviderRequestInfo;
    use crate::providers::test_helpers::WEATHER_PROVIDER_TOOL_CONFIG;
    use crate::tool::ToolChoice as TensorZeroToolChoice;
    use tensorzero_inference_types::ProviderToolCallConfig;
    use tensorzero_types_providers::aws_bedrock::{
        ConverseOutput, ConverseResponse, ResponseContentBlock, ResponseMessage,
    };

    fn sample_model_provider() -> ModelProviderRequestInfo {
        ModelProviderRequestInfo {
            provider_name: PROVIDER_TYPE.into(),
            extra_headers: None,
            extra_body: None,
            discard_unknown_chunks: false,
        }
    }

    fn sample_inference_request(messages: Vec<RequestMessage>) -> ModelInferenceRequest<'static> {
        ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages,
            system: Some("test_system".to_string()),
            tool_config: None,
            max_tokens: Some(100),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            ..Default::default()
        }
    }

    #[gtest]
    fn test_needs_json_prefill() {
        expect_that!(
            needs_json_prefill(
                "anthropic.claude-sonnet-4-20250514-v1:0",
                &FunctionType::Json,
                ModelInferenceRequestJsonMode::On,
            ),
            eq(true)
        );
        expect_that!(
            needs_json_prefill(
                "amazon.nova-lite-v1:0",
                &FunctionType::Json,
                ModelInferenceRequestJsonMode::On,
            ),
            eq(false)
        );
    }

    #[gtest]
    fn test_apply_inference_params_thinking_budget_tokens() {
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: None,
            service_tier: Some(ServiceTier::Auto),
            thinking_budget_tokens: Some(1024),
            verbosity: None,
        };

        let Some(AdditionalModelRequestFields {
            thinking:
                Some(ThinkingConfig {
                    thinking_type: ThinkingType::Enabled,
                    budget_tokens: 1024,
                }),
        }) = apply_inference_params(&inference_params)
        else {
            panic!("expected thinking budget tokens in additional model request fields");
        };
    }

    #[tokio::test]
    async fn test_build_request_with_tools() {
        let messages = vec![RequestMessage {
            role: TensorZeroRole::User,
            content: vec!["What's the weather?".to_string().into()],
        }];
        let mut request = sample_inference_request(messages);
        request.tool_config = Some(Cow::Owned(WEATHER_PROVIDER_TOOL_CONFIG.clone()));

        let converse_request = build_request(&request, &request.inference_params_v2)
            .await
            .expect("request should build");

        let tool_config = converse_request
            .tool_config
            .expect("tool config should be present");
        assert_eq!(tool_config.tools.len(), 1);
        assert!(matches!(
            tool_config.tool_choice,
            Some(ToolChoice::Tool(types::SpecificToolChoice { name }))
                if name == "get_temperature"
        ));
    }

    #[tokio::test]
    async fn test_build_request_tool_choice_none_omits_tool_config() {
        let messages = vec![RequestMessage {
            role: TensorZeroRole::User,
            content: vec!["hello".to_string().into()],
        }];
        let mut request = sample_inference_request(messages);
        request.tool_config = Some(Cow::Owned(ProviderToolCallConfig {
            tool_choice: TensorZeroToolChoice::None,
            ..WEATHER_PROVIDER_TOOL_CONFIG.clone()
        }));

        let converse_request = build_request(&request, &request.inference_params_v2)
            .await
            .expect("request should build");
        assert!(converse_request.tool_config.is_none());
    }

    #[tokio::test]
    async fn test_prepare_request_body_json_prefill_for_claude() {
        let messages = vec![RequestMessage {
            role: TensorZeroRole::User,
            content: vec!["Give me JSON".to_string().into()],
        }];
        let mut request = sample_inference_request(messages);
        request.function_type = FunctionType::Json;
        request.json_mode = ModelInferenceRequestJsonMode::On;

        let prepared = prepare_request_body(
            "anthropic.claude-sonnet-4-20250514-v1:0",
            &request,
            &sample_model_provider(),
            "test-model",
        )
        .await
        .expect("request body should prepare");

        let body: serde_json::Value =
            serde_json::from_str(&prepared.raw_request).expect("valid converse request JSON");
        assert_eq!(
            body["messages"].as_array().expect("messages array").len(),
            2
        );
        assert_eq!(body["messages"][1]["role"], "assistant");
    }

    #[gtest]
    fn test_convert_converse_response_with_cache_tokens() {
        let model_inference_id = Uuid::now_v7();
        let response = ConverseResponse {
            output: ConverseOutput {
                message: Some(ResponseMessage {
                    role: "assistant".to_string(),
                    content: vec![ResponseContentBlock::Text("hello".to_string())],
                }),
            },
            stop_reason: StopReason::EndTurn,
            usage: types::Usage {
                input_tokens: 50,
                output_tokens: 30,
                total_tokens: Some(80),
                cache_read_input_tokens: Some(40),
                cache_write_input_tokens: Some(10),
            },
            metrics: None,
        };
        let raw_response = r#"{"usage":{"inputTokens":50,"outputTokens":30,"cacheReadInputTokens":40,"cacheWriteInputTokens":10}}"#.to_string();
        let ctx = ResponseContext {
            system: Some("system".to_string()),
            input_messages: vec![RequestMessage {
                role: TensorZeroRole::User,
                content: vec!["hi".to_string().into()],
            }],
            model_id: "anthropic.claude-sonnet-4-20250514-v1:0",
            function_type: &FunctionType::Chat,
            json_mode: ModelInferenceRequestJsonMode::Off,
        };

        let provider_response = convert_converse_response(
            response,
            Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            "{}".to_string(),
            raw_response,
            ctx,
            model_inference_id,
        )
        .expect("response should convert");

        assert_eq!(provider_response.usage.input_tokens, Some(100));
        assert_eq!(
            provider_response.usage.provider_cache_read_input_tokens,
            Some(40)
        );
        assert_eq!(
            provider_response.usage.provider_cache_write_input_tokens,
            Some(10)
        );
    }

    #[gtest]
    fn test_convert_converse_response_json_prefill() {
        let model_inference_id = Uuid::now_v7();
        let response = ConverseResponse {
            output: ConverseOutput {
                message: Some(ResponseMessage {
                    role: "assistant".to_string(),
                    content: vec![ResponseContentBlock::Text(
                        "\"key\": \"value\"}".to_string(),
                    )],
                }),
            },
            stop_reason: StopReason::EndTurn,
            usage: types::Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: Some(15),
                cache_read_input_tokens: None,
                cache_write_input_tokens: None,
            },
            metrics: None,
        };
        let raw_response = r#"{"usage":{"inputTokens":10,"outputTokens":5}}"#.to_string();
        let ctx = ResponseContext {
            system: None,
            input_messages: vec![],
            model_id: "anthropic.claude-sonnet-4-20250514-v1:0",
            function_type: &FunctionType::Json,
            json_mode: ModelInferenceRequestJsonMode::On,
        };

        let provider_response = convert_converse_response(
            response,
            Latency::NonStreaming {
                response_time: Duration::from_millis(50),
            },
            "{}".to_string(),
            raw_response,
            ctx,
            model_inference_id,
        )
        .expect("response should convert");

        assert!(matches!(
            &provider_response.output[0],
            ContentBlockOutput::Text(Text { text }) if text == "{\"key\": \"value\"}"
        ));
    }

    #[gtest]
    fn test_process_stream_event_tool_use_flow() {
        let start_payload =
            br#"{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"tool_1","name":"weather"}}}"#;
        let mut current_tool_id = None;
        process_stream_event(
            Some("contentBlockStart"),
            start_payload,
            Duration::from_millis(10),
            &mut current_tool_id,
            Uuid::now_v7(),
        )
        .expect("start event should process")
        .expect("start chunk should be present");
        assert_eq!(current_tool_id, Some("tool_1".to_string()));

        let delta_payload = br#"{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"loc\""}}}"#;
        let delta_chunk = process_stream_event(
            Some("contentBlockDelta"),
            delta_payload,
            Duration::from_millis(10),
            &mut current_tool_id,
            Uuid::now_v7(),
        )
        .expect("delta event should process")
        .expect("delta chunk should be present");
        assert!(matches!(
            &delta_chunk.content[0],
            ContentBlockChunk::ToolCall(ToolCallChunk {
                id,
                raw_name: None,
                raw_arguments,
            }) if id == "tool_1" && raw_arguments == "{\"loc\""
        ));
    }

    #[gtest]
    fn test_process_stream_event_metadata_usage() {
        let payload = br#"{"usage":{"inputTokens":10,"outputTokens":5,"cacheReadInputTokens":20,"cacheWriteInputTokens":3}}"#;
        let chunk = process_stream_event(
            Some("metadata"),
            payload,
            Duration::from_millis(10),
            &mut None,
            Uuid::now_v7(),
        )
        .expect("metadata event should process")
        .expect("metadata chunk should be present");

        let usage = chunk.usage.expect("usage should be present");
        assert_eq!(usage.input_tokens, Some(33));
        assert_eq!(usage.provider_cache_read_input_tokens, Some(20));
        assert_eq!(usage.provider_cache_write_input_tokens, Some(3));
    }

    #[gtest]
    fn test_convert_converse_response_empty_message_errors() {
        let response = ConverseResponse {
            output: ConverseOutput { message: None },
            stop_reason: StopReason::EndTurn,
            usage: types::Usage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: None,
                cache_read_input_tokens: None,
                cache_write_input_tokens: None,
            },
            metrics: None,
        };
        let ctx = ResponseContext {
            system: None,
            input_messages: vec![],
            model_id: "amazon.nova-lite-v1:0",
            function_type: &FunctionType::Chat,
            json_mode: ModelInferenceRequestJsonMode::Off,
        };
        let err = convert_converse_response(
            response,
            Latency::NonStreaming {
                response_time: Duration::from_millis(1),
            },
            "{}".to_string(),
            "{}".to_string(),
            ctx,
            Uuid::now_v7(),
        )
        .expect_err("empty message should error");
        expect_that!(
            err.get_details(),
            matches_pattern!(ErrorDetails::InferenceServer { .. })
        );
    }
}
