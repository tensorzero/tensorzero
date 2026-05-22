//! AWS Bedrock's Anthropic Messages compatible API via Bedrock mantle endpoint.
//!
//! See:
//! - https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
//! - https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock

use std::borrow::Cow;

use futures::StreamExt;
use futures::future::try_join_all;
use secrecy::ExposeSecret;
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::ModelInferenceRequest;
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
};
use crate::inference::types::{
    ApiType, Latency, ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse,
};
use crate::model::{ModelProviderRequestInfo, ProviderInferenceRequest};

use super::anthropic::{
    AnthropicMessage, AnthropicMessagesConfig, AnthropicOutputFormat,
    AnthropicResponseWithMetadata, AnthropicSystemBlock, AnthropicTool, AnthropicToolChoice,
    build_anthropic_tools, collect_all_provider_tools, needs_json_prefill,
    prefill_json_chunk_response, prefill_json_message, stream_anthropic,
};
use super::aws_bedrock::{AWSBedrockProvider, PROVIDER_TYPE};
use super::aws_common::send_aws_request;
use super::helpers::{inject_extra_request_data, peek_first_chunk};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const PROVIDER_NAME: &str = "AWS Bedrock Messages";

/// Build HTTP headers for the Anthropic Messages API on Bedrock mantle endpoint.
///
/// Adds `anthropic-version`, `Content-Type`, and `x-api-key` (if provided) headers to the provided header map.
fn add_request_headers_messages_api(
    mut headers: http::HeaderMap,
    api_key: Option<&secrecy::SecretString>,
) -> Result<http::HeaderMap, Error> {
    headers.insert(
        http::header::HeaderName::from_static("anthropic-version"),
        http::header::HeaderValue::from_static(ANTHROPIC_VERSION),
    );
    headers.insert(
        http::header::CONTENT_TYPE,
        http::header::HeaderValue::from_static("application/json"),
    );
    if let Some(api_key) = api_key {
        headers.insert(
            http::header::HeaderName::from_static("x-api-key"),
            http::header::HeaderValue::from_str(api_key.expose_secret()).map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: format!("Invalid API key format: {e}"),
                })
            })?,
        );
    }
    Ok(headers)
}

impl AWSBedrockProvider {
    /// Send a non-streaming inference request via the Anthropic Messages API on Bedrock.
    pub(super) async fn infer_anthropic<'a>(
        &self,
        ProviderInferenceRequest {
            request,
            provider_name,
            model_name,
            model_inference_id,
        }: ProviderInferenceRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProviderRequestInfo,
    ) -> Result<ProviderInferenceResponse, Error> {
        let all_provider_tools =
            collect_all_provider_tools(self.provider_tools(), request, model_name, provider_name);
        let PreparedAnthropicRequestBody {
            raw_request,
            body_bytes,
            http_extra_headers,
        } = prepare_anthropic_request_body(
            self.model_id(),
            request,
            model_provider,
            model_name,
            &all_provider_tools,
        )
        .await?;

        let base_url = self.get_base_url(dynamic_api_keys, ApiType::ChatCompletions)?;
        let url = format!("{base_url}/anthropic/v1/messages");

        let request_headers = self
            .build_request_headers(
                dynamic_api_keys,
                &url,
                http_extra_headers,
                &body_bytes,
                PROVIDER_NAME,
                add_request_headers_messages_api,
            )
            .await?;

        let aws_response = send_aws_request(
            http_client,
            &url,
            request_headers,
            body_bytes,
            "bedrock-mantle",
            PROVIDER_TYPE,
            &raw_request,
            ApiType::ChatCompletions,
        )
        .await?;

        let latency = Latency::NonStreaming {
            response_time: aws_response.response_time,
        };

        let raw_response = aws_response.raw_response;

        let response = serde_json::from_str(&raw_response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error parsing JSON response: {}: {raw_response}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                api_type: ApiType::ChatCompletions,
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
            })
        })?;

        let response_with_latency = AnthropicResponseWithMetadata::new(
            response,
            raw_response,
            latency,
            raw_request,
            request,
            request.messages.clone(),
            model_name,
            PROVIDER_NAME,
            model_inference_id,
        );
        Ok(response_with_latency.try_into()?)
    }

    /// Send a streaming inference request via the Anthropic Messages API on Bedrock.
    pub(super) async fn infer_stream_anthropic<'a>(
        &self,
        ProviderInferenceRequest {
            request,
            provider_name,
            model_name,
            model_inference_id,
        }: ProviderInferenceRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProviderRequestInfo,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let all_provider_tools =
            collect_all_provider_tools(self.provider_tools(), request, model_name, provider_name);
        let PreparedAnthropicRequestBody {
            raw_request,
            body_bytes,
            http_extra_headers,
        } = prepare_anthropic_request_body(
            self.model_id(),
            request,
            model_provider,
            model_name,
            &all_provider_tools,
        )
        .await?;

        let base_url = self.get_base_url(dynamic_api_keys, ApiType::ChatCompletions)?;
        let url = format!("{base_url}/anthropic/v1/messages");
        let request_headers = self
            .build_request_headers(
                dynamic_api_keys,
                &url,
                http_extra_headers,
                &body_bytes,
                PROVIDER_NAME,
                add_request_headers_messages_api,
            )
            .await?;

        let start_time = Instant::now();
        let event_source = http_client
            .post(&url)
            .headers(request_headers)
            .body(body_bytes)
            .eventsource()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error sending streaming request to {PROVIDER_NAME} mantle Anthropic: {e}"
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    api_type: ApiType::ChatCompletions,
                })
            })?;

        let mut stream = stream_anthropic(
            event_source,
            start_time,
            model_provider,
            model_name,
            PROVIDER_NAME,
            &raw_request,
            model_inference_id,
        )
        .peekable();

        let chunk = peek_first_chunk(
            &mut stream,
            &raw_request,
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
        )
        .await?;

        if needs_json_prefill(request) {
            prefill_json_chunk_response(chunk);
        }

        Ok((stream, raw_request))
    }
}

pub(crate) struct PreparedAnthropicRequestBody {
    pub raw_request: String,
    pub body_bytes: Vec<u8>,
    pub http_extra_headers: http::HeaderMap,
}

pub(crate) async fn prepare_anthropic_request_body(
    model_id: &str,
    request: &ModelInferenceRequest<'_>,
    model_provider: &ModelProviderRequestInfo,
    model_name: &str,
    provider_tools: &[Value],
) -> Result<PreparedAnthropicRequestBody, Error> {
    let request_body = BedrockAnthropicRequestBody::new(model_id, request, provider_tools).await?;
    let mut body_json = serde_json::to_value(&request_body).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize {PROVIDER_NAME} request: {e}"),
        })
    })?;

    let http_extra_headers = inject_extra_request_data(
        &request.extra_body,
        &request.extra_headers,
        model_provider,
        model_name,
        &mut body_json,
    )?;

    if cfg!(feature = "e2e_tests") {
        body_json.sort_all_objects();
    }

    let raw_request = serde_json::to_string(&body_json).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Failed to serialize {PROVIDER_NAME} request: {e}"),
        })
    })?;
    let body_bytes = raw_request.as_bytes().to_vec();

    Ok(PreparedAnthropicRequestBody {
        raw_request,
        body_bytes,
        http_extra_headers,
    })
}

#[derive(Debug, PartialEq, Serialize)]
struct BedrockAnthropicOutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<AnthropicOutputFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(untagged)]
enum BedrockAnthropicThinkingConfig {
    Enabled {
        r#type: &'static str,
        budget_tokens: i32,
    },
    Adaptive {
        r#type: &'static str,
    },
}

#[derive(Debug, Default, PartialEq, Serialize)]
struct BedrockAnthropicRequestBody<'a> {
    model: String,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<BedrockAnthropicOutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<AnthropicSystemBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<BedrockAnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
}

impl<'a> BedrockAnthropicRequestBody<'a> {
    async fn new(
        model_id: &str,
        request: &'a ModelInferenceRequest<'a>,
        provider_tools: &'a [Value],
    ) -> Result<BedrockAnthropicRequestBody<'a>, Error> {
        if request.messages.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
            .into());
        }

        let messages_config = AnthropicMessagesConfig {
            fetch_and_encode_input_files_before_inference: request
                .fetch_and_encode_input_files_before_inference,
        };
        let system = request
            .system
            .as_deref()
            .map(|text| vec![AnthropicSystemBlock::Text { text }]);
        let mut messages: Vec<AnthropicMessage> =
            try_join_all(request.messages.iter().map(|m| {
                AnthropicMessage::from_request_message(m, messages_config, PROVIDER_TYPE)
            }))
            .await?
            .into_iter()
            .collect();

        if needs_json_prefill(request) {
            messages = prefill_json_message(messages);
        }

        let tools = build_anthropic_tools(request.tool_config.as_ref(), provider_tools, true)?;
        let tool_choice: Option<AnthropicToolChoice> = tools
            .as_ref()
            .filter(|t| !t.is_empty())
            .and(request.tool_config.as_ref())
            .and_then(|c| c.as_ref().try_into().ok());

        let max_tokens = match request.max_tokens {
            Some(max_tokens) => Ok(max_tokens),
            None => get_default_max_tokens(model_id),
        }?;

        let mut bedrock_request = BedrockAnthropicRequestBody {
            model: model_id.to_string(),
            messages,
            max_tokens,
            stream: Some(request.stream),
            output_config: match request.json_mode {
                ModelInferenceRequestJsonMode::Strict => {
                    request
                        .output_schema
                        .map(|schema| BedrockAnthropicOutputConfig {
                            format: Some(AnthropicOutputFormat::JsonSchema {
                                schema: schema.clone(),
                            }),
                            effort: None,
                        })
                }
                ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Off => None,
            },
            system,
            temperature: request.temperature,
            thinking: None,
            top_p: request.top_p,
            stop_sequences: request.borrow_stop_sequences(),
            tool_choice,
            tools,
        };

        apply_inference_params(&mut bedrock_request, &request.inference_params_v2)?;

        Ok(bedrock_request)
    }
}

fn apply_inference_params(
    request: &mut BedrockAnthropicRequestBody<'_>,
    inference_params: &ChatCompletionInferenceParamsV2,
) -> Result<(), Error> {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() && thinking_budget_tokens.is_some() {
        return Err(ErrorDetails::InvalidRequest {
            message: format!("Cannot specify both `reasoning_effort` and `thinking_budget_tokens` for {PROVIDER_NAME}. Use `reasoning_effort` for adaptive thinking or `thinking_budget_tokens` for manual thinking."),
        }
        .into());
    }

    if let Some(effort) = reasoning_effort {
        request.thinking = Some(BedrockAnthropicThinkingConfig::Adaptive { r#type: "adaptive" });
        match &mut request.output_config {
            Some(config) => {
                config.effort = Some(effort.clone());
            }
            None => {
                request.output_config = Some(BedrockAnthropicOutputConfig {
                    format: None,
                    effort: Some(effort.clone()),
                });
            }
        }
    }

    if let Some(budget_tokens) = thinking_budget_tokens {
        request.thinking = Some(BedrockAnthropicThinkingConfig::Enabled {
            r#type: "enabled",
            budget_tokens: *budget_tokens,
        });
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
    Ok(())
}

fn get_default_max_tokens(model_id: &str) -> Result<u32, Error> {
    if model_id.contains("claude-3-haiku") || model_id.contains("claude-3-opus") {
        Ok(4_096)
    } else if model_id.contains("claude-3-5-haiku") || model_id.contains("claude-3-5-sonnet") {
        Ok(8_192)
    } else if model_id.contains("claude-3-7-sonnet")
        || model_id.contains("claude-sonnet-4")
        || model_id.contains("claude-haiku-4-5")
        || model_id.contains("claude-sonnet-4-5")
        || model_id.contains("claude-sonnet-4-6")
        || model_id.contains("claude-opus-4-5")
    {
        Ok(64_000)
    } else if model_id.contains("claude-opus-4-1") || model_id.contains("claude-opus-4-0") {
        Ok(32_000)
    } else if model_id.contains("claude-opus-4-6") || model_id.contains("claude-opus-4-7") {
        Ok(128_000)
    } else {
        Err(Error::new(ErrorDetails::InferenceClient {
            message: format!(
                "The TensorZero Gateway doesn't know the output token limit for `{model_id}` and {PROVIDER_NAME} requires you to provide a `max_tokens` value. Please set `max_tokens` in your configuration or inference request."
            ),
            status_code: None,
            provider_type: PROVIDER_TYPE.into(),
            api_type: ApiType::ChatCompletions,
            raw_request: None,
            raw_response: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use uuid::Uuid;

    use super::*;
    use crate::error::ErrorDetails;
    use crate::inference::types::chat_completion_inference_params::ChatCompletionInferenceParamsV2;
    use crate::inference::types::{
        FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role as TensorZeroRole,
    };
    use crate::providers::anthropic::{AnthropicFunctionTool, AnthropicTool};
    use crate::providers::test_helpers::WEATHER_PROVIDER_TOOL_CONFIG;

    #[tokio::test]
    async fn test_bedrock_anthropic_request_body_requires_messages() {
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            ..Default::default()
        };
        let err = BedrockAnthropicRequestBody::new(
            "anthropic.claude-sonnet-4-20250514-v1:0",
            &request,
            &[],
        )
        .await
        .expect_err("empty messages should error");
        assert_eq!(
            err.get_details(),
            &ErrorDetails::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn test_bedrock_anthropic_request_body_json_prefill() {
        let messages = vec![RequestMessage {
            role: TensorZeroRole::User,
            content: vec!["Give JSON".to_string().into()],
        }];
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages,
            function_type: FunctionType::Json,
            json_mode: ModelInferenceRequestJsonMode::On,
            ..Default::default()
        };

        let body = BedrockAnthropicRequestBody::new(
            "anthropic.claude-sonnet-4-20250514-v1:0",
            &request,
            &[],
        )
        .await
        .expect("request body should build");
        assert_eq!(body.messages.len(), 2);
    }

    #[tokio::test]
    async fn test_bedrock_anthropic_request_body_with_tools() {
        let messages = vec![RequestMessage {
            role: TensorZeroRole::User,
            content: vec!["What's the weather?".to_string().into()],
        }];
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages,
            tool_config: Some(Cow::Owned(WEATHER_PROVIDER_TOOL_CONFIG.clone())),
            ..Default::default()
        };

        let body = BedrockAnthropicRequestBody::new(
            "anthropic.claude-sonnet-4-20250514-v1:0",
            &request,
            &[],
        )
        .await
        .expect("request body should build");

        let tools = body.tools.expect("tools should be present");
        assert_eq!(tools.len(), 1);
        assert!(matches!(
            tools[0],
            AnthropicTool::Function(AnthropicFunctionTool {
                name: "get_temperature",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_get_default_max_tokens_for_bedrock_model_ids() {
        let messages = vec![RequestMessage {
            role: TensorZeroRole::User,
            content: vec!["hello".to_string().into()],
        }];
        let request = ModelInferenceRequest {
            messages: messages.clone(),
            ..Default::default()
        };
        let request_with_max_tokens = ModelInferenceRequest {
            messages,
            max_tokens: Some(100),
            ..Default::default()
        };

        let model = "anthropic.claude-sonnet-4-20250514-v1:0";
        let body = BedrockAnthropicRequestBody::new(model, &request, &[])
            .await
            .expect("sonnet-4 should default max tokens");
        assert_eq!(body.max_tokens, 64_000);

        let model = "anthropic.claude-opus-4-6-20260101-v1:0";
        let body = BedrockAnthropicRequestBody::new(model, &request, &[])
            .await
            .expect("opus-4-6 should default max tokens");
        assert_eq!(body.max_tokens, 128_000);

        let unknown_model = "anthropic.claude-unknown-v1:0";
        let err = BedrockAnthropicRequestBody::new(unknown_model, &request, &[])
            .await
            .expect_err("unknown model without max_tokens should error");
        assert!(matches!(
            err.get_details(),
            ErrorDetails::InferenceClient { .. }
        ));

        let body = BedrockAnthropicRequestBody::new(unknown_model, &request_with_max_tokens, &[])
            .await
            .expect("unknown model with explicit max_tokens should succeed");
        assert_eq!(body.max_tokens, 100);
    }

    #[test]
    fn test_bedrock_messages_apply_inference_params_reasoning_effort() {
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("low".to_string()),
            service_tier: None,
            thinking_budget_tokens: None,
            verbosity: None,
        };
        let mut request = BedrockAnthropicRequestBody {
            model: "anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
            messages: vec![],
            max_tokens: 100,
            ..Default::default()
        };

        apply_inference_params(&mut request, &inference_params).expect("should succeed");
        assert_eq!(
            request.thinking,
            Some(BedrockAnthropicThinkingConfig::Adaptive { r#type: "adaptive" })
        );
        assert_eq!(
            request.output_config,
            Some(BedrockAnthropicOutputConfig {
                format: None,
                effort: Some("low".to_string()),
            })
        );
    }

    #[test]
    fn test_bedrock_messages_apply_inference_params_mutual_exclusivity() {
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: None,
        };
        let mut request = BedrockAnthropicRequestBody {
            model: "anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
            messages: vec![],
            max_tokens: 100,
            ..Default::default()
        };

        let err = apply_inference_params(&mut request, &inference_params)
            .expect_err("both thinking params should error");
        assert!(
            err.to_string()
                .contains("Cannot specify both `reasoning_effort` and `thinking_budget_tokens`")
        );
    }
}
