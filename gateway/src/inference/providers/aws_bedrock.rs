use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, ConverseOutput as ConverseOutputType, InferenceConfiguration,
    Message, SystemContentBlock,
};
use tokio::sync::OnceCell;
use tokio::time::Instant;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    InferenceRequestMessage, InferenceResponseStream, Latency, ModelInferenceRequest,
    ModelInferenceResponse, ModelInferenceResponseChunk, ToolCall, Usage,
};

static AWS_BEDROCK_CLIENT: OnceCell<aws_sdk_bedrockruntime::Client> = OnceCell::const_new();

async fn get_aws_bedrock_client() -> &'static aws_sdk_bedrockruntime::Client {
    // TODO (#73): we should be able to customize the region per model provider => will require a client per region

    AWS_BEDROCK_CLIENT
        .get_or_init(|| async {
            let config = aws_config::load_from_env().await;
            aws_sdk_bedrockruntime::Client::new(&config)
        })
        .await
}

#[derive(Clone, Debug)]
pub struct AWSBedrockProvider {
    pub model_id: String,
}

impl InferenceProvider for AWSBedrockProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let aws_bedrock_client = get_aws_bedrock_client().await;

        let first_message = &request.messages[0];
        let (system, request_messages) = match first_message {
            InferenceRequestMessage::System(message) => {
                let system = SystemContentBlock::Text(message.content.clone());
                (Some(system), &request.messages[1..])
            }
            _ => (None, &request.messages[..]),
        };

        // TODO (#55): add support for guardrails and additional fields

        let messages: Vec<Message> = request_messages
            .iter()
            .map(Message::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let mut inference_config = InferenceConfiguration::builder();
        // TODO (#55): add support for top_p, stop_sequences, etc.
        if let Some(max_tokens) = request.max_tokens {
            inference_config = inference_config.max_tokens(max_tokens as i32);
        }
        if let Some(temperature) = request.temperature {
            inference_config = inference_config.temperature(temperature);
        }

        let mut request = aws_bedrock_client
            .converse()
            .model_id(&self.model_id)
            .set_messages(Some(messages))
            .inference_config(inference_config.build());

        if let Some(system) = system {
            request = request.system(system);
        }

        // TODO (#18, #30): .tool_config(...)

        // TODO (#88): add more granularity to error handling
        let start_time = Instant::now();
        let output = request.send().await.map_err(|e| Error::AWSBedrockServer {
            message: e.to_string(),
        })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };

        ConverseOutputWithLatency { output, latency }.try_into()
    }

    async fn infer_stream<'a>(
        &'a self,
        _request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        todo!() // TODO (#30): implement streaming inference
    }
}

impl TryFrom<&InferenceRequestMessage> for Message {
    type Error = Error;

    fn try_from(inference_message: &InferenceRequestMessage) -> Result<Message, Error> {
        let message_builder = match inference_message {
            InferenceRequestMessage::System(_) => {
                return Err(Error::InvalidMessage {
                    message: "Can't convert System message to AWS Bedrock message. Don't pass System message in except as the first message in the chat.".to_string(),
                })
            }
            InferenceRequestMessage::User(message) => {
                Message::builder()
                    .role(ConversationRole::User)
                    .content(ContentBlock::Text(message.content.clone()))
            }
            InferenceRequestMessage::Assistant(message) => {
                let mut message_builder = Message::builder().role(ConversationRole::Assistant);

                if let Some(text) = &message.content {
                    message_builder = message_builder.content(ContentBlock::Text(text.clone()));
                }

                // TODO (#30): handle tool calls (like Anthropic)

                message_builder
            }
            InferenceRequestMessage::Tool(_) => {
                todo!(); // TODO (#30): handle tool calls (like Anthropic)
            }
        };

        let message = message_builder.build().map_err(|e| Error::InvalidMessage {
            message: e.to_string(),
        })?;

        Ok(message)
    }
}

struct ConverseOutputWithLatency {
    output: ConverseOutput,
    latency: Latency,
}

impl TryFrom<ConverseOutputWithLatency> for ModelInferenceResponse {
    type Error = Error;

    fn try_from(value: ConverseOutputWithLatency) -> Result<Self, Self::Error> {
        let ConverseOutputWithLatency { output, latency } = value;

        // NOTE: AWS SDK doesn't implement Serialize :(
        // Therefore, we construct this unusual JSON object to store the raw output
        //
        // This feature request has been pending since 2022:
        // https://github.com/awslabs/aws-sdk-rust/issues/645
        let raw = serde_json::to_string(&serde_json::json!({"debug": format!("{:?}", output)}))
            .map_err(|e| Error::AWSBedrockServer {
                message: format!("Error parsing response from AWS Bedrock: {e}"),
            })?;

        let message = match output.output {
            Some(ConverseOutputType::Message(message)) => Some(message),
            _ => {
                return Err(Error::AWSBedrockServer {
                    message: "AWS Bedrock returned an unknown output type.".to_string(),
                });
            }
        };

        let mut message_text: Option<String> = None;
        let tool_calls: Option<Vec<ToolCall>> = None;

        if let Some(message) = message {
            for block in message.content {
                match block {
                    ContentBlock::Text(text) => match message_text {
                        Some(message) => message_text = Some(format!("{}\n{}", message, text)),
                        None => message_text = Some(text),
                    },
                    _ => todo!(), // TODO (#18): handle tool use and other blocks
                }
            }
        }

        let usage = match output.usage {
            Some(usage) => Usage {
                prompt_tokens: usage.input_tokens as u32,
                completion_tokens: usage.output_tokens as u32,
            },
            None => todo!(), // TODO (#18): this should be nullable
        };

        Ok(ModelInferenceResponse::new(
            message_text,
            tool_calls,
            raw,
            usage,
            latency,
        ))
    }
}
