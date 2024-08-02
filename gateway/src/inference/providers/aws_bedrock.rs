use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, ConverseOutput as ConverseOutputType, InferenceConfiguration,
    Message, SystemContentBlock,
};
use tokio::sync::OnceCell;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    InferenceRequestMessage, InferenceResponseStream, ModelInferenceRequest,
    ModelInferenceResponse, ModelInferenceResponseChunk, ToolCall, Usage,
};
use crate::model::ProviderConfig;

static AWS_BEDROCK_CLIENT: OnceCell<aws_sdk_bedrockruntime::Client> = OnceCell::const_new();

async fn get_aws_bedrock_client() -> &'static aws_sdk_bedrockruntime::Client {
    // TODO: we should be able to customize the region per model provider => will require a client per region

    AWS_BEDROCK_CLIENT
        .get_or_init(|| async {
            let config = aws_config::load_from_env().await;
            aws_sdk_bedrockruntime::Client::new(&config)
        })
        .await
}

pub struct AWSBedrockProvider;

impl InferenceProvider for AWSBedrockProvider {
    async fn infer<'a>(
        request: &'a ModelInferenceRequest<'a>,
        config: &'a ProviderConfig,
        _http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let aws_bedrock_client = get_aws_bedrock_client().await;

        let model_id = match config {
            ProviderConfig::AWSBedrock { model_id } => model_id,
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected AWS Bedrock provider config".to_string(),
                })
            }
        };

        let first_message = &request.messages[0];
        let (system, request_messages) = match first_message {
            InferenceRequestMessage::System(message) => {
                let system = SystemContentBlock::Text(message.content.clone());
                (Some(system), &request.messages[1..])
            }
            _ => (None, &request.messages[..]),
        };

        // TODO: add support for guardrails and additional fields

        let messages: Vec<Message> = request_messages
            .iter()
            .map(Message::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let mut inference_config = InferenceConfiguration::builder();
        // TODO: add support for top_p, stop_sequences, etc.
        if let Some(max_tokens) = request.max_tokens {
            inference_config = inference_config.max_tokens(max_tokens as i32);
        }
        if let Some(temperature) = request.temperature {
            inference_config = inference_config.temperature(temperature);
        }

        let mut request = aws_bedrock_client
            .converse()
            .model_id(model_id)
            .set_messages(Some(messages))
            .inference_config(inference_config.build());

        if let Some(system) = system {
            request = request.system(system);
        }

        // TODO: .tool_config(...)

        // TODO: add more granularity to error handling
        let output = request.send().await.map_err(|e| Error::AWSBedrockServer {
            message: e.to_string(),
        })?;

        output.try_into()
    }

    async fn infer_stream<'a>(
        _request: &'a ModelInferenceRequest<'a>,
        _config: &'a ProviderConfig,
        _http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        todo!()
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

                // TODO: handle tool calls (like Anthropic)

                message_builder
            }
            InferenceRequestMessage::Tool(_) => {
                todo!();
            }
        };

        let message = message_builder.build().map_err(|e| Error::InvalidMessage {
            message: e.to_string(),
        })?;

        Ok(message)
    }
}

impl TryFrom<ConverseOutput> for ModelInferenceResponse {
    type Error = Error;

    fn try_from(output: ConverseOutput) -> Result<Self, Self::Error> {
        // TODO: is there something we can do about this?
        let raw = format!("{:?}", output); // AWS SDK doesn't implement Serialize :(

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
                    _ => todo!(), // TODO: handle tool use and other blocks
                }
            }
        }

        let usage = match output.usage {
            Some(usage) => Usage {
                prompt_tokens: usage.input_tokens as u32,
                completion_tokens: usage.output_tokens as u32,
            },
            None => todo!(), // TODO: this should be nullable
        };

        Ok(ModelInferenceResponse::new(
            message_text,
            tool_calls,
            raw,
            usage,
        ))
    }
}
