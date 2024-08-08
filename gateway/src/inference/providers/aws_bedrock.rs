use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole, ConverseOutput as ConverseOutputType,
    InferenceConfiguration, Message, SystemContentBlock,
};
use tokio::sync::OnceCell;
use tokio::time::Instant;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    ContentBlock, InferenceResponseStream, Latency, ModelInferenceRequest, ModelInferenceResponse,
    ModelInferenceResponseChunk, RequestMessage, Role, Text, Usage,
};
use crate::model::ProviderConfig;

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

        // TODO (#55): add support for guardrails and additional fields

        let messages: Vec<Message> = request
            .messages
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

        let mut bedrock_request = aws_bedrock_client
            .converse()
            .model_id(model_id)
            .set_messages(Some(messages))
            .inference_config(inference_config.build());

        if let Some(system) = &request.system_instructions {
            let system_block = SystemContentBlock::Text(system.clone());
            bedrock_request = bedrock_request.system(system_block);
        }

        // TODO (#18, #30): .tool_config(...)

        // TODO (#88): add more granularity to error handling
        let start_time = Instant::now();
        let output = bedrock_request
            .send()
            .await
            .map_err(|e| Error::AWSBedrockServer {
                message: e.to_string(),
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };

        ConverseOutputWithLatency { output, latency }.try_into()
    }

    async fn infer_stream<'a>(
        _request: &'a ModelInferenceRequest<'a>,
        _config: &'a ProviderConfig,
        _http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        todo!() // TODO (#30): implement streaming inference
    }
}

impl From<Role> for ConversationRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => ConversationRole::User,
            Role::Assistant => ConversationRole::Assistant,
        }
    }
}

impl TryFrom<&ContentBlock> for BedrockContentBlock {
    type Error = Error;

    fn try_from(block: &ContentBlock) -> Result<Self, Self::Error> {
        match block {
            ContentBlock::Text(Text { text }) => Ok(BedrockContentBlock::Text(text.clone())),
            _ => Err(Error::TypeConversion {
                message: "Unsupported content block type for AWS Bedrock.".to_string(),
            }), // TODO (#18, #30): handle tool use and tool call blocks
        }
    }
}

impl TryFrom<BedrockContentBlock> for ContentBlock {
    type Error = Error;

    fn try_from(block: BedrockContentBlock) -> Result<Self, Self::Error> {
        match block {
            BedrockContentBlock::Text(text) => Ok(text.into()),
            _ => Err(Error::TypeConversion {
                message: "Unsupported content block type for AWS Bedrock.".to_string(),
            }), // TODO (#18, #30): handle tool use and tool call blocks
        }
    }
}

impl TryFrom<&RequestMessage> for Message {
    type Error = Error;

    fn try_from(inference_message: &RequestMessage) -> Result<Message, Error> {
        let role: ConversationRole = inference_message.role.into();
        let blocks: Vec<BedrockContentBlock> = inference_message
            .content
            .iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        let mut message_builder = Message::builder().role(role);
        for block in blocks {
            message_builder = message_builder.content(block);
        }
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
        // TODO (#79): is there something we can do about this?
        let raw = format!("{:?}", output); // AWS SDK doesn't implement Serialize :(

        let message = match output.output {
            Some(ConverseOutputType::Message(message)) => Some(message),
            _ => {
                return Err(Error::AWSBedrockServer {
                    message: "AWS Bedrock returned an unknown output type.".to_string(),
                });
            }
        };

        let content: Vec<ContentBlock> = message
            .ok_or(Error::AWSBedrockServer {
                message: "AWS Bedrock returned an empty message.".to_string(),
            })?
            .content
            .into_iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<ContentBlock>, _>>()?;

        let usage = match output.usage {
            Some(usage) => Usage {
                prompt_tokens: usage.input_tokens as u32,
                completion_tokens: usage.output_tokens as u32,
            },
            None => todo!(), // TODO (#18): this should be nullable
        };

        Ok(ModelInferenceResponse::new(content, raw, usage, latency))
    }
}
