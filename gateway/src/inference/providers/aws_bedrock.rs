use aws_config::meta::region::RegionProviderChain;
use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole, ConverseOutput as ConverseOutputType,
    InferenceConfiguration, Message, SystemContentBlock,
};
use aws_smithy_types::error::display::DisplayErrorContext;
use aws_types::region::Region;
use dashmap::{DashMap, Entry as DashMapEntry};
use lazy_static::lazy_static;
use reqwest::StatusCode;
use tokio::time::Instant;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    ContentBlock, InferenceResultStream, Latency, ModelInferenceRequest, ModelInferenceResponse,
    ModelInferenceResponseChunk, RequestMessage, Role, Text, Usage,
};

lazy_static! {
     /// NOTE: The AWS client is thread-safe but not safe across Tokio runtimes. By default, `tokio::test`
     /// spawns a new runtime for each test, causing intermittent issues with the AWS client. For tests,
     /// use a shared runtime like in our integration tests. This isn't an issue when running the gateway
     /// normally since that uses a single runtime.
    static ref AWS_BEDROCK_CLIENTS: DashMap<Region, &'static aws_sdk_bedrockruntime::Client> = DashMap::new();
}

async fn get_aws_bedrock_client(
    region: Option<Region>,
) -> Result<&'static aws_sdk_bedrockruntime::Client, Error> {
    // Decide which AWS region to use. We try the following in order:
    // - The provided `region` argument
    // - The region defined by the credentials (e.g. `AWS_REGION` environment variable)
    // - The default region (us-east-1)
    let region = RegionProviderChain::first_try(region)
        .or_default_provider()
        .or_else(Region::new("us-east-1"))
        .region()
        .await
        .ok_or(Error::AWSBedrockClient {
            status_code: StatusCode::INTERNAL_SERVER_ERROR,
            message: "Failed to determine AWS region.".to_string(),
        })?;

    let client: &aws_sdk_bedrockruntime::Client = match AWS_BEDROCK_CLIENTS.entry(region) {
        DashMapEntry::Occupied(entry) => entry.get(),
        DashMapEntry::Vacant(entry) => {
            tracing::trace!(
                "Creating new AWS Bedrock client for region: {}",
                entry.key()
            );
            let config = aws_config::from_env()
                .region(entry.key().clone())
                .load()
                .await;
            let client = Box::leak(Box::new(aws_sdk_bedrockruntime::Client::new(&config)));
            entry.insert(client);
            client
        }
    };

    Ok(client)
}

#[derive(Clone, Debug)]
pub struct AWSBedrockProvider {
    pub model_id: String,
    pub region: Option<Region>,
}

impl InferenceProvider for AWSBedrockProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let aws_bedrock_client = get_aws_bedrock_client(self.region.clone()).await?;

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
            .model_id(&self.model_id)
            .set_messages(Some(messages))
            .inference_config(inference_config.build());

        if let Some(system) = &request.system {
            let system_block = SystemContentBlock::Text(system.clone());
            bedrock_request = bedrock_request.system(system_block);
        }

        // TODO (#18, #30): .tool_config(...)

        let start_time = Instant::now();
        let output = bedrock_request
            .send()
            .await
            .map_err(|e| Error::AWSBedrockServer {
                message: format!(
                    "Error sending request to AWS Bedrock: {}",
                    DisplayErrorContext(&e)
                ),
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
    ) -> Result<(ModelInferenceResponseChunk, InferenceResultStream), Error> {
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

#[cfg(test)]
mod tests {
    use super::*;

    use tracing_test::traced_test;

    #[tokio::test]
    async fn test_get_aws_bedrock_client() {
        #[traced_test]
        async fn first_run() {
            get_aws_bedrock_client(Some(Region::new("uk-hogwarts-1")))
                .await
                .unwrap();

            assert!(logs_contain(
                "Creating new AWS Bedrock client for region: uk-hogwarts-1"
            ));
        }

        #[traced_test]
        async fn second_run() {
            get_aws_bedrock_client(Some(Region::new("uk-hogwarts-1")))
                .await
                .unwrap();

            assert!(!logs_contain(
                "Creating new AWS Bedrock client for region: uk-hogwarts-1"
            ));
        }

        #[traced_test]
        async fn third_run() {
            get_aws_bedrock_client(None).await.unwrap();
        }

        #[traced_test]
        async fn fourth_run() {
            get_aws_bedrock_client(Some(Region::new("me-shire-2")))
                .await
                .unwrap();

            assert!(logs_contain(
                "Creating new AWS Bedrock client for region: me-shire-2"
            ));
        }

        #[traced_test]
        async fn fifth_run() {
            get_aws_bedrock_client(None).await.unwrap();

            assert!(!logs_contain("Creating new AWS Bedrock client for region:"));
        }

        // First call ("uk-hogwarts-1") should trigger client creation
        first_run().await;

        // Second call ("uk-hogwarts-1") should not trigger client creation
        second_run().await;

        // Third call (None) is unclear: we also can't guarantee that it will create a provider because another test might have already created it.
        third_run().await;

        // Fourth call ("me-shire-2") should trigger client creation
        fourth_run().await;

        // Fifth call (None) should not trigger client creation
        fifth_run().await;

        // NOTE: There isn't an easy way to test the fallback between the default provider and the last-resort fallback region.
        // We can only test that the method returns either of these regions when an explicit region is not provided.
    }
}
