use aws_config::meta::region::RegionProviderChain;
use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamOutput;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ContentBlockDelta, ContentBlockStart, ConversationRole,
    ConverseOutput as ConverseOutputType, ConverseStreamOutput as ConverseStreamOutputType,
    InferenceConfiguration, Message, SystemContentBlock,
};
use aws_smithy_types::error::display::DisplayErrorContext;
use aws_types::region::Region;
use dashmap::{DashMap, Entry as DashMapEntry};
use futures::{Stream, StreamExt};
use lazy_static::lazy_static;
use reqwest::StatusCode;
use std::time::Duration;
use tokio::time::Instant;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequest, ModelInferenceResponse,
    ModelInferenceResponseChunk, ModelInferenceResponseStream, RequestMessage, Role, Text,
    TextChunk, Usage,
};
use crate::tool::ToolCallChunk;

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
        // Note: AWS Bedrock does not support seed

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
        request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, ModelInferenceResponseStream), Error> {
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
            .converse_stream()
            .model_id(&self.model_id)
            .set_messages(Some(messages))
            .inference_config(inference_config.build());

        if let Some(system) = &request.system {
            let system_block = SystemContentBlock::Text(system.clone());
            bedrock_request = bedrock_request.system(system_block);
        }

        // TODO (#18, #30): .tool_config(...)

        let start_time = Instant::now();
        let stream = bedrock_request
            .send()
            .await
            .map_err(|e| Error::AWSBedrockServer {
                message: format!(
                    "Error sending request to AWS Bedrock: {}",
                    DisplayErrorContext(&e)
                ),
            })?;

        let mut stream = Box::pin(stream_bedrock(stream, start_time));
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::AWSBedrockServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };

        Ok((chunk, stream))
    }
}

fn stream_bedrock(
    mut stream: ConverseStreamOutput,
    start_time: Instant,
) -> impl Stream<Item = Result<ModelInferenceResponseChunk, Error>> {
    async_stream::stream! {
        let inference_id = Uuid::now_v7();
        let mut current_tool_id : Option<String> = None;
        let mut current_tool_name: Option<String> = None;

        loop {
            let ev = stream.stream.recv().await;

            match ev {
                Err(e) => {
                    yield Err(Error::AWSBedrockServer {
                        message: e.to_string(),
                    });
                }
                Ok(ev) => match ev {
                    None => break,
                    Some(output) => {
                        // NOTE: AWS Bedrock returns usage (ConverseStreamMetadataEvent) AFTER MessageStop.

                        // Convert the event to a tensorzero stream message
                        let stream_message = bedrock_to_tensorzero_stream_message(output, inference_id, start_time.elapsed(), &mut current_tool_id, &mut current_tool_name);

                        match stream_message {
                            Ok(None) => {},
                            Ok(Some(stream_message)) => yield Ok(stream_message),
                            Err(e) => yield Err(e),
                        }
                    }
                }
            }
        }
    }
}

fn bedrock_to_tensorzero_stream_message(
    output: ConverseStreamOutputType,
    inference_id: Uuid,
    message_latency: Duration,
    current_tool_id: &mut Option<String>,
    current_tool_name: &mut Option<String>,
) -> Result<Option<ModelInferenceResponseChunk>, Error> {
    match output {
        ConverseStreamOutputType::ContentBlockDelta(message) => {
            let raw_message = serialize_aws_bedrock_struct(&message)?;

            match message.delta {
                Some(delta) => match delta {
                    ContentBlockDelta::Text(text) => Ok(Some(ModelInferenceResponseChunk::new(
                        inference_id,
                        vec![ContentBlockChunk::Text(TextChunk {
                            text,
                            id: message.content_block_index.to_string(),
                        })],
                        None,
                        raw_message,
                        message_latency,
                    ))),
                    ContentBlockDelta::ToolUse(tool_use) => {
                        Ok(Some(ModelInferenceResponseChunk::new(
                            inference_id,
                            // Take the current tool name and ID and use them to create a ToolCallChunk
                            // This is necessary because the ToolCallChunk must always contain the tool name and ID
                            // even though AWS Bedrock only sends the tool ID and name in the ToolUse chunk and not InputJSONDelta
                            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                                name: current_tool_name.clone().ok_or(Error::AWSBedrockServer {
                                    message: "Got InputJsonDelta chunk from AWS Bedrock without current tool name being set by a ToolUse".to_string(),
                                })?,
                                id: current_tool_id.clone().ok_or(Error::AWSBedrockServer {
                                    message: "Got InputJsonDelta chunk from AWS Bedrock without current tool id being set by a ToolUse".to_string(),
                                })?,
                                arguments: tool_use.input,
                            })],
                            None,
                            raw_message,
                            message_latency,
                        )))
                    }
                    _ => Err(Error::AWSBedrockServer {
                        message: "Unsupported content block delta type for AWS Bedrock".to_string(),
                    }),
                },
                None => Ok(None),
            }
        }
        ConverseStreamOutputType::ContentBlockStart(message) => {
            let raw_message = serialize_aws_bedrock_struct(&message)?;

            match message.start {
                None => Ok(None),
                Some(ContentBlockStart::ToolUse(tool_use)) => {
                    // This is a new tool call, update the ID for future chunks
                    *current_tool_id = Some(tool_use.tool_use_id.clone());
                    *current_tool_name = Some(tool_use.name.clone());
                    Ok(Some(ModelInferenceResponseChunk::new(
                        inference_id,
                        vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                            id: tool_use.tool_use_id,
                            name: tool_use.name,
                            arguments: "".to_string(),
                        })],
                        None,
                        raw_message,
                        message_latency,
                    )))
                }
                _ => Err(Error::AWSBedrockServer {
                    message: "Unsupported content block start type for AWS Bedrock".to_string(),
                }),
            }
        }
        ConverseStreamOutputType::ContentBlockStop(_) => Ok(None),
        ConverseStreamOutputType::MessageStart(_) => Ok(None),
        ConverseStreamOutputType::MessageStop(_) => Ok(None),
        ConverseStreamOutputType::Metadata(message) => {
            let raw_message = serialize_aws_bedrock_struct(&message)?;

            // Note: There are other types of metadata (e.g. traces) but for now we're only interested in usage

            match message.usage {
                None => Ok(None),
                Some(usage) => {
                    let usage = Some(Usage {
                        prompt_tokens: usage.input_tokens as u32,
                        completion_tokens: usage.output_tokens as u32,
                    });

                    Ok(Some(ModelInferenceResponseChunk::new(
                        inference_id,
                        vec![],
                        usage,
                        raw_message,
                        message_latency,
                    )))
                }
            }
        }
        _ => Err(Error::AWSBedrockServer {
            message: "Unknown event type from AWS Bedrock".to_string(),
        }),
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

        let raw = serialize_aws_bedrock_struct(&output)?;

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

/// Serialize a struct to a JSON string.
///
/// This is necessary because the AWS SDK doesn't implement Serialize.
/// Therefore, we construct this unusual JSON object to store the raw output
///
/// This feature request has been pending since 2022:
/// https://github.com/awslabs/aws-sdk-rust/issues/645
fn serialize_aws_bedrock_struct<T: std::fmt::Debug>(output: &T) -> Result<String, Error> {
    serde_json::to_string(&serde_json::json!({"debug": format!("{:?}", output)})).map_err(|e| {
        Error::AWSBedrockServer {
            message: format!("Error parsing response from AWS Bedrock: {e}"),
        }
    })
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
