use aws_config::meta::region::RegionProviderChain;
use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamOutput;
use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, AutoToolChoice, ContentBlock as BedrockContentBlock, ContentBlockDelta,
    ContentBlockStart, ConversationRole, ConverseOutput as ConverseOutputType,
    ConverseStreamOutput as ConverseStreamOutputType, InferenceConfiguration, Message,
    SpecificToolChoice, SystemContentBlock, Tool, ToolChoice as AWSBedrockToolChoice,
    ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolSpecification,
    ToolUseBlock,
};
use aws_smithy_types::error::display::DisplayErrorContext;
use aws_types::region::Region;
use futures::{Stream, StreamExt};
use lazy_static::lazy_static;
use reqwest::StatusCode;
use secrecy::SecretString;
use std::borrow::Cow;
use std::time::Duration;
use tokio::time::Instant;
use uuid::Uuid;

use crate::endpoints::inference::InferenceApiKeys;
use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequest, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, ProviderInferenceResponseStream, RequestMessage, Role, Text,
    TextChunk, Usage,
};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

use super::provider_trait::HasCredentials;

// NB: If you add `Clone` someday, you'll need to wrap client in Arc
#[derive(Debug)]
pub struct AWSBedrockProvider {
    pub model_id: String,
    client: aws_sdk_bedrockruntime::Client,
}

impl AWSBedrockProvider {
    pub async fn new(model_id: String, region: Option<Region>) -> Result<Self, Error> {
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

        tracing::trace!("Creating new AWS Bedrock client for region: {region}",);

        let config = aws_config::from_env().region(region).load().await;
        let client = aws_sdk_bedrockruntime::Client::new(&config);

        Ok(Self { model_id, client })
    }
}

impl InferenceProvider for AWSBedrockProvider {
    async fn _infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
        _api_key: Cow<'a, SecretString>,
    ) -> Result<ProviderInferenceResponse, Error> {
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

        let mut bedrock_request = self
            .client
            .converse()
            .model_id(&self.model_id)
            .set_messages(Some(messages))
            .inference_config(inference_config.build());

        if let Some(system) = &request.system {
            let system_block = SystemContentBlock::Text(system.clone());
            bedrock_request = bedrock_request.system(system_block);
        }

        if let Some(tool_config) = &request.tool_config {
            let tools: Vec<Tool> = tool_config
                .tools_available
                .iter()
                .map(Tool::try_from)
                .collect::<Result<Vec<_>, _>>()?;

            let tool_choice: AWSBedrockToolChoice = tool_config.tool_choice.clone().try_into()?;

            let aws_bedrock_tool_config = ToolConfiguration::builder()
                .set_tools(Some(tools))
                .tool_choice(tool_choice)
                .build()
                .map_err(|e| Error::AWSBedrockClient {
                    status_code: StatusCode::INTERNAL_SERVER_ERROR,
                    message: format!("Error configuring AWS Bedrock tool config: {e}"),
                })?;

            bedrock_request = bedrock_request.tool_config(aws_bedrock_tool_config);
        }

        // We serialize here because the ConverseFluidBuilder type is not one you can import I guess
        let raw_request = serialize_aws_bedrock_struct(&bedrock_request)?;

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

        ConverseOutputWithMetadata {
            output,
            latency,
            raw_request,
        }
        .try_into()
    }

    async fn _infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
        _api_key: Cow<'a, SecretString>,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
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

        let mut bedrock_request = self
            .client
            .converse_stream()
            .model_id(&self.model_id)
            .set_messages(Some(messages))
            .inference_config(inference_config.build());

        if let Some(system) = &request.system {
            let system_block = SystemContentBlock::Text(system.clone());
            bedrock_request = bedrock_request.system(system_block);
        }

        if let Some(tool_config) = &request.tool_config {
            let tools: Vec<Tool> = tool_config
                .tools_available
                .iter()
                .map(Tool::try_from)
                .collect::<Result<Vec<_>, _>>()?;

            let tool_choice: AWSBedrockToolChoice = tool_config.tool_choice.clone().try_into()?;

            let aws_bedrock_tool_config = ToolConfiguration::builder()
                .set_tools(Some(tools))
                .tool_choice(tool_choice)
                .build()
                .map_err(|e| Error::AWSBedrockClient {
                    status_code: StatusCode::INTERNAL_SERVER_ERROR,
                    message: format!("Error configuring AWS Bedrock tool config: {e}"),
                })?;

            bedrock_request = bedrock_request.tool_config(aws_bedrock_tool_config);
        }

        let raw_request = serialize_aws_bedrock_struct(&bedrock_request)?;

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

        Ok((chunk, stream, raw_request))
    }
}

lazy_static! {
    static ref EMPTY_SECRET: SecretString = SecretString::from(String::new());
}

impl HasCredentials for AWSBedrockProvider {
    fn has_credentials(&self) -> bool {
        // TODO (#313): Actually check if the AWS Bedrock client is configured with credentials
        true
    }

    fn get_api_key<'a>(
        &'a self,
        _api_keys: &'a InferenceApiKeys,
    ) -> Result<Cow<'a, SecretString>, Error> {
        Ok(Cow::Borrowed(&EMPTY_SECRET))
    }
}

fn stream_bedrock(
    mut stream: ConverseStreamOutput,
    start_time: Instant,
) -> impl Stream<Item = Result<ProviderInferenceResponseChunk, Error>> {
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
) -> Result<Option<ProviderInferenceResponseChunk>, Error> {
    match output {
        ConverseStreamOutputType::ContentBlockDelta(message) => {
            let raw_message = serialize_aws_bedrock_struct(&message)?;

            match message.delta {
                Some(delta) => match delta {
                    ContentBlockDelta::Text(text) => Ok(Some(ProviderInferenceResponseChunk::new(
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
                        Ok(Some(ProviderInferenceResponseChunk::new(
                            inference_id,
                            // Take the current tool name and ID and use them to create a ToolCallChunk
                            // This is necessary because the ToolCallChunk must always contain the tool name and ID
                            // even though AWS Bedrock only sends the tool ID and name in the ToolUse chunk and not InputJSONDelta
                            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                                raw_name: current_tool_name.clone().ok_or(Error::AWSBedrockServer {
                                    message: "Got InputJsonDelta chunk from AWS Bedrock without current tool name being set by a ToolUse".to_string(),
                                })?,
                                id: current_tool_id.clone().ok_or(Error::AWSBedrockServer {
                                    message: "Got InputJsonDelta chunk from AWS Bedrock without current tool id being set by a ToolUse".to_string(),
                                })?,
                                raw_arguments: tool_use.input,
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
                    Ok(Some(ProviderInferenceResponseChunk::new(
                        inference_id,
                        vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                            id: tool_use.tool_use_id,
                            raw_name: tool_use.name,
                            raw_arguments: "".to_string(),
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
                        input_tokens: usage.input_tokens as u32,
                        output_tokens: usage.output_tokens as u32,
                    });

                    Ok(Some(ProviderInferenceResponseChunk::new(
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
            ContentBlock::ToolCall(tool_call) => {
                // Convert the tool call arguments from String to JSON Value...
                let input = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::AWSBedrockClient {
                        status_code: StatusCode::BAD_REQUEST,
                        message: format!("Error parsing tool call arguments as JSON Value: {e}"),
                    }
                })?;

                // ...then convert the JSON Value to an AWS SDK Document
                let input = serde_json::from_value(input).map_err(|e| Error::AWSBedrockServer {
                    message: format!(
                        "Error converting tool call arguments to AWS SDK Document: {e}"
                    ),
                })?;

                let tool_use_block = ToolUseBlock::builder()
                    .name(tool_call.name.clone())
                    .input(input)
                    .tool_use_id(tool_call.id.clone())
                    .build()
                    .map_err(|_| Error::AWSBedrockClient {
                        status_code: StatusCode::BAD_REQUEST,
                        message: "Error serializing tool call block".to_string(),
                    })?;

                Ok(BedrockContentBlock::ToolUse(tool_use_block))
            }
            ContentBlock::ToolResult(tool_result) => {
                let tool_result_block = ToolResultBlock::builder()
                    .tool_use_id(tool_result.id.clone())
                    .content(ToolResultContentBlock::Text(tool_result.result.clone()))
                    // NOTE: The AWS Bedrock SDK doesn't include `name` in the ToolResultBlock
                    .build()
                    .map_err(|_| Error::AWSBedrockClient {
                        status_code: StatusCode::BAD_REQUEST,
                        message: "Error serializing tool result block".to_string(),
                    })?;

                Ok(BedrockContentBlock::ToolResult(tool_result_block))
            }
        }
    }
}

impl TryFrom<BedrockContentBlock> for ContentBlock {
    type Error = Error;

    fn try_from(block: BedrockContentBlock) -> Result<Self, Self::Error> {
        match block {
            BedrockContentBlock::Text(text) => Ok(text.into()),
            BedrockContentBlock::ToolUse(tool_use) => {
                let arguments = serde_json::to_string(&tool_use.input).map_err(|e| {
                    Error::AWSBedrockServer {
                        message: format!("Error parsing tool call arguments from AWS Bedrock: {e}"),
                    }
                })?;

                Ok(ContentBlock::ToolCall(ToolCall {
                    name: tool_use.name,
                    arguments,
                    id: tool_use.tool_use_id,
                }))
            }
            _ => Err(Error::TypeConversion {
                message: format!(
                    "Unsupported content block type for AWS Bedrock: {}",
                    std::any::type_name_of_val(&block)
                ),
            }),
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

struct ConverseOutputWithMetadata {
    output: ConverseOutput,
    latency: Latency,
    raw_request: String,
}

impl TryFrom<ConverseOutputWithMetadata> for ProviderInferenceResponse {
    type Error = Error;

    fn try_from(value: ConverseOutputWithMetadata) -> Result<Self, Self::Error> {
        let ConverseOutputWithMetadata {
            output,
            latency,
            raw_request,
        } = value;

        let raw_response = serialize_aws_bedrock_struct(&output)?;

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

        let usage = output
            .usage
            .map(|u| Usage {
                input_tokens: u.input_tokens as u32,
                output_tokens: u.output_tokens as u32,
            })
            .ok_or(Error::AWSBedrockServer {
                message: "AWS Bedrock returned a message without usage information.".to_string(),
            })?;

        Ok(ProviderInferenceResponse::new(
            content,
            raw_request,
            raw_response,
            usage,
            latency,
        ))
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

impl TryFrom<&ToolConfig> for Tool {
    type Error = Error;

    fn try_from(tool_config: &ToolConfig) -> Result<Self, Error> {
        let tool_input_schema = ToolInputSchema::Json(
            serde_json::from_value(tool_config.parameters().clone()).map_err(|e| {
                Error::AWSBedrockClient {
                    status_code: StatusCode::INTERNAL_SERVER_ERROR,
                    message: format!("Error parsing tool input schema: {e}"),
                }
            })?,
        );

        let tool_spec = ToolSpecification::builder()
            .name(tool_config.name())
            .description(tool_config.description())
            .input_schema(tool_input_schema)
            .build()
            .map_err(|_| Error::AWSBedrockClient {
                status_code: StatusCode::INTERNAL_SERVER_ERROR,
                message: "Error configuring AWS Bedrock tool choice (this should never happen)"
                    .to_string(),
            })?;

        Ok(Tool::ToolSpec(tool_spec))
    }
}

impl TryFrom<ToolChoice> for AWSBedrockToolChoice {
    type Error = Error;

    fn try_from(tool_choice: ToolChoice) -> Result<Self, Error> {
        match tool_choice {
            // TODO (#204): Implement ToolChoice::None workaround for AWS Bedrock.
            //              MAKE SURE TO UPDATE THE E2E TESTS WHEN THIS IS DONE.
            ToolChoice::None => Err(Error::InvalidTool {
                message: "Tool choice is None. AWS Bedrock does not support tool choice None."
                    .to_string(),
            }),
            ToolChoice::Auto => Ok(AWSBedrockToolChoice::Auto(
                AutoToolChoice::builder().build(),
            )),
            ToolChoice::Required => Ok(AWSBedrockToolChoice::Any(AnyToolChoice::builder().build())),
            ToolChoice::Specific(tool_name) => Ok(AWSBedrockToolChoice::Tool(
                SpecificToolChoice::builder()
                    .name(tool_name)
                    .build()
                    .map_err(|_| Error::AWSBedrockClient {
                        status_code: StatusCode::INTERNAL_SERVER_ERROR,
                        message:
                            "Error configuring AWS Bedrock tool choice (this should never happen)"
                                .to_string(),
                    })?,
            )),
        }
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
            AWSBedrockProvider::new("test".to_string(), Some(Region::new("uk-hogwarts-1")))
                .await
                .unwrap();

            assert!(logs_contain(
                "Creating new AWS Bedrock client for region: uk-hogwarts-1"
            ));
        }

        #[traced_test]
        async fn second_run() {
            AWSBedrockProvider::new("test".to_string(), Some(Region::new("uk-hogwarts-1")))
                .await
                .unwrap();

            assert!(logs_contain(
                "Creating new AWS Bedrock client for region: uk-hogwarts-1"
            ));
        }

        #[traced_test]
        async fn third_run() {
            AWSBedrockProvider::new("test".to_string(), None)
                .await
                .unwrap();

            assert!(logs_contain("Creating new AWS Bedrock client for region:"));
        }

        #[traced_test]
        async fn fourth_run() {
            AWSBedrockProvider::new("test".to_string(), Some(Region::new("me-shire-2")))
                .await
                .unwrap();

            assert!(logs_contain(
                "Creating new AWS Bedrock client for region: me-shire-2"
            ));
        }

        // Every call should trigger client creation since each provider has its own AWS Bedrock client
        first_run().await;
        second_run().await;
        third_run().await;
        fourth_run().await;

        // NOTE: There isn't an easy way to test the fallback between the default provider and the last-resort fallback region.
        // We can only test that the method returns either of these regions when an explicit region is not provided.
    }
}
