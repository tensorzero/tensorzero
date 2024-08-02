use crate::integration::providers::common::create_simple_inference_request;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::{inference::providers::aws_bedrock::AWSBedrockProvider, model::ProviderConfig};

#[tokio::test]
async fn test_infer() {
    let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
    let config = ProviderConfig::AWSBedrock { model_id };
    let client = reqwest::Client::new();
    let inference_request = create_simple_inference_request();
    let result = AWSBedrockProvider::infer(&inference_request, &config, &client).await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
    assert!(result.unwrap().content.is_some());
}

// TODO: add tests for streaming and tool calls
// #[tokio::test]
// async fn test_infer_stream() {
// }

// #[tokio::test]
// async fn test_infer_with_tool_calls() {
// }
