use futures::StreamExt;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::{ContentBlock, Text};
use gateway::{inference::providers::aws_bedrock::AWSBedrockProvider, model::ProviderConfig};
use std::sync::OnceLock;
use tokio::runtime::Runtime;

use crate::providers::common::{
    create_simple_inference_request, create_streaming_inference_request,
};

/// NOTE: The `lazy_static` AWS client is thread-safe but not safe across Tokio runtimes. By default,
/// `tokio::test` spawns a new runtime for each test, causing intermittent issues with the AWS client.
/// Instead, we use a shared runtime for all tests. This runtime should be used for every integration
/// test that needs to make AWS calls. (This is not necessary for E2E tests since the AWS client runs
/// on a separate process then.)
static TEST_RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub fn get_test_runtime() -> &'static Runtime {
    TEST_RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create test runtime"))
}

#[test]
fn test_infer() {
    get_test_runtime().block_on(async {
        let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
        let provider = ProviderConfig::AWSBedrock(AWSBedrockProvider {
            model_id,
            region: None,
        });
        let client = reqwest::Client::new();
        let inference_request = create_simple_inference_request();
        let result = provider.infer(&inference_request, &client).await;
        assert!(result.is_ok(), "{}", result.unwrap_err());
        let result = result.unwrap();
        assert!(result.content.len() == 1);
        let content = result.content.first().unwrap();
        match content {
            ContentBlock::Text(Text { text }) => {
                assert!(!text.is_empty());
            }
            _ => unreachable!(),
        }
    });
}

#[test]
fn test_infer_with_region() {
    get_test_runtime().block_on(async {
        let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
        let provider = ProviderConfig::AWSBedrock(AWSBedrockProvider {
            model_id,
            region: Some(aws_types::region::Region::new("us-east-1")),
        });
        let client = reqwest::Client::new();
        let inference_request = create_simple_inference_request();
        let result = provider.infer(&inference_request, &client).await;
        assert!(result.is_ok(), "{}", result.unwrap_err());
        let result = result.unwrap();
        assert!(result.content.len() == 1);
        let content = result.content.first().unwrap();
        match content {
            ContentBlock::Text(Text { text }) => {
                assert!(!text.is_empty());
            }
            _ => unreachable!(),
        }
    });
}

#[test]
fn test_infer_with_broken_region() {
    get_test_runtime().block_on(async {
        let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
        let provider = ProviderConfig::AWSBedrock(AWSBedrockProvider {
            model_id,
            region: Some(aws_types::region::Region::new("uk-hogwarts-1")),
        });
        let client = reqwest::Client::new();
        let inference_request = create_simple_inference_request();
        let result = provider.infer(&inference_request, &client).await;
        assert!(result.is_err(), "{}", result.unwrap_err());
    });
}

#[test]
fn test_infer_stream() {
    get_test_runtime().block_on(async {
        let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
        let provider = ProviderConfig::AWSBedrock(AWSBedrockProvider {
            model_id,
            region: None,
        });
        let client = reqwest::Client::new();
        let inference_request = create_streaming_inference_request();

        let result = provider.infer_stream(&inference_request, &client).await;
        assert!(result.is_ok());
        let (chunk, mut stream) = result.unwrap();
        let mut collected_chunks = vec![chunk];
        while let Some(chunk) = stream.next().await {
            collected_chunks.push(chunk.unwrap());
        }
        assert!(!collected_chunks.is_empty());
        // Fourth as an arbitrary middle chunk
        assert!(collected_chunks[4].content.len() == 1);
        assert!(collected_chunks.last().unwrap().usage.is_some());
    });
}

// TODO (#81): add tests for tool calls and JSON mode
// #[tokio::test]
// async fn test_infer_with_tool_calls() {
// }
