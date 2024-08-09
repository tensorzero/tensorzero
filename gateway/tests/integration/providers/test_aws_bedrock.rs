use crate::integration::providers::common::create_simple_inference_request;
use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::{ContentBlock, Text};
use gateway::{inference::providers::aws_bedrock::AWSBedrockProvider, model::ProviderConfig};
use std::sync::OnceLock;
use tokio::runtime::Runtime;

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
    let rt = get_test_runtime();

    rt.block_on(async {
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
    let rt = get_test_runtime();

    rt.block_on(async {
        let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
        let provider = ProviderConfig::AWSBedrock(AWSBedrockProvider {
            model_id,
            region: Some("us-east-1".to_string()),
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
    let rt = get_test_runtime();

    rt.block_on(async {
        let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
        let provider = ProviderConfig::AWSBedrock(AWSBedrockProvider {
            model_id,
            region: Some("uk-hogwarts-1".to_string()),
        });
        let client = reqwest::Client::new();
        let inference_request = create_simple_inference_request();
        let result = provider.infer(&inference_request, &client).await;
        assert!(result.is_err(), "{}", result.unwrap_err());
    });
}

// TODO (#81): add tests for streaming and tool calls
// #[tokio::test]
// async fn test_infer_stream() {
// }

// #[tokio::test]
// async fn test_infer_with_tool_calls() {
// }
