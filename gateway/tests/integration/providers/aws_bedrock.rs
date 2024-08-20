use futures::StreamExt;

use gateway::inference::providers::provider_trait::InferenceProvider;
use gateway::inference::types::ContentBlockChunk;
use gateway::{inference::providers::aws_bedrock::AWSBedrockProvider, model::ProviderConfig};

use crate::providers::common::{create_streaming_tool_result_inference_request, TestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> TestProviders {
    // Generic provider for testing
    let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
    let provider =
        ProviderConfig::AWSBedrock(AWSBedrockProvider::new(model_id, None).await.unwrap());

    TestProviders::with_provider(provider)
}

#[tokio::test]
async fn test_simple_inference_request_with_region() {
    let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
    let provider = ProviderConfig::AWSBedrock(
        AWSBedrockProvider::new(model_id, Some(aws_types::region::Region::new("us-east-1")))
            .await
            .unwrap(),
    );

    test_simple_inference_request_with_provider(&provider).await;
}

#[tokio::test]
#[should_panic]
async fn test_simple_inference_request_with_broken_region() {
    let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
    let provider = ProviderConfig::AWSBedrock(
        AWSBedrockProvider::new(
            model_id,
            Some(aws_types::region::Region::new("uk-hogwarts-1")),
        )
        .await
        .unwrap(),
    );

    test_simple_inference_request_with_provider(&provider).await;
}

#[tokio::test]
async fn test_infer_with_tool_result_stream() {
    let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
    let provider =
        ProviderConfig::AWSBedrock(AWSBedrockProvider::new(model_id, None).await.unwrap());
    let client = reqwest::Client::new();
    let inference_request = create_streaming_tool_result_inference_request();
    let result = provider.infer_stream(&inference_request, &client).await;

    let (chunk, mut stream) = result.unwrap();
    let mut collected_chunks = vec![chunk];
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        collected_chunks.push(chunk);
    }

    assert!(!collected_chunks.is_empty());
    // Fourth as an arbitrary middle chunk
    assert!(collected_chunks[4].content.len() == 1);
    assert!(collected_chunks.last().unwrap().usage.is_some());

    // Collect all chunks and retrieve the generation
    let generation = collected_chunks
        .iter()
        .filter(|chunk| !chunk.content.is_empty())
        .map(|chunk| chunk.content.first().unwrap())
        .map(|content| match content {
            ContentBlockChunk::Text(block) => block.text.clone(),
            b => panic!("Unexpected content block: {:?}", b),
        })
        .collect::<Vec<String>>()
        .join("");

    assert!(generation.contains("New York"));
    assert!(generation.contains("70"));
}
