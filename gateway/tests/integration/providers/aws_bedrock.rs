use gateway::{inference::providers::aws_bedrock::AWSBedrockProvider, model::ProviderConfig};

use crate::providers::common::IntegrationTestProviders;

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> IntegrationTestProviders {
    // Generic provider for testing
    let model_id = "anthropic.claude-3-haiku-20240307-v1:0".to_string();
    let provider =
        ProviderConfig::AWSBedrock(AWSBedrockProvider::new(model_id, None).await.unwrap());

    IntegrationTestProviders::with_provider(provider)
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
