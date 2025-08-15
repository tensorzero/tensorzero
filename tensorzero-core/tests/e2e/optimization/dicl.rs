use std::collections::HashMap;
use tensorzero_core::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::inference::types::{
    ContentBlock, ContentBlockChatOutput, ModelInput, RequestMessage, Role, Text,
};
use tensorzero_core::optimization::dicl::insert_dicl_examples_with_batching;
use tensorzero_core::stored_inference::RenderedSample;
use uuid::Uuid;

fn create_test_rendered_sample(input: &str, output: &str) -> RenderedSample {
    RenderedSample {
        function_name: "test_function".to_string(),
        input: ModelInput {
            system: None,
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: output.to_string(),
        })]),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: None,
        output_schema: None,
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    }
}

#[tokio::test]
async fn test_insert_dicl_examples_with_batching_success() {
    let clickhouse = get_clickhouse().await;

    let samples = vec![
        create_test_rendered_sample("test input 1", "test output 1"),
        create_test_rendered_sample("test input 2", "test output 2"),
    ];
    let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> =
        samples.into_iter().zip(embeddings.into_iter()).collect();

    let result = insert_dicl_examples_with_batching(
        &clickhouse,
        examples_with_embeddings,
        "test_function_e2e",
        "test_variant_e2e",
        10, // batch_size
    )
    .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_insert_dicl_examples_batching_logic() {
    let clickhouse = get_clickhouse().await;

    // Create 5 examples with batch size of 2, should create 3 batches (2+2+1)
    let samples = (0..5)
        .map(|i| {
            create_test_rendered_sample(&format!("test input {i}"), &format!("test output {i}"))
        })
        .collect::<Vec<_>>();
    let embeddings = (0..5)
        .map(|i| vec![i as f64, (i + 1) as f64, (i + 2) as f64])
        .collect::<Vec<_>>();
    let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> =
        samples.into_iter().zip(embeddings.into_iter()).collect();

    let result = insert_dicl_examples_with_batching(
        &clickhouse,
        examples_with_embeddings,
        "test_function_batching_e2e",
        "test_variant_batching_e2e",
        2, // Small batch size to force multiple batches
    )
    .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_insert_dicl_examples_with_empty_batch() {
    let clickhouse = get_clickhouse().await;

    let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> = vec![];

    let result = insert_dicl_examples_with_batching(
        &clickhouse,
        examples_with_embeddings,
        "test_function_empty_e2e",
        "test_variant_empty_e2e",
        10,
    )
    .await;

    // Should handle empty batch gracefully
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_insert_dicl_examples_json_serialization() {
    let clickhouse = get_clickhouse().await;

    // Test with special characters that need proper JSON encoding
    let sample = create_test_rendered_sample(
        "input with \"quotes\" and \n newlines",
        "output with special chars: àáâã",
    );
    let embedding = vec![1.0, 2.0, 3.0];
    let examples_with_embeddings = vec![(sample, embedding)];

    let result = insert_dicl_examples_with_batching(
        &clickhouse,
        examples_with_embeddings,
        "test_function_json_e2e",
        "test_variant_json_e2e",
        10,
    )
    .await;

    assert!(result.is_ok());
}
