use std::collections::HashMap;
use tensorzero::DynamicToolParams;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, ModelInput, ResolvedContentBlock, ResolvedRequestMessage, Role,
    StoredInput, StoredInputMessage, StoredInputMessageContent, Text,
};
use tensorzero_core::stored_inference::RenderedSample;
use tensorzero_optimizers::dicl::{dicl_examples_exist, insert_dicl_examples_with_batching};
use uuid::Uuid;

fn create_test_rendered_sample(input: &str, output: &str) -> RenderedSample {
    let output_vec = vec![ContentBlockChatOutput::Text(Text {
        text: output.to_string(),
    })];
    RenderedSample {
        function_name: "test_function".to_string(),
        input: ModelInput {
            system: None,
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        output: Some(output_vec.clone()),
        stored_output: Some(tensorzero_core::stored_inference::StoredOutput::Chat(
            output_vec,
        )),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: DynamicToolParams::default(),
        output_schema: None,
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    }
}

#[tokio::test]
async fn test_insert_dicl_examples_success() {
    let clickhouse = get_clickhouse().await;

    // Generate unique names to ensure test isolation
    let function_name = format!("test_function_e2e_{}", Uuid::now_v7());
    let variant_name = format!("test_variant_e2e_{}", Uuid::now_v7());

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
        &function_name,
        &variant_name,
        10, // batch_size
    )
    .await;

    assert!(result.is_ok());

    // Verify the data was actually written to ClickHouse by querying the DynamicInContextLearningExample table
    // We should find exactly 2 rows with the expected function_name and variant_name
    let query = format!(
        "SELECT COUNT(*) as count FROM DynamicInContextLearningExample WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
    );
    let count_result = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let count_str = count_result.response;
    let count: u32 = count_str.trim().parse().unwrap();
    assert_eq!(
        count, 2,
        "Expected 2 DICL examples to be inserted, but found {count}"
    );
}

#[tokio::test]
async fn test_insert_dicl_examples_multiple_chunks() {
    let clickhouse = get_clickhouse().await;

    // Generate unique names to ensure test isolation
    let function_name = format!("test_function_chunks_e2e_{}", Uuid::now_v7());
    let variant_name = format!("test_variant_chunks_e2e_{}", Uuid::now_v7());

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
        &function_name,
        &variant_name,
        2, // Small batch size to force multiple batches
    )
    .await;

    assert!(result.is_ok());

    // Verify the chunking logic worked correctly by checking that all 5 examples were inserted
    // This test specifically validates that the chunking with size 2 creates 3 chunks (2+2+1) and all data is preserved
    let query = format!(
        "SELECT COUNT(*) as count FROM DynamicInContextLearningExample WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
    );
    let count_result = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let count_str = count_result.response;
    let count: u32 = count_str.trim().parse().unwrap();
    assert_eq!(
        count, 5,
        "Expected 5 DICL examples to be inserted across 3 chunks, but found {count}"
    );
}

#[tokio::test]
async fn test_insert_dicl_examples_with_empty_input() {
    let clickhouse = get_clickhouse().await;

    // Generate unique names to ensure test isolation
    let function_name = format!("test_function_empty_e2e_{}", Uuid::now_v7());
    let variant_name = format!("test_variant_empty_e2e_{}", Uuid::now_v7());

    let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> = vec![];

    let result = insert_dicl_examples_with_batching(
        &clickhouse,
        examples_with_embeddings,
        &function_name,
        &variant_name,
        10,
    )
    .await;

    // Should handle empty batch gracefully
    assert!(result.is_ok());

    // Verify that no data was written to ClickHouse for empty input
    // This confirms that empty inputs are handled properly without creating spurious database entries
    let query = format!(
        "SELECT COUNT(*) as count FROM DynamicInContextLearningExample WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
    );
    let count_result = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let count_str = count_result.response;
    let count: u32 = count_str.trim().parse().unwrap();
    assert_eq!(
        count, 0,
        "Expected 0 DICL examples for empty input, but found {count}"
    );
}

#[tokio::test]
async fn test_insert_dicl_examples_json_serialization() {
    let clickhouse = get_clickhouse().await;

    // Generate unique variant name to ensure test isolation
    let variant_name = format!("test_variant_json_e2e_{}", Uuid::now_v7());

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
        &variant_name,
        10,
    )
    .await;

    assert!(result.is_ok());

    // Verify that the data with special characters was correctly inserted and can be retrieved
    // This test ensures JSON serialization handles quotes, newlines, and Unicode characters properly
    let count_query = format!(
        "SELECT COUNT(*) as count FROM DynamicInContextLearningExample WHERE function_name = 'test_function_json_e2e' AND variant_name = '{variant_name}'"
    );
    let count_result = clickhouse
        .run_query_synchronous_no_params(count_query)
        .await
        .unwrap();
    let count_str = count_result.response;
    let count: u32 = count_str.trim().parse().unwrap();
    assert_eq!(
        count, 1,
        "Expected 1 DICL example with special characters to be inserted, but found {count}"
    );

    // Also verify that the input and output contain the expected special characters
    let content_query = format!(
        "SELECT input, output FROM DynamicInContextLearningExample WHERE function_name = 'test_function_json_e2e' AND variant_name = '{variant_name}' FORMAT JSONEachRow"
    );
    let content_result = clickhouse
        .run_query_synchronous_no_params(content_query)
        .await
        .unwrap();
    let content_str = content_result.response;

    // Parse the JSON line to verify the content
    let parsed: serde_json::Value = serde_json::from_str(content_str.trim()).unwrap();
    let input_str = parsed["input"].as_str().unwrap();
    let output_str = parsed["output"].as_str().unwrap();

    assert!(
        input_str.contains("\\\"quotes\\\""),
        "Input should contain escaped quotes: {input_str}"
    );
    assert!(
        input_str.contains("\\n"),
        "Input should contain escaped newlines: {input_str}"
    );
    assert!(
        output_str.contains("àáâã"),
        "Output should contain Unicode chars: {output_str}"
    );
}

#[tokio::test]
async fn test_dicl_examples_exist() {
    let clickhouse = get_clickhouse().await;

    // Generate unique names to ensure test isolation
    let function_name = format!("test_function_exists_{}", Uuid::now_v7());
    let variant_name = format!("test_variant_exists_{}", Uuid::now_v7());
    let non_existent_variant = format!("test_variant_non_existent_{}", Uuid::now_v7());

    // First, check that no examples exist for the variant
    let exists_before = dicl_examples_exist(&clickhouse, &function_name, &variant_name)
        .await
        .unwrap();
    assert!(
        !exists_before,
        "Expected no DICL examples to exist initially for variant '{variant_name}'"
    );

    // Insert some examples
    let samples = vec![
        create_test_rendered_sample("test input 1", "test output 1"),
        create_test_rendered_sample("test input 2", "test output 2"),
    ];
    let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let examples_with_embeddings: Vec<(RenderedSample, Vec<f64>)> =
        samples.into_iter().zip(embeddings.into_iter()).collect();

    let insert_result = insert_dicl_examples_with_batching(
        &clickhouse,
        examples_with_embeddings,
        &function_name,
        &variant_name,
        10,
    )
    .await;
    assert!(insert_result.is_ok(), "Failed to insert DICL examples");

    // Now check that examples exist for the variant
    let exists_after = dicl_examples_exist(&clickhouse, &function_name, &variant_name)
        .await
        .unwrap();
    assert!(
        exists_after,
        "Expected DICL examples to exist after insertion for variant '{variant_name}'"
    );

    // Check that a non-existent variant returns false
    let non_existent = dicl_examples_exist(&clickhouse, &function_name, &non_existent_variant)
        .await
        .unwrap();
    assert!(
        !non_existent,
        "Expected no DICL examples to exist for non-existent variant '{non_existent_variant}'"
    );

    // Also check that a different function name returns false
    let different_function = format!("different_function_{}", Uuid::now_v7());
    let different_function_exists =
        dicl_examples_exist(&clickhouse, &different_function, &variant_name)
            .await
            .unwrap();
    assert!(
        !different_function_exists,
        "Expected no DICL examples to exist for different function '{different_function}'"
    );
}
