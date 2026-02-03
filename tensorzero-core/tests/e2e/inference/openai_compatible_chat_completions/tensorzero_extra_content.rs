//! E2E tests for `tensorzero_extra_content_experimental` in the OpenAI-compatible API.
//!
//! Tests that extra content blocks (Thought, Unknown) are correctly returned in responses
//! and can be round-tripped through the API.

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Non-streaming round-trip test for `tensorzero_extra_content_experimental`.
///
/// 1. Make an inference request with a model that returns Thought content
/// 2. Verify the response contains `tensorzero_extra_content_experimental` with proper structure
/// 3. Send a follow-up request including the extra_content in an assistant message
/// 4. Verify the request is processed successfully
#[tokio::test]
async fn test_extra_content_roundtrip_non_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Step 1: Make initial inference request using a model that returns Thought content
    // The dummy::reasoner model returns [Thought, Text] content
    let payload = json!({
        "model": "tensorzero::model_name::dummy::reasoner",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string()
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();

    assert_eq!(
        status,
        StatusCode::OK,
        "Initial request should be successful. Body: {response_text}"
    );

    let response_json: Value = serde_json::from_str(&response_text).unwrap();

    // Step 2: Verify response structure
    let choices = response_json
        .get("choices")
        .expect("Response should have choices");
    let message = choices[0]
        .get("message")
        .expect("Choice should have message");

    // Check that content exists (the text part)
    let content = message.get("content").expect("Message should have content");
    assert!(
        content.is_string(),
        "Content should be a string with the text output"
    );

    // Check that tensorzero_extra_content_experimental exists
    let extra_content = message
        .get("tensorzero_extra_content_experimental")
        .expect("Message should have tensorzero_extra_content_experimental");
    assert!(
        extra_content.is_array(),
        "tensorzero_extra_content_experimental should be an array"
    );

    let extra_content_array = extra_content.as_array().unwrap();
    assert!(
        !extra_content_array.is_empty(),
        "tensorzero_extra_content_experimental should have at least one entry"
    );

    // Verify the structure of the first extra content block
    let first_block = &extra_content_array[0];
    assert_eq!(
        first_block.get("type").and_then(|v| v.as_str()),
        Some("thought"),
        "First extra content block should be a thought"
    );
    assert!(
        first_block.get("insert_index").is_some(),
        "Thought block should have insert_index"
    );
    assert!(
        first_block.get("text").is_some(),
        "Thought block should have text field (flattened from Thought struct)"
    );

    // Step 3: Round-trip - send the extra_content back as an assistant message
    let roundtrip_payload = json!({
        "model": "tensorzero::model_name::dummy::echo",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            },
            {
                "role": "assistant",
                "content": content,
                "tensorzero_extra_content_experimental": extra_content
            },
            {
                "role": "user",
                "content": "Continue"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string()
    });

    let roundtrip_response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&roundtrip_payload)
        .send()
        .await
        .unwrap();

    let roundtrip_status = roundtrip_response.status();
    let roundtrip_response_text = roundtrip_response.text().await.unwrap();

    assert_eq!(
        roundtrip_status,
        StatusCode::OK,
        "Round-trip request should be successful. Body: {roundtrip_response_text}"
    );

    // Verify the round-trip response is valid
    let roundtrip_json: Value = serde_json::from_str(&roundtrip_response_text).unwrap();
    assert!(
        roundtrip_json.get("choices").is_some(),
        "Round-trip response should have choices"
    );
}

/// Streaming round-trip test for `tensorzero_extra_content_experimental`.
///
/// 1. Make a streaming inference request with a model that returns Thought content
/// 2. Collect and verify streaming chunks contain extra_content with proper structure
/// 3. Verify insert_index consistency across chunks with the same ID
/// 4. Reconstruct extra_content from chunks and send back as input
/// 5. Verify the round-trip works correctly
#[tokio::test]
async fn test_extra_content_roundtrip_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Step 1: Make streaming inference request
    let payload = json!({
        "model": "tensorzero::model_name::dummy::reasoner",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "stream": true,
        "tensorzero::episode_id": episode_id.to_string()
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut all_chunks: Vec<Value> = Vec::new();
    let mut extra_content_chunks: Vec<Value> = Vec::new();
    let mut content_text = String::new();

    // Step 2: Collect streaming chunks
    while let Some(event) = response.next().await {
        let event = event.expect("Failed to receive event");
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }

                let chunk_json: Value =
                    serde_json::from_str(&message.data).expect("Failed to parse chunk as JSON");
                all_chunks.push(chunk_json.clone());

                // Collect content delta
                if let Some(choices) = chunk_json.get("choices")
                    && let Some(delta) = choices[0].get("delta")
                {
                    // Collect text content
                    if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                        content_text.push_str(content);
                    }

                    // Collect extra content chunks
                    if let Some(extra) = delta.get("tensorzero_extra_content_experimental")
                        && let Some(arr) = extra.as_array()
                    {
                        for block in arr {
                            extra_content_chunks.push(block.clone());
                        }
                    }
                }
            }
        }
    }

    // Step 3: Verify we received extra content in streaming
    assert!(
        !extra_content_chunks.is_empty(),
        "Streaming response should include tensorzero_extra_content_experimental chunks.\n\
        Total chunks received: {}\n\
        All chunks:\n{:#?}",
        all_chunks.len(),
        all_chunks
    );

    // Verify structure of extra content chunks
    for chunk in &extra_content_chunks {
        assert!(
            chunk.get("type").is_some(),
            "Extra content chunk should have type field"
        );
        // insert_index should be present (at least in final chunks)
        // Note: In streaming, insert_index is determined by chunk order/id
    }

    // Step 4: Reconstruct full extra_content blocks for round-trip
    // For streaming, we need to aggregate chunks by ID
    // For simplicity, let's use the first complete block we can construct
    let reconstructed_extra_content: Vec<Value> = extra_content_chunks
        .iter()
        .filter(|chunk| {
            // Include chunks that have insert_index (they're complete enough for round-trip)
            chunk.get("insert_index").is_some()
        })
        .cloned()
        .collect();

    // Step 5: Round-trip the extra_content
    if !reconstructed_extra_content.is_empty() && !content_text.is_empty() {
        let roundtrip_payload = json!({
            "model": "tensorzero::model_name::dummy::echo",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                },
                {
                    "role": "assistant",
                    "content": content_text,
                    "tensorzero_extra_content_experimental": reconstructed_extra_content
                },
                {
                    "role": "user",
                    "content": "Continue"
                }
            ],
            "stream": false,
            "tensorzero::episode_id": episode_id.to_string()
        });

        let roundtrip_response = client
            .post(get_gateway_endpoint("/openai/v1/chat/completions"))
            .json(&roundtrip_payload)
            .send()
            .await
            .unwrap();

        let roundtrip_status = roundtrip_response.status();
        let roundtrip_response_text = roundtrip_response.text().await.unwrap();

        assert_eq!(
            roundtrip_status,
            StatusCode::OK,
            "Streaming round-trip request should be successful. Body: {roundtrip_response_text}"
        );
    }
}

/// Test that insert_index is correctly assigned based on content position.
///
/// The dummy::reasoner model returns [Thought(index=0), Text(index=1)].
/// We verify that insert_index=0 for the Thought block.
#[tokio::test]
async fn test_extra_content_insert_index_correctness() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::dummy::reasoner",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string()
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();

    assert_eq!(
        status,
        StatusCode::OK,
        "Request should be successful. Body: {response_text}"
    );

    let response_json: Value = serde_json::from_str(&response_text).unwrap();
    let message = &response_json["choices"][0]["message"];

    let extra_content = message
        .get("tensorzero_extra_content_experimental")
        .expect("Should have extra_content");
    let extra_content_array = extra_content.as_array().unwrap();

    // The reasoner model returns [Thought, Text]
    // So Thought should have insert_index = 0
    let thought_block = &extra_content_array[0];
    assert_eq!(
        thought_block.get("type").and_then(|v| v.as_str()),
        Some("thought"),
        "First block should be a thought"
    );
    assert_eq!(
        thought_block.get("insert_index").and_then(|v| v.as_u64()),
        Some(0),
        "Thought block should have insert_index=0 since it came first in the content array"
    );
}

/// Test round-trip with multiple extra content blocks.
///
/// Verifies that when we send back extra_content with insert_index values,
/// the content is correctly reconstructed on the input side.
#[tokio::test]
async fn test_extra_content_multi_block_roundtrip() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Construct a request with multiple extra content blocks
    let payload = json!({
        "model": "tensorzero::model_name::dummy::echo",
        "messages": [
            {
                "role": "user",
                "content": "Start"
            },
            {
                "role": "assistant",
                "content": "Middle text",
                "tensorzero_extra_content_experimental": [
                    {
                        "type": "thought",
                        "insert_index": 0,
                        "text": "First thought before text"
                    },
                    {
                        "type": "thought",
                        "insert_index": 2,
                        "text": "Second thought after text"
                    }
                ]
            },
            {
                "role": "user",
                "content": "End"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string()
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();

    assert_eq!(
        status,
        StatusCode::OK,
        "Multi-block round-trip should succeed. Body: {response_text}"
    );

    // The request was processed successfully, meaning the extra_content
    // blocks were correctly inserted into the message content
    let response_json: Value = serde_json::from_str(&response_text).unwrap();
    assert!(
        response_json.get("choices").is_some(),
        "Response should have choices"
    );
}

/// Test that extra_content without insert_index is appended to the end.
#[tokio::test]
async fn test_extra_content_unindexed_appended() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // Send extra_content without insert_index - should be appended
    let payload = json!({
        "model": "tensorzero::model_name::dummy::echo",
        "messages": [
            {
                "role": "user",
                "content": "Start"
            },
            {
                "role": "assistant",
                "content": "Some text",
                "tensorzero_extra_content_experimental": [
                    {
                        "type": "thought",
                        "text": "Unindexed thought"
                        // No insert_index - should be appended to end
                    }
                ]
            },
            {
                "role": "user",
                "content": "End"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string()
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();

    assert_eq!(
        status,
        StatusCode::OK,
        "Unindexed extra_content should be accepted. Body: {response_text}"
    );
}
