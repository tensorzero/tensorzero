use futures::StreamExt;
use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero_core::{
    db::delegating_connection::DelegatingDatabaseConnection,
    db::inferences::{InferenceQueries, ListInferencesParams},
    db::model_inferences::ModelInferenceQueries,
    db::test_helpers::TestDatabaseHelpers,
    inference::types::{
        ContentBlockChatOutput, ContentBlockOutput, Role, StoredContentBlock, StoredRequestMessage,
        Text,
    },
    providers::dummy::{
        DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW, DUMMY_RAW_REQUEST,
        DUMMY_STREAMING_RESPONSE,
    },
    stored_inference::StoredInferenceDatabase,
    test_helpers::get_e2e_config,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// This test calls a function which calls a model where the provider is flaky but with retries.
#[gtest]
#[tokio::test]
async fn test_inference_flaky() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "flaky",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    expect_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
    assert_that!(content_blocks.len(), eq(1));
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    expect_that!(content_block_type, eq("text"));
    let content = content_block.get("text").unwrap().as_str().unwrap();
    expect_that!(content, eq(DUMMY_INFER_RESPONSE_CONTENT));

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    expect_that!(input_tokens, eq(10));
    expect_that!(output_tokens, eq(1));
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences.len(), eq(1));
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    expect_that!(chat.inference_id, eq(inference_id));
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, world!"}]
                }
            ]
        }
    );
    expect_that!(input, eq(&correct_input));
    // Check that content blocks are correct
    let output = chat.output.as_ref().unwrap();
    assert_that!(output.len(), eq(1));
    expect_that!(
        output[0],
        eq(&ContentBlockChatOutput::Text(Text {
            text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
        }))
    );
    // Check that episode_id is here and correct
    expect_that!(chat.episode_id, eq(episode_id));
    // Check the variant name
    expect_that!(chat.variant_name, eq("flaky"));

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences.len(), eq(1));
    let mi = &model_inferences[0];
    expect_that!(mi.inference_id, eq(inference_id));

    expect_that!(mi.input_tokens, eq(Some(10)));
    expect_that!(mi.output_tokens, eq(Some(1)));
    expect_that!(mi.response_time_ms.unwrap(), gt(0));
    expect_that!(mi.ttft_ms, none());
    expect_that!(
        mi.raw_response.as_deref().unwrap(),
        eq(DUMMY_INFER_RESPONSE_RAW)
    );
    expect_that!(mi.raw_request.as_deref().unwrap(), eq(DUMMY_RAW_REQUEST));
    expect_that!(
        mi.system.as_deref().unwrap(),
        eq("You are a helpful and friendly assistant named AskJeeves")
    );
    expect_that!(
        mi.input_messages.as_ref().unwrap(),
        eq(&vec![StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "Hello, world!".to_string()
            })],
        }])
    );
    let output = mi.output.as_ref().unwrap();
    expect_that!(
        output,
        eq(&vec![ContentBlockOutput::Text(Text {
            text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
        })])
    );
}

/// This test checks that streaming inference works as expected with a flaky provider and retries.
#[gtest]
#[tokio::test]
async fn test_streaming_flaky() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "flaky",
        "episode_id": episode_id,
        "input":
            {
                "system": {
                    "assistant_name": "AskJeeves"
                },
                "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": true,
        "params": {
            "chat_completion": {
                "temperature": 2.0,
            "max_tokens": 200,
            "seed": 420
        }}
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();
    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    let mut inference_id = None;
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_array().unwrap();
            assert_that!(content.len(), eq(1));
            let content_block = content.first().unwrap();
            let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
            expect_that!(content_block_type, eq("text"));
            let content = content_block.get("text").unwrap().as_str().unwrap();
            expect_that!(content, eq(DUMMY_STREAMING_RESPONSE[i]));
        } else {
            expect_that!(
                chunk_json
                    .get("content")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .is_empty(),
                eq(true)
            );
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
            let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
            expect_that!(input_tokens, eq(10));
            expect_that!(output_tokens, eq(16));
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }
    let inference_id = inference_id.unwrap();

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences.len(), eq(1));
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    expect_that!(chat.inference_id, eq(inference_id));
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, world!"}]
            }
        ]}
    );
    expect_that!(input, eq(&correct_input));
    // Check content blocks
    let output = chat.output.as_ref().unwrap();
    assert_that!(output.len(), eq(1));
    expect_that!(
        output[0],
        eq(&ContentBlockChatOutput::Text(Text {
            text: DUMMY_STREAMING_RESPONSE.join(""),
        }))
    );
    expect_that!(chat.episode_id, eq(episode_id));
    // Check the variant name
    expect_that!(chat.variant_name, eq("flaky"));
    // Check the inference_params (set via payload)
    let inference_params = serde_json::to_value(&chat.inference_params).unwrap();
    let chat_completion_inference_params = inference_params
        .get("chat_completion")
        .unwrap()
        .as_object()
        .unwrap();
    let temperature = chat_completion_inference_params
        .get("temperature")
        .unwrap()
        .as_f64()
        .unwrap();
    expect_that!(temperature, eq(2.0));
    let max_tokens = chat_completion_inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    expect_that!(max_tokens, eq(200));
    let seed = chat_completion_inference_params
        .get("seed")
        .unwrap()
        .as_u64()
        .unwrap();
    expect_that!(seed, eq(420));

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences.len(), eq(1));
    let mi = &model_inferences[0];
    expect_that!(mi.input_tokens, eq(Some(10)));
    expect_that!(mi.output_tokens, eq(Some(16)));
    let response_time_ms = mi.response_time_ms.unwrap();
    expect_that!(response_time_ms, gt(0));
    let ttft = mi.ttft_ms.unwrap();
    expect_that!(ttft, gt(0));
    expect_that!(ttft, le(response_time_ms));
    expect_that!(mi.raw_response, some(anything()));
    expect_that!(mi.raw_request.as_deref().unwrap(), eq(DUMMY_RAW_REQUEST));
    expect_that!(
        mi.system.as_deref().unwrap(),
        eq("You are a helpful and friendly assistant named AskJeeves")
    );
    expect_that!(
        mi.input_messages.as_ref().unwrap(),
        eq(&vec![StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "Hello, world!".to_string()
            })],
        }])
    );
    expect_that!(
        mi.output.as_ref().unwrap(),
        eq(&vec![ContentBlockOutput::Text(Text {
            text: DUMMY_STREAMING_RESPONSE.join(""),
        })])
    );
}

/// This test calls a function which currently uses best of n.
/// We call 2 models, one that gives the usual good response, one that
/// gives a JSON response, and then use a flaky judge to select the best one (it should fail the first time but succeed the second).
/// We check that the good response is selected and that the other responses are not
/// but they get stored to the ModelInference table.
#[gtest]
#[tokio::test]
async fn test_best_of_n_dummy_candidates_flaky_judge() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n",
        "variant_name": "flaky_best_of_n_variant",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Please write me a sentence about Megumin making an explosion."
                }
            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    expect_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
    assert_that!(content_blocks.len(), eq(1));
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    expect_that!(content_block_type, eq("text"));
    let content = content_block.get("text").unwrap().as_str().unwrap();
    expect_that!(content, eq(DUMMY_INFER_RESPONSE_CONTENT));

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    expect_that!(
        input_tokens,
        gt(10),
        "Unexpected input tokens: {input_tokens}"
    );
    expect_that!(output_tokens, eq(3), "Unexpected output tokens");
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                ids: Some(&[inference_id]),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_that!(inferences.len(), eq(1));
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    expect_that!(chat.inference_id, eq(inference_id));
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Please write me a sentence about Megumin making an explosion."}]
                }
            ]
        }
    );
    expect_that!(input, eq(&correct_input));
    // Check that content blocks are correct
    let output = chat.output.as_ref().unwrap();
    assert_that!(output.len(), eq(1));
    expect_that!(
        output[0],
        eq(&ContentBlockChatOutput::Text(Text {
            text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
        }))
    );
    // Check that episode_id is here and correct
    expect_that!(chat.episode_id, eq(episode_id));
    // Check the variant name
    expect_that!(chat.variant_name, eq("flaky_best_of_n_variant"));

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    expect_that!(model_inferences.len(), eq(3));

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for mi in &model_inferences {
        expect_that!(mi.inference_id, eq(inference_id));
        model_names.insert(mi.model_name.clone());

        expect_that!(mi.input_tokens.unwrap(), gt(0));
        expect_that!(mi.output_tokens.unwrap(), gt(0));
        expect_that!(mi.response_time_ms.unwrap(), gt(0));
        expect_that!(mi.ttft_ms, none());
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "json", "flaky_best_of_n_judge"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
    expect_that!(model_names, eq(&expected_model_names));
}
