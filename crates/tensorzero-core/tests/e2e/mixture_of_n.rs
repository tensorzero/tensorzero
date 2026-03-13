use futures::StreamExt;
use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use rust_decimal::Decimal;
use serde_json::{Value, json};
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::inferences::{InferenceQueries, ListInferencesParams};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, ContentBlockOutput, FinishReason, Role, StoredContentBlock,
    StoredModelInference, StoredRequestMessage, Text, Unknown, Usage,
};
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tensorzero_core::test_helpers::get_e2e_config;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[gtest]
#[tokio::test]
async fn test_mixture_of_n_dummy_candidates_dummy_judge_non_stream() {
    // Include randomness in put to make sure that the first request is a cache miss
    let random_input = Uuid::now_v7();
    test_mixture_of_n_dummy_candidates_dummy_judge_inner(random_input, false, false).await;
    test_mixture_of_n_dummy_candidates_dummy_judge_inner(random_input, true, false).await;
}

#[gtest]
#[tokio::test]
async fn test_mixture_of_n_dummy_candidates_dummy_judge_streaming() {
    // Include randomness in put to make sure that the first request is a cache miss
    let random_input = Uuid::now_v7();
    test_mixture_of_n_dummy_candidates_dummy_judge_inner(random_input, false, true).await;
    test_mixture_of_n_dummy_candidates_dummy_judge_inner(random_input, true, true).await;
}

async fn test_mixture_of_n_dummy_candidates_dummy_judge_inner(
    random_input: Uuid,
    should_be_cached: bool,
    stream: bool,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n_json_repeated",
        "variant_name": "mixture_of_n_variant",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": format!("Please write me a sentence about Megumin making an explosion: {random_input}")},
                    ]
                }
            ]},
        "stream": stream,
        "cache_options": {"enabled": "on", "lookback_s": 10}
    });

    let builder = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload);
    let (inference_id, output_usage) = if stream {
        let mut chunks = builder.eventsource().await.unwrap();
        let mut first_inference_id = None;
        let mut last_chunk = None;
        while let Some(chunk) = chunks.next().await {
            println!("chunk: {chunk:?}");
            let chunk = chunk.unwrap();
            let Event::Message(chunk) = chunk else {
                continue;
            };
            if chunk.data == "[DONE]" {
                break;
            }
            let chunk_json = chunk.data;
            let chunk_json: Value = serde_json::from_str(&chunk_json).unwrap();
            let inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
            let inference_id = Uuid::parse_str(inference_id).unwrap();
            if first_inference_id.is_none() {
                first_inference_id = Some(inference_id);
            }
            last_chunk = Some(chunk_json);
        }
        let usage = last_chunk.unwrap().get("usage").unwrap().clone();
        let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap() as u32;
        let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap() as u32;
        (
            first_inference_id.unwrap(),
            Usage {
                input_tokens: Some(input_tokens),
                output_tokens: Some(output_tokens),
                cache_read_input_tokens: None,
                cache_write_input_tokens: None,
                cost: None,
            },
        )
    } else {
        let response = builder.send().await.unwrap();
        // Check Response is OK
        assert_eq!(response.status(), StatusCode::OK);
        let response_json = response.json::<Value>().await.unwrap();

        let usage = response_json.get("usage").unwrap();
        let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap() as u32;
        let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap() as u32;

        // Check that inference_id is here
        let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
        (
            Uuid::parse_str(inference_id).unwrap(),
            Usage {
                input_tokens: Some(input_tokens),
                output_tokens: Some(output_tokens),
                cache_read_input_tokens: None,
                cache_write_input_tokens: None,
                cost: None,
            },
        )
    };

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
    assert_eq!(inferences.len(), 1);
    let json_inf = match &inferences[0] {
        StoredInferenceDatabase::Json(j) => j,
        StoredInferenceDatabase::Chat(_) => panic!("Expected json inference"),
    };
    assert_eq!(json_inf.inference_id, inference_id);
    let input = serde_json::to_value(&json_inf.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": format!("Please write me a sentence about Megumin making an explosion: {random_input}")},
                    ]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 4);

    // Collect model names
    let mut model_names = std::collections::HashSet::new();
    let mut dummy_uuids = vec![];

    let mut usage_sum = Usage {
        input_tokens: Some(0),
        output_tokens: Some(0),
        cache_read_input_tokens: None,
        cache_write_input_tokens: None,
        cost: None,
    };

    for mi in &model_inferences {
        assert_eq!(mi.inference_id, inference_id);

        // Collect model_name
        model_names.insert(mi.model_name.clone());

        // Check that all expected fields are present
        assert!(!mi.model_provider_name.is_empty());
        assert!(mi.raw_request.is_some());
        assert!(mi.raw_response.is_some());
        assert!(mi.input_tokens.is_some());
        assert!(mi.output_tokens.is_some());
        assert!(mi.response_time_ms.is_some());

        let input_tokens = mi.input_tokens.unwrap();
        let output_tokens = mi.output_tokens.unwrap();
        usage_sum.input_tokens = Some(usage_sum.input_tokens.unwrap() + input_tokens);
        usage_sum.output_tokens = Some(usage_sum.output_tokens.unwrap() + output_tokens);

        // We just check the output here, since we already have several tests covering the other fields
        // for mixture_of_n
        if mi.model_name == "dummy::random_answer" {
            assert_eq!(mi.cached, should_be_cached);
            let output = mi.output.as_ref().unwrap();
            assert_eq!(output.len(), 1);
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let uuid = parsed.get("answer").unwrap().as_str().unwrap();
                    dummy_uuids.push(uuid.to_string());
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
    }

    // Each model stream response uses 2 output tokens
    // We have 3 candidates and 1 fuser, so 4*2=8 output tokens
    if stream {
        assert_eq!(
            usage_sum,
            Usage {
                input_tokens: Some(40),
                output_tokens: Some(8),
                cache_read_input_tokens: None,
                cache_write_input_tokens: None,
                cost: None,
            }
        );
    } else {
        // Non-streaming: dummy provider returns 1 output token per model (content.len() = 1 Vec element)
        assert_eq!(
            usage_sum,
            Usage {
                input_tokens: Some(40),
                output_tokens: Some(4),
                cache_read_input_tokens: None,
                cache_write_input_tokens: None,
                cost: None,
            }
        );
    }

    // When all of the candidates are cached, the reported HTTP usage should be 0 (since no tokens were billed),
    // even though we'll store the original cached usage in the database.
    if should_be_cached {
        assert_eq!(
            output_usage,
            Usage {
                input_tokens: Some(0),
                output_tokens: Some(0),
                cache_read_input_tokens: None,
                cache_write_input_tokens: None,
                cost: None,
            }
        );
    } else {
        assert_eq!(usage_sum, output_usage);
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> = ["dummy::random_answer", "json"]
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    assert_eq!(model_names, expected_model_names);

    // Test that the model was actually invoked twice (producing a different UUID each time)
    assert_eq!(dummy_uuids.len(), 2);
    assert_ne!(dummy_uuids[0], dummy_uuids[1]);
}

#[gtest]
#[tokio::test]
async fn test_mixture_of_n_dummy_candidates_real_judge_non_stream() {
    test_mixture_of_n_dummy_candidates_real_judge_inner(false).await;
}

#[gtest]
#[tokio::test]
async fn test_mixture_of_n_dummy_candidates_real_judge_streaming() {
    test_mixture_of_n_dummy_candidates_real_judge_inner(true).await;
}

/// This test calls a function which currently uses mixture of n.
/// We call 2 models that each give a different response, and then use GPT4o-mini to fuse them.
/// Besides checking that the response is well-formed and everything is stored correctly,
/// we also check that the input to GPT4o-mini is correct (as this is the most critical part).
async fn test_mixture_of_n_dummy_candidates_real_judge_inner(stream: bool) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n",
        "variant_name": "mixture_of_n_variant",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please write me a sentence about the anime character Megumin."},
                        {"type": "unknown", "model_name": "test", "provider_name": "good", "data": {"type": "text", "text": "My extra test-model input"}}
                    ]
                }
            ]},
        "stream": stream,
    });

    let builder = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload);

    let (content, inference_id) = if stream {
        let mut chunks = builder.eventsource().await.unwrap();
        let mut first_inference_id = None;
        while let Some(chunk) = chunks.next().await {
            println!("chunk: {chunk:?}");
            let chunk = chunk.unwrap();
            let Event::Message(chunk) = chunk else {
                continue;
            };
            if chunk.data == "[DONE]" {
                break;
            }
            let chunk_json = chunk.data;
            let chunk_json: Value = serde_json::from_str(&chunk_json).unwrap();
            let inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
            let inference_id = Uuid::parse_str(inference_id).unwrap();
            if first_inference_id.is_none() {
                first_inference_id = Some(inference_id);
            }
        }
        // TODO - expand this test to build up 'content' from the chunks
        (None, first_inference_id.unwrap())
    } else {
        let response = builder.send().await.unwrap();
        // Check Response is OK, then fields in order
        assert_eq!(response.status(), StatusCode::OK);
        let response_json = response.json::<Value>().await.unwrap();
        // Check that inference_id is here
        let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
        let inference_id = Uuid::parse_str(inference_id).unwrap();
        // Check that raw_content is same as content
        let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
        assert_eq!(content_blocks.len(), 1);
        let content_block = content_blocks.first().unwrap();
        let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
        assert_eq!(content_block_type, "text");
        let content = content_block.get("text").unwrap().as_str().unwrap();
        // Can't check content directly, as it's generated here

        // Check that usage is correct
        let usage = response_json.get("usage").unwrap();
        let usage = usage.as_object().unwrap();
        let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
        let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
        // We're invoking a real judge, so we can't assert the exact number of tokens used or produced.
        assert!(
            input_tokens > 100,
            "Unexpected input tokens: {input_tokens}"
        );
        assert!(
            output_tokens > 20,
            "Unexpected output tokens: {output_tokens}"
        );
        (Some(content.to_string()), inference_id)
    };

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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please write me a sentence about the anime character Megumin."},
                        {"type": "unknown", "model_name": "test", "provider_name": "good", "data": {"type": "text", "text": "My extra test-model input"}},
                    ]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let output = chat.output.as_ref().unwrap();
    assert_eq!(output.len(), 1);
    let db_content = match &output[0] {
        ContentBlockChatOutput::Text(text) => &text.text,
        _ => panic!("Expected a text block, got {:?}", output[0]),
    };
    if let Some(content) = content {
        assert_eq!(db_content, &content);
    }
    // Check that episode_id is here and correct
    assert_eq!(chat.episode_id, episode_id);
    // Check the variant name
    assert_eq!(chat.variant_name, "mixture_of_n_variant");

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_eq!(model_inferences.len(), 3);

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for mi in &model_inferences {
        assert_eq!(mi.inference_id, inference_id);

        // Collect model_name
        model_names.insert(mi.model_name.clone());

        // Check that all expected fields are present
        assert!(!mi.model_provider_name.is_empty());
        assert!(mi.raw_request.is_some());
        assert!(mi.raw_response.is_some());
        assert!(mi.input_tokens.is_some());
        assert!(mi.output_tokens.is_some());
        assert!(mi.response_time_ms.is_some());

        // For the judge model we want to check that the `raw_request` is correct
        if mi.model_name == "gpt-4o-mini-2024-07-18" {
            let raw_request: Value =
                serde_json::from_str(mi.raw_request.as_deref().unwrap()).unwrap();
            let mut expected_request = json!({
              "messages": [
                {
                  "role": "system",
                  "content": "You have been provided with a set of responses from various models to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nYour task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses."
                },
                {
                  "role": "user",
                  "content": "Please write me a sentence about the anime character Megumin."
                },
                {
                  "role": "user",
                  "content": "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]\n------\n1:\n[{\"type\":\"text\",\"text\":\"Megumin chanted her spell, but instead of an explosion, a gentle rain began to fall.\"}]\n------"
                }
              ],
              "model": "gpt-4o-mini-2024-07-18",
              "stream": stream,
            });
            if stream {
                expected_request["stream_options"] = serde_json::json!({
                    "include_usage": true,
                });
            }
            assert_eq!(raw_request, expected_request);
            assert_eq!(
                mi.system.as_deref().unwrap(),
                "You have been provided with a set of responses from various models to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nYour task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses."
            );
            let input_messages = mi.input_messages.as_ref().unwrap();
            assert_eq!(input_messages.len(), 2);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![StoredContentBlock::Text(Text {
                        text: "Please write me a sentence about the anime character Megumin."
                            .to_string()
                    })],
                }
            );
            assert_eq!(input_messages[1], StoredRequestMessage {
                role: Role::User,
                content: vec![
                    StoredContentBlock::Text(Text { text: "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]\n------\n1:\n[{\"type\":\"text\",\"text\":\"Megumin chanted her spell, but instead of an explosion, a gentle rain began to fall.\"}]\n------".to_string() })
                ],
            });
            let output = mi.output.as_ref().unwrap();
            assert_eq!(output.len(), 1);
            match &output[0] {
                ContentBlockOutput::Text(_) => {
                    // We don't need to check the exact content since this is a fuser model
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        } else if mi.model_name == "test" {
            let input_messages = mi.input_messages.as_ref().unwrap();
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text {
                            text: "Please write me a sentence about the anime character Megumin."
                                .to_string()
                        }),
                        StoredContentBlock::Unknown(Unknown {
                            model_name: Some("test".to_string()),
                            provider_name: Some("good".to_string()),
                            data: serde_json::json!({"type": "text", "text": "My extra test-model input"})
                        })
                    ],
                }
            );
        } else if mi.model_name == "alternate" {
            let input_messages = mi.input_messages.as_ref().unwrap();
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![StoredContentBlock::Text(Text {
                        text: "Please write me a sentence about the anime character Megumin."
                            .to_string()
                    }),],
                }
            );
        }

        assert!(mi.input_tokens.unwrap() > 0);
        assert!(mi.output_tokens.unwrap() > 0);
        assert!(mi.response_time_ms.unwrap() > 0);

        // In streaming mode, only the judge model should have a ttft_ms
        // (all of the other models should have received non-streaming requests,
        // since their responses need to be concatenated into the judge input).
        if stream && mi.model_name == "gpt-4o-mini-2024-07-18" {
            println!("ttft_ms: {:?}", mi.ttft_ms);
            let ttft_ms = mi.ttft_ms.expect("Missing ttft_ms");
            assert!(ttft_ms > 0);
        } else {
            assert!(mi.ttft_ms.is_none());
        }
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "alternate", "gpt-4o-mini-2024-07-18"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
    assert_eq!(model_names, expected_model_names);
}

/// This test calls a function which currently uses best of n.
/// We call 3 dummy models, one that gives malformed JSON, one that gives a correct JSON response,
/// and one that gives an incorrect but well-formed JSON response.
/// We check that the good response is selected and that the other responses are not
/// but they get stored to the ModelInference table.
#[gtest]
#[tokio::test]
async fn test_mixture_of_n_json_real_judge() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n_json",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "What are the first names of the Beatles? Respond in the format {\"names\": List[str]}"
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
    let output = response_json.get("output").unwrap();
    let parsed_output = output.get("parsed").unwrap();
    let names = parsed_output.get("names").unwrap().as_array().unwrap();
    expect_that!(names.len(), eq(4));
    expect_that!(names.contains(&"John".into()), eq(true));
    expect_that!(names.contains(&"Paul".into()), eq(true));
    expect_that!(names.contains(&"Ringo".into()), eq(true));
    expect_that!(names.contains(&"George".into()), eq(true));
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    expect_that!(input_tokens, gt(100));
    expect_that!(output_tokens, gt(10), "output_tokens: {output_tokens}");
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
    let json_inf = match &inferences[0] {
        StoredInferenceDatabase::Json(j) => j,
        StoredInferenceDatabase::Chat(_) => panic!("Expected json inference"),
    };
    expect_that!(json_inf.inference_id, eq(inference_id));
    let input = serde_json::to_value(&json_inf.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What are the first names of the Beatles? Respond in the format {\"names\": List[str]}"}]
                }
            ]
        }
    );
    expect_that!(input, eq(&correct_input));
    // Check that json parsed output is correct
    let output = json_inf.output.as_ref().unwrap();
    let parsed_output = output.parsed.as_ref().unwrap();
    let names = parsed_output.get("names").unwrap().as_array().unwrap();
    expect_that!(names.len(), eq(4));
    expect_that!(names.contains(&"John".into()), eq(true));
    expect_that!(names.contains(&"Paul".into()), eq(true));
    expect_that!(names.contains(&"Ringo".into()), eq(true));
    expect_that!(names.contains(&"George".into()), eq(true));
    // Check that episode_id is here and correct
    expect_that!(json_inf.episode_id, eq(episode_id));
    // Check the variant name
    expect_that!(&json_inf.variant_name, eq("mixture_of_n_variant"));

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

        // Collect model_name
        model_names.insert(mi.model_name.clone());

        // Check that all expected fields are present
        expect_that!(mi.model_provider_name.is_empty(), eq(false));
        expect_that!(&mi.raw_request, some(anything()));
        expect_that!(&mi.raw_response, some(anything()));
        expect_that!(mi.input_tokens, some(anything()));
        expect_that!(mi.output_tokens, some(anything()));
        expect_that!(mi.response_time_ms, some(anything()));
        // For the judge model we want to check that the `raw_request` is correct
        if mi.model_name == "gpt-4o-mini-2024-07-18" {
            let raw_request: Value =
                serde_json::from_str(mi.raw_request.as_deref().unwrap()).unwrap();
            let expected_request = json!({
              "messages": [
                {
                  "role": "system",
                  "content": "You have been provided with a set of responses from various models to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nYour task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses."
                },
                {
                  "role": "user",
                  "content": "What are the first names of the Beatles? Respond in the format {\"names\": List[str]}"
                },
                {
                  "role": "user",
                  "content": "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n{\"names\":[\"John\", \"George\"]}\n------\n1:\n{\"names\":[\"Paul\", \"Ringo\"]}\n------"
                }
              ],
              "model": "gpt-4o-mini-2024-07-18",
              "stream": false,
              "response_format": {
                "type": "json_schema",
                "json_schema": {
                  "name": "response",
                  "strict": true,
                  "schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                      "names": {
                        "type": "array",
                        "items": {
                          "type": "string"
                        }
                      }
                    },
                    "required": [
                      "names"
                    ],
                    "additionalProperties": false
                  }
                }
              }
            });
            expect_that!(raw_request, eq(&expected_request));
            expect_that!(
                mi.system.as_deref().unwrap(),
                eq(
                    "You have been provided with a set of responses from various models to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nYour task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses."
                )
            );
            let input_messages = mi.input_messages.as_ref().unwrap();
            assert_that!(input_messages.len(), eq(2));
            expect_that!(
                &input_messages[0],
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text { text: "What are the first names of the Beatles? Respond in the format {\"names\": List[str]}".to_string() })
                    ],
                })
            );
            expect_that!(&input_messages[1], eq(&StoredRequestMessage {
                role: Role::User,
                content: vec![
                    StoredContentBlock::Text(Text { text: "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n{\"names\":[\"John\", \"George\"]}\n------\n1:\n{\"names\":[\"Paul\", \"Ringo\"]}\n------".to_string() })
                ],
            }));
            let output = mi.output.as_ref().unwrap();
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(_) => {
                    // We don't need to check the exact content since this is a fuser model
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }

        expect_that!(mi.input_tokens.unwrap(), gt(0));
        expect_that!(mi.output_tokens.unwrap(), gt(0));
        expect_that!(mi.response_time_ms.unwrap(), gt(0));
        expect_that!(mi.ttft_ms, none());
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["json_beatles_1", "json_beatles_2", "gpt-4o-mini-2024-07-18"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
    expect_that!(model_names, eq(&expected_model_names));
}

#[gtest]
#[tokio::test]
async fn test_mixture_of_n_extra_body() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n_extra_body",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Please write me a sentence about the anime character Megumin."
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
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

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

        // Collect model_name
        model_names.insert(mi.model_name.clone());

        // Check that all expected fields are present
        expect_that!(mi.model_provider_name.is_empty(), eq(false));
        expect_that!(&mi.raw_request, some(anything()));
        expect_that!(&mi.raw_response, some(anything()));
        expect_that!(mi.input_tokens, some(anything()));
        expect_that!(mi.output_tokens, some(anything()));
        expect_that!(mi.response_time_ms, some(anything()));

        // Check that the judge model gets 'temperature' injected from 'fuser.extra_body'
        if mi.model_name == "gpt-4o-mini-2024-07-18" {
            let mut raw_request: Value =
                serde_json::from_str(mi.raw_request.as_deref().unwrap()).unwrap();

            // This message depends on the particular output of the candidate model, so just check that
            // it has the expected prefix.
            let candidate_msg = raw_request
                .get_mut("messages")
                .unwrap()
                .as_array_mut()
                .unwrap()
                .pop()
                .unwrap();
            expect_that!(
                candidate_msg
                    .get("content")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .contains("Here are the candidate answers"),
                eq(true),
                "Unexpected candidate msg: {candidate_msg:?}"
            );

            let expected_request = json!({
              "messages": [
                {
                  "role": "system",
                  "content": "You have been provided with a set of responses from various models to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nYour task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses."
                },
                {
                  "role": "user",
                  "content": "Please write me a sentence about the anime character Megumin."
                },
              ],
              "model": "gpt-4o-mini-2024-07-18",
              "stream": false,
              "temperature": 0.123
            });
            expect_that!(raw_request, eq(&expected_request));
        // Check that the other model does not get 'temperature' injected from 'fuser.extra_body'
        } else if mi.model_name == "o1-2024-12-17" {
            let raw_request: Value =
                serde_json::from_str(mi.raw_request.as_deref().unwrap()).unwrap();
            let expected_request = json!({
              "messages": [
                {
                  "role": "system",
                  "content": "You are a helpful and friendly assistant named AskJeeves"
                },
                {
                  "role": "user",
                  "content": "Please write me a sentence about the anime character Megumin."
                },
              ],
              "model": "o1-2024-12-17",
              "stream": false,
            });
            expect_that!(raw_request, eq(&expected_request));
        }
    }
    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "o1-2024-12-17", "gpt-4o-mini-2024-07-18"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
    expect_that!(model_names, eq(&expected_model_names));
}

#[gtest]
#[tokio::test]
async fn test_mixture_of_n_bad_fuser_streaming() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "mixture_of_n",
        "variant_name": "mixture_of_n_variant_bad_fuser",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": format!("Please write me a sentence about Megumin making an explosion")},
                    ]
                }
            ]},
        "stream": true,
    });

    let builder = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload);

    let mut chunks = builder.eventsource().await.unwrap();
    let mut first_inference_id = None;
    let mut chunk_data = vec![];
    while let Some(chunk) = chunks.next().await {
        println!("chunk: {chunk:?}");
        let chunk = chunk.unwrap();
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }
        let chunk_json = chunk.data;
        let chunk_json: Value = serde_json::from_str(&chunk_json).unwrap();
        let inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let inference_id = Uuid::parse_str(inference_id).unwrap();
        if first_inference_id.is_none() {
            first_inference_id = Some(inference_id);
        }
        chunk_data.push(chunk_json);
    }
    assert_that!(chunk_data.len(), eq(2));
    // First chunk contains content only (usage/finish_reason are in the final chunk)
    let expected_chunk_0 = serde_json::json!({
        "inference_id": first_inference_id.unwrap().to_string(),
        "episode_id": episode_id.to_string(),
        "variant_name":"mixture_of_n_variant_bad_fuser",
        "content":[{"type": "text", "id": "0", "text": "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."}],
    });
    expect_that!(chunk_data[0], eq(&expected_chunk_0));
    // Final chunk contains usage (accumulated from all candidates) and finish_reason
    let expected_chunk_1 = serde_json::json!({
        "inference_id": first_inference_id.unwrap().to_string(),
        "episode_id": episode_id.to_string(),
        "variant_name":"mixture_of_n_variant_bad_fuser",
        "content":[],
        // Usage is accumulated from all 2 candidates (10+10 input tokens, 1+1 output tokens)
        "usage":{"input_tokens":20,"output_tokens":2,"cost":0.00036},
        "finish_reason": "stop"
    });
    expect_that!(chunk_data[1], eq(&expected_chunk_1));

    let inference_id = first_inference_id.unwrap();

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
                    "content": [
                        {"type": "text", "text": "Please write me a sentence about Megumin making an explosion"},
                    ]
                }
            ]
        }
    );
    expect_that!(input, eq(&correct_input));

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    // Both candidates should be present (but not the fuser, since it failed)
    println!("model_inferences: {model_inferences:#?}");
    assert_that!(model_inferences.len(), eq(2));

    let expected_raw_response = "{\n  \"id\": \"id\",\n  \"object\": \"text.completion\",\n  \"created\": 1618870400,\n  \"model\": \"text-davinci-002\",\n  \"choices\": [\n    {\n      \"text\": \"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\",\n      \"index\": 0,\n      \"logprobs\": null,\n      \"finish_reason\": null\n    }\n  ],\n  \"usage\": {\n    \"prompt_tokens\": 10,\n    \"completion_tokens\": 10,\n    \"total_tokens\": 20\n  }\n}";
    let expected_input_messages = vec![StoredRequestMessage {
        role: Role::User,
        content: vec![StoredContentBlock::Text(Text {
            text: "Please write me a sentence about Megumin making an explosion".to_string(),
        })],
    }];
    let expected_output = vec![ContentBlockOutput::Text(Text {
        text: "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.".to_string(),
    })];

    // Each model inference should have its INDIVIDUAL usage, not aggregated.
    // Row order is not guaranteed, so check common fields in a loop
    // and use unordered_elements_are! for the distinguishing field (ttft_ms).
    let inference_cost = Decimal::from(18) / Decimal::from(100_000);
    for mi in &model_inferences {
        expect_that!(
            mi,
            matches_pattern!(StoredModelInference {
                snapshot_hash: some(anything()),
                inference_id: eq(&inference_id),
                raw_request: some(eq("raw request")),
                raw_response: some(eq(expected_raw_response)),
                model_name: eq("test"),
                model_provider_name: eq("good"),
                input_tokens: some(eq(&10)),
                output_tokens: some(eq(&1)),
                response_time_ms: some(eq(&100)),
                system: some(eq(
                    "You are a helpful and friendly assistant named AskJeeves"
                )),
                input_messages: some(eq(&expected_input_messages)),
                output: some(eq(&expected_output)),
                cached: eq(&false),
                finish_reason: some(eq(&FinishReason::Stop)),
                cost: some(eq(&inference_cost)),
                ..
            })
        );
    }

    // One candidate should have ttft_ms (the first streamed response) and the other should not.
    let ttft_values: Vec<_> = model_inferences.iter().map(|mi| mi.ttft_ms).collect();
    expect_that!(
        ttft_values,
        unordered_elements_are![none(), some(eq(&100u32))],
    );
}

#[gtest]
#[tokio::test]
async fn test_mixture_of_n_single_candidate_streaming() {
    let episode_id = Uuid::now_v7();
    test_mixture_of_n_single_candidate_inner(true, episode_id, json!({
        "function_name": "mixture_of_n_single_candidate",
        "variant_name": "mixture_of_n_variant",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": format!("Please write me a sentence about Megumin making an explosion")},
                    ]
                }
            ]},
        "stream": true,
    })).await;
}

async fn test_mixture_of_n_single_candidate_inner(stream: bool, episode_id: Uuid, payload: Value) {
    let builder = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload);
    let inference_id = if stream {
        let mut chunks = builder.eventsource().await.unwrap();
        let mut first_inference_id = None;
        let mut chunk_data = vec![];
        while let Some(chunk) = chunks.next().await {
            println!("chunk: {chunk:?}");
            let chunk = chunk.unwrap();
            let Event::Message(chunk) = chunk else {
                continue;
            };
            if chunk.data == "[DONE]" {
                break;
            }
            let chunk_json = chunk.data;
            let chunk_json: Value = serde_json::from_str(&chunk_json).unwrap();
            let inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
            let inference_id = Uuid::parse_str(inference_id).unwrap();
            if first_inference_id.is_none() {
                first_inference_id = Some(inference_id);
            }
            chunk_data.push(chunk_json);
        }
        assert_eq!(chunk_data.len(), 2);
        // First chunk contains content only
        assert_eq!(
            chunk_data[0],
            serde_json::json!({
                "inference_id": first_inference_id.unwrap().to_string(),
                "episode_id": episode_id.to_string(),
                "variant_name":"mixture_of_n_variant",
                "content":[{"type": "text", "id": "0", "text": "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."}],
            })
        );
        // Final chunk contains usage and finish_reason
        assert_eq!(
            chunk_data[1],
            serde_json::json!({
                "inference_id": first_inference_id.unwrap().to_string(),
                "episode_id": episode_id.to_string(),
                "variant_name":"mixture_of_n_variant",
                "content":[],
                "usage":{"input_tokens":10,"output_tokens":1,"cost":0.00018},
                "finish_reason": "stop"
            })
        );

        first_inference_id.unwrap()
    } else {
        let response = builder.send().await.unwrap();
        // Check Response is OK
        assert_eq!(response.status(), StatusCode::OK);
        let response_json = response.json::<Value>().await.unwrap();
        // Check that inference_id is here
        let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
        Uuid::parse_str(inference_id).unwrap()
    };

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
    assert_eq!(inferences.len(), 1);
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    assert_eq!(chat.inference_id, inference_id);
    let input = serde_json::to_value(&chat.input).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please write me a sentence about Megumin making an explosion"},
                    ]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    // With only a single candidate, the fuser should not be used
    println!("model_inferences: {model_inferences:#?}");
    assert_eq!(model_inferences.len(), 1);

    let mi = &model_inferences[0];
    println!("mi: {mi:#?}");
    assert!(mi.snapshot_hash.is_some());
    assert_eq!(mi.inference_id, inference_id);
    assert_eq!(mi.raw_request.as_deref().unwrap(), "raw request");
    assert_eq!(
        mi.raw_response.as_deref().unwrap(),
        "{\n  \"id\": \"id\",\n  \"object\": \"text.completion\",\n  \"created\": 1618870400,\n  \"model\": \"text-davinci-002\",\n  \"choices\": [\n    {\n      \"text\": \"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\",\n      \"index\": 0,\n      \"logprobs\": null,\n      \"finish_reason\": null\n    }\n  ],\n  \"usage\": {\n    \"prompt_tokens\": 10,\n    \"completion_tokens\": 10,\n    \"total_tokens\": 20\n  }\n}"
    );
    assert_eq!(mi.model_name, "test");
    assert_eq!(mi.model_provider_name, "good");
    assert_eq!(mi.input_tokens, Some(10));
    assert_eq!(mi.output_tokens, Some(1));
    assert_eq!(mi.response_time_ms, Some(100));
    assert_eq!(mi.ttft_ms, Some(100));
    assert_eq!(
        mi.system.as_deref().unwrap(),
        "You are a helpful and friendly assistant named AskJeeves"
    );
    assert_eq!(
        mi.input_messages.as_ref().unwrap(),
        &vec![StoredRequestMessage {
            role: Role::User,
            content: vec![StoredContentBlock::Text(Text {
                text: "Please write me a sentence about Megumin making an explosion".to_string(),
            })],
        }]
    );
    assert_eq!(
        mi.output.as_ref().unwrap(),
        &vec![ContentBlockOutput::Text(Text {
            text: "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.".to_string(),
        })]
    );
    assert!(!mi.cached);
}
