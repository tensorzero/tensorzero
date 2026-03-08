use futures::StreamExt;
use googletest::prelude::*;
use googletest_matchers::{matches_json_literal, partially};
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero_core::{
    db::{
        delegating_connection::DelegatingDatabaseConnection,
        inferences::{InferenceQueries, ListInferencesParams},
        model_inferences::ModelInferenceQueries,
        test_helpers::TestDatabaseHelpers,
    },
    inference::types::{
        ContentBlockChatOutput, ContentBlockOutput, Role, StoredContentBlock, StoredModelInference,
        StoredRequestMessage, Text, Unknown,
    },
    providers::dummy::DUMMY_INFER_RESPONSE_CONTENT,
    stored_inference::{StoredChatInferenceDatabase, StoredInferenceDatabase},
    test_helpers::get_e2e_config,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[gtest]
#[tokio::test]
async fn test_best_of_n_dummy_candidates_dummy_judge_non_stream() {
    // Include randomness in put to make sure that the first request is a cache miss
    let random_input = Uuid::now_v7();
    test_best_of_n_dummy_candidates_dummy_judge_inner(random_input, false, false).await;
    test_best_of_n_dummy_candidates_dummy_judge_inner(random_input, true, false).await;
}

#[gtest]
#[tokio::test]
async fn test_best_of_n_dummy_candidates_dummy_judge_streaming() {
    // Include randomness in put to make sure that the first request is a cache miss
    let random_input = Uuid::now_v7();
    test_best_of_n_dummy_candidates_dummy_judge_inner(random_input, false, true).await;
    test_best_of_n_dummy_candidates_dummy_judge_inner(random_input, true, true).await;
}

async fn test_best_of_n_dummy_candidates_dummy_judge_inner(
    random_input: Uuid,
    should_be_cached: bool,
    stream: bool,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n_json_repeated",
        "variant_name": "best_of_n_variant",
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

    let inference_id = if stream {
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
        first_inference_id.unwrap()
    } else {
        let response = builder.send().await.unwrap();
        // Check Response is OK, then fields in order
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
    assert_that!(inferences, len(eq(1)));

    let json_inf = match &inferences[0] {
        StoredInferenceDatabase::Json(j) => j,
        StoredInferenceDatabase::Chat(_) => panic!("Expected json inference"),
    };

    expect_that!(json_inf.inference_id, eq(inference_id));
    expect_that!(
        serde_json::to_value(&json_inf.input),
        ok(matches_json_literal!({
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
        }))
    );

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(4)));

    // Collect model names
    let mut model_names = std::collections::HashSet::new();
    let mut dummy_uuids = vec![];

    for mi in &model_inferences {
        // Collect model_name
        model_names.insert(mi.model_name.clone());

        expect_that!(
            mi,
            matches_pattern!(StoredModelInference {
                inference_id: eq(&inference_id),
                model_provider_name: not(eq("")),
                raw_request: some(anything()),
                raw_response: some(anything()),
                input_tokens: some(anything()),
                output_tokens: some(anything()),
                response_time_ms: some(anything()),
                ..
            })
        );

        // We just check the output here, since we already have several tests covering the other fields
        // for best_of_n
        if mi.model_name == "dummy::random_answer" {
            expect_that!(mi.cached, eq(should_be_cached));
            let output = mi.output.as_ref().unwrap();
            assert_that!(output, len(eq(1)));
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

    // Check that all expected model names are present
    expect_that!(
        model_names,
        unordered_elements_are![
            eq("dummy::random_answer"),
            eq("dummy::best_of_n_0"),
            eq("json"),
        ]
    );

    // Test that the model was actually invoked twice (producing a different UUID each time)
    assert_that!(dummy_uuids, len(eq(2)));
    expect_ne!(dummy_uuids[0], dummy_uuids[1]);
}

/// This test calls a function which currently uses best of n.
/// We call 2 models, one that gives the usual good response, one that
/// gives a JSON response, and then use Gemini to select the best one.
/// We check that the good response is selected and that the other responses are not
/// but they get stored to the ModelInference table.
#[gtest]
#[tokio::test]
async fn test_best_of_n_dummy_candidates_real_judge() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n",
        "variant_name": "best_of_n_variant",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please write me a sentence about Megumin making an explosion."},
                        {"type": "unknown", "model_name": "json", "provider_name": "json", "data": {"type": "text", "text": "My extra json-model input", "my": {"other": "keys"}}},
                        {"type": "unknown", "model_name": "gcp-gemini-2.5-flash", "provider_name": "gcp_vertex_gemini", "data": {"text": "My extra gemini text"}}
                    ]
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
    expect_that!(
        response_json,
        partially(matches_json_literal!({
            "content": [{"type": "text", "text": DUMMY_INFER_RESPONSE_CONTENT}]
        }))
    );
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    expect_that!(input_tokens, gt(100));
    expect_that!(output_tokens, gt(20));

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
    assert_that!(inferences, len(eq(1)));
    let chat = match &inferences[0] {
        StoredInferenceDatabase::Chat(c) => c,
        StoredInferenceDatabase::Json(_) => panic!("Expected chat inference"),
    };
    expect_that!(
        chat,
        matches_pattern!(StoredChatInferenceDatabase {
            inference_id: eq(&inference_id),
            episode_id: eq(&episode_id),
            variant_name: eq("best_of_n_variant"),
            output: some(elements_are![eq(&ContentBlockChatOutput::Text(Text {
                text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
            }))]),
            ..
        })
    );

    expect_that!(
        serde_json::to_value(&chat.input),
        ok(matches_json_literal!({
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please write me a sentence about Megumin making an explosion."},
                        {"type": "unknown", "model_name": "json", "provider_name": "json", "data": {"type": "text", "text": "My extra json-model input", "my": {"other": "keys"}}},
                        {"type": "unknown", "model_name": "gcp-gemini-2.5-flash", "provider_name": "gcp_vertex_gemini", "data": {"text": "My extra gemini text"}}
                    ]
                }
            ]
        }))
    );

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences, len(eq(3)));

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for mi in &model_inferences {
        // Collect model_name
        model_names.insert(mi.model_name.clone());

        expect_that!(
            mi,
            matches_pattern!(StoredModelInference {
                inference_id: eq(&inference_id),
                model_provider_name: not(eq("")),
                raw_request: some(anything()),
                raw_response: some(anything()),
                input_tokens: some(gt(&0)),
                output_tokens: some(gt(&0)),
                response_time_ms: some(gt(&0)),
                ttft_ms: none(),
                ..
            })
        );

        // For the judge model we want to check that the `raw_request` is correct
        if mi.model_name == "gcp-gemini-2.5-flash" {
            let raw_request: Value =
                serde_json::from_str(mi.raw_request.as_deref().unwrap()).unwrap();
            let expected_request = json!({
                "contents": [
                  {
                    "role": "user",
                    "parts": [
                      {
                        "text": "Please write me a sentence about Megumin making an explosion."
                      },
                      {
                        "text": "My extra gemini text"
                      }
                    ]
                  },
                  {
                    "role": "user",
                    "parts": [
                      {
                        "text": "Here are the candidate answers (with the index and a row of ------ separating):\n0: [{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]\n------\n1: [{\"type\":\"text\",\"text\":\"{\\\"answer\\\":\\\"Hello\\\"}\"}]\n------\nPlease evaluate these candidates and provide the index of the best one."
                      }
                    ]
                  }
                ],
                "generationConfig": {
                  "responseMimeType": "application/json",
                  "responseSchema": {
                    "type": "object",
                    "properties": {
                      "thinking": {
                        "type": "string"
                      },
                      "answer_choice": {
                        "type": "integer"
                      }
                    },
                    "required": ["thinking", "answer_choice"]
                  }
                },
                "systemInstruction": {
                  "role": "model",
                  "parts": [
                    {
                      "text": "You are an assistant tasked with re-ranking candidate answers to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nThe messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.\nPlease evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:\n{\n    \"thinking\": \"your reasoning here\",\n    \"answer_choice\": int  // Range: 0 to 1\n}\nIn the \"thinking\" block:\nFirst, you should analyze each response itself against the conversation history and determine if it is a good response or not.\nThen you should think out loud about which is best and most faithful to instructions.\nIn the \"answer_choice\" block: you should output the index of the best response."
                    }
                  ]
                },
            });
            expect_that!(raw_request, eq(&expected_request));
            expect_that!(
                mi.system.as_deref().unwrap(),
                eq(
                    "You are an assistant tasked with re-ranking candidate answers to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nThe messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.\nPlease evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:\n{\n    \"thinking\": \"your reasoning here\",\n    \"answer_choice\": int  // Range: 0 to 1\n}\nIn the \"thinking\" block:\nFirst, you should analyze each response itself against the conversation history and determine if it is a good response or not.\nThen you should think out loud about which is best and most faithful to instructions.\nIn the \"answer_choice\" block: you should output the index of the best response."
                )
            );
            let input_messages = mi.input_messages.as_ref().unwrap();
            assert_that!(input_messages.len(), eq(2));
            expect_that!(
                &input_messages[0],
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text {
                            text: "Please write me a sentence about Megumin making an explosion."
                                .to_string()
                        }),
                        StoredContentBlock::Unknown(Unknown {
                            model_name: Some("gcp-gemini-2.5-flash".into()),
                            provider_name: Some("gcp_vertex_gemini".into()),
                            data: serde_json::json!({"text": "My extra gemini text"})
                        })
                    ],
                })
            );
            expect_that!(&input_messages[1], eq(&StoredRequestMessage {
                role: Role::User,
                content: vec![
                    StoredContentBlock::Text(Text { text: "Here are the candidate answers (with the index and a row of ------ separating):\n0: [{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]\n------\n1: [{\"type\":\"text\",\"text\":\"{\\\"answer\\\":\\\"Hello\\\"}\"}]\n------\nPlease evaluate these candidates and provide the index of the best one.".to_string() })
                ],
            }));
            let output = mi.output.as_ref().unwrap();
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer_choice").unwrap().as_u64().unwrap();
                    expect_that!(answer, eq(0));
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }

        let system = mi.system.as_deref().unwrap();
        let input_messages = mi.input_messages.as_ref().unwrap();
        let output = mi.output.as_ref().unwrap();
        if mi.model_name == "json" {
            expect_that!(
                system,
                eq("You are a helpful and friendly assistant named AskJeeves")
            );
            assert_that!(input_messages.len(), eq(1));
            expect_that!(
                &input_messages[0],
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text {
                            text: "Please write me a sentence about Megumin making an explosion."
                                .to_string()
                        }),
                        StoredContentBlock::Unknown(Unknown {
                            model_name: Some("json".into()),
                            provider_name: Some("json".into()),
                            data: serde_json::json!({"type": "text", "text": "My extra json-model input", "my": {"other": "keys"}})
                        })
                    ],
                })
            );
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    expect_that!(answer, eq("Hello"));
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if mi.model_name == "test" {
            expect_that!(
                system,
                eq("You are a helpful and friendly assistant named AskJeeves")
            );
            assert_that!(input_messages.len(), eq(1));
            expect_that!(
                &input_messages[0],
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        "Please write me a sentence about Megumin making an explosion."
                            .to_string()
                            .into()
                    ],
                })
            );
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    expect_that!(
                        &text.text,
                        eq(
                            "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
                        )
                    );
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "json", "gcp-gemini-2.5-flash"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
    expect_that!(model_names, eq(&expected_model_names));
}

/// This test calls a function which currently uses best of n.
/// We call 3 dummy models, one that gives malformed JSON, one that gives a correct JSON response,
/// and one that gives an incorrect but well-formed JSON response.
/// We check that the good response is selected and that the other responses are not
/// but they get stored to the ModelInference table.
#[gtest]
#[tokio::test]
async fn test_best_of_n_json_real_judge() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n_json",
        "variant_name": "best_of_n_variant",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."
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
    expect_that!(
        response_json,
        partially(matches_json_literal!({
            "output": {"parsed": {"answer": "Hello"}}
        }))
    );
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    expect_that!(input_tokens, gt(100));
    expect_that!(output_tokens, gt(20));

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
    expect_that!(
        serde_json::to_value(&json_inf.input),
        ok(matches_json_literal!({
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."}]
                }
            ]
        }))
    );
    // Check that json parsed output is correct
    let output = json_inf.output.as_ref().unwrap();
    let parsed_output = output.parsed.as_ref().unwrap();
    let answer = parsed_output.get("answer").unwrap().as_str().unwrap();
    expect_that!(answer, eq("Hello"));
    // Check that episode_id is here and correct
    expect_that!(json_inf.episode_id, eq(episode_id));
    // Check the variant name
    expect_that!(&json_inf.variant_name, eq("best_of_n_variant"));

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences.len(), eq(4));

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for mi in &model_inferences {
        expect_that!(
            mi,
            matches_pattern!(StoredModelInference {
                inference_id: eq(&inference_id),
                model_provider_name: not(eq("")),
                raw_request: some(anything()),
                raw_response: some(anything()),
                input_tokens: some(gt(&0)),
                output_tokens: some(gt(&0)),
                response_time_ms: some(gt(&0)),
                ttft_ms: none(),
                ..
            })
        );

        // Collect model_name
        model_names.insert(mi.model_name.clone());
        let system = mi.system.as_deref().unwrap();
        let input_messages = mi.input_messages.as_ref().unwrap();
        let output = mi.output.as_ref().unwrap();
        if mi.model_name == "json" {
            expect_that!(
                system,
                eq("You are a helpful and friendly assistant named AskJeeves")
            );
            assert_that!(input_messages.len(), eq(1));
            expect_that!(
                &input_messages[0],
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text { text: "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string.".to_string() })
                    ],
                })
            );
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    expect_that!(answer, eq("Hello"));
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if mi.model_name == "test" {
            expect_that!(
                system,
                eq("You are a helpful and friendly assistant named AskJeeves")
            );
            assert_that!(input_messages.len(), eq(1));
            expect_that!(
                &input_messages[0],
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."
                            .to_string()
                            .into()
                    ],
                })
            );
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    expect_that!(
                        &text.text,
                        eq(
                            "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
                        )
                    );
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if mi.model_name == "json_goodbye" {
            expect_that!(
                system,
                eq("You are a helpful and friendly assistant named AskJeeves")
            );
            assert_that!(input_messages.len(), eq(1));
            expect_that!(
              &input_messages[0],
              eq(&StoredRequestMessage {
                  role: Role::User,
                  content: vec![
                      "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."
                          .to_string()
                          .into()
                  ],
              })
          );
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    expect_that!(answer, eq("Goodbye"));
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        // For the judge model we want to check that the `raw_request` is correct
        if mi.model_name == "gcp-gemini-2.5-flash" {
            let raw_request: Value =
                serde_json::from_str(mi.raw_request.as_deref().unwrap()).unwrap();
            let expected_request = json!({
              "contents": [
                {
                  "role": "user",
                  "parts": [
                    {
                      "text": "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."
                    }
                  ]
                },
                {
                  "role": "user",
                  "parts": [
                    {
                      "text": "Here are the candidate answers (with the index and a row of ------ separating):\n0: {\"answer\":\"Hello\"}\n------\n1: {\"answer\":\"Goodbye\"}\n------\nPlease evaluate these candidates and provide the index of the best one."
                    }
                  ]
                }
              ],
              "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "thinking": { "type": "string" },
                        "answer_choice": { "type": "integer" }
                    },
                    "required": ["thinking", "answer_choice"]
                  }
              },
              "systemInstruction": {
                "role": "model",
                "parts": [
                  {
                    "text": "You are an assistant tasked with re-ranking candidate answers to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nThe messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.\nPlease evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:\n{\n    \"thinking\": \"your reasoning here\",\n    \"answer_choice\": int  // Range: 0 to 1\n}\nIn the \"thinking\" block:\nFirst, you should analyze each response itself against the conversation history and determine if it is a good response or not.\nThen you should think out loud about which is best and most faithful to instructions.\nIn the \"answer_choice\" block: you should output the index of the best response.",
                  }
                ]
              },
            });
            expect_that!(raw_request, eq(&expected_request));
        }
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "json", "json_goodbye", "gcp-gemini-2.5-flash"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
    expect_that!(model_names, eq(&expected_model_names));
}

/// This test calls a function which currently uses best of n.
/// We call 3 dummy models, one that gives malformed JSON, one that gives a correct JSON response,
/// and one that gives an incorrect but well-formed JSON response.
/// We check that the good response is selected and that the other responses are not
/// but they get stored to the ModelInference table.
/// This test uses `json_mode="tool"` in the evaluator, so we also check that there was actually a tool call made under the hood.
#[gtest]
#[tokio::test]
async fn test_best_of_n_json_real_judge_implicit_tool() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n_json",
        "variant_name": "best_of_n_variant_implicit_tool",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."
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
    expect_that!(
        response_json,
        partially(matches_json_literal!({
            "output": {"parsed": {"answer": "Hello"}}
        }))
    );
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    expect_that!(input_tokens, gt(100));
    expect_that!(output_tokens, gt(20));

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
    expect_that!(
        serde_json::to_value(&json_inf.input),
        ok(matches_json_literal!({
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."}]
                }
            ]
        }))
    );
    // Check that json parsed output is correct
    let output = json_inf.output.as_ref().unwrap();
    let parsed_output = output.parsed.as_ref().unwrap();
    let answer = parsed_output.get("answer").unwrap().as_str().unwrap();
    expect_that!(answer, eq("Hello"));
    // Check that episode_id is here and correct
    expect_that!(json_inf.episode_id, eq(episode_id));
    // Check the variant name
    expect_that!(
        &json_inf.variant_name,
        eq("best_of_n_variant_implicit_tool")
    );

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences.len(), eq(4));

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for mi in &model_inferences {
        expect_that!(
            mi,
            matches_pattern!(StoredModelInference {
                inference_id: eq(&inference_id),
                model_provider_name: not(eq("")),
                raw_request: some(anything()),
                raw_response: some(anything()),
                input_tokens: some(gt(&0)),
                output_tokens: some(gt(&0)),
                response_time_ms: some(gt(&0)),
                ttft_ms: none(),
                ..
            })
        );

        // Collect model_name
        model_names.insert(mi.model_name.clone());

        // Check that all expected fields are present
        let system = mi.system.as_deref().unwrap();
        let input_messages = mi.input_messages.as_ref().unwrap();
        let output = mi.output.as_ref().unwrap();
        if mi.model_name == "json" {
            expect_that!(
                system,
                eq("You are a helpful and friendly assistant named AskJeeves")
            );
            assert_that!(input_messages.len(), eq(1));
            expect_that!(
                &input_messages[0],
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text { text: "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string.".to_string() })
                    ],
                })
            );
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    expect_that!(answer, eq("Hello"));
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if mi.model_name == "test" {
            expect_that!(
                system,
                eq("You are a helpful and friendly assistant named AskJeeves")
            );
            assert_that!(input_messages.len(), eq(1));
            expect_that!(
                &input_messages[0],
                eq(&StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text { text: "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string.".to_string() })
                    ],
                })
            );
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    expect_that!(
                        &text.text,
                        eq(
                            "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake."
                        )
                    );
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if mi.model_name == "json_goodbye" {
            expect_that!(
                system,
                eq("You are a helpful and friendly assistant named AskJeeves")
            );
            assert_that!(input_messages.len(), eq(1));
            expect_that!(
              &input_messages[0],
              eq(&StoredRequestMessage {
                  role: Role::User,
                  content: vec![
                      StoredContentBlock::Text(Text { text: "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string.".to_string() })
                  ],
              })
          );
            assert_that!(output.len(), eq(1));
            match &output[0] {
                ContentBlockOutput::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    expect_that!(answer, eq("Goodbye"));
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        // For the judge model we want to check that the `raw_request` is correct
        if mi.model_name == "claude-haiku-4-5-anthropic" {
            let raw_request: Value =
                serde_json::from_str(mi.raw_request.as_deref().unwrap()).unwrap();
            let expected_request = json!({
                "model": "claude-haiku-4-5",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Here are the candidate answers (with the index and a row of ------ separating):\n0: {\"answer\":\"Hello\"}\n------\n1: {\"answer\":\"Goodbye\"}\n------\nPlease evaluate these candidates and provide the index of the best one."
                            }
                        ]
                    }
                ],
                "max_tokens": 64000,
                "stream": false,
                "system": [
                    {
                        "type": "text",
                        "text": "You are an assistant tasked with re-ranking candidate answers to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nThe messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.\nPlease evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:\n{\n    \"thinking\": \"your reasoning here\",\n    \"answer_choice\": int  // Range: 0 to 1\n}\nIn the \"thinking\" block:\nFirst, you should analyze each response itself against the conversation history and determine if it is a good response or not.\nThen you should think out loud about which is best and most faithful to instructions.\nIn the \"answer_choice\" block: you should output the index of the best response."
                    }
                ],
                "tool_choice": {
                    "type": "tool",
                    "name": "respond",
                    "disable_parallel_tool_use": false,
                },
                "tools": [
                    {
                        "name": "respond",
                        "description": "Respond to the user using the output schema provided.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "thinking": {"type": "string"},
                                "answer_choice": {"type": "integer"}
                            },
                            "required": ["thinking", "answer_choice"],
                            "additionalProperties": false,
                        },
                        "strict": false
                    }
                ]
            });
            expect_that!(raw_request, eq(&expected_request));
        }
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "json", "json_goodbye", "claude-haiku-4-5-anthropic"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
    expect_that!(model_names, eq(&expected_model_names));
}

#[gtest]
#[tokio::test]
async fn test_best_of_n_judge_extra_body() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n",
        "variant_name": "best_of_n_variant_extra_body",
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
    expect_that!(
        response_json,
        partially(matches_json_literal!({
            "content": [{"type": "text", "text": DUMMY_INFER_RESPONSE_CONTENT}]
        }))
    );
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    expect_that!(input_tokens, gt(100));
    expect_that!(output_tokens, gt(20));

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check the ModelInference Table
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    assert_that!(model_inferences.len(), eq(3));

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for mi in &model_inferences {
        expect_that!(
            mi,
            matches_pattern!(StoredModelInference {
                inference_id: eq(&inference_id),
                model_provider_name: not(eq("")),
                raw_request: some(anything()),
                raw_response: some(anything()),
                input_tokens: some(anything()),
                output_tokens: some(anything()),
                response_time_ms: some(anything()),
                ttft_ms: none(),
                ..
            })
        );

        // Collect model_name
        model_names.insert(mi.model_name.clone());

        // For the judge model we want to check that the `raw_request` is correct
        if mi.model_name == "gcp-gemini-2.5-flash" {
            let raw_request: Value =
                serde_json::from_str(mi.raw_request.as_deref().unwrap()).unwrap();
            let expected_request = json!({
                "contents": [
                  {
                    "role": "user",
                    "parts": [
                      {
                        "text": "Please write me a sentence about Megumin making an explosion."
                      }
                    ]
                  },
                  {
                    "role": "user",
                    "parts": [
                      {
                        "text": "Here are the candidate answers (with the index and a row of ------ separating):\n0: [{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]\n------\n1: [{\"type\":\"text\",\"text\":\"{\\\"answer\\\":\\\"Hello\\\"}\"}]\n------\nPlease evaluate these candidates and provide the index of the best one."
                      }
                    ]
                  }
                ],
                "generationConfig": {
                    "temperature": 0.123,
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "object",
                        "properties": {
                          "thinking": {
                            "type": "string"
                          },
                          "answer_choice": {
                            "type": "integer"
                          }
                        },
                        "required": ["thinking", "answer_choice"]
                      }
                },
                "systemInstruction": {
                  "role": "model",
                  "parts": [
                    {
                      "text": "You are an assistant tasked with re-ranking candidate answers to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nThe messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.\nPlease evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:\n{\n    \"thinking\": \"your reasoning here\",\n    \"answer_choice\": int  // Range: 0 to 1\n}\nIn the \"thinking\" block:\nFirst, you should analyze each response itself against the conversation history and determine if it is a good response or not.\nThen you should think out loud about which is best and most faithful to instructions.\nIn the \"answer_choice\" block: you should output the index of the best response."
                    }
                  ]
                },
            });
            expect_that!(raw_request, eq(&expected_request));
        }
    }
}
