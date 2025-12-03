use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use tensorzero_core::{
    inference::types::{Role, StoredContentBlock, StoredRequestMessage, Text, Unknown},
    providers::dummy::DUMMY_INFER_RESPONSE_CONTENT,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_json_inference_clickhouse,
    select_model_inferences_clickhouse,
};

#[tokio::test]
async fn e2e_test_best_of_n_dummy_candidates_dummy_judge_non_stream() {
    // Include randomness in put to make sure that the first request is a cache miss
    let random_input = Uuid::now_v7();
    e2e_test_best_of_n_dummy_candidates_dummy_judge_inner(random_input, false, false).await;
    e2e_test_best_of_n_dummy_candidates_dummy_judge_inner(random_input, true, false).await;
}

#[tokio::test]
async fn e2e_test_best_of_n_dummy_candidates_dummy_judge_streaming() {
    // Include randomness in put to make sure that the first request is a cache miss
    let random_input = Uuid::now_v7();
    e2e_test_best_of_n_dummy_candidates_dummy_judge_inner(random_input, false, true).await;
    e2e_test_best_of_n_dummy_candidates_dummy_judge_inner(random_input, true, true).await;
}

async fn e2e_test_best_of_n_dummy_candidates_dummy_judge_inner(
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
        let mut chunks = builder.eventsource().unwrap();
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

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
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
    let results: Vec<Value> = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 4);

    // Collect model names
    let mut model_names = std::collections::HashSet::new();
    let mut dummy_uuids = vec![];

    for result in results {
        let id = result.get("id").unwrap().as_str().unwrap();
        let _ = Uuid::parse_str(id).unwrap();
        let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        // Collect model_name
        let model_name = result.get("model_name").unwrap().as_str().unwrap();
        model_names.insert(model_name.to_string());

        // Check that all expected fields are present
        assert!(result.get("model_provider_name").is_some());
        assert!(result.get("raw_request").is_some());
        assert!(result.get("raw_response").is_some());
        assert!(result.get("input_tokens").is_some());
        assert!(result.get("output_tokens").is_some());
        assert!(result.get("response_time_ms").is_some());
        assert!(result.get("ttft_ms").is_some());

        // We just check the output here, since we already have several tests covering the other fields
        // for best_of_n
        if model_name == "dummy::random_answer" {
            let cached = result.get("cached").unwrap().as_bool().unwrap();
            assert_eq!(cached, should_be_cached);
            let output = result.get("output").unwrap().as_str().unwrap();
            let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
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
    let expected_model_names: std::collections::HashSet<String> =
        ["dummy::random_answer", "dummy::best_of_n_0", "json"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
    assert_eq!(model_names, expected_model_names);

    // Test that the model was actually invoked twice (producing a different UUID each time)
    assert_eq!(dummy_uuids.len(), 2);
    assert_ne!(dummy_uuids[0], dummy_uuids[1]);
}

/// This test calls a function which currently uses best of n.
/// We call 2 models, one that gives the usual good response, one that
/// gives a JSON response, and then use Gemini to select the best one.
/// We check that the good response is selected and that the other responses are not
/// but they get stored to the ModelInference table.
#[tokio::test]
async fn e2e_test_best_of_n_dummy_candidates_real_judge() {
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
                        {"type": "unknown", "model_name": "gemini-2.0-flash-001", "provider_name": "gcp_vertex_gemini", "data": {"text": "My extra gemini text"}}
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
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 100);
    assert!(output_tokens > 20);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please write me a sentence about Megumin making an explosion."},
                        {"type": "unknown", "model_name": "json", "provider_name": "json", "data": {"type": "text", "text": "My extra json-model input", "my": {"other": "keys"}}},
                        {"type": "unknown", "model_name": "gemini-2.0-flash-001", "provider_name": "gcp_vertex_gemini", "data": {"text": "My extra gemini text"}}
                    ]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "best_of_n_variant");

    // Check the ModelInference Table
    let results: Vec<Value> = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 3);

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for result in results {
        let id = result.get("id").unwrap().as_str().unwrap();
        let _ = Uuid::parse_str(id).unwrap();
        let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        // Collect model_name
        let model_name = result.get("model_name").unwrap().as_str().unwrap();
        model_names.insert(model_name.to_string());

        // Check that all expected fields are present
        assert!(result.get("model_provider_name").is_some());
        assert!(result.get("raw_request").is_some());
        assert!(result.get("raw_response").is_some());
        assert!(result.get("input_tokens").is_some());
        assert!(result.get("output_tokens").is_some());
        assert!(result.get("response_time_ms").is_some());
        assert!(result.get("ttft_ms").is_some());

        // For the judge model we want to check that the `raw_request` is correct
        if model_name == "gemini-2.0-flash-001" {
            let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
            let raw_request: Value = serde_json::from_str(raw_request).unwrap();
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
            assert_eq!(raw_request, expected_request);
            let system = result.get("system").unwrap().as_str().unwrap();
            assert_eq!(system, "You are an assistant tasked with re-ranking candidate answers to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nThe messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.\nPlease evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:\n{\n    \"thinking\": \"your reasoning here\",\n    \"answer_choice\": int  // Range: 0 to 1\n}\nIn the \"thinking\" block:\nFirst, you should analyze each response itself against the conversation history and determine if it is a good response or not.\nThen you should think out loud about which is best and most faithful to instructions.\nIn the \"answer_choice\" block: you should output the index of the best response.");
            let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
            let input_messages: Vec<StoredRequestMessage> =
                serde_json::from_str(input_messages).unwrap();
            assert_eq!(input_messages.len(), 2);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text {
                            text: "Please write me a sentence about Megumin making an explosion."
                                .to_string()
                        }),
                        StoredContentBlock::Unknown(Unknown {
                            model_name: Some("gemini-2.0-flash-001".into()),
                            provider_name: Some("gcp_vertex_gemini".into()),
                            data: serde_json::json!({"text": "My extra gemini text"})
                        })
                    ],
                }
            );
            assert_eq!(input_messages[1], StoredRequestMessage {
                role: Role::User,
                content: vec![
                    StoredContentBlock::Text(Text { text: "Here are the candidate answers (with the index and a row of ------ separating):\n0: [{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]\n------\n1: [{\"type\":\"text\",\"text\":\"{\\\"answer\\\":\\\"Hello\\\"}\"}]\n------\nPlease evaluate these candidates and provide the index of the best one.".to_string() })
                ],
            });
            let output = result.get("output").unwrap().as_str().unwrap();
            let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer_choice").unwrap().as_u64().unwrap();
                    assert_eq!(answer, 0);
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }

        let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
        assert!(input_tokens > 0);
        let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
        assert!(output_tokens > 0);
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
        let system = result.get("system").unwrap().as_str().unwrap();
        let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
        let input_messages: Vec<StoredRequestMessage> =
            serde_json::from_str(input_messages).unwrap();
        let output = result.get("output").unwrap().as_str().unwrap();
        let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
        if model_name == "json" {
            assert_eq!(
                system,
                "You are a helpful and friendly assistant named AskJeeves"
            );
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
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
                }
            );
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    assert_eq!(answer, "Hello");
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if model_name == "test" {
            assert_eq!(
                system,
                "You are a helpful and friendly assistant named AskJeeves"
            );
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        "Please write me a sentence about Megumin making an explosion."
                            .to_string()
                            .into()
                    ],
                }
            );
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    assert_eq!(text.text, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.");
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "json", "gemini-2.0-flash-001"]
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
#[tokio::test]
async fn e2e_test_best_of_n_json_real_judge() {
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
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let output = response_json.get("output").unwrap();
    let parsed_output = output.get("parsed").unwrap();
    let answer = parsed_output.get("answer").unwrap().as_str().unwrap();
    assert_eq!(answer, "Hello");
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 100);
    assert!(output_tokens > 20);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that json parsed output is correct
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let parsed_output = output.get("parsed").unwrap();
    let answer = parsed_output.get("answer").unwrap().as_str().unwrap();
    assert_eq!(answer, "Hello");
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "best_of_n_variant");

    // Check the ModelInference Table
    let results = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 4);

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for result in results {
        let id = result.get("id").unwrap().as_str().unwrap();
        let _ = Uuid::parse_str(id).unwrap();
        let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        // Collect model_name
        let model_name = result.get("model_name").unwrap().as_str().unwrap();
        model_names.insert(model_name.to_string());

        // Check that all expected fields are present
        assert!(result.get("model_provider_name").is_some());
        assert!(result.get("raw_request").is_some());
        assert!(result.get("raw_response").is_some());
        assert!(result.get("input_tokens").is_some());
        assert!(result.get("output_tokens").is_some());
        assert!(result.get("response_time_ms").is_some());
        assert!(result.get("ttft_ms").is_some());

        let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
        assert!(input_tokens > 0);
        let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
        assert!(output_tokens > 0);
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
        let system = result.get("system").unwrap().as_str().unwrap();
        let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
        let input_messages: Vec<StoredRequestMessage> =
            serde_json::from_str(input_messages).unwrap();
        let output = result.get("output").unwrap().as_str().unwrap();
        let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
        if model_name == "json" {
            assert_eq!(
                system,
                "You are a helpful and friendly assistant named AskJeeves"
            );
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text { text: "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string.".to_string() })
                    ],
                }
            );
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    assert_eq!(answer, "Hello");
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if model_name == "test" {
            assert_eq!(
                system,
                "You are a helpful and friendly assistant named AskJeeves"
            );
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."
                            .to_string()
                            .into()
                    ],
                }
            );
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    assert_eq!(text.text, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.");
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if model_name == "json_goodbye" {
            assert_eq!(
                system,
                "You are a helpful and friendly assistant named AskJeeves"
            );
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
              input_messages[0],
              StoredRequestMessage {
                  role: Role::User,
                  content: vec![
                      "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."
                          .to_string()
                          .into()
                  ],
              }
          );
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    assert_eq!(answer, "Goodbye");
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        // For the judge model we want to check that the `raw_request` is correct
        if model_name == "gemini-2.0-flash-001" {
            let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
            let raw_request: Value = serde_json::from_str(raw_request).unwrap();
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
            assert_eq!(raw_request, expected_request);
        }
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "json", "json_goodbye", "gemini-2.0-flash-001"]
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
/// This test uses `json_mode="tool"` in the evaluator, so we also check that there was actually a tool call made under the hood.
#[tokio::test]
async fn e2e_test_best_of_n_json_real_judge_implicit_tool() {
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
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let output = response_json.get("output").unwrap();
    let parsed_output = output.get("parsed").unwrap();
    let answer = parsed_output.get("answer").unwrap().as_str().unwrap();
    assert_eq!(answer, "Hello");
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 100);
    assert!(output_tokens > 20);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that json parsed output is correct
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let parsed_output = output.get("parsed").unwrap();
    let answer = parsed_output.get("answer").unwrap().as_str().unwrap();
    assert_eq!(answer, "Hello");
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "best_of_n_variant_implicit_tool");

    // Check the ModelInference Table
    let results = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 4);

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for result in results {
        let id = result.get("id").unwrap().as_str().unwrap();
        let _ = Uuid::parse_str(id).unwrap();
        let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        // Collect model_name
        let model_name = result.get("model_name").unwrap().as_str().unwrap();
        model_names.insert(model_name.to_string());

        // Check that all expected fields are present
        assert!(result.get("model_provider_name").is_some());
        assert!(result.get("raw_request").is_some());
        assert!(result.get("raw_response").is_some());
        assert!(result.get("input_tokens").is_some());
        assert!(result.get("output_tokens").is_some());
        assert!(result.get("response_time_ms").is_some());
        assert!(result.get("ttft_ms").is_some());

        let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
        assert!(input_tokens > 0);
        let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
        assert!(output_tokens > 0);
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
        let system = result.get("system").unwrap().as_str().unwrap();
        let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
        let input_messages: Vec<StoredRequestMessage> =
            serde_json::from_str(input_messages).unwrap();
        let output = result.get("output").unwrap().as_str().unwrap();
        let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
        if model_name == "json" {
            assert_eq!(
                system,
                "You are a helpful and friendly assistant named AskJeeves"
            );
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text { text: "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string.".to_string() })
                    ],
                }
            );
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    assert_eq!(answer, "Hello");
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if model_name == "test" {
            assert_eq!(
                system,
                "You are a helpful and friendly assistant named AskJeeves"
            );
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
                input_messages[0],
                StoredRequestMessage {
                    role: Role::User,
                    content: vec![
                        StoredContentBlock::Text(Text { text: "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string.".to_string() })
                    ],
                }
            );
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    assert_eq!(text.text, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.");
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        if model_name == "json_goodbye" {
            assert_eq!(
                system,
                "You are a helpful and friendly assistant named AskJeeves"
            );
            assert_eq!(input_messages.len(), 1);
            assert_eq!(
              input_messages[0],
              StoredRequestMessage {
                  role: Role::User,
                  content: vec![
                      StoredContentBlock::Text(Text { text: "What's the first word in the typical output of one's first program. Answer as a json object with a single field 'answer' containing the string.".to_string() })
                  ],
              }
          );
            assert_eq!(output.len(), 1);
            match &output[0] {
                StoredContentBlock::Text(text) => {
                    let parsed: Value = serde_json::from_str(&text.text).unwrap();
                    let answer = parsed.get("answer").unwrap().as_str().unwrap();
                    assert_eq!(answer, "Goodbye");
                }
                _ => {
                    panic!("Expected a text block, got {:?}", output[0]);
                }
            }
        }
        // For the judge model we want to check that the `raw_request` is correct
        if model_name == "claude-3-haiku-20240307-anthropic" {
            let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
            let raw_request: Value = serde_json::from_str(raw_request).unwrap();
            let expected_request = json!({
                "model": "claude-3-haiku-20240307",
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
                "max_tokens": 4096,
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
                            "additionalProperties": false
                        }
                    }
                ]
            });
            assert_eq!(raw_request, expected_request);
        }
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> = [
        "test",
        "json",
        "json_goodbye",
        "claude-3-haiku-20240307-anthropic",
    ]
    .iter()
    .map(std::string::ToString::to_string)
    .collect();
    assert_eq!(model_names, expected_model_names);
}

#[tokio::test]
async fn e2e_test_best_of_n_judge_extra_body() {
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
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 100);
    assert!(output_tokens > 20);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let clickhouse = get_clickhouse().await;
    // Check the ModelInference Table
    let results = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 3);

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for result in results {
        let id = result.get("id").unwrap().as_str().unwrap();
        let _ = Uuid::parse_str(id).unwrap();
        let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        // Collect model_name
        let model_name = result.get("model_name").unwrap().as_str().unwrap();
        model_names.insert(model_name.to_string());

        // Check that all expected fields are present
        assert!(result.get("model_provider_name").is_some());
        assert!(result.get("raw_request").is_some());
        assert!(result.get("raw_response").is_some());
        assert!(result.get("input_tokens").is_some());
        assert!(result.get("output_tokens").is_some());
        assert!(result.get("response_time_ms").is_some());
        assert!(result.get("ttft_ms").is_some());

        // For the judge model we want to check that the `raw_request` is correct
        if model_name == "gemini-2.0-flash-001" {
            let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
            let raw_request: Value = serde_json::from_str(raw_request).unwrap();
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
            assert_eq!(raw_request, expected_request);
        }
    }
}
