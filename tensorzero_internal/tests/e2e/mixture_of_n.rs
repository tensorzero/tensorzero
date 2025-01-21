use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero_internal::inference::types::{ContentBlock, RequestMessage, Role};
use uuid::Uuid;

use crate::common::{
    get_clickhouse, get_gateway_endpoint, select_chat_inference_clickhouse,
    select_json_inference_clickhouse, select_model_inferences_clickhouse,
};

/// This test calls a function which currently uses mixture of n.
/// We call 2 models that each give a different response, and then use GPT4o-mini to fuse them.
/// Besides checking that the response is well-formed and everything is stored correctly,
/// we also check that the input to GPT4o-mini is correct (as this is the most critical part).
#[tokio::test]
async fn e2e_test_mixture_of_n_dummy_candidates_real_judge() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "mixture_of_n",
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
                    "content": [{"type": "text", "value": "Please write me a sentence about the anime character Megumin."}]
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
    let db_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(db_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "mixture_of_n_variant");

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

        // For the judge model we want to check that the raw_request is corredt
        if model_name == "gpt-4o-mini-2024-07-18" {
            let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
            let raw_request: Value = serde_json::from_str(raw_request).unwrap();
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
                {
                  "role": "user",
                  "content": "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]\n------1:\n[{\"type\":\"text\",\"text\":\"Megumin chanted her spell, but instead of an explosion, a gentle rain began to fall.\"}]\n------"
                }
              ],
              "model": "gpt-4o-mini-2024-07-18",
              "stream": false,
              "response_format": {
                "type": "text"
              }
            });
            assert_eq!(raw_request, expected_request);
            let system = result.get("system").unwrap().as_str().unwrap();
            assert_eq!(system, "You have been provided with a set of responses from various models to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nYour task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses.");
            let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
            let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
            assert_eq!(input_messages.len(), 2);
            assert_eq!(
                input_messages[0],
                RequestMessage {
                    role: Role::User,
                    content: vec![
                        "Please write me a sentence about the anime character Megumin."
                            .to_string()
                            .into()
                    ],
                }
            );
            assert_eq!(input_messages[1], RequestMessage {
                role: Role::User,
                content: vec![
                    "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]\n------1:\n[{\"type\":\"text\",\"text\":\"Megumin chanted her spell, but instead of an explosion, a gentle rain began to fall.\"}]\n------"
                        .to_string()
                        .into()
                ],
            });
            let output = result.get("output").unwrap().as_str().unwrap();
            let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
            assert_eq!(output.len(), 1);
            match &output[0] {
                ContentBlock::Text(_) => {
                    // We don't need to check the exact content since this is a fuser model
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
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "alternate", "gpt-4o-mini-2024-07-18"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    assert_eq!(model_names, expected_model_names);
}

/// This test calls a function which currently uses best of n.
/// We call 3 dummy models, one that gives malformed JSON, one that gives a correct JSON response,
/// and one that gives an incorrect but well-formed JSON response.
/// We check that the good response is selected and that the other responses are not
/// but they get stored to the ModelInference table.
#[tokio::test]
async fn e2e_test_mixture_of_n_json_real_judge() {
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
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let output = response_json.get("output").unwrap();
    let parsed_output = output.get("parsed").unwrap();
    let names = parsed_output.get("names").unwrap().as_array().unwrap();
    assert_eq!(names.len(), 4);
    assert!(names.contains(&"John".into()));
    assert!(names.contains(&"Paul".into()));
    assert!(names.contains(&"Ringo".into()));
    assert!(names.contains(&"George".into()));
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
                    "content": [{"type": "text", "value": "What are the first names of the Beatles? Respond in the format {\"names\": List[str]}"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that json parsed output is correct
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let parsed_output = output.get("parsed").unwrap();
    let names = parsed_output.get("names").unwrap().as_array().unwrap();
    assert_eq!(names.len(), 4);
    assert!(names.contains(&"John".into()));
    assert!(names.contains(&"Paul".into()));
    assert!(names.contains(&"Ringo".into()));
    assert!(names.contains(&"George".into()));
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "mixture_of_n_variant");

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
        // For the judge model we want to check that the raw_request is corredt
        if model_name == "gpt-4o-mini-2024-07-18" {
            let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
            let raw_request: Value = serde_json::from_str(raw_request).unwrap();
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
                  "content": "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n{\"names\":[\"John\", \"George\"]}\n------1:\n{\"names\":[\"Paul\", \"Ringo\"]}\n------"
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
            assert_eq!(raw_request, expected_request);
            let system = result.get("system").unwrap().as_str().unwrap();
            assert_eq!(system, "You have been provided with a set of responses from various models to the following problem:\n------\nYou are a helpful and friendly assistant named AskJeeves\n------\nYour task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses.");
            let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
            let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
            assert_eq!(input_messages.len(), 2);
            assert_eq!(
                input_messages[0],
                RequestMessage {
                    role: Role::User,
                    content: vec![
                        "What are the first names of the Beatles? Respond in the format {\"names\": List[str]}"
                            .to_string()
                            .into()
                    ],
                }
            );
            assert_eq!(input_messages[1], RequestMessage {
                role: Role::User,
                content: vec![
                    "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n{\"names\":[\"John\", \"George\"]}\n------1:\n{\"names\":[\"Paul\", \"Ringo\"]}\n------"
                        .to_string()
                        .into()
                ],
            });
            let output = result.get("output").unwrap().as_str().unwrap();
            let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
            assert_eq!(output.len(), 1);
            match &output[0] {
                ContentBlock::Text(_) => {
                    // We don't need to check the exact content since this is a fuser model
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
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["json_beatles_1", "json_beatles_2", "gpt-4o-mini-2024-07-18"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    assert_eq!(model_names, expected_model_names);
}
