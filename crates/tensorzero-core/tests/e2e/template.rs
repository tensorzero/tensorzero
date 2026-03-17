use googletest::prelude::*;
use http::StatusCode;
use reqwest::Client;
use serde_json::{Value, json};
use tensorzero::{
    InferenceOutput, InferenceResponse, test_helpers::make_embedded_gateway_with_config_path,
};
use tensorzero_core::{
    db::delegating_connection::DelegatingDatabaseConnection,
    db::inferences::{InferenceQueries, ListInferencesParams},
    db::model_inferences::ModelInferenceQueries,
    db::test_helpers::TestDatabaseHelpers,
    inference::types::{Arguments, ContentBlockChatOutput, System, Text},
    stored_inference::StoredInferenceDatabase,
    test_helpers::get_e2e_config,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[gtest]
#[tokio::test]
async fn test_template_no_schema() {
    let payload = json!({
        "function_name": "basic_test_template_no_schema",
        "variant_name": "test",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First user message"},
                        {"type": "text", "text": "Second user message"},
                        {
                          "type": "template",
                          "name": "my_custom_template",
                          "arguments": {
                            "first_variable": "my_content",
                            "second_variable": "my_other_content"
                          }
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First assistant message"},
                        {"type": "text", "text": "Second assistant message"},
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

    expect_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let content = &response_json["content"][0]["text"].as_str().unwrap();
    let echoed_content = serde_json::from_str::<Value>(content).unwrap();
    let expected_content = json!({
        "system": "The system text was `My system message`",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "User content: `First user message`"
              },
              {
                "type": "text",
                "text": "User content: `Second user message`"
              },
              {
                "type": "text",
                "text": "New template: first_variable=my_content second_variable=my_other_content"
              }
            ]
          },
          {
            "role": "assistant",
            "content": [
              {
                "type": "text",
                "text": "Assistant content: `First assistant message`"
              },
              {
                "type": "text",
                "text": "Assistant content: `Second assistant message`"
              }
            ]
          }
        ]
    });
    expect_that!(echoed_content, eq(&expected_content));

    let inference_id = response_json["inference_id"].as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

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
    let input = serde_json::to_value(&chat.input).unwrap();
    println!("Input: {input}");
    let expected_input = serde_json::json!({
        "system":"My system message",
        "messages":[
          {"role":"user","content":[
            {"type":"text","text":"First user message"},
            {"type":"text","text":"Second user message"},
            {"type":"template","name":"my_custom_template","arguments":{"first_variable":"my_content","second_variable":"my_other_content"}}
          ]},
          {"role":"assistant","content":[{"type":"text","text":"First assistant message"},{"type":"text","text":"Second assistant message"}]}]
    });
    expect_that!(input, eq(&expected_input));
}

#[gtest]
#[tokio::test]
async fn test_mixture_of_n_template_no_schema() {
    let payload = json!({
        "function_name": "basic_test_template_no_schema",
        "variant_name": "mixture_of_n",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First user message"},
                        {"type": "text", "text": "Second user message"},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First assistant message"},
                        {"type": "text", "text": "Second assistant message"},
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

    expect_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let content = &response_json["content"][0]["text"].as_str().unwrap();
    let echoed_content = serde_json::from_str::<Value>(content).unwrap();
    println!("echoed_content: {echoed_content}");
    let expected_content = json!({
      "system": "You have been provided with a set of responses from various models to the following problem:\n------\nouter template system text: `My system message`\n------\nYour task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction and take the best from all the responses. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.  Below will be: first, any messages leading up to this point, and then, a final message containing the set of candidate responses.",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "outer template user text: `First user message`"
            },
            {
              "type": "text",
              "text": "outer template user text: `Second user message`"
            }
          ]
        },
        {
          "role": "assistant",
          "content": [
            {
              "type": "text",
              "text": "outer template assistant text: `First assistant message`"
            },
            {
              "type": "text",
              "text": "outer template assistant text: `Second assistant message`"
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n[{\"type\":\"text\",\"text\":\"{\\\"system\\\":\\\"The system text was `My system message`\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"User content: `First user message`\\\"},{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"User content: `Second user message`\\\"}]},{\\\"role\\\":\\\"assistant\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Assistant content: `First assistant message`\\\"},{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Assistant content: `Second assistant message`\\\"}]}]}\"}]\n------\n1:\n[{\"type\":\"text\",\"text\":\"{\\\"system\\\":\\\"The system text was `My system message`\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"User content: `First user message`\\\"},{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"User content: `Second user message`\\\"}]},{\\\"role\\\":\\\"assistant\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Assistant content: `First assistant message`\\\"},{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Assistant content: `Second assistant message`\\\"}]}]}\"}]\n------"
            }
          ]
        }
      ]
    });
    expect_that!(echoed_content, eq(&expected_content));
    // We don't check ClickHouse, as we already do that in lots of other tests
}

#[gtest]
#[tokio::test]
async fn test_best_of_n_template_no_schema() {
    let payload = json!({
        "function_name": "basic_test_template_no_schema",
        "variant_name": "best_of_n",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First user message"},
                        {"type": "text", "text": "Second user message"},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First assistant message"},
                        {"type": "text", "text": "Second assistant message"},
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

    expect_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json["inference_id"].as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let content = &response_json["content"][0]["text"].as_str().unwrap();
    let echoed_content = serde_json::from_str::<Value>(content).unwrap();
    let expected_content = json!({
        "system": "The system text was `My system message`",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "User content: `First user message`"
              },
              {
                "type": "text",
                "text": "User content: `Second user message`"
              }
            ]
          },
          {
            "role": "assistant",
            "content": [
              {
                "type": "text",
                "text": "Assistant content: `First assistant message`"
              },
              {
                "type": "text",
                "text": "Assistant content: `Second assistant message`"
              }
            ]
          }
        ]
    });
    expect_that!(echoed_content, eq(&expected_content));

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .unwrap();
    expect_that!(model_inferences.len(), eq(3));
    expect_that!(
        model_inferences
            .iter()
            .filter(|mi| mi.model_name == "dummy::best_of_n_0")
            .count(),
        eq(1)
    );
    // Just check the input to 'dummy::best_of_n_0' - we already have lots of other 'best_of_n' tests
    for mi in &model_inferences {
        if mi.model_name == "dummy::best_of_n_0" {
            expect_that!(
                mi.system.as_deref().unwrap(),
                eq(
                    "You are an assistant tasked with re-ranking candidate answers to the following problem:\n------\nouter template system text: `My system message`\n------\nThe messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.\nPlease evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:\n{\n    \"thinking\": \"your reasoning here\",\n    \"answer_choice\": int  // Range: 0 to 1\n}\nIn the \"thinking\" block:\nFirst, you should analyze each response itself against the conversation history and determine if it is a good response or not.\nThen you should think out loud about which is best and most faithful to instructions.\nIn the \"answer_choice\" block: you should output the index of the best response."
                )
            );
            let input_messages = serde_json::to_value(&mi.input_messages).unwrap();
            let expected_input_messages = serde_json::json!([
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": "outer template user text: `First user message`"
                  },
                  {
                    "type": "text",
                    "text": "outer template user text: `Second user message`"
                  }
                ]
              },
              {
                "role": "assistant",
                "content": [
                  {
                    "type": "text",
                    "text": "outer template assistant text: `First assistant message`"
                  },
                  {
                    "type": "text",
                    "text": "outer template assistant text: `Second assistant message`"
                  }
                ]
              },
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": "Here are the candidate answers (with the index and a row of ------ separating):\n0: [{\"type\":\"text\",\"text\":\"{\\\"system\\\":\\\"The system text was `My system message`\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"User content: `First user message`\\\"},{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"User content: `Second user message`\\\"}]},{\\\"role\\\":\\\"assistant\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Assistant content: `First assistant message`\\\"},{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Assistant content: `Second assistant message`\\\"}]}]}\"}]\n------\n1: [{\"type\":\"text\",\"text\":\"{\\\"system\\\":\\\"The system text was `My system message`\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"User content: `First user message`\\\"},{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"User content: `Second user message`\\\"}]},{\\\"role\\\":\\\"assistant\\\",\\\"content\\\":[{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Assistant content: `First assistant message`\\\"},{\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"Assistant content: `Second assistant message`\\\"}]}]}\"}]\n------\nPlease evaluate these candidates and provide the index of the best one."
                  }
                ]
              }
            ]);
            expect_that!(input_messages, eq(&expected_input_messages));
        }
    }
}

#[gtest]
#[tokio::test]
async fn test_invalid_system_input_template_no_schema() {
    let payload = json!({
        "function_name": "basic_test_template_no_schema",
        "input":{
            "system": { "my_invalid": "system message"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First user message"},
                        {"type": "text", "text": "Second user message"},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First assistant message"},
                        {"type": "text", "text": "Second assistant message"},
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

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    println!("Response JSON: {response_json}");
    expect_that!(status, eq(StatusCode::BAD_REQUEST));
    let error = response_json["error"].as_str().unwrap();
    expect_that!(
        error,
        eq(
            "System message has non-string content but there is no template `system` in any variant"
        )
    );
}

#[gtest]
#[tokio::test]
async fn test_invalid_json_user_input_template_no_schema() {
    let payload = json!({
        "function_name": "null_json",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First user message"},
                        {"type": "template", "name": "user", "arguments": {"my_invalid": "user message"}},
                    ]
                },

            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json["error"].as_str().unwrap();
    expect_that!(
        error,
        eq(
            "Message at index 0 has non-string content but there is no template `user` in any variant"
        )
    );
}

#[gtest]
#[tokio::test]
async fn test_invalid_user_input_template_no_schema() {
    let payload = json!({
        "function_name": "basic_test_template_no_schema",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First user message"},
                        {"type": "template", "name": "user", "arguments": {"my_invalid": "user message"}},
                    ]
                },

            ]},
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json["error"].as_str().unwrap();
    expect_that!(
        error,
        eq(
            "Message at index 0 has non-string content but there is no template `user` in any variant"
        )
    );
}

#[gtest]
#[tokio::test]
async fn test_invalid_assistant_input_template_no_schema() {
    let payload = json!({
        "function_name": "basic_test_template_no_schema",
        "variant_name": "test",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "template", "name": "assistant", "arguments": {"my_invalid": "assistant message"}},
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

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json["error"].as_str().unwrap();
    expect_that!(
        error,
        eq(
            "Message at index 0 has non-string content but there is no template `assistant` in any variant"
        )
    );
    expect_that!(status, eq(StatusCode::BAD_REQUEST));
}

#[gtest]
#[tokio::test]
async fn test_invalid_json_assistant_input_template_no_schema() {
    let payload = json!({
        "function_name": "null_json",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "template", "name": "assistant", "arguments": {"my_invalid": "assistant message"}},
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

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json["error"].as_str().unwrap();
    expect_that!(
        error,
        eq(
            "Message at index 0 has non-string content but there is no template `assistant` in any variant"
        )
    );
    expect_that!(status, eq(StatusCode::BAD_REQUEST));
}

#[gtest]
#[tokio::test]
async fn test_named_system_template_no_schema() {
    let config_dir = tempfile::tempdir().unwrap();
    let config_path = config_dir.path().join("tensorzero.toml");
    let config = r#"
  [functions.test_system_template]
  type = "chat"

  [functions.test_system_template.variants.test]
  type = "chat_completion"
  model = "dummy::echo_request_messages"
  templates.system.path = "./system_template.minijinja"
  "#;
    std::fs::write(&config_path, config).unwrap();
    std::fs::write(
        config_dir.path().join("system_template.minijinja"),
        "You are a helpful and friendly assistant named {{ assistant_name }}",
    )
    .unwrap();

    let client = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path.to_owned()),
        clickhouse_url: None,
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();

    let res = client
        .inference(tensorzero::ClientInferenceParams {
            function_name: Some("test_system_template".to_string()),
            variant_name: Some("test".to_string()),
            input: tensorzero::Input {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "AskJeeves".into(),
                )])))),
                messages: vec![],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(InferenceResponse::Chat(res)) = res else {
        panic!("Expected non-streaming response, got {res:?}");
    };

    let expected_content = vec![
      ContentBlockChatOutput::Text(Text {
        text: "{\"system\":\"You are a helpful and friendly assistant named AskJeeves\",\"messages\":[]}".to_string(),
      })
    ];
    expect_that!(res.content, eq(&expected_content));
}

#[gtest]
#[tokio::test]
async fn test_named_system_template_with_schema() {
    let config_dir = tempfile::tempdir().unwrap();
    let config_path = config_dir.path().join("tensorzero.toml");
    let config = r#"
  [functions.test_system_template]
  type = "chat"
  schemas.system.path = "./system_schema.json"

  [functions.test_system_template.variants.test]
  type = "chat_completion"
  model = "dummy::echo_request_messages"
  templates.system.path = "./system_template.minijinja"
  "#;
    std::fs::write(&config_path, config).unwrap();
    std::fs::write(
        config_dir.path().join("system_template.minijinja"),
        "You are a helpful and friendly assistant named {{ assistant_name }}",
    )
    .unwrap();
    std::fs::write(
        config_dir.path().join("system_schema.json"),
        "{\"type\":\"object\",\"properties\":{\"assistant_name\":{\"type\":\"string\"}},\"required\":[\"assistant_name\"]}",
    )
    .unwrap();

    let client = make_embedded_gateway_with_config_path(Some(config_path.as_path())).await;

    let res = client
        .inference(tensorzero::ClientInferenceParams {
            function_name: Some("test_system_template".to_string()),
            variant_name: Some("test".to_string()),
            input: tensorzero::Input {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "AskJeeves".into(),
                )])))),
                messages: vec![],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(InferenceResponse::Chat(res)) = res else {
        panic!("Expected non-streaming response, got {res:?}");
    };

    let expected_content = vec![
      ContentBlockChatOutput::Text(Text {
        text: "{\"system\":\"You are a helpful and friendly assistant named AskJeeves\",\"messages\":[]}".to_string(),
      })
    ];
    expect_that!(res.content, eq(&expected_content));

    let inference_id = res.inference_id;
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let e2e_config = get_e2e_config().await;
    let inferences = conn
        .list_inferences(
            &e2e_config,
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
    let input = serde_json::to_value(&chat.input).unwrap();
    let expected_input = serde_json::json!({"system":{"assistant_name":"AskJeeves"},"messages":[]});
    expect_that!(input, eq(&expected_input));

    let error = client
        .inference(tensorzero::ClientInferenceParams {
            function_name: Some("test_system_template".to_string()),
            variant_name: Some("test".to_string()),
            input: tensorzero::Input {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    123.into(),
                )])))),
                messages: vec![],
            },
            ..Default::default()
        })
        .await
        .unwrap_err();
    expect_that!(error.to_string(), contains_substring("123 is not of type"));
}
