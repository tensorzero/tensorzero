#![allow(clippy::print_stdout)]

use http::StatusCode;
use reqwest::Client;
use serde_json::{json, Value};
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_model_inferences_clickhouse,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn e2e_test_template_no_schema() {
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

    assert_eq!(response.status(), StatusCode::OK);
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
    assert_eq!(echoed_content, expected_content);
}

#[tokio::test]
async fn e2e_test_mixture_of_n_template_no_schema() {
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

    assert_eq!(response.status(), StatusCode::OK);
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
    assert_eq!(echoed_content, expected_content);
    // We don't check ClickHouse, as we already do that in lots of other tests
}

#[tokio::test]
async fn e2e_test_best_of_n_template_no_schema() {
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

    assert_eq!(response.status(), StatusCode::OK);
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
    assert_eq!(echoed_content, expected_content);

    let clickhouse = get_clickhouse().await;
    let results: Vec<Value> = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 3);
    assert!(
        results
            .iter()
            .filter(|r| r["model_name"] == "dummy::best_of_n_0")
            .count()
            == 1
    );
    // Just check the input to 'dummy::best_of_n_0' - we already have lots of other 'best_of_n' tests
    for result in results {
        let model_name = result.get("model_name").unwrap().as_str().unwrap();
        if model_name == "dummy::best_of_n_0" {
            let system = &result["system"];
            assert_eq!(system, "You are an assistant tasked with re-ranking candidate answers to the following problem:\n------\nouter template system text: `My system message`\n------\nThe messages below are the conversation history between the user and the assistant along with a final message giving a set of candidate responses.\nPlease evaluate the following candidate responses and provide your reasoning along with the index of the best candidate in the following JSON format:\n{\n    \"thinking\": \"your reasoning here\",\n    \"answer_choice\": int  // Range: 0 to 1\n}\nIn the \"thinking\" block:\nFirst, you should analyze each response itself against the conversation history and determine if it is a good response or not.\nThen you should think out loud about which is best and most faithful to instructions.\nIn the \"answer_choice\" block: you should output the index of the best response.");
            let input_messages = result["input_messages"].as_str().unwrap();
            let input_messages = serde_json::from_str::<Value>(input_messages).unwrap();
            assert_eq!(
                input_messages,
                serde_json::json!([
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
                ])
            );
        }
    }
}

#[tokio::test]
async fn e2e_test_invalid_system_input_template_no_schema() {
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

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json["error"].as_str().unwrap();
    assert_eq!(
        error,
        "Message has non-string content but there is no schema given for role system."
    );
}

#[tokio::test]
async fn e2e_test_invalid_user_input_template_no_schema() {
    let payload = json!({
        "function_name": "basic_test_template_no_schema",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First user message"},
                        {"type": "text", "arguments": {"my_invalid": "user message"}},
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

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json["error"].as_str().unwrap();
    assert_eq!(
        error,
        "Message at index 0 has non-string content but there is no schema given for role user."
    );
}

#[tokio::test]
async fn e2e_test_invalid_assistant_input_template_no_schema() {
    let payload = json!({
        "function_name": "basic_test_template_no_schema",
        "input":{
            "system": "My system message",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "arguments": {"my_invalid": "assistant message"}},
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

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json["error"].as_str().unwrap();
    assert_eq!(
        error,
        "Message at index 0 has non-string content but there is no schema given for role assistant."
    );
}
