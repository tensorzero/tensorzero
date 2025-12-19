use reqwest::{Client, StatusCode};
use serde_json::{Value, json};

use crate::common::get_gateway_endpoint;

async fn test_payload_produces_error(payload: Value, expected_err: &str) {
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    let error_msg = response_json["error"].as_str().unwrap();
    assert_eq!(expected_err, error_msg);
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_bad_text_input_unknown_field() {
    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "bad_field": "Blah"
                    }]
                }]
            },
        }),
        "input.messages[0].content[0]: bad_field: unknown field `bad_field`, expected `text`",
    )
    .await;
}

#[tokio::test]
async fn test_bad_text_input_multiple_unknown_fields() {
    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "bad_field": "Blah",
                        "bad_field_2": "Other",
                    }]
                }]
            },
        }),
        "input.messages[0].content[0]: bad_field_2: unknown field `bad_field_2`, expected `text`",
    )
    .await;
}

#[tokio::test]
async fn test_bad_text_input_missing_text_field() {
    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                    }]
                }]
            },
        }),
        "input.messages[0].content[0]: missing field `text`",
    )
    .await;
}

#[tokio::test]
async fn test_bad_text_input_wrong_type() {
    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": ["Not", "a", "string"]
                    }]
                }]
            },
        }),
        "input.messages[0].content[0]: text: invalid type: sequence, expected a string",
    )
    .await;
}
