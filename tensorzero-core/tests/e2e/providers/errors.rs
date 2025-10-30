use reqwest::{Client, StatusCode};
use serde_json::{json, Value};

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
async fn test_bad_text_input() {
    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "bad_field": "Blah"
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Unknown key `bad_field` in text content",
    )
    .await;

    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "bad_field": "Blah",
                            "bad_field_2": "Other",
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Expected exactly one other key in text content, found 2 other keys",
    )
    .await;

    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Expected exactly one other key in text content, found 0 other keys",
    )
    .await;

    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": ["Not", "a", "string"]
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Error deserializing `text`: invalid type: sequence, expected a string",
    )
    .await;

    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "arguments": "Not an object"
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Error deserializing `arguments`: invalid type: string \"Not an object\", expected a map",
    )
    .await;
}
