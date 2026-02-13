//! Mid-stream error tests for `include_raw_response`.
//!
//! These tests exercise streaming requests where a provider error occurs after
//! some SSE chunks have already been sent. The error should be emitted as an SSE
//! event (not silently dropped), and `raw_chunk` / `tensorzero_raw_chunk` should
//! be included when `include_raw_response=true`.

use futures::StreamExt;
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero::InferenceResponseChunk;

use crate::common::get_gateway_endpoint;

// =============================================================================
// T0 native /inference endpoint -- streaming -- include_raw_response=true
// =============================================================================

#[tokio::test]
async fn test_raw_response_error_mid_stream_inference() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "err_in_stream_with_raw_response",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Please write me a sentence about Megumin making an explosion."
                }
            ]},
        "stream": true,
        "include_raw_response": true,
    });

    let mut event_stream = reqwest::Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut good_chunks = 0;
    loop {
        match event_stream.next().await {
            Some(Ok(e)) => match e {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        break;
                    }
                    let obj: Value = serde_json::from_str(&message.data).unwrap();
                    if let Some(error) = obj.get("error") {
                        let error_str = error.as_str().expect(
                            "T0 native error field should be a string in mid-stream error event",
                        );
                        assert!(
                            error_str.contains("Dummy error in stream with raw response"),
                            "Unexpected error: {error_str}"
                        );
                        assert_eq!(good_chunks, 3, "Error should appear after 3 good chunks");

                        // Should have raw_chunk (matching success chunk format)
                        let raw_chunk = obj
                            .get("raw_chunk")
                            .expect(
                                "Mid-stream error event should have `raw_chunk` when include_raw_response=true",
                            )
                            .as_str()
                            .expect("raw_chunk should be a string");
                        assert!(
                            raw_chunk.contains("dummy client raw response"),
                            "raw_chunk should contain the dummy raw response"
                        );
                    } else {
                        let _chunk: InferenceResponseChunk =
                            serde_json::from_str(&message.data).unwrap();
                    }
                    good_chunks += 1;
                }
            },
            Some(Err(e)) => {
                panic!("Unexpected error: {e:?}");
            }
            None => {
                break;
            }
        }
    }
    assert_eq!(
        good_chunks, 17,
        "Stream should continue after error and produce 17 total chunks"
    );
}

// =============================================================================
// T0 native /inference endpoint -- streaming -- include_raw_response not set
// =============================================================================

#[tokio::test]
async fn test_raw_response_error_mid_stream_inference_not_requested() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "err_in_stream_with_raw_response",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Please write me a sentence about Megumin making an explosion."
                }
            ]},
        "stream": true,
    });

    let mut event_stream = reqwest::Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut good_chunks = 0;
    loop {
        match event_stream.next().await {
            Some(Ok(e)) => match e {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        break;
                    }
                    let obj: Value = serde_json::from_str(&message.data).unwrap();
                    if let Some(error) = obj.get("error") {
                        let error_str = error.as_str().expect(
                            "T0 native error field should be a string in mid-stream error event",
                        );
                        assert!(
                            error_str.contains("Dummy error in stream with raw response"),
                            "Unexpected error: {error_str}"
                        );
                        assert_eq!(good_chunks, 3, "Error should appear after 3 good chunks");

                        // Should NOT have raw_chunk when include_raw_response is not set
                        assert!(
                            obj.get("raw_chunk").is_none(),
                            "raw_chunk should not be present when include_raw_response is not set"
                        );
                    } else {
                        let _chunk: InferenceResponseChunk =
                            serde_json::from_str(&message.data).unwrap();
                    }
                    good_chunks += 1;
                }
            },
            Some(Err(e)) => {
                panic!("Unexpected error: {e:?}");
            }
            None => {
                break;
            }
        }
    }
    assert_eq!(
        good_chunks, 17,
        "Stream should continue after error and produce 17 total chunks"
    );
}

// =============================================================================
// OAI-compatible /openai/v1/chat/completions -- streaming -- include_raw_response=true
// =============================================================================

#[tokio::test]
async fn test_raw_response_error_mid_stream_openai() {
    let payload = json!({
        "model": "tensorzero::function_name::basic_test",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "tensorzero::arguments": {"assistant_name": "AskJeeves"}}]
            },
            {
                "role": "user",
                "content": "Please write me a sentence about Megumin making an explosion."
            }
        ],
        "stream": true,
        "tensorzero::variant_name": "err_in_stream_with_raw_response",
        "tensorzero::include_raw_response": true,
    });

    let mut event_stream = reqwest::Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut good_chunks = 0;
    let mut saw_error = false;
    loop {
        match event_stream.next().await {
            Some(Ok(e)) => match e {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        break;
                    }
                    let obj: Value = serde_json::from_str(&message.data).unwrap();
                    if let Some(error_obj) = obj.get("error") {
                        saw_error = true;

                        // OAI format: {"error": {"message": "..."}}
                        let error_message = error_obj
                            .get("message")
                            .expect("OAI error should have `message` field")
                            .as_str()
                            .expect("OAI error message should be a string");
                        assert!(
                            error_message.contains("Dummy error in stream with raw response"),
                            "Unexpected error: {error_message}"
                        );

                        // Should have tensorzero_raw_chunk (matching success chunk format)
                        let raw_chunk = obj
                            .get("tensorzero_raw_chunk")
                            .expect(
                                "Mid-stream error event should have `tensorzero_raw_chunk` when include_raw_response=true",
                            )
                            .as_str()
                            .expect("tensorzero_raw_chunk should be a string");
                        assert!(
                            raw_chunk.contains("dummy client raw response"),
                            "tensorzero_raw_chunk should contain the dummy raw response"
                        );
                    } else {
                        good_chunks += 1;
                    }
                }
            },
            Some(Err(e)) => {
                panic!("Unexpected error: {e:?}");
            }
            None => {
                break;
            }
        }
    }
    assert!(
        saw_error,
        "Should have received an error event in the stream"
    );
    assert!(
        good_chunks > 0,
        "Stream should continue after error and produce good chunks"
    );
}

// =============================================================================
// OAI-compatible /openai/v1/chat/completions -- streaming -- include_raw_response not set
// =============================================================================

#[tokio::test]
async fn test_raw_response_error_mid_stream_openai_not_requested() {
    let payload = json!({
        "model": "tensorzero::function_name::basic_test",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "tensorzero::arguments": {"assistant_name": "AskJeeves"}}]
            },
            {
                "role": "user",
                "content": "Please write me a sentence about Megumin making an explosion."
            }
        ],
        "stream": true,
        "tensorzero::variant_name": "err_in_stream_with_raw_response",
    });

    let mut event_stream = reqwest::Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut good_chunks = 0;
    let mut saw_error = false;
    loop {
        match event_stream.next().await {
            Some(Ok(e)) => match e {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        break;
                    }
                    let obj: Value = serde_json::from_str(&message.data).unwrap();
                    if let Some(error_obj) = obj.get("error") {
                        saw_error = true;

                        // OAI format: {"error": {"message": "..."}}
                        let error_message = error_obj
                            .get("message")
                            .expect("OAI error should have `message` field")
                            .as_str()
                            .expect("OAI error message should be a string");
                        assert!(
                            error_message.contains("Dummy error in stream with raw response"),
                            "Unexpected error: {error_message}"
                        );

                        // Should NOT have tensorzero_raw_chunk when include_raw_response is not set
                        assert!(
                            obj.get("tensorzero_raw_chunk").is_none(),
                            "tensorzero_raw_chunk should not be present when include_raw_response is not set"
                        );
                    } else {
                        good_chunks += 1;
                    }
                }
            },
            Some(Err(e)) => {
                panic!("Unexpected error: {e:?}");
            }
            None => {
                break;
            }
        }
    }
    assert!(
        saw_error,
        "Should have received an error event in the stream"
    );
    assert!(
        good_chunks > 0,
        "Stream should continue after error and produce good chunks"
    );
}
