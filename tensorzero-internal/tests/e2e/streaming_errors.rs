use futures::StreamExt;
use serde_json::json;
use tensorzero::InferenceResponseChunk;

use crate::common::get_gateway_endpoint;
use reqwest_eventsource::{Event, RequestBuilderExt};

#[tokio::test]
async fn test_stream_with_error() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "err_in_stream",
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
        .unwrap();

    let mut good_chunks = 0;
    // Check we receive all client chunks correctly
    loop {
        match event_stream.next().await {
            Some(Ok(e)) => match e {
                Event::Open => continue,
                Event::Message(message) => {
                    if message.data == "[DONE]" {
                        break;
                    }
                    let obj: serde_json::Value = serde_json::from_str(&message.data).unwrap();
                    if let Some(error) = obj.get("error") {
                        let error_str: &str = error.as_str().unwrap();
                        assert!(
                            error_str.contains("Dummy error in stream"),
                            "Unexpected error: {error_str}"
                        );
                        assert_eq!(good_chunks, 3);
                    } else {
                        let _chunk: InferenceResponseChunk =
                            serde_json::from_str(&message.data).unwrap();
                    }
                    good_chunks += 1;
                }
            },
            Some(Err(e)) => {
                if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                    break;
                }
                panic!("Unexpected error: {e:?}");
            }
            None => {
                panic!("Stream ended unexpectedly");
            }
        }
    }
    assert_eq!(good_chunks, 17);
}
