use futures::StreamExt;
use serde_json::json;
use tensorzero::{
    Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    InferenceOutput, InferenceResponseChunk, Role,
};
use tensorzero_core::inference::types::{Arguments, System, TextKind};

use crate::common::get_gateway_endpoint;
use reqwest_eventsource::{Event, RequestBuilderExt};

#[tokio::test]
async fn test_client_stream_with_error_http_gateway() {
    test_client_stream_with_error(tensorzero::test_helpers::make_http_gateway().await).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_client_stream_with_error_embedded_gateway() {
    test_client_stream_with_error(tensorzero::test_helpers::make_embedded_gateway().await).await;
}

async fn test_client_stream_with_error(client: Client) {
    let res = client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            variant_name: Some("err_in_stream".to_string()),
            input: ClientInput {
                system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
                    "assistant_name".to_string(),
                    "AskJeeves".into(),
                )])))),
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Please write me a sentence about Megumin making an explosion."
                            .into(),
                    })],
                }],
            },
            stream: Some(true),
            ..Default::default()
        })
        .await
        .unwrap();
    let InferenceOutput::Streaming(stream) = res else {
        panic!("Expected a stream");
    };
    let stream = stream.enumerate().collect::<Vec<_>>().await;
    assert_eq!(stream.len(), 17);

    for (i, chunk) in stream {
        if i == 3 {
            let err = chunk
                .expect_err("Expected error after 3 chunks")
                .to_string();
            assert!(
                err.contains("Dummy error in stream"),
                "Unexpected error: `{err}`"
            );
        } else {
            chunk.expect("Expected first few chunks to be Ok");
        }
    }
}

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
