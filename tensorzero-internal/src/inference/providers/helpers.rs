use std::pin::Pin;

use futures::{stream::Peekable, Stream};

use crate::{
    error::{Error, ErrorDetails},
    inference::types::ProviderInferenceResponseChunk,
    model::ModelProviderRequestInfo,
    variant::chat_completion::ExtraBodyConfig,
};

pub fn inject_extra_body(
    config: Option<&ExtraBodyConfig>,
    model_provider_data: impl Into<ModelProviderRequestInfo>,
    body: &mut serde_json::Value,
) -> Result<(), Error> {
    let Some(body_map) = body.as_object_mut() else {
        return Err(Error::new(ErrorDetails::Serialization {
            message: "Body is not a map".to_string(),
        }));
    };
    let model_provider: ModelProviderRequestInfo = model_provider_data.into();
    // Write the variant extra_body first, then the model_provider extra_body.
    // This way, the model_provider extra_body will overwrite any keys in the
    // variant extra_body.
    for extra_body in [config, model_provider.extra_body.as_ref()]
        .into_iter()
        .flatten()
    {
        for (key, value) in extra_body.data.iter() {
            // We intentionally overwrite any existing keys in the body.
            body_map.insert(key.to_string(), value.clone());
        }
    }
    Ok(())
}

/// Gives mutable access to the first chunk of a stream, returning an error if the stream is empty
pub async fn peek_first_chunk<
    'a,
    T: Stream<Item = Result<ProviderInferenceResponseChunk, Error>> + ?Sized,
>(
    stream: &'a mut Peekable<Pin<Box<T>>>,
    raw_request: &str,
    provider_type: &str,
) -> Result<&'a mut ProviderInferenceResponseChunk, Error> {
    // If the next stream item is an error, consume and return it
    if let Some(err) = Pin::new(&mut *stream).next_if(Result::is_err).await {
        match err {
            Err(e) => {
                return Err(e)
            }
            Ok(_) => {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: "Stream `next_if` produced wrong value (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                 }))
            }
        }
    }
    // Peek at the same item - we already checked that it's not an error.
    match Pin::new(stream).peek_mut().await {
        // Returning `chunk` extends the lifetime of 'stream.as_mut() to 'a,
        // which blocks us from using 'stream' in the other branches of
        // this match.
        Some(Ok(chunk)) => Ok(chunk),
        None => {
            Err(Error::new(ErrorDetails::InferenceServer {
                message: "Stream ended before first chunk".to_string(),
                provider_type: provider_type.to_string(),
                raw_request: Some(raw_request.to_string()),
                raw_response: None,
            }))
        }
        // Due to a borrow-checker limitation, we can't use 'stream' here
        // (since returning `chunk` above will cause `stream` to still be borrowed here.)
        // We check for an error before the `match` block, which makes this unreachable
        Some(Err(_)) => {
            Err(Error::new(ErrorDetails::InternalError {
                message: "Stream produced error after we peeked non-error (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string()
             }))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::{stream, StreamExt};

    use crate::inference::types::{ContentBlockChunk, TextChunk};

    use super::*;

    #[tokio::test]
    async fn test_peek_empty() {
        let mut stream = Box::pin(stream::empty()).peekable();
        let err = peek_first_chunk(&mut stream, "test", "test")
            .await
            .expect_err("Peeking empty stream should fail");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Stream ended before first chunk"),
            "Unexpected error message: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_peek_err() {
        let mut stream = Box::pin(stream::iter([Err(Error::new(
            ErrorDetails::InternalError {
                message: "My test error".to_string(),
            },
        ))]))
        .peekable();
        let err = peek_first_chunk(&mut stream, "test", "test")
            .await
            .expect_err("Peeking errored stream should fail");
        assert_eq!(
            err,
            Error::new(ErrorDetails::InternalError {
                message: "My test error".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn test_peek_good() {
        let chunk = ProviderInferenceResponseChunk {
            content: vec![ContentBlockChunk::Text(TextChunk {
                id: "0".to_string(),
                text: "Hello, world!".to_string(),
            })],
            created: 0,
            usage: None,
            raw_response: "My raw response".to_string(),
            latency: Duration::from_secs(0),
        };
        let mut stream = Box::pin(stream::iter([
            Ok(chunk.clone()),
            Err(Error::new(ErrorDetails::InternalError {
                message: "My test error".to_string(),
            })),
        ]))
        .peekable();
        let peeked_chunk: &mut ProviderInferenceResponseChunk =
            peek_first_chunk(&mut stream, "test", "test")
                .await
                .expect("Peeking stream should succeed");
        assert_eq!(&chunk, peeked_chunk);
    }

    #[test]
    fn test_inject_nothing() {
        let mut body = serde_json::json!({});
        inject_extra_body(
            None,
            ModelProviderRequestInfo { extra_body: None },
            &mut body,
        )
        .unwrap();
        assert_eq!(body, serde_json::json!({}));
    }

    #[test]
    fn test_inject_to_non_map() {
        let err = inject_extra_body(
            None,
            ModelProviderRequestInfo { extra_body: None },
            &mut serde_json::Value::String("test".to_string()),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("Body is not a map"),
            "Unexpected error message: {err:?}"
        );
    }

    #[test]
    fn test_inject_overwrite_object() {
        let mut body = serde_json::json!({
            "otherKey": "otherValue",
            "generationConfig": {
                "temperature": 123
            }
        });
        inject_extra_body(
            Some(&ExtraBodyConfig {
                data: serde_json::json!({
                    "generationConfig": {
                        "otherNestedKey": "otherNestedValue"
                    }
                })
                .as_object()
                .unwrap()
                .clone(),
            }),
            ModelProviderRequestInfo { extra_body: None },
            &mut body,
        )
        .unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "otherKey": "otherValue",
                "generationConfig": {
                    "otherNestedKey": "otherNestedValue"
                }
            })
        );
    }

    #[test]
    fn test_inject_both() {
        let mut body = serde_json::json!({
            "otherKey": "otherValue",
            "generationConfig": {
                "temperature": 123
            }
        });
        inject_extra_body(
            Some(&ExtraBodyConfig {
                data: serde_json::json!({
                    "variantKey": "variantValue",
                    "generationConfig": {
                        "otherNestedKey": "otherNestedValue"
                    }
                })
                .as_object()
                .unwrap()
                .clone(),
            }),
            ModelProviderRequestInfo {
                extra_body: Some(ExtraBodyConfig {
                    data: serde_json::json!({
                        "variantKey": "modelProviderOverride",
                        "modelProviderKey": "modelProviderValue"
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                }),
            },
            &mut body,
        )
        .unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "otherKey": "otherValue",
                "modelProviderKey": "modelProviderValue",
                "variantKey": "modelProviderOverride",
                "generationConfig": {
                    "otherNestedKey": "otherNestedValue"
                }
            })
        );
    }
}
