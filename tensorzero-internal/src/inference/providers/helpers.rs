use std::pin::Pin;

use futures::{stream::Peekable, Stream};
use serde_json::{map::Entry, Map, Value};

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
    if !body.is_object() {
        return Err(Error::new(ErrorDetails::Serialization {
            message: "Body is not a map".to_string(),
        }));
    }
    let model_provider: ModelProviderRequestInfo = model_provider_data.into();
    // Write the variant extra_body first, then the model_provider extra_body.
    // This way, the model_provider extra_body will overwrite any keys in the
    // variant extra_body.
    for extra_body in [config, model_provider.extra_body.as_ref()]
        .into_iter()
        .flatten()
    {
        for replacement in &extra_body.data {
            write_json_pointer_with_parent_creation(
                body,
                &replacement.pointer,
                replacement.value.clone(),
            )?;
        }
    }
    Ok(())
}

// Copied from serde_json (MIT-licensed): https://github.com/serde-rs/json/blob/400eaa977f1f0a1c9ad5e35d634ed2226bf1218c/src/value/mod.rs#L259
// This accepts positive integers, rejecting integers with a leading plus or extra leading zero.
// We use this to parse integers according to the JSON pointer spec
fn parse_index(s: &str) -> Option<usize> {
    if s.starts_with('+') || (s.starts_with('0') && s.len() != 1) {
        return None;
    }
    s.parse().ok()
}

// Based on https://github.com/serde-rs/json/blob/400eaa977f1f0a1c9ad5e35d634ed2226bf1218c/src/value/mod.rs#L834
fn write_json_pointer_with_parent_creation(
    mut value: &mut serde_json::Value,
    pointer: &str,
    target_value: Value,
) -> Result<(), Error> {
    if pointer.is_empty() {
        return Err(Error::new(ErrorDetails::ExtraBodyReplacement {
            message: "Pointer cannot be empty".to_string(),
            pointer: pointer.to_string(),
        }));
    }
    if !pointer.starts_with('/') {
        return Err(Error::new(ErrorDetails::ExtraBodyReplacement {
            message: "Pointer must start with '/'".to_string(),
            pointer: pointer.to_string(),
        }));
    }

    if pointer.ends_with('/') {
        return Err(Error::new(ErrorDetails::ExtraBodyReplacement {
            message: "Pointer cannot end with '/'".to_string(),
            pointer: pointer.to_string(),
        }));
    }

    let components = pointer
        .split('/')
        .skip(1)
        .map(|x| x.replace("~1", "/").replace("~0", "~"));
    for token in components {
        match value {
            Value::Object(map) => match map.entry(token.clone()) {
                // Move inside an object if the current pointer component is a valid key
                Entry::Occupied(occupied) => value = occupied.into_mut(),
                Entry::Vacant(vacant) => {
                    // Edge case - we reject json paths like `/existing-key/new-key/<n>`, where:
                    // * 'existing-key' already exists in the object
                    // * 'new-key' does not already exist
                    // * <n> is an integer
                    //
                    // We cannot create an entry for 'new-key', as it's ambiguous whether it should be an object {"n": some_value}
                    // or an array [.., some_value] with `some_value` at index `n`.
                    if parse_index(&token).is_some() {
                        return Err(Error::new(ErrorDetails::ExtraBodyReplacement {
                        message: format!("TensorZero doesn't support pointing an index ({token}) if its container doesn't exist. We'd love to hear about your use case (& help)! Please open a GitHub Discussion: https://github.com/tensorzero/tensorzero/discussions/new"),
                        pointer: pointer.to_string(),
                        }));
                    } else {
                        // For non-integer keys, create a new object. This allows writing things like
                        // `/generationConfig/temperature`, which will create a `generationConfig` object
                        // if we don't already have `generationConfig` as a key in the object.
                        value = vacant.insert(Value::Object(Map::new()));
                    }
                }
            },
            Value::Array(list) => {
                let len = list.len();
                value = parse_index(&token)
                    .and_then(move |x| list.get_mut(x))
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::ExtraBodyReplacement {
                            message: format!(
                                "Could not find array index {token} in target array (len {len})",
                            ),
                            pointer: pointer.to_string(),
                        })
                    })?;
            }
            other => {
                return Err(Error::new(ErrorDetails::ExtraBodyReplacement {
                    message: format!("Can only index into object or array - found target {other}"),
                    pointer: pointer.to_string(),
                }))
            }
        }
    }
    *value = target_value;
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

    use crate::{
        inference::types::{ContentBlockChunk, TextChunk},
        variant::chat_completion::ExtraBodyReplacement,
    };

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
                data: vec![
                    ExtraBodyReplacement {
                        pointer: "/generationConfig".to_string(),
                        value: serde_json::json!({
                            "otherNestedKey": "otherNestedValue"
                        }),
                    },
                    ExtraBodyReplacement {
                        pointer: "/generationConfig/temperature".to_string(),
                        value: serde_json::json!(0.123),
                    },
                ],
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
                    "otherNestedKey": "otherNestedValue",
                    "temperature": 0.123
                }
            })
        );
    }

    // Tests that we inject fields in the correct order when `extra_body`
    // is set at both the variant and model provider level. Keys set in the
    // model provider should override keys set in the variant
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
                data: vec![
                    ExtraBodyReplacement {
                        pointer: "/generationConfig/otherNestedKey".to_string(),
                        value: Value::String("otherNestedValue".to_string()),
                    },
                    ExtraBodyReplacement {
                        pointer: "/variantKey".to_string(),
                        value: Value::String("variantValue".to_string()),
                    },
                ],
            }),
            ModelProviderRequestInfo {
                extra_body: Some(ExtraBodyConfig {
                    data: vec![
                        ExtraBodyReplacement {
                            pointer: "/variantKey".to_string(),
                            value: Value::String("modelProviderOverride".to_string()),
                        },
                        ExtraBodyReplacement {
                            pointer: "/modelProviderKey".to_string(),
                            value: Value::String("modelProviderValue".to_string()),
                        },
                    ],
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
                    "temperature": 123,
                    "otherNestedKey": "otherNestedValue"
                }
            })
        );
    }

    #[test]
    fn test_json_pointer_write_simple() {
        let mut obj1 = serde_json::json!({
            "object1": "value1",
            "object2": {
                "key1": "value1",
            }
        });
        write_json_pointer_with_parent_creation(
            &mut obj1,
            "/object1",
            serde_json::json!("new_value"),
        )
        .unwrap();
        assert_eq!(
            obj1,
            serde_json::json!({
                "object1": "new_value",
                "object2": {
                    "key1": "value1",
                }
            })
        );

        write_json_pointer_with_parent_creation(
            &mut obj1,
            "/object2/key1",
            serde_json::json!("new_key_value"),
        )
        .unwrap();
        assert_eq!(
            obj1,
            serde_json::json!({
                "object1": "new_value",
                "object2": {
                    "key1": "new_key_value",
                }
            })
        );

        write_json_pointer_with_parent_creation(&mut obj1, "/object2", serde_json::json!(42.1))
            .unwrap();
        assert_eq!(
            obj1,
            serde_json::json!({
                "object1": "new_value",
                "object2": 42.1
            })
        );

        write_json_pointer_with_parent_creation(
            &mut obj1,
            "/new-top-level",
            serde_json::json!(["Hello", 100]),
        )
        .unwrap();
        assert_eq!(
            obj1,
            serde_json::json!({
                "object1": "new_value",
                "object2": 42.1,
                "new-top-level": ["Hello", 100]
            })
        );

        write_json_pointer_with_parent_creation(
            &mut obj1,
            "/new-top-level/1",
            serde_json::json!("Replaced array value"),
        )
        .unwrap();
        assert_eq!(
            obj1,
            serde_json::json!({
                "object1": "new_value",
                "object2": 42.1,
                "new-top-level": ["Hello", "Replaced array value"]
            })
        );

        write_json_pointer_with_parent_creation(
            &mut obj1,
            "/some/new/object/path",
            serde_json::json!("Inserted a deeply nested string"),
        )
        .unwrap();
        assert_eq!(
            obj1,
            serde_json::json!({
                "object1": "new_value",
                "object2": 42.1,
                "new-top-level": ["Hello", "Replaced array value"],
                "some": {
                    "new": {
                        "object": {
                            "path": "Inserted a deeply nested string"
                        }
                    }
                }
            })
        );
    }

    #[test]
    fn test_json_pointer_errors() {
        let mut obj1 = serde_json::json!({});
        let err =
            write_json_pointer_with_parent_creation(&mut obj1, "", serde_json::json!("new_value"))
                .unwrap_err()
                .to_string();
        assert!(
            err.contains("Pointer cannot be empty"),
            "Unexpected error message: {err:?}"
        );

        let err = write_json_pointer_with_parent_creation(
            &mut obj1,
            "object1",
            serde_json::json!("new_value"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("Pointer must start with '/'"),
            "Unexpected error message: {err:?}"
        );

        let err = write_json_pointer_with_parent_creation(
            &mut obj1,
            "/object1/",
            serde_json::json!("new_value"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("Pointer cannot end with '/'"),
            "Unexpected error message: {err:?}"
        );

        let mut array_val = serde_json::json!(["First", "Second"]);
        let err = write_json_pointer_with_parent_creation(
            &mut array_val,
            "/2",
            serde_json::json!("Replaced array value"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("Could not find array index 2 in target array (len 2)"),
            "Unexpected error message: {err:?}"
        );

        let err = write_json_pointer_with_parent_creation(
            &mut array_val,
            "/non-int-index",
            serde_json::json!("Replaced array value"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("Could not find array index non-int-index in target array (len 2)"),
            "Unexpected error message: {err:?}"
        );

        let mut obj = serde_json::json!({});
        let err = write_json_pointer_with_parent_creation(
            &mut obj,
            "/new-key/0",
            serde_json::json!("Replaced array value"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("TensorZero doesn't support pointing an index (0) if its container doesn't exist. We'd love to hear about your use case (& help)! Please open a GitHub Discussion: https://github.com/tensorzero/tensorzero/discussions/new` with pointer: `/new-key/0`"),
            "Unexpected error message: {err:?}"
        );
    }
}
