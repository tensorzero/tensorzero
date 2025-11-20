use axum::http;
use bytes::Bytes;
use futures::{stream::Peekable, Stream};
use serde::de::DeserializeOwned;
use serde_json::{map::Entry, Map, Value};
use std::{collections::HashMap, pin::Pin};
use uuid::Uuid;

use crate::{
    error::{DisplayOrDebugGateway, Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    http::{TensorZeroEventSource, TensorzeroRequestBuilder},
    inference::types::{
        batch::{ProviderBatchInferenceOutput, ProviderBatchInferenceResponse},
        extra_body::{DynamicExtraBody, ExtraBodyReplacementKind, FullExtraBodyConfig},
        extra_headers::{DynamicExtraHeader, ExtraHeader, ExtraHeaderKind, FullExtraHeadersConfig},
        resolved_input::{FileUrl, LazyFile},
        ProviderInferenceResponseChunk,
    },
    model::{fully_qualified_name, ModelProviderRequestInfo},
};

pub struct JsonlBatchFileInfo {
    pub provider_type: String,
    pub raw_request: String,
    pub raw_response: String,
    pub file_id: String,
}

pub async fn convert_stream_error(
    raw_request: String,
    provider_type: String,
    e: reqwest_eventsource::Error,
) -> Error {
    let message = e.to_string();
    // If we get an invalid status code, content type, or generic transport error,
    // then we assume that we're never going to be able to read more chunks from the stream,
    // The `wrap_provider_stream` function will bail out when it sees this error,
    // to avoid holding open a broken stream (which will delay gateway shutdown when we
    // wait on the parent `Span` to finish)
    match e {
        reqwest_eventsource::Error::InvalidStatusCode(_, resp)
        | reqwest_eventsource::Error::InvalidContentType(_, resp) => {
            ErrorDetails::FatalStreamError {
                message,
                provider_type,
                raw_request: Some(raw_request),
                raw_response: resp.text().await.ok(),
            }
            .into()
        }
        reqwest_eventsource::Error::Transport(_) => ErrorDetails::FatalStreamError {
            message,
            provider_type,
            raw_request: Some(raw_request),
            raw_response: None,
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message,
            raw_request: Some(raw_request),
            raw_response: None,
            provider_type,
        }
        .into(),
    }
}

// If we could have forwarded an file/image (except for the fact that we're missing the mime_type), log a warning.
pub fn warn_cannot_forward_url_if_missing_mime_type(
    file: &LazyFile,
    fetch_and_encode_input_files_before_inference: bool,
    provider_type: &str,
) {
    // We're not forwarding any urls, so it doesn't matter whether or not we have a mime type
    if fetch_and_encode_input_files_before_inference {
        return;
    }
    if matches!(
        file,
        LazyFile::Url {
            file_url: FileUrl {
                url: _,
                mime_type: None,
                ..
            },
            future: _
        }
    ) {
        tracing::warn!("Cannot forward image_url to {provider_type} because no mime_type was provided. Specify `mime_type` (or `tensorzero::mime_type` for openai-compatible requests) when sending files to allow URL forwarding.");
    }
}

/// A helper function to parse lines from a JSONL file into a batch response.
/// The provided type `T` is used to parse each line.
pub async fn parse_jsonl_batch_file<T: DeserializeOwned, E: std::error::Error>(
    bytes: Result<Bytes, E>,
    JsonlBatchFileInfo {
        provider_type,
        raw_request,
        raw_response,
        file_id,
    }: JsonlBatchFileInfo,
    mut make_output: impl FnMut(T) -> Result<ProviderBatchInferenceOutput, Error>,
) -> Result<ProviderBatchInferenceResponse, Error> {
    let bytes = bytes.map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!(
                "Error reading batch results response for file {file_id}: {}",
                DisplayOrDebugGateway::new(e)
            ),
            raw_request: None,
            raw_response: None,
            provider_type: provider_type.to_string(),
        })
    })?;
    let mut elements: HashMap<Uuid, ProviderBatchInferenceOutput> = HashMap::new();
    let text = std::str::from_utf8(&bytes).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!(
                "Error parsing batch results response for file {file_id}: {}",
                DisplayOrDebugGateway::new(e)
            ),
            raw_request: None,
            raw_response: None,
            provider_type: provider_type.to_string(),
        })
    })?;
    for line in text.lines() {
        let row = match serde_json::from_str::<T>(line) {
            Ok(row) => row,
            Err(e) => {
                // Construct error for logging but don't return it
                let _ = Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing batch results row for file {file_id}: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: None,
                    raw_response: Some(line.to_string()),
                    provider_type: provider_type.to_string(),
                });
                continue;
            }
        };
        let output = match make_output(row) {
            Ok(output) => output,
            Err(_) => {
                // Construct error for logging but don't return it
                continue;
            }
        };
        elements.insert(output.id, output);
    }

    Ok(ProviderBatchInferenceResponse {
        raw_request,
        raw_response,
        elements,
    })
}

/// Injects extra headers/body fields into a request builder, and sends the request.
/// This is used when implementing non-streaming inference for a model provider,
/// and is responsible for actually submitting the HTTP request.
pub async fn inject_extra_request_data_and_send(
    provider_type: &str,
    config: &FullExtraBodyConfig,
    extra_headers_config: &FullExtraHeadersConfig,
    model_provider_data: impl Into<ModelProviderRequestInfo>,
    model_name: &str,
    mut body: serde_json::Value,
    builder: TensorzeroRequestBuilder<'_>,
) -> Result<(reqwest::Response, String), Error> {
    let headers = inject_extra_request_data(
        config,
        extra_headers_config,
        model_provider_data,
        model_name,
        &mut body,
    )?;
    let raw_request = body.to_string();
    // Apply the headers as the very last step, so that they can overwrite all
    // other headers (including things like `Authorization` and `Content-Type`)
    Ok((
        builder
            .body(raw_request.clone())
            .header("content-type", "application/json")
            .headers(headers)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!("Error sending request: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: provider_type.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?,
        raw_request,
    ))
}

/// Like `inject_extra_request_data_and_send`, but for streaming requests
/// Produces an `EventSource` instead of a `Response`.
pub async fn inject_extra_request_data_and_send_eventsource(
    provider_type: &str,
    config: &FullExtraBodyConfig,
    extra_headers_config: &FullExtraHeadersConfig,
    model_provider_data: impl Into<ModelProviderRequestInfo>,
    model_name: &str,
    mut body: serde_json::Value,
    builder: TensorzeroRequestBuilder<'_>,
) -> Result<(TensorZeroEventSource, String), Error> {
    let headers = inject_extra_request_data(
        config,
        extra_headers_config,
        model_provider_data,
        model_name,
        &mut body,
    )?;
    let raw_request = body.to_string();
    // Apply the headers as the very last step, so that they can overwrite all
    // other headers (including things like `Authorization` and `Content-Type`)
    Ok((
        builder
            .body(raw_request.clone())
            .header("content-type", "application/json")
            .headers(headers)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request: {}", DisplayOrDebugGateway::new(e)),
                    status_code: None,
                    provider_type: provider_type.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?,
        raw_request,
    ))
}

/// A helper method to inject extra_body fields into a request, and
/// construct the `HeaderMap` for the applicable extra_headers.
///
/// You should almost always use
/// `inject_extra_request_data_and_send` and `inject_extra_request_data_and_send_eventsource`
/// instead of calling this directly. The one exception is for providers which use
/// external SDK crates (e.g. aws), which may not have a `reqwest` builder available.
#[must_use = "Extra headers must be inserted into request builder"]
pub fn inject_extra_request_data(
    config: &FullExtraBodyConfig,
    extra_headers_config: &FullExtraHeadersConfig,
    model_provider_data: impl Into<ModelProviderRequestInfo>,
    model_name: &str,
    body: &mut serde_json::Value,
) -> Result<http::HeaderMap, Error> {
    if !body.is_object() {
        return Err(Error::new(ErrorDetails::Serialization {
            message: "Body is not a map".to_string(),
        }));
    }
    let model_provider: ModelProviderRequestInfo = model_provider_data.into();
    // Write the variant extra_body first, then the model_provider extra_body.
    // This way, the model_provider extra_body will overwrite any keys in the
    // variant extra_body.
    for replacement in config
        .extra_body
        .iter()
        .flat_map(|c| &c.data)
        .chain(model_provider.extra_body.iter().flat_map(|c| &c.data))
    {
        match &replacement.kind {
            ExtraBodyReplacementKind::Value(value) => {
                write_json_pointer_with_parent_creation(body, &replacement.pointer, value.clone())?;
            }
            ExtraBodyReplacementKind::Delete => {
                delete_json_pointer(body, &replacement.pointer)?;
            }
        }
    }

    let expected_model_name = model_name;
    let expected_provider_name_plain = &model_provider.provider_name;
    let expected_provider_name_fully_qualified =
        fully_qualified_name(model_name, &model_provider.provider_name);

    // Finally, write the inference-level extra_body information. This can overwrite values set from the config-level extra_body.
    for extra_body in &config.inference_extra_body.data {
        match extra_body {
            DynamicExtraBody::Variant {
                // We're iterating over a 'FilteredInferenceExtraBody', so we've already removed any non-matching variant names.
                // Any remaining `InferenceExtraBody::Variant` values should be applied to the current request
                pointer,
                value,
                ..
            } => {
                write_json_pointer_with_parent_creation(body, pointer, value.clone())?;
            }
            DynamicExtraBody::VariantDelete { pointer, .. } => {
                delete_json_pointer(body, pointer)?;
            }
            #[expect(deprecated)]
            DynamicExtraBody::Provider {
                model_provider_name,
                pointer,
                value,
            } => {
                if *model_provider_name == expected_provider_name_fully_qualified {
                    write_json_pointer_with_parent_creation(body, pointer, value.clone())?;
                }
            }
            #[expect(deprecated)]
            DynamicExtraBody::ProviderDelete {
                model_provider_name,
                pointer,
                ..
            } => {
                if *model_provider_name == expected_provider_name_fully_qualified {
                    delete_json_pointer(body, pointer)?;
                }
            }
            DynamicExtraBody::ModelProvider {
                model_name: filter_model_name,
                provider_name: filter_provider_name,
                pointer,
                value,
            } => {
                if filter_model_name == expected_model_name
                    && filter_provider_name
                        .as_deref()
                        .is_none_or(|name| name == expected_provider_name_plain.as_ref())
                {
                    write_json_pointer_with_parent_creation(body, pointer, value.clone())?;
                }
            }
            DynamicExtraBody::ModelProviderDelete {
                model_name: filter_model_name,
                provider_name: filter_provider_name,
                pointer,
                ..
            } => {
                if filter_model_name == expected_model_name
                    && filter_provider_name
                        .as_deref()
                        .is_none_or(|name| name == expected_provider_name_plain.as_ref())
                {
                    delete_json_pointer(body, pointer)?;
                }
            }
            DynamicExtraBody::Always { pointer, value } => {
                write_json_pointer_with_parent_creation(body, pointer, value.clone())?;
            }
            DynamicExtraBody::AlwaysDelete { pointer, .. } => {
                delete_json_pointer(body, pointer)?;
            }
        }
    }

    let mut headers = http::HeaderMap::new();
    // Write the variant extra_headers first, then the model_provider extra_headers.
    // This way, the model_provider extra_headers will overwrite keys in the
    // variant extra_headers.
    for extra_headers in [
        &extra_headers_config.variant_extra_headers,
        &model_provider.extra_headers,
    ]
    .into_iter()
    .flatten()
    {
        for ExtraHeader { name, kind } in &extra_headers.data {
            let name = http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Invalid header name `{name}`: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
            match kind {
                ExtraHeaderKind::Value(value) => {
                    let value =
                        http::header::HeaderValue::from_bytes(value.as_bytes()).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Invalid header value `{value}`: {}",
                                    DisplayOrDebugGateway::new(e)
                                ),
                            })
                        })?;
                    headers.insert(name, value);
                }
                ExtraHeaderKind::Delete => {
                    headers.remove(name);
                }
            }
        }
    }

    // Finally, write the inference-level extra_headers information. This can overwrite header set from the config-level extra_headers.
    for extra_header in &extra_headers_config.inference_extra_headers.data {
        match extra_header {
            DynamicExtraHeader::Variant { name, value, .. } => {
                // We're iterating over a 'FilteredInferenceExtraHeaders', so we've already removed any non-matching variant names.
                // Any remaining `InferenceExtraHeader::Variant` values should be applied to the current request
                let name = http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Invalid header name `{name}`: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;
                headers.insert(
                    name,
                    http::header::HeaderValue::from_bytes(value.as_bytes()).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Invalid header value `{value}`: {}",
                                DisplayOrDebugGateway::new(e)
                            ),
                        })
                    })?,
                );
            }
            DynamicExtraHeader::VariantDelete { name, .. } => {
                let name = http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Invalid header name `{name}`: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;
                headers.remove(name);
            }
            #[expect(deprecated)]
            DynamicExtraHeader::Provider {
                model_provider_name,
                name,
                value,
            } => {
                if *model_provider_name == expected_provider_name_fully_qualified {
                    let name =
                        http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Invalid header name `{name}`: {}",
                                    DisplayOrDebugGateway::new(e)
                                ),
                            })
                        })?;
                    headers.insert(
                        name,
                        http::header::HeaderValue::from_bytes(value.as_bytes()).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Invalid header value `{value}`: {}",
                                    DisplayOrDebugGateway::new(e)
                                ),
                            })
                        })?,
                    );
                }
            }
            #[expect(deprecated)]
            DynamicExtraHeader::ProviderDelete {
                model_provider_name,
                name,
                ..
            } => {
                if *model_provider_name == expected_provider_name_fully_qualified {
                    let name =
                        http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Invalid header name `{name}`: {}",
                                    DisplayOrDebugGateway::new(e)
                                ),
                            })
                        })?;
                    headers.remove(name);
                }
            }
            DynamicExtraHeader::ModelProvider {
                model_name: filter_model_name,
                provider_name: filter_provider_name,
                name,
                value,
            } => {
                if filter_model_name == expected_model_name
                    && filter_provider_name
                        .as_deref()
                        .is_none_or(|name| name == expected_provider_name_plain.as_ref())
                {
                    let name =
                        http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Invalid header name `{name}`: {}",
                                    DisplayOrDebugGateway::new(e)
                                ),
                            })
                        })?;
                    headers.insert(
                        name,
                        http::header::HeaderValue::from_bytes(value.as_bytes()).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Invalid header value `{value}`: {}",
                                    DisplayOrDebugGateway::new(e)
                                ),
                            })
                        })?,
                    );
                }
            }
            DynamicExtraHeader::ModelProviderDelete {
                model_name: filter_model_name,
                provider_name: filter_provider_name,
                name,
                ..
            } => {
                if filter_model_name == expected_model_name
                    && filter_provider_name
                        .as_deref()
                        .is_none_or(|name| name == expected_provider_name_plain.as_ref())
                {
                    let name =
                        http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Invalid header name `{name}`: {}",
                                    DisplayOrDebugGateway::new(e)
                                ),
                            })
                        })?;
                    headers.remove(name);
                }
            }
            DynamicExtraHeader::Always { name, value } => {
                let name = http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Invalid header name `{name}`: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;
                headers.insert(
                    name,
                    http::header::HeaderValue::from_bytes(value.as_bytes()).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Invalid header value `{value}`: {}",
                                DisplayOrDebugGateway::new(e)
                            ),
                        })
                    })?,
                );
            }
            DynamicExtraHeader::AlwaysDelete { name, .. } => {
                let name = http::header::HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Invalid header name `{name}`: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;
                headers.remove(name);
            }
        }
    }
    Ok(headers)
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

fn delete_json_pointer(mut value: &mut serde_json::Value, pointer: &str) -> Result<(), Error> {
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

    let mut components = pointer
        .split('/')
        .skip(1)
        .map(|x| x.replace("~1", "/").replace("~0", "~"))
        .peekable();
    while let Some(token) = components.next() {
        // This isn't the last component, so navigate deeper
        if components.peek().is_some() {
            match value {
                Value::Object(map) => match map.entry(token.clone()) {
                    // Move inside an object if the current pointer component is a valid key
                    Entry::Occupied(occupied) => value = occupied.into_mut(),
                    Entry::Vacant(_) => {
                        tracing::warn!("Skipping deletion of extra_body pointer `{pointer}` - parent of pointer doesn't exist");
                        // If a parent of our target pointer doesn't exist, then do nothing,
                        // since `value`` is already an object where the target pointer doesn't exist
                        return Ok(());
                    }
                },
                Value::Array(list) => {
                    match parse_index(&token) {
                        Some(index) => {
                            if let Some(target) = list.get_mut(index) {
                                value = target;
                            } else {
                                tracing::warn!("Skipping deletion of extra_body pointer `{pointer}` - index `{token}` out of bounds");
                                // If a parent of our target pointer doesn't exist, then do nothing,
                                // since `value`` is already an object where the target pointer doesn't exist
                                return Ok(());
                            }
                        }
                        None => {
                            tracing::warn!("Skipping deletion of extra_body pointer `{pointer}` - non-numeric array index `{token}`");
                            // If a parent of our target pointer doesn't exist, then do nothing,
                            // since `value`` is already an object where the target pointer doesn't exist
                            return Ok(());
                        }
                    }
                }
                other => {
                    tracing::warn!("Skipping deletion of extra_body pointer `{pointer}` - found non array/object target {other}");
                    return Ok(());
                }
            }
        } else {
            match value {
                Value::Object(map) => {
                    if map.remove(&token).is_none() {
                        tracing::warn!("Skipping deletion of extra_body pointer `{pointer}` - key `{token}` doesn't exist");
                    }
                }
                Value::Array(list) => match parse_index(&token) {
                    Some(index) => {
                        if index < list.len() {
                            list.remove(index);
                        } else {
                            tracing::warn!("Skipping deletion of extra_body pointer `{pointer}` - index `{token}` out of bounds");
                        }
                    }
                    None => {
                        tracing::warn!("Skipping deletion of extra_body pointer `{pointer}` - non-numeric array index `{token}`");
                    }
                },
                other => {
                    tracing::warn!("Skipping deletion of extra_body pointer `{pointer}` - found non array/object target {other}");
                    return Ok(());
                }
            }
            return Ok(());
        }
    }
    Ok(())
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

/// For providers that return the tool call name in every chunk, we only want to send the name when it changes.
/// Gemini & Mistral do this.
/// If the tool call name changes, we return the new name.
/// We also update the last_tool_name to the new name.
/// If the tool call name does not change, we return None.
/// We do not update the last_tool_name in this case.
pub(crate) fn check_new_tool_call_name(
    new_name: String,
    last_tool_name: &mut Option<String>,
) -> Option<String> {
    match last_tool_name {
        None => {
            *last_tool_name = Some(new_name.to_string());
            Some(new_name)
        }
        Some(last_tool_name) => {
            if last_tool_name == &new_name {
                // If the previous tool name was the same as the old name, we can just return None as it will have already been sent
                return None;
            }
            *last_tool_name = new_name.clone();
            Some(new_name)
        }
    }
}

pub trait TensorZeroRequestBuilderExt {}

impl TensorZeroRequestBuilderExt for TensorzeroRequestBuilder<'_> {}

pub trait UrlParseErrExt<T> {
    fn convert_parse_error(self) -> Result<T, Error>;
}

impl<T> UrlParseErrExt<T> for Result<T, url::ParseError> {
    fn convert_parse_error(self) -> Result<T, Error> {
        self.map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Error parsing URL: {}. {IMPOSSIBLE_ERROR_MESSAGE}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;

    use crate::inference::types::{
        extra_body::{ExtraBodyConfig, ExtraBodyReplacement, FilteredInferenceExtraBody},
        extra_headers::{DynamicExtraHeader, ExtraHeadersConfig, FilteredInferenceExtraHeaders},
        ContentBlockChunk, TextChunk,
    };
    use futures::{stream, StreamExt};

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
            finish_reason: None,
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
        inject_extra_request_data(
            &Default::default(),
            &Default::default(),
            ModelProviderRequestInfo {
                provider_name: "dummy_provider".into(),
                extra_body: Default::default(),
                extra_headers: None,
            },
            "dummy_model",
            &mut body,
        )
        .unwrap();
        assert_eq!(body, serde_json::json!({}));
    }

    #[test]
    fn test_inject_no_matches() {
        let mut body = serde_json::json!({});
        let headers = inject_extra_request_data(
            &FullExtraBodyConfig {
                extra_body: Some(ExtraBodyConfig { data: vec![] }),
                inference_extra_body: FilteredInferenceExtraBody {
                    #[expect(deprecated)]
                    data: vec![DynamicExtraBody::Provider {
                        model_provider_name: "wrong_provider".to_string(),
                        pointer: "/my_key".to_string(),
                        value: Value::String("My Value".to_string()),
                    }],
                },
            },
            &FullExtraHeadersConfig {
                variant_extra_headers: Some(ExtraHeadersConfig { data: vec![] }),
                inference_extra_headers: FilteredInferenceExtraHeaders {
                    #[expect(deprecated)]
                    data: vec![DynamicExtraHeader::Provider {
                        model_provider_name: "wrong_provider".to_string(),
                        name: "X-My-Header".to_string(),
                        value: "My Value".to_string(),
                    }],
                },
            },
            ModelProviderRequestInfo {
                extra_headers: None,
                provider_name: "dummy_provider".into(),
                extra_body: Default::default(),
            },
            "dummy_model",
            &mut body,
        )
        .unwrap();
        assert_eq!(body, serde_json::json!({}));
        assert_eq!(headers.len(), 0);
    }

    #[test]
    fn test_inject_to_non_map() {
        let err = inject_extra_request_data(
            &Default::default(),
            &Default::default(),
            ModelProviderRequestInfo {
                provider_name: "dummy_provider".into(),
                extra_body: Default::default(),
                extra_headers: None,
            },
            "dummy_model",
            &mut "test".into(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("Body is not a map"),
            "Unexpected error message: {err:?}"
        );
    }

    #[test]
    fn test_inject_headers() {
        let headers = inject_extra_request_data(
            &Default::default(),
            &FullExtraHeadersConfig {
                variant_extra_headers: Some(ExtraHeadersConfig {
                    data: vec![
                        ExtraHeader {
                            name: "X-My-Overridden".to_string(),
                            kind: ExtraHeaderKind::Value("My variant value".to_string()),
                        },
                        ExtraHeader {
                            name: "X-My-Overridden-Inference".to_string(),
                            kind: ExtraHeaderKind::Value("My variant value".to_string()),
                        },
                        ExtraHeader {
                            name: "X-My-Variant".to_string(),
                            kind: ExtraHeaderKind::Value("My variant header".to_string()),
                        },
                        ExtraHeader {
                            name: "X-My-Delete".to_string(),
                            kind: ExtraHeaderKind::Value("Should be deleted".to_string()),
                        },
                        ExtraHeader {
                            name: "X-My-Delete-Inference".to_string(),
                            kind: ExtraHeaderKind::Delete,
                        },
                    ],
                }),
                inference_extra_headers: FilteredInferenceExtraHeaders {
                    #[expect(deprecated)]
                    data: vec![
                        DynamicExtraHeader::Provider {
                            model_provider_name:
                                "tensorzero::model_name::dummy_model::provider_name::dummy_provider"
                                    .to_string(),
                            name: "X-My-Inference".to_string(),
                            value: "My inference header value".to_string(),
                        },
                        DynamicExtraHeader::Variant {
                            variant_name: "dummy_variant".to_string(),
                            name: "X-My-Overridden-Inference".to_string(),
                            value: "My inference value".to_string(),
                        },
                    ],
                },
            },
            ModelProviderRequestInfo {
                provider_name: "dummy_provider".into(),
                extra_body: Default::default(),
                extra_headers: Some(ExtraHeadersConfig {
                    data: vec![
                        ExtraHeader {
                            name: "X-My-Overridden".to_string(),
                            kind: ExtraHeaderKind::Value("My model provider value".to_string()),
                        },
                        ExtraHeader {
                            name: "X-My-Overridden-Inference".to_string(),
                            kind: ExtraHeaderKind::Value("My model provider value".to_string()),
                        },
                        ExtraHeader {
                            name: "X-My-ModelProvider".to_string(),
                            kind: ExtraHeaderKind::Value("My model provider header".to_string()),
                        },
                    ],
                }),
            },
            "dummy_model",
            &mut serde_json::json!({}),
        )
        .unwrap();
        assert_eq!(
            headers.get("X-My-Overridden").unwrap(),
            "My model provider value"
        );
        assert_eq!(headers.get("X-My-Variant").unwrap(), "My variant header");
        assert_eq!(
            headers.get("X-My-ModelProvider").unwrap(),
            "My model provider header"
        );
        assert_eq!(
            headers.get("X-My-Inference").unwrap(),
            "My inference header value"
        );
        assert_eq!(
            headers.get("X-My-Overridden-Inference").unwrap(),
            "My inference value"
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
        inject_extra_request_data(
            &FullExtraBodyConfig {
                extra_body: Some(ExtraBodyConfig {
                    data: vec![
                        ExtraBodyReplacement {
                            pointer: "/generationConfig".to_string(),
                            kind: ExtraBodyReplacementKind::Value(serde_json::json!({
                                "otherNestedKey": "otherNestedValue"
                            })),
                        },
                        ExtraBodyReplacement {
                            pointer: "/generationConfig/temperature".to_string(),
                            kind: ExtraBodyReplacementKind::Value(serde_json::json!(0.123)),
                        },
                    ],
                }),
                inference_extra_body: FilteredInferenceExtraBody {
                    #[expect(deprecated)]
                    data: vec![DynamicExtraBody::Provider {
                        model_provider_name:
                            "tensorzero::model_name::dummy_model::provider_name::dummy_provider"
                                .to_string(),
                        pointer: "/generationConfig/valueFromInference".to_string(),
                        value: Value::String("inferenceValue".to_string()),
                    }],
                },
            },
            &Default::default(),
            ModelProviderRequestInfo {
                provider_name: "dummy_provider".into(),
                extra_body: Default::default(),
                extra_headers: None,
            },
            "dummy_model",
            &mut body,
        )
        .unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "otherKey": "otherValue",
                "generationConfig": {
                    "otherNestedKey": "otherNestedValue",
                    "temperature": 0.123,
                    "valueFromInference": "inferenceValue"
                }
            })
        );
    }

    // Tests that we inject fields in the correct order when `extra_body`
    // is set at both the variant and model provider level,
    // and `inference_extra_body` is provided.
    // The correct priority is inference -> model provider -> variant.
    #[test]
    fn test_inject_all() {
        let mut body = serde_json::json!({
            "otherKey": "otherValue",
            "generationConfig": {
                "temperature": 123
            },
            "delete_me": {"some": "value"}
        });
        inject_extra_request_data(
            &FullExtraBodyConfig {
                extra_body: Some(ExtraBodyConfig {
                    data: vec![
                        ExtraBodyReplacement {
                            pointer: "/generationConfig/otherNestedKey".to_string(),
                            kind: ExtraBodyReplacementKind::Value(Value::String(
                                "otherNestedValue".to_string(),
                            )),
                        },
                        ExtraBodyReplacement {
                            pointer: "/variantKey".to_string(),
                            kind: ExtraBodyReplacementKind::Value(Value::String(
                                "variantValue".to_string(),
                            )),
                        },
                        ExtraBodyReplacement {
                            pointer: "/multiOverride".to_string(),
                            kind: ExtraBodyReplacementKind::Value(Value::String(
                                "from variant".to_string(),
                            )),
                        },
                        ExtraBodyReplacement {
                            pointer: "/delete_me".to_string(),
                            kind: ExtraBodyReplacementKind::Delete,
                        },
                    ],
                }),
                inference_extra_body: FilteredInferenceExtraBody {
                    #[expect(deprecated)]
                    data: vec![DynamicExtraBody::Provider {
                        model_provider_name:
                            "tensorzero::model_name::dummy_model::provider_name::dummy_provider"
                                .to_string(),
                        pointer: "/multiOverride".to_string(),
                        value: Value::String("from inference".to_string()),
                    }],
                },
            },
            &Default::default(),
            ModelProviderRequestInfo {
                provider_name: "dummy_provider".into(),
                extra_body: Some(ExtraBodyConfig {
                    data: vec![
                        ExtraBodyReplacement {
                            pointer: "/variantKey".to_string(),
                            kind: ExtraBodyReplacementKind::Value(Value::String(
                                "modelProviderOverride".to_string(),
                            )),
                        },
                        ExtraBodyReplacement {
                            pointer: "/modelProviderKey".to_string(),
                            kind: ExtraBodyReplacementKind::Value(Value::String(
                                "modelProviderValue".to_string(),
                            )),
                        },
                        ExtraBodyReplacement {
                            pointer: "/multiOverride".to_string(),
                            kind: ExtraBodyReplacementKind::Value(Value::String(
                                "from model provider".to_string(),
                            )),
                        },
                    ],
                }),
                extra_headers: None,
            },
            "dummy_model",
            &mut body,
        )
        .unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "otherKey": "otherValue",
                "modelProviderKey": "modelProviderValue",
                "variantKey": "modelProviderOverride",
                "multiOverride": "from inference",
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

    #[test]
    fn test_delete_json_pointer_simple() {
        let mut obj = serde_json::json!({
            "object1": "value1",
            "my_array": ["value1", true],
            "object2": {
                "key1": "value1",
            }
        });
        delete_json_pointer(&mut obj, "/object1").unwrap();
        assert_eq!(
            obj,
            serde_json::json!({
                "my_array": ["value1", true],
                "object2": {
                    "key1": "value1",
                }
            })
        );

        delete_json_pointer(&mut obj, "/object2/key1").unwrap();
        assert_eq!(
            obj,
            serde_json::json!({
                "my_array": ["value1", true],
                "object2": {
                }
            })
        );

        delete_json_pointer(&mut obj, "/object2").unwrap();
        assert_eq!(
            obj,
            serde_json::json!({
                "my_array": ["value1", true],
            })
        );

        delete_json_pointer(&mut obj, "/my_array/0").unwrap();
        assert_eq!(
            obj,
            serde_json::json!({
                "my_array": [true],
            })
        );

        delete_json_pointer(&mut obj, "/my_array").unwrap();
        assert_eq!(obj, serde_json::json!({}));
    }

    #[test]
    fn test_delete_json_pointer_errors() {
        let logs_contain = crate::utils::testing::capture_logs();
        let mut obj = serde_json::json!({"other": "value"});
        delete_json_pointer(&mut obj, "/object1").unwrap();
        assert!(logs_contain(
            "Skipping deletion of extra_body pointer `/object1` - key `object1` doesn't exist"
        ));
        assert_eq!(obj, serde_json::json!({"other": "value"}));

        let mut obj = serde_json::json!({
            "my_array": ["value1", true],
        });
        delete_json_pointer(&mut obj, "/my_array/2").unwrap();
        assert!(logs_contain(
            "Skipping deletion of extra_body pointer `/my_array/2` - index `2` out of bounds"
        ));
        assert_eq!(
            obj,
            serde_json::json!({
                "my_array": ["value1", true],
            })
        );

        delete_json_pointer(&mut obj, "/fake/pointer").unwrap();
        assert!(logs_contain(
            "Skipping deletion of extra_body pointer `/fake/pointer` - parent of pointer doesn't exist"
        ));
        assert_eq!(
            obj,
            serde_json::json!({
                "my_array": ["value1", true],
            })
        );

        delete_json_pointer(&mut obj, "/my_array/non-int-index").unwrap();
        assert!(logs_contain(
            "Skipping deletion of extra_body pointer `/my_array/non-int-index` - non-numeric array index `non-int-index`"
        ));
        assert_eq!(
            obj,
            serde_json::json!({
                "my_array": ["value1", true],
            })
        );

        delete_json_pointer(&mut obj, "/my_array/0/bad-index").unwrap();
        assert!(logs_contain(
            "Skipping deletion of extra_body pointer `/my_array/0/bad-index` - found non array/object target \"value1\""
        ));
        assert_eq!(
            obj,
            serde_json::json!({
                "my_array": ["value1", true],
            })
        );
    }

    #[test]
    fn test_check_new_tool_call_name() {
        let mut last_tool_name = None;
        assert_eq!(
            check_new_tool_call_name("get_temperature".to_string(), &mut last_tool_name),
            Some("get_temperature".to_string())
        );
        assert_eq!(
            check_new_tool_call_name("get_temperature".to_string(), &mut last_tool_name),
            None
        );
        assert_eq!(
            check_new_tool_call_name("get_humidity".to_string(), &mut last_tool_name),
            Some("get_humidity".to_string())
        );
        assert_eq!(
            check_new_tool_call_name("get_temperature".to_string(), &mut last_tool_name),
            Some("get_temperature".to_string())
        );
    }

    #[test]
    fn test_inject_extra_body_model_provider_without_shorthand() {
        let mut body = serde_json::json!({});
        let config = FullExtraBodyConfig {
            extra_body: None,
            inference_extra_body: FilteredInferenceExtraBody {
                data: vec![DynamicExtraBody::ModelProvider {
                    model_name: "gpt-4o".to_string(), // NOT using shorthand
                    provider_name: Some("openai".to_string()),
                    pointer: "/test_no_shorthand".to_string(),
                    value: json!(99),
                }],
            },
        };
        let model_provider = ModelProviderRequestInfo {
            provider_name: "openai".into(),
            extra_headers: None,
            extra_body: None,
        };

        inject_extra_request_data(
            &config,
            &Default::default(),
            model_provider,
            "gpt-4o",
            &mut body,
        )
        .unwrap();

        // Should have applied the filter
        assert_eq!(body.get("test_no_shorthand").unwrap(), &json!(99));
    }

    #[test]
    fn test_inject_extra_body_model_provider_wrong_prefix() {
        let mut body = serde_json::json!({});
        let config = FullExtraBodyConfig {
            extra_body: None,
            inference_extra_body: FilteredInferenceExtraBody {
                data: vec![DynamicExtraBody::ModelProvider {
                    model_name: "anthropic::claude-3".to_string(), // Wrong prefix
                    provider_name: Some("openai".to_string()),
                    pointer: "/test_wrong".to_string(),
                    value: json!(1),
                }],
            },
        };
        let model_provider = ModelProviderRequestInfo {
            provider_name: "openai".into(),
            extra_headers: None,
            extra_body: None,
        };

        inject_extra_request_data(
            &config,
            &Default::default(),
            model_provider,
            "gpt-4o",
            &mut body,
        )
        .unwrap();

        // Should NOT have applied the filter
        assert!(!body.as_object().unwrap().contains_key("test_wrong"));
    }

    #[test]
    fn test_inject_extra_body_model_provider_external_model_with_colons() {
        let mut body = serde_json::json!({});
        let config = FullExtraBodyConfig {
            extra_body: None,
            inference_extra_body: FilteredInferenceExtraBody {
                data: vec![DynamicExtraBody::ModelProvider {
                    // External model with :: but not a known prefix
                    model_name: "custom::deployment::model".to_string(),
                    provider_name: Some("custom".to_string()),
                    pointer: "/test_external".to_string(),
                    value: json!(7),
                }],
            },
        };
        let model_provider = ModelProviderRequestInfo {
            provider_name: "custom".into(),
            extra_headers: None,
            extra_body: None,
        };

        inject_extra_request_data(
            &config,
            &Default::default(),
            model_provider,
            "custom::deployment::model", // Full name without stripping
            &mut body,
        )
        .unwrap();

        // Should have applied the filter (matched exactly)
        assert_eq!(body.get("test_external").unwrap(), &json!(7));
    }

    #[test]
    fn test_inject_extra_headers_model_provider_without_shorthand() {
        let mut body = serde_json::json!({});
        let config = FullExtraBodyConfig::default();
        let headers_config = FullExtraHeadersConfig {
            variant_extra_headers: None,
            inference_extra_headers: FilteredInferenceExtraHeaders {
                data: vec![DynamicExtraHeader::ModelProvider {
                    model_name: "gpt-4o".to_string(), // NOT using shorthand
                    provider_name: Some("openai".to_string()),
                    name: "X-Custom-Header-2".to_string(),
                    value: "test-value-2".to_string(),
                }],
            },
        };
        let model_provider = ModelProviderRequestInfo {
            provider_name: "openai".into(),
            extra_headers: None,
            extra_body: None,
        };

        let headers = inject_extra_request_data(
            &config,
            &headers_config,
            model_provider,
            "gpt-4o",
            &mut body,
        )
        .unwrap();

        // Should have applied the header
        assert_eq!(
            headers.get("X-Custom-Header-2").unwrap().to_str().unwrap(),
            "test-value-2"
        );
    }
}
