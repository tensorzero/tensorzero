use std::pin::Pin;

use futures::{Stream, StreamExt};
use http_body::Body;
use thiserror::Error;

pub use sse_stream::{Sse, SseStream};

/// A boxed event stream that is `Unpin` and `Send`.
pub type EventStream =
    Pin<Box<dyn Stream<Item = Result<Event, ReqwestSseStreamError>> + Send + 'static>>;

#[derive(Debug, Error)]
pub enum ReqwestSseStreamError {
    #[error("Reqwest error: {0}")]
    ReqwestError(reqwest::Error),
    #[error("Invalid status code: {0}")]
    InvalidStatusCode(reqwest::StatusCode, reqwest::Response),
    #[error("Expected content-type 'text/event-stream', got {0:?}")]
    InvalidContentType(http::header::HeaderValue, reqwest::Response),
    #[error("SSE error: {0}")]
    SseError(sse_stream::Error),
}

/// An SSE message event with guaranteed data field.
/// This mimics the API of `eventsource_stream::Event` (used by `reqwest_eventsource`),
/// where `data` is a non-optional `String`.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageEvent {
    pub event: String,
    pub data: String,
    pub id: String,
}

impl MessageEvent {
    /// Creates a MessageEvent from an Sse event if data is present.
    /// Returns None if the Sse event has no data field.
    fn from_sse(sse: Sse) -> Option<Self> {
        // Only convert if data is present - events without data are not meaningful messages
        sse.data.map(|data| MessageEvent {
            event: sse.event.unwrap_or_default(),
            data,
            id: sse.id.unwrap_or_default(),
        })
    }
}

/// Mimics the api of `reqwest_eventsource::Event`
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Open,
    Message(MessageEvent),
}

/// Mimics the api of `reqwest_eventsource::RequestBuilderExt`
#[expect(async_fn_in_trait)]
pub trait RequestBuilderExt {
    async fn eventsource(self) -> Result<EventStream, ReqwestSseStreamError>;

    /// Sends the request and returns the event stream along with response headers.
    /// Returns the headers even on error for cases where they are needed for error handling.
    async fn eventsource_with_headers(
        self,
    ) -> Result<(EventStream, http::HeaderMap), (ReqwestSseStreamError, Option<http::HeaderMap>)>;
}

async fn start_stream_with_headers(
    builder: reqwest::RequestBuilder,
) -> Result<
    (
        SseStream<impl Body<Error = reqwest::Error>>,
        http::HeaderMap,
    ),
    (ReqwestSseStreamError, Option<http::HeaderMap>),
> {
    let response = builder
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .send()
        .await
        .map_err(|e| (ReqwestSseStreamError::ReqwestError(e), None))?;

    let headers = response.headers().clone();
    let status = response.status();

    if !status.is_success() {
        return Err((
            ReqwestSseStreamError::InvalidStatusCode(status, response),
            Some(headers),
        ));
    }

    if let Some(content_type) = response.headers().get(reqwest::header::CONTENT_TYPE) {
        let content_type_valid = content_type
            .to_str()
            .ok()
            .is_some_and(|s| s.to_ascii_lowercase().starts_with("text/event-stream"));
        if !content_type_valid {
            return Err((
                ReqwestSseStreamError::InvalidContentType(content_type.clone(), response),
                Some(headers),
            ));
        }
    } else {
        return Err((
            ReqwestSseStreamError::InvalidContentType(
                http::header::HeaderValue::from_static(""),
                response,
            ),
            Some(headers),
        ));
    }
    Ok((
        SseStream::from_byte_stream(response.bytes_stream()),
        headers,
    ))
}

impl RequestBuilderExt for reqwest::RequestBuilder {
    async fn eventsource(self) -> Result<EventStream, ReqwestSseStreamError> {
        match self.eventsource_with_headers().await {
            Ok((event_stream, _headers)) => Ok(event_stream),
            Err((e, _headers)) => Err(e),
        }
    }

    async fn eventsource_with_headers(
        self,
    ) -> Result<(EventStream, http::HeaderMap), (ReqwestSseStreamError, Option<http::HeaderMap>)>
    {
        match start_stream_with_headers(self).await {
            Ok((mut sse_stream, headers)) => Ok((
                Box::pin(async_stream::stream! {
                    while let Some(event) = sse_stream.next().await {
                        match event {
                            Ok(sse) => {
                                if let Some(message_event) = MessageEvent::from_sse(sse) {
                                    yield Ok(Event::Message(message_event));
                                }
                            }
                            Err(e) => {
                                yield Err(ReqwestSseStreamError::SseError(e));
                            }
                        }
                    }
                }),
                headers,
            )),
            // For backwards compatibility with our existing stream consumers, turn connection errors into a stream
            // with a single error event.
            Err((e, headers)) => Ok((
                Box::pin(futures::stream::once(async { Err(e) })),
                headers.unwrap_or_default(),
            )),
        }
    }
}
