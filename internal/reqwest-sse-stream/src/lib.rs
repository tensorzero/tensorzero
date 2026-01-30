use futures::{Stream, StreamExt};
use http_body::Body;
use thiserror::Error;

pub use sse_stream::{Sse, SseStream};

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
pub trait RequestBuilderExt {
    fn eventsource(self) -> impl Stream<Item = Result<Event, ReqwestSseStreamError>>;

    /// Sends the request and returns the event stream along with response headers.
    /// Returns the headers even on error for cases where they are needed for error handling.
    #[expect(clippy::type_complexity)]
    fn eventsource_with_headers(
        self,
    ) -> impl std::future::Future<
        Output = Result<
            (
                impl Stream<Item = Result<Event, ReqwestSseStreamError>>,
                http::HeaderMap,
            ),
            (ReqwestSseStreamError, Option<http::HeaderMap>),
        >,
    >;
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

fn sse_stream_to_event_stream(
    sse_stream: SseStream<impl Body<Error = reqwest::Error>>,
) -> impl Stream<Item = Result<Event, ReqwestSseStreamError>> {
    futures::stream::once(async { Ok(Event::Open) }).chain(sse_stream.filter_map(|event| async {
        match event {
            Ok(sse) => {
                // Only yield Message events when data is present
                MessageEvent::from_sse(sse).map(|message| Ok(Event::Message(message)))
            }
            Err(e) => Some(Err(ReqwestSseStreamError::SseError(e))),
        }
    }))
}

/// Sends the request and returns a stream of raw SSE events.
/// This is useful when you want to handle SSE events directly without
/// the `Event::Open`/`Event::Message` wrapper.
pub async fn into_sse_stream(
    builder: reqwest::RequestBuilder,
) -> Result<impl Stream<Item = Result<Sse, ReqwestSseStreamError>>, ReqwestSseStreamError> {
    let (sse_stream, _headers) = start_stream_with_headers(builder)
        .await
        .map_err(|(e, _headers)| e)?;
    Ok(sse_stream.map(|r| r.map_err(ReqwestSseStreamError::SseError)))
}

impl RequestBuilderExt for reqwest::RequestBuilder {
    fn eventsource(self) -> impl Stream<Item = Result<Event, ReqwestSseStreamError>> {
        async_stream::stream! {
            match self.eventsource_with_headers().await {
                Ok((event_stream, _headers)) => {
                    let mut event_stream = std::pin::pin!(event_stream);
                    while let Some(event) = event_stream.next().await {
                        yield event;
                    }
                }
                Err((e, _headers)) => yield Err(e),
            }
        }
    }

    async fn eventsource_with_headers(
        self,
    ) -> Result<
        (
            impl Stream<Item = Result<Event, ReqwestSseStreamError>>,
            http::HeaderMap,
        ),
        (ReqwestSseStreamError, Option<http::HeaderMap>),
    > {
        let (sse_stream, headers) = start_stream_with_headers(self).await?;
        Ok((sse_stream_to_event_stream(sse_stream), headers))
    }
}
