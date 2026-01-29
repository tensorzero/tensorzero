use futures::{Stream, StreamExt};
use http_body::Body;
use sse_stream::{Sse, SseStream};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReqwestSseStreamError {
    #[error("Reqwest error: {0}")]
    ReqwestError(reqwest::Error),
    #[error("Expected content-type 'text/event-stream', got {0:?}")]
    InvalidContentType(Option<http::header::HeaderValue>),
    #[error("SSE error: {0}")]
    SseError(sse_stream::Error),
}

/// Mimics the api of `reqwest_eventsource::Event`
pub enum Event {
    Open,
    Message(Sse),
}

//// Mimics the api of `reqwest_eventsource::RequestBuilderExt`
pub trait RequestBuilderExt {
    fn eventsource(self) -> impl Stream<Item = Result<Event, ReqwestSseStreamError>>;
}

async fn start_stream(
    builder: reqwest::RequestBuilder,
) -> Result<SseStream<impl Body<Error = reqwest::Error>>, ReqwestSseStreamError> {
    let response = builder
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .send()
        .await
        .map_err(ReqwestSseStreamError::ReqwestError)?
        .error_for_status()
        .map_err(ReqwestSseStreamError::ReqwestError)?;

    if let Some(content_type) = response.headers().get(reqwest::header::CONTENT_TYPE) {
        if content_type
            .to_str()
            .ok()
            .is_none_or(|s| !s.eq_ignore_ascii_case("text/event-stream"))
        {
            return Err(ReqwestSseStreamError::InvalidContentType(Some(
                content_type.clone(),
            )));
        }
    } else {
        return Err(ReqwestSseStreamError::InvalidContentType(None));
    }
    Ok(SseStream::from_byte_stream(response.bytes_stream()))
}

impl RequestBuilderExt for reqwest::RequestBuilder {
    fn eventsource(self) -> impl Stream<Item = Result<Event, ReqwestSseStreamError>> {
        async_stream::stream! {
            match start_stream(self).await {
                Ok(mut sse_stream) => {
                    yield Ok(Event::Open);
                    while let Some(event) = sse_stream.next().await {
                        match event {
                            Ok(sse) => yield Ok(Event::Message(sse)),
                            Err(e) => yield Err(ReqwestSseStreamError::SseError(e)),
                        }
                    }
                }
                Err(e) => yield Err(e),
            }
        }
    }
}
