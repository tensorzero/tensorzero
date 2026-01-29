use http_body::Body;
use sse_stream::SseStream;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReqwestSseStreamError {
    #[error("Reqwest error: {0}")]
    ReqwestError(reqwest::Error),
    #[error("Expected content-type 'text/event-stream', got {0:?}")]
    InvalidContentType(Option<http::header::HeaderValue>),
}

pub async fn into_sse_stream(
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
