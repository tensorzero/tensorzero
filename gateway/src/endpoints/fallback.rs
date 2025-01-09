use crate::error::{Error, ErrorDetails};
use axum::{
    body::Body,
    http::Request,
    response::{IntoResponse, Response},
};

pub async fn handle_404(req: Request<Body>) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().to_string();

    Error::new(ErrorDetails::RouteNotFound { path, method }).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::to_bytes,
        http::{Method, StatusCode, Uri},
    };
    use serde_json::Value;

    #[tokio::test]
    async fn test_handle_404_get() {
        let req = Request::builder()
            .method(Method::GET)
            .uri(Uri::from_static("/unknown/path"))
            .body(Body::empty())
            .unwrap();

        let response = handle_404(req).await;

        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body_bytes = to_bytes(response.into_body(), 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&body_bytes).unwrap();

        let error_msg = body.get("error").and_then(Value::as_str).unwrap();
        assert!(error_msg.contains("GET"));
        assert!(error_msg.contains("/unknown/path"));
    }

    #[tokio::test]
    async fn test_handle_404_post() {
        let json_body = serde_json::json!({
            "message": "Hello world",
            "number": 42,
            "active": true
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri(Uri::from_static("/unknown/path"))
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&json_body).unwrap()))
            .unwrap();

        let response = handle_404(req).await;

        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body_bytes = to_bytes(response.into_body(), 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&body_bytes).unwrap();

        let error_msg = body.get("error").and_then(Value::as_str).unwrap();
        assert!(error_msg.contains("POST"));
        assert!(error_msg.contains("/unknown/path"));
    }
}
