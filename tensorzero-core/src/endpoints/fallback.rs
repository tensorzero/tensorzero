use crate::error::{AxumResponseError, Error, ErrorDetails};
use crate::utils::gateway::AppState;
use axum::{
    body::Body,
    http::Request,
    response::{IntoResponse, Response},
};

pub async fn handle_404(app_state: AppState, req: Request<Body>) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().to_string();

    AxumResponseError::new(
        Error::new(ErrorDetails::RouteNotFound { path, method }),
        app_state.0,
    )
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::testing::get_unit_test_gateway_handle;
    use axum::{
        body::to_bytes,
        extract::State,
        http::{Method, StatusCode, Uri},
    };
    use serde_json::Value;
    use std::sync::Arc;

    fn create_test_app_state() -> AppState {
        let config = Arc::new(Config::default());
        let gateway_handle = get_unit_test_gateway_handle(config);
        State(gateway_handle.app_state.clone())
    }

    #[tokio::test]
    async fn test_handle_404_get() {
        let req = Request::builder()
            .method(Method::GET)
            .uri(Uri::from_static("/unknown/path"))
            .body(Body::empty())
            .unwrap();

        let app_state = create_test_app_state();
        let response = handle_404(app_state, req).await;

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

        let app_state = create_test_app_state();
        let response = handle_404(app_state, req).await;

        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body_bytes = to_bytes(response.into_body(), 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&body_bytes).unwrap();

        let error_msg = body.get("error").and_then(Value::as_str).unwrap();
        assert!(error_msg.contains("POST"));
        assert!(error_msg.contains("/unknown/path"));
    }
}
