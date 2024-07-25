#![forbid(unsafe_code)]

use axum::routing::{get, post};
use axum::Router;

use api::api_util;
use api::endpoints;

#[tokio::main]
async fn main() {
    let router = Router::new()
        .route("/inference", post(endpoints::inference::inference_handler))
        .route("/feedback", post(endpoints::feedback::feedback_handler))
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
        .with_state(api_util::AppStateData::default());

    // TODO: allow the user to configure the port
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("Failed to bind to port 3000");

    axum::serve(listener, router)
        .await
        .expect("Failed to start server")
}
