#![forbid(unsafe_code)]

use axum::routing::{get, post};
use axum::Router;

use ::api::api;
use ::api::endpoints;

#[tokio::main]
async fn main() {
    let router = Router::new()
        .route("/inference", post(endpoints::inference::inference_handler))
        .route("/status", get(endpoints::status::status_handler))
        .with_state(api::AppStateData::default());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("Failed to bind to port 3000");

    axum::serve(listener, router)
        .await
        .expect("Failed to start server")
}
