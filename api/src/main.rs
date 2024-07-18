#![forbid(unsafe_code)]

use axum::response::IntoResponse;
use axum::routing::get;
use axum::{debug_handler, Router};

mod error;
mod inference;
mod status;

#[tokio::main]
async fn main() {
    let router = Router::new()
        .route("/", get(hello_world))
        .route("/status", get(status::status_handler));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("Failed to bind to port 3000");

    axum::serve(listener, router)
        .await
        .expect("Failed to start server")
}

#[debug_handler]
async fn hello_world() -> impl IntoResponse {
    "HELL0 W0RLD"
}
