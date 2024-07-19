#![forbid(unsafe_code)]

use axum::routing::{get, post};
use axum::Router;

mod api; // utilities for API
mod config_parser; // TensorZero config file
mod endpoints; // API endpoints
mod error; // error handling
mod function; // types and methods for working with TensorZero functions
mod jsonschema_util; // utilities for working with JSON schemas

#[tokio::main]
async fn main() {
    let router = Router::new()
        .route("/inference", post(endpoints::inference::inference_handler))
        .route("/status", get(endpoints::status::status_handler))
        .with_state(api::AppStateData::new());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("Failed to bind to port 3000");

    axum::serve(listener, router)
        .await
        .expect("Failed to start server")
}
