//! OpenAI-compatible API endpoints.
//!
//! This module provides compatibility with OpenAI's API format, supporting both
//! chat completions and embeddings endpoints. It handles routing, request/response
//! conversion, and provides the main entry points for OpenAI-compatible requests.

pub mod chat_completions;
pub mod embeddings;
pub mod types;

use chat_completions::chat_completions_handler;
use embeddings::embeddings_handler;

use axum::routing::post;
use axum::Router;

use crate::endpoints::RouteHandlers;
use crate::utils::gateway::AppStateData;

/// Constructs (but does not register) all of our OpenAI-compatible endpoints.
/// The `RouterExt::register_openai_compatible_routes` is a convenience method
/// to register all of the routes on a router.
///
/// Alternatively, the returned `RouteHandlers` can be inspected (e.g. to allow middleware to see the route paths)
/// and then manually registered on a router.
pub fn build_openai_compatible_routes() -> RouteHandlers {
    RouteHandlers {
        routes: vec![
            (
                "/openai/v1/chat/completions",
                post(chat_completions_handler),
            ),
            ("/openai/v1/embeddings", post(embeddings_handler)),
        ],
    }
}

pub trait RouterExt {
    /// Applies our OpenAI-compatible endpoints to the router.
    /// This is used by the the gateway for the patched OpenAI python client (`start_openai_compatible_gateway`),
    /// as well as the normal standalone TensorZero gateway.
    fn register_openai_compatible_routes(self) -> Self;
}

impl RouterExt for Router<AppStateData> {
    fn register_openai_compatible_routes(mut self) -> Self {
        for (path, handler) in build_openai_compatible_routes().routes {
            self = self.route(path, handler);
        }
        self
    }
}
