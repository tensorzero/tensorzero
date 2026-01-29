//! Anthropic-compatible API endpoints.
//!
//! This module provides compatibility with Anthropic's Messages API format.
//! It handles routing, request/response conversion, and provides the main entry
//! points for Anthropic-compatible requests.

pub mod messages;
pub mod models;
pub mod types;

use messages::messages_handler;
use models::models_handler;

use axum::Router;
use axum::routing::{get, post};

use crate::endpoints::RouteHandlers;
use crate::utils::gateway::AppStateData;

/// Constructs (but does not register) all of our Anthropic-compatible endpoints.
/// The `RouterExt::register_anthropic_compatible_routes` is a convenience method
/// to register all of the routes on a router.
///
/// Alternatively, the returned `RouteHandlers` can be inspected (e.g. to allow middleware to see the route paths)
/// and then manually registered on a router.
pub fn build_anthropic_compatible_routes() -> RouteHandlers {
    RouteHandlers {
        routes: vec![
            ("/anthropic/v1/messages", post(messages_handler)),
            ("/anthropic/v1/models", get(models_handler)),
        ],
    }
}

pub trait RouterExt {
    /// Applies our Anthropic-compatible endpoints to the router.
    fn register_anthropic_compatible_routes(self) -> Self;
}

impl RouterExt for Router<AppStateData> {
    fn register_anthropic_compatible_routes(mut self) -> Self {
        for (path, handler) in build_anthropic_compatible_routes().routes {
            self = self.route(path, handler);
        }
        self
    }
}
