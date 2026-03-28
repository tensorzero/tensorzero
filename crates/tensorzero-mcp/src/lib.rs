use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::Arc;

use axum::middleware;
use rmcp::ServiceExt;
use rmcp::transport::io::stdio;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use tensorzero_auth::middleware::TensorzeroAuthMiddlewareStateInner;
use tensorzero_core::endpoints::TensorzeroAuthMiddlewareState;
use tensorzero_core::utils::gateway::AppStateData;

mod handler;

use handler::TensorZeroMcpServer;

/// Spawn the MCP HTTP server on its own listener.
///
/// The server runs as a background task tracked by the provided `TaskTracker`
/// and shuts down when the `CancellationToken` is cancelled.
///
/// If the gateway has auth enabled, the same Bearer-token middleware is applied
/// to all MCP HTTP requests.
pub async fn spawn_mcp_http_server(
    bind_address: SocketAddr,
    app_state: Arc<AppStateData>,
    deferred_tasks: &TaskTracker,
    shutdown_token: CancellationToken,
) -> Result<(), ExitCode> {
    let ct = shutdown_token.clone();

    // Apply the same auth middleware as the gateway when auth is enabled.
    let auth_enabled = app_state.config.gateway.auth.enabled;
    let auth_state = if auth_enabled {
        Some(TensorzeroAuthMiddlewareState::new(
            TensorzeroAuthMiddlewareStateInner {
                unauthenticated_routes: &[],
                auth_cache: app_state.auth_cache.clone(),
                pool: app_state.postgres_connection_info.get_pool().cloned(),
                error_json: app_state.config.gateway.unstable_error_json,
                base_path: None,
            },
        ))
    } else {
        None
    };

    let service = StreamableHttpService::new(
        move || Ok(TensorZeroMcpServer::new(app_state.clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default().with_cancellation_token(ct.child_token()),
    );

    let mut router = axum::Router::new().nest_service("/mcp", service);

    if let Some(state) = auth_state {
        router = router.layer(middleware::from_fn_with_state(
            state,
            tensorzero_auth::middleware::tensorzero_auth_middleware,
        ));
    }

    let listener = tokio::net::TcpListener::bind(bind_address)
        .await
        .map_err(|e| {
            tracing::error!("Failed to bind MCP server to {bind_address}: {e}");
            ExitCode::FAILURE
        })?;

    let actual_address = listener.local_addr().map_err(|e| {
        tracing::error!("Failed to get MCP server bind address: {e}");
        ExitCode::FAILURE
    })?;

    tracing::info!("MCP server listening on {actual_address}");

    deferred_tasks.spawn(async move {
        let result = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_token.cancelled_owned())
            .await;
        if let Err(e) = result {
            tracing::error!("MCP HTTP server error: {e}");
        }
    });

    Ok(())
}

/// Run the MCP server over stdio. This blocks until the client disconnects.
pub async fn run_mcp_stdio(app_state: Arc<AppStateData>) -> Result<(), ExitCode> {
    let server = TensorZeroMcpServer::new(app_state);
    let service = server.serve(stdio()).await.map_err(|e| {
        tracing::error!("Failed to start MCP stdio server: {e}");
        ExitCode::FAILURE
    })?;

    service.waiting().await.map_err(|e| {
        tracing::error!("MCP stdio server error: {e}");
        ExitCode::FAILURE
    })?;

    Ok(())
}
