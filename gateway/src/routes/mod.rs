//! Route definitions and endpoint mappings for the TensorZero Gateway API.

mod external;
mod internal;

use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use tensorzero_core::observability::{RouterExt as _, TracerWrapper};
use tensorzero_core::utils::gateway::AppStateData;

pub fn build_api_routes(
    otel_tracer: Option<Arc<TracerWrapper>>,
    metrics_handle: PrometheusHandle,
) -> Router<AppStateData> {
    let (otel_enabled_routes, otel_enabled_router) = external::build_otel_enabled_routes();
    Router::new()
        .merge(otel_enabled_router)
        .merge(external::build_non_otel_enabled_routes(metrics_handle))
        .merge(internal::build_internal_routes())
        .apply_top_level_otel_http_trace_layer(otel_tracer, otel_enabled_routes)
}
