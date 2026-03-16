//! Route definitions and endpoint mappings for the TensorZero Gateway API.

mod action;
pub mod evaluations;
mod external;
mod internal;

use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;
use std::sync::Arc;
use tensorzero_core::observability::{RouterExt as _, TracerWrapper};
use tensorzero_core::utils::gateway::AppStateData;
use utoipa::openapi::OpenApi;

struct ApiRoutes {
    external_openapi: OpenApi,
    internal_openapi: OpenApi,
    otel_enabled_routes: tensorzero_core::observability::OtelEnabledRoutes,
    router: Router<AppStateData>,
}

fn build_api_routes_parts(metrics_handle: PrometheusHandle) -> ApiRoutes {
    let (otel_enabled_routes, otel_enabled_router) = external::build_otel_enabled_routes();
    let (otel_enabled_router, otel_enabled_openapi) = otel_enabled_router.split_for_parts();
    let (external_router, external_openapi) =
        external::build_non_otel_enabled_routes(metrics_handle).split_for_parts();
    let (internal_router, internal_openapi) =
        internal::build_internal_non_otel_enabled_routes().split_for_parts();
    let mut external_openapi = external_openapi;
    external_openapi.merge(otel_enabled_openapi);

    ApiRoutes {
        external_openapi,
        internal_openapi,
        otel_enabled_routes,
        router: Router::new()
            .merge(otel_enabled_router)
            .merge(external_router)
            .merge(internal_router),
    }
}

pub fn build_api_routes(
    otel_tracer: Option<Arc<TracerWrapper>>,
    metrics_handle: PrometheusHandle,
) -> Router<AppStateData> {
    let _ = build_external_openapi_spec as fn(PrometheusHandle) -> OpenApi;
    let _ = build_internal_openapi_spec as fn() -> OpenApi;
    let api_routes = build_api_routes_parts(metrics_handle);
    let _ = (&api_routes.external_openapi, &api_routes.internal_openapi);
    api_routes
        .router
        .apply_top_level_otel_http_trace_layer(otel_tracer, api_routes.otel_enabled_routes)
}

pub fn build_external_openapi_spec(metrics_handle: PrometheusHandle) -> OpenApi {
    let api_routes = build_api_routes_parts(metrics_handle);
    let _ = (
        &api_routes.internal_openapi,
        &api_routes.otel_enabled_routes,
        &api_routes.router,
    );
    api_routes.external_openapi
}

pub fn build_internal_openapi_spec() -> OpenApi {
    let (_, openapi) = internal::build_internal_non_otel_enabled_routes().split_for_parts();
    openapi
}
