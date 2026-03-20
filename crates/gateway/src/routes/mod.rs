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

#[cfg(test)]
mod tests {
    use super::{build_external_openapi_spec, build_internal_openapi_spec};
    use googletest::prelude::*;
    use metrics_exporter_prometheus::PrometheusBuilder;
    use serde_json::Value;

    fn find_branch_with_title<'a>(branches: &'a [Value], title: &str) -> Option<&'a Value> {
        branches
            .iter()
            .find(|branch| branch.get("title").and_then(Value::as_str) == Some(title))
    }

    fn first_all_of_ref(branch: &Value) -> Option<&str> {
        branch
            .get("allOf")?
            .as_array()?
            .first()?
            .get("$ref")?
            .as_str()
    }

    fn test_metrics_handle() -> metrics_exporter_prometheus::PrometheusHandle {
        PrometheusBuilder::new().build_recorder().handle()
    }

    #[gtest]
    fn internal_openapi_extracts_named_refs_for_autopilot_event_unions() {
        let openapi = serde_json::to_value(build_internal_openapi_spec())
            .expect("internal OpenAPI should serialize to JSON");
        let schemas = openapi["components"]["schemas"]
            .as_object()
            .expect("OpenAPI components.schemas should be an object");

        let status_update_branches = schemas["StatusUpdate"]["oneOf"]
            .as_array()
            .expect("StatusUpdate.oneOf should be an array");
        expect_that!(
            first_all_of_ref(
                find_branch_with_title(status_update_branches, "StatusUpdateText")
                    .expect("StatusUpdateText branch should exist"),
            ),
            some(eq("#/components/schemas/StatusUpdateText"))
        );

        let content_block_branches = schemas["AutoEvalContentBlock"]["oneOf"]
            .as_array()
            .expect("AutoEvalContentBlock.oneOf should be an array");
        expect_that!(
            first_all_of_ref(
                find_branch_with_title(content_block_branches, "AutoEvalContentBlockMarkdown")
                    .expect("AutoEvalContentBlockMarkdown branch should exist"),
            ),
            some(eq("#/components/schemas/AutoEvalMarkdownContentBlock"))
        );
        expect_that!(
            first_all_of_ref(
                find_branch_with_title(content_block_branches, "AutoEvalContentBlockJson")
                    .expect("AutoEvalContentBlockJson branch should exist"),
            ),
            some(eq("#/components/schemas/AutoEvalJsonContentBlock"))
        );

        let tool_outcome_branches = schemas["ToolOutcome"]["oneOf"]
            .as_array()
            .expect("ToolOutcome.oneOf should be an array");
        expect_that!(
            first_all_of_ref(
                find_branch_with_title(tool_outcome_branches, "ToolOutcomeRejected")
                    .expect("ToolOutcomeRejected branch should exist"),
            ),
            some(eq("#/components/schemas/ToolOutcomeRejected"))
        );
        expect_that!(
            first_all_of_ref(
                find_branch_with_title(tool_outcome_branches, "ToolOutcomeFailure")
                    .expect("ToolOutcomeFailure branch should exist"),
            ),
            some(eq("#/components/schemas/ToolOutcomeFailure"))
        );
    }

    #[gtest]
    fn external_openapi_titles_create_datapoints_inference_query_branch() {
        let openapi = serde_json::to_value(build_external_openapi_spec(test_metrics_handle()))
            .expect("external OpenAPI should serialize to JSON");
        let schemas = openapi["components"]["schemas"]
            .as_object()
            .expect("OpenAPI components.schemas should be an object");
        let branches = schemas["CreateDatapointsFromInferenceRequestParams"]["oneOf"]
            .as_array()
            .expect("CreateDatapointsFromInferenceRequestParams.oneOf should be an array");

        let branch = find_branch_with_title(
            branches,
            "CreateDatapointsFromInferenceRequestParamsInferenceQuery",
        )
        .expect("InferenceQuery branch should have a title");
        expect_that!(
            first_all_of_ref(branch),
            some(eq("#/components/schemas/ListInferencesRequest"))
        );
    }
}
