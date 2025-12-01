//! Internal route definitions for the TensorZero Gateway API.
//!
//! These routes are for internal use. They are unstable and might change without notice,
//! and do not export any OpenTelemetry spans.

use axum::{
    routing::{get, post, put},
    Router,
};
use tensorzero_core::endpoints;
use tensorzero_core::utils::gateway::AppStateData;

pub fn build_internal_non_otel_enabled_routes() -> Router<AppStateData> {
    Router::new()
        // Deprecated (#4652): Remove the endpoint without the `/internal` prefix.
        .route(
            "/variant_sampling_probabilities",
            get(endpoints::variant_probabilities::get_variant_sampling_probabilities_handler),
        )
        .route(
            "/internal/functions/{function_name}/variant_sampling_probabilities",
            get(endpoints::variant_probabilities::get_variant_sampling_probabilities_by_function_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints",
            post(endpoints::datasets::insert_from_existing_datapoint_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints/clone",
            post(endpoints::datasets::internal::clone_datapoints_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
            put(endpoints::datasets::update_datapoint_handler),
        )
        .route(
            "/internal/object_storage",
            get(endpoints::object_storage::get_object_handler),
        )
        .route(
            "/internal/inferences/bounds",
            get(endpoints::stored_inferences::v1::get_inference_bounds_handler),
        )
        .route(
            "/internal/inferences",
            get(endpoints::stored_inferences::v1::list_inferences_by_id_handler),
        )
}
