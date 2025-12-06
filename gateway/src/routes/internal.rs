//! Internal route definitions for the TensorZero Gateway API.
//!
//! These routes are for internal use. They are unstable and might change without notice,
//! and do not export any OpenTelemetry spans.

use axum::{
    Router,
    routing::{get, post, put},
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
            "/internal/ui-config",
            get(endpoints::ui::get_config::ui_config_handler),
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
            "/internal/datasets",
            get(endpoints::datasets::v1::list_datasets_handler),
        )
}
