//! Evaluation-related internal endpoints.
//!
//! These endpoints support the UI for viewing and managing evaluation runs and results.

mod count_datapoints;
mod count_runs;
mod get_run_infos;
mod list_runs;
mod search_runs;
pub mod types;

pub use count_datapoints::count_datapoints_handler;
pub use count_runs::get_evaluation_run_stats_handler;
pub use get_run_infos::{GetEvaluationRunInfosResponse, get_evaluation_run_infos_handler};
pub use list_runs::list_evaluation_runs_handler;
pub use search_runs::search_evaluation_runs_handler;
