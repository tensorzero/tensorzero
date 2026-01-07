//! Evaluation-related internal endpoints.
//!
//! These endpoints support the UI for viewing and managing evaluation runs and results.

mod count_datapoints;
mod count_runs;
mod get_evaluation_results;
mod get_human_feedback;
mod get_run_infos;
mod get_statistics;
mod list_runs;
mod search_runs;
pub mod types;

pub use count_datapoints::count_datapoints_handler;
pub use count_runs::count_evaluation_runs_handler;
pub use get_evaluation_results::{GetEvaluationResultsResponse, get_evaluation_results_handler};
pub use get_human_feedback::{GetHumanFeedbackResponse, get_human_feedback_handler};
pub use get_run_infos::{
    GetEvaluationRunInfosResponse, get_evaluation_run_infos_for_datapoint_handler,
    get_evaluation_run_infos_handler,
};
pub use get_statistics::get_evaluation_statistics_handler;
pub use list_runs::list_evaluation_runs_handler;
pub use search_runs::search_evaluation_runs_handler;
