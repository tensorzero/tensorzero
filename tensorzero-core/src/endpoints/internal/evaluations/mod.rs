//! Evaluation-related internal endpoints.
//!
//! These endpoints support the UI for viewing and managing evaluation runs and results.

mod count_datapoints;
mod count_runs;
mod list_runs;
pub mod types;

pub use count_datapoints::count_datapoints_handler;
pub use count_runs::get_evaluation_run_stats_handler;
pub use list_runs::list_evaluation_runs_handler;
