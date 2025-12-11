//! Evaluation-related internal endpoints.
//!
//! These endpoints support the UI for viewing and managing evaluation runs and results.

mod count_runs;
pub mod types;

pub use count_runs::get_evaluation_run_stats_handler;
