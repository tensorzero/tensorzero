//! Workflow evaluation-related internal endpoints.
//!
//! These endpoints support the UI for viewing and managing workflow evaluation runs and results.

mod get_projects;
pub mod types;

pub use get_projects::get_workflow_evaluation_projects_handler;
