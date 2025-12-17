//! Workflow evaluation-related internal endpoints.
//!
//! These endpoints support the UI for viewing and managing workflow evaluation runs and results.

mod get_project_count;
mod get_projects;
mod types;

pub use get_project_count::get_workflow_evaluation_project_count;
pub use get_project_count::get_workflow_evaluation_project_count_handler;
pub use get_projects::get_workflow_evaluation_projects_handler;
pub use types::*;
