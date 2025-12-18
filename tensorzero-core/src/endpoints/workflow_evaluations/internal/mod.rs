//! Workflow evaluation-related internal endpoints.
//!
//! These endpoints support the UI for viewing and managing workflow evaluation runs and results.

mod count_runs;
mod get_project_count;
mod get_projects;
mod get_runs;
mod list_runs;
mod search_runs;
mod types;

pub use count_runs::count_workflow_evaluation_runs;
pub use count_runs::count_workflow_evaluation_runs_handler;
pub use get_project_count::get_workflow_evaluation_project_count;
pub use get_project_count::get_workflow_evaluation_project_count_handler;
pub use get_projects::get_workflow_evaluation_projects_handler;
pub use get_runs::get_workflow_evaluation_runs;
pub use get_runs::get_workflow_evaluation_runs_handler;
pub use list_runs::list_workflow_evaluation_runs;
pub use list_runs::list_workflow_evaluation_runs_handler;
pub use search_runs::search_workflow_evaluation_runs;
pub use search_runs::search_workflow_evaluation_runs_handler;
pub use types::*;
