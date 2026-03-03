//! Workflow evaluation-related internal endpoints.
//!
//! These endpoints support the UI for viewing and managing workflow evaluation runs and results.

pub mod count_episodes_by_task_name;
pub mod count_run_episodes;
pub mod count_runs;
pub mod get_project_count;
pub mod get_projects;
pub mod get_run_episodes;
pub mod get_run_statistics;
pub mod get_runs;
pub mod list_episodes_by_task_name;
pub mod list_runs;
pub mod search_runs;
pub mod types;

pub use count_episodes_by_task_name::count_workflow_evaluation_run_episodes;
pub use count_episodes_by_task_name::count_workflow_evaluation_run_episodes_handler;
pub use count_run_episodes::count_workflow_evaluation_run_episodes_total;
pub use count_run_episodes::count_workflow_evaluation_run_episodes_total_handler;
pub use count_runs::count_workflow_evaluation_runs;
pub use count_runs::count_workflow_evaluation_runs_handler;
pub use get_project_count::get_workflow_evaluation_project_count;
pub use get_project_count::get_workflow_evaluation_project_count_handler;
pub use get_projects::get_workflow_evaluation_projects_handler;
pub use get_run_episodes::get_workflow_evaluation_run_episodes;
pub use get_run_episodes::get_workflow_evaluation_run_episodes_handler;
pub use get_run_statistics::get_workflow_evaluation_run_statistics;
pub use get_run_statistics::get_workflow_evaluation_run_statistics_handler;
pub use get_runs::get_workflow_evaluation_runs;
pub use get_runs::get_workflow_evaluation_runs_handler;
pub use list_episodes_by_task_name::list_workflow_evaluation_run_episodes_by_task_name;
pub use list_episodes_by_task_name::list_workflow_evaluation_run_episodes_by_task_name_handler;
pub use list_runs::list_workflow_evaluation_runs;
pub use list_runs::list_workflow_evaluation_runs_handler;
pub use search_runs::search_workflow_evaluation_runs;
pub use search_runs::search_workflow_evaluation_runs_handler;
pub use types::*;
