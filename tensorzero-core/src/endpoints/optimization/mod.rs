pub mod helpers;
pub mod launch_optimization_workflow;
pub mod legacy;
pub mod types;

// Re-export legacy public API for backward compatibility
// Keep the old name for the legacy type to avoid breaking existing code
pub use legacy::{
    launch_optimization, launch_optimization_workflow as launch_optimization_workflow_legacy,
    launch_optimization_workflow_handler, poll_optimization, poll_optimization_handler,
    LaunchOptimizationParams, LaunchOptimizationWorkflowParams,
};

// Re-export new API with different names to avoid conflicts
pub use launch_optimization_workflow::launch_optimization_workflow;
pub use types::{
    LaunchOptimizationWorkflowParams as LaunchOptimizationWorkflowInternalParams,
    ListDatapointsData, OptimizationData,
};
