//! Worker pool for JsRuntime execution.
//!
//! This module provides a pool of worker threads, each owning a long-lived JsRuntime.
//! This architecture is necessary because:
//! 1. JsRuntime is `!Send` - cannot be held across `.await` in `TaskTool::execute`
//! 2. JsRuntime creation is expensive (~50-200ms per instance)
//! 3. We need O(1) runtimes for process lifetime, not O(n) per execution

use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Sender, bounded};
use deno_core::PollEventLoopOptions;
use serde_json::Value as JsonValue;
use tokio::runtime::Handle;
use tokio::sync::oneshot;
use uuid::Uuid;

use super::error::TypeScriptToolError;
use super::runtime::{create_runtime, update_runtime_context};
use crate::ToolContext;

/// A pool of worker threads that execute TypeScript code.
///
/// Each worker thread owns a long-lived `JsRuntime` that processes
/// work items from a shared queue. The pool uses the main Tokio runtime's
/// handle for async operations (database, network, etc.).
///
/// # Architecture
///
/// ```text
/// Main Tokio Runtime
///   │
///   │  ┌──────────────────────────────────────────────────────┐
///   │  │  JsRuntimePool                                       │
///   │  │                                                       │
///   │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐       │
///   │  │  │ Worker 1   │ │ Worker 2   │ │ Worker N   │       │
///   │  │  │ JsRuntime  │ │ JsRuntime  │ │ JsRuntime  │       │
///   │  │  │ (lives     │ │ (lives     │ │ (lives     │       │
///   │  │  │  forever)  │ │  forever)  │ │  forever)  │       │
///   │  │  └────────────┘ └────────────┘ └────────────┘       │
///   │  │        ▲              ▲              ▲               │
///   │  │        │   channels   │              │               │
///   │  │        ▼              ▼              ▼               │
///   └──┼───  work queue / result channels  ───────────────────┤
///      └──────────────────────────────────────────────────────┘
/// ```
pub struct JsRuntimePool {
    work_sender: Sender<WorkItem>,
    _workers: Vec<JoinHandle<()>>,
}

/// A work item to be executed on a worker thread.
struct WorkItem {
    /// Pre-transpiled JavaScript code
    js_code: String,
    /// LLM-provided parameters as JSON string
    params_json: String,
    /// Side information as JSON string
    side_info_json: String,
    /// The ToolContext for this execution
    ctx: ToolContext,
    /// The task ID for this execution
    task_id: Uuid,
    /// The episode ID for this execution
    episode_id: Uuid,
    /// Channel to send the result back
    result_sender: oneshot::Sender<Result<String, String>>,
}

impl JsRuntimePool {
    /// Create a new pool with the specified number of workers.
    ///
    /// # Arguments
    ///
    /// * `num_workers` - Number of worker threads to create
    /// * `tokio_handle` - Handle to the main Tokio runtime for async operations
    ///
    /// # Panics
    ///
    /// Panics if `num_workers` is 0.
    pub fn new(num_workers: usize, tokio_handle: Handle) -> Self {
        assert!(num_workers > 0, "num_workers must be > 0");

        // Buffer size is 2x workers to allow some queueing
        let (work_sender, work_receiver) = bounded::<WorkItem>(num_workers * 2);
        let work_receiver = Arc::new(work_receiver);

        #[expect(clippy::expect_used, reason = "Thread spawn failure is unrecoverable")]
        let workers: Vec<_> = (0..num_workers)
            .map(|worker_id| {
                let receiver = Arc::clone(&work_receiver);
                let handle = tokio_handle.clone();

                thread::Builder::new()
                    .name(format!("js-worker-{worker_id}"))
                    .spawn(move || {
                        // Create JsRuntime ONCE per worker thread - lives for thread lifetime
                        let mut runtime = create_runtime();

                        // Process work items until channel is closed
                        while let Ok(work) = receiver.recv() {
                            execute_on_runtime(&mut runtime, &handle, work);
                        }
                    })
                    .expect("Failed to spawn worker thread")
            })
            .collect();

        Self {
            work_sender,
            _workers: workers,
        }
    }

    /// Create a new pool with a default number of workers (number of CPUs).
    ///
    /// # Arguments
    ///
    /// * `tokio_handle` - Handle to the main Tokio runtime for async operations
    pub fn new_default(tokio_handle: Handle) -> Self {
        let num_workers = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self::new(num_workers, tokio_handle)
    }

    /// Execute JavaScript code on a worker.
    ///
    /// This method sends the work to an available worker and waits for the result.
    ///
    /// # Arguments
    ///
    /// * `js_code` - Pre-transpiled JavaScript code to execute
    /// * `params` - LLM-provided parameters
    /// * `side_info` - Side information (hidden from LLM)
    /// * `ctx` - The ToolContext for this execution
    /// * `task_id` - The task ID
    /// * `episode_id` - The episode ID
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The pool is shut down
    /// - The worker thread panicked
    /// - JavaScript execution failed
    pub async fn execute(
        &self,
        js_code: String,
        params: JsonValue,
        side_info: JsonValue,
        ctx: ToolContext,
        task_id: Uuid,
        episode_id: Uuid,
    ) -> Result<JsonValue, TypeScriptToolError> {
        let (result_sender, result_receiver) = oneshot::channel();

        let params_json = serde_json::to_string(&params)
            .map_err(|e| TypeScriptToolError::Serialization(e.to_string()))?;
        let side_info_json = serde_json::to_string(&side_info)
            .map_err(|e| TypeScriptToolError::Serialization(e.to_string()))?;

        self.work_sender
            .send(WorkItem {
                js_code,
                params_json,
                side_info_json,
                ctx,
                task_id,
                episode_id,
                result_sender,
            })
            .map_err(|_| TypeScriptToolError::PoolShutdown)?;

        let result_str = result_receiver
            .await
            .map_err(|_| TypeScriptToolError::WorkerPanicked)?
            .map_err(TypeScriptToolError::Execution)?;

        serde_json::from_str(&result_str)
            .map_err(|e| TypeScriptToolError::Serialization(e.to_string()))
    }

    /// Get the number of workers in the pool.
    pub fn num_workers(&self) -> usize {
        self._workers.len()
    }
}

/// Execute JavaScript code on a JsRuntime.
///
/// This function is called on the worker thread. It:
/// 1. Updates the runtime's OpState with the current context
/// 2. Wraps the user code to call the tool's run function
/// 3. Executes the code and runs the event loop
/// 4. Extracts and returns the result
fn execute_on_runtime(runtime: &mut deno_core::JsRuntime, tokio_handle: &Handle, work: WorkItem) {
    // Update the runtime context for this execution
    update_runtime_context(runtime, work.ctx, work.task_id, work.episode_id);

    // Build wrapped code that calls the tool's run function
    let wrapped_code = format!(
        r"
        // User's transpiled code
        {js_code}

        // Find the default export (the tool object)
        const tool = (typeof module !== 'undefined' && module.exports && module.exports.default)
            ? module.exports.default
            : globalThis.default;

        if (!tool || typeof tool.run !== 'function') {{
            throw new Error('Tool must export a default object with a run() function');
        }}

        // Execute the tool and store result
        (async () => {{
            const llmParams = {params_json};
            const sideInfo = {side_info_json};
            const result = await tool.run(llmParams, sideInfo);
            return JSON.stringify(result);
        }})();
        ",
        js_code = work.js_code,
        params_json = work.params_json,
        side_info_json = work.side_info_json,
    );

    // Run with main runtime's async infrastructure
    let result = tokio_handle.block_on(async {
        // Execute the script
        let result_handle = match runtime.execute_script("<user_tool>", wrapped_code) {
            Ok(h) => h,
            Err(e) => return Err(e.to_string()),
        };

        // Run the event loop until all promises resolve
        if let Err(e) = runtime
            .run_event_loop(PollEventLoopOptions::default())
            .await
        {
            return Err(e.to_string());
        }

        // Extract the result using v8
        extract_result(runtime, result_handle)
    });

    // Send the result back
    let _ = work.result_sender.send(result);
}

/// Extract the result from a completed promise.
fn extract_result(
    runtime: &mut deno_core::JsRuntime,
    result_handle: deno_core::v8::Global<deno_core::v8::Value>,
) -> Result<String, String> {
    let context = runtime.main_context();
    let isolate = runtime.v8_isolate();

    // Create handle scope
    let hs = std::pin::pin!(deno_core::v8::HandleScope::new(isolate));
    let mut hs = hs.init();

    // Get context as local
    let ctx = deno_core::v8::Local::new(&hs, context);

    // Create context scope
    let scope = deno_core::v8::ContextScope::new(&mut hs, ctx);

    // Get the result value
    let local = deno_core::v8::Local::new(&scope, result_handle);

    // Check if it's a promise
    if let Ok(promise) = deno_core::v8::Local::<deno_core::v8::Promise>::try_from(local) {
        if promise.state() != deno_core::v8::PromiseState::Fulfilled {
            let exception = promise.result(&scope);
            let exception_str = exception.to_rust_string_lossy(&scope);
            return Err(format!("Tool execution failed: {exception_str}"));
        }

        let result_value = promise.result(&scope);
        Ok(result_value.to_rust_string_lossy(&scope))
    } else {
        // Direct string result
        Ok(local.to_rust_string_lossy(&scope))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let handle = rt.handle().clone();
        let pool = JsRuntimePool::new(2, handle);

        assert_eq!(pool.num_workers(), 2);
    }

    #[test]
    fn test_default_pool_creation() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let handle = rt.handle().clone();
        let pool = JsRuntimePool::new_default(handle);

        assert!(pool.num_workers() > 0);
    }

    #[test]
    #[should_panic(expected = "num_workers must be > 0")]
    fn test_pool_zero_workers_panics() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let handle = rt.handle().clone();
        let _ = JsRuntimePool::new(0, handle);
    }
}
