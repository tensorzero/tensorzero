//! Deno runtime creation and custom ops for the RLM sandbox.
//!
//! Provides a sandboxed JavaScript execution environment with ops for:
//! - `op_llm_query` / `op_llm_query_batched` — invoke LLM inference from JS
//! - `op_set_final` — mark a value as the final RLM answer
//! - `op_console_log` — capture console output

// Imports like Rc, RefCell, and JsErrorBox are used inside the #[op2] macro
// expansion, not directly in our code.

mod ops;

use crate::ExtraInferenceTags;
use crate::runtime::ops::rlm_ext;
use crate::tensorzero_client::TensorZeroClient;
use crate::ts_checker::{ExposedToolData, TsCheckerPool};
use crate::{SES_INIT_JS, SES_JS, TsError};
use deno_core::v8::IsolateHandle;
use deno_core::{JsRuntime, OpState, PollEventLoopOptions, RuntimeOptions};
use deno_error::JsErrorBox;
use durable::{ControlFlow, async_trait};
use tensorzero_types::tool_error::ToolResult;
use tokio::runtime::Handle;
use tracing::Span;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::sync::{Mutex, OwnedSemaphorePermit};
use uuid::Uuid;

use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

use crate::RlmConfig;
use crate::state::OomSnapshotConfig;

// ---------------------------------------------------------------------------
// RLM loop types
// ---------------------------------------------------------------------------

/// The data source for the RLM loop, encoding both what data is available
/// and how the LLM prompt should reference it.
pub enum RlmDataSource<'a> {
    /// Context management strategy path: structured JSON available as `toolOutput` in JS.
    ToolOutput,
    /// `llm_query` child runtime: parent LLM output as a text string available as `context` in JS.
    Context(&'a str),
    /// No data to process (e.g. `ExploreTool`): prompt focuses on instructions and tools only.
    InstructionsOnly,
}

/// Arguments for the RLM loop.
pub struct RlmLoopParams<'a> {
    pub handle: &'a JsRuntimeHandle,
    pub data_source: RlmDataSource<'a>,
    pub t0_client: Arc<dyn TensorZeroClient>,
    pub extra_inference_tags: &'a ExtraInferenceTags,
    pub episode_id: Uuid,
    pub config: &'a RlmConfig,
    pub instructions: &'a str,
    pub ts_checker: &'a TsCheckerPool,
    pub exposed_tools: Option<&'a ExposedTools>,
    /// The TensorZero function name to use for code generation inference calls.
    pub function_name: &'a str,
}

/// Trait abstracting the RLM loop execution within `llm_query`.
///
/// Called after the child JS runtime has been spawned. The default
/// implementation delegates to `run_rlm_loop`.
#[async_trait]
pub trait RlmQuery: Send + Sync {
    async fn run(&self, params: RlmLoopParams<'_>) -> Result<Result<String, ControlFlow>, TsError>;
}

// ---------------------------------------------------------------------------
// Runtime parameters
// ---------------------------------------------------------------------------

/// Parameters for spawning a new JS runtime.
///
/// Groups the arguments shared by [`RlmPool::spawn_runtime`],
/// [`spawn_runtime_handle`], [`spawn_child_runtime`], and
/// [`create_rlm_runtime`].
pub struct RuntimeParams {
    pub t0_client: Arc<dyn TensorZeroClient>,
    pub extra_inference_tags: ExtraInferenceTags,
    pub mode: RuntimeMode,
    pub ts_checker: Arc<TsCheckerPool>,
    pub exposed_tools: Option<ExposedTools>,
    pub oom_snapshot_config: Option<OomSnapshotConfig>,
}

/// State stored in `OpState` for the RLM runtime.
///
/// Contains metadata about the current RLM session that ops can read.
#[derive(Clone)]
pub struct RlmRuntimeState {
    /// Current recursion depth (0 = top level).
    pub depth: u32,
    /// Maximum allowed recursion depth.
    pub max_depth: u32,
    /// Maximum iterations per RLM loop level.
    pub max_iterations: usize,
    /// Episode ID for linking inference calls.
    pub episode_id: Uuid,
    /// Caller-provided instructions propagated to every level.
    pub instructions: String,
    /// Per-block execution timeout in seconds.
    pub execution_timeout_secs: u64,
    /// Optional recursive LLM query implementation.
    /// When present and depth < max_depth, `llm_query` spawns a child RLM loop
    /// via this trait. When absent, the leaf (single-shot) path is always taken.
    pub rlm_query: Option<Arc<dyn RlmQuery>>,
}

/// RLM-specific data exposed to JavaScript and referenced in prompts.
#[derive(Clone)]
pub enum RlmRuntimeInput {
    /// Parent-provided text exposed as `context`.
    Context(String),
    /// No input text beyond the instructions.
    InstructionsOnly,
    /// Structured JSON exposed as `toolOutput`.
    ToolOutput(serde_json::Value),
}

impl RlmRuntimeInput {
    fn context_for_runtime(&self) -> &str {
        match self {
            Self::Context(context) => context,
            Self::InstructionsOnly | Self::ToolOutput(_) => "",
        }
    }

    fn tool_output(&self) -> Option<&serde_json::Value> {
        match self {
            Self::ToolOutput(tool_output) => Some(tool_output),
            Self::Context(_) | Self::InstructionsOnly => None,
        }
    }
}

/// The final answer value, set by `op_set_final` from JavaScript.
///
/// Stored in `OpState` as `FinalAnswer`. After JS execution, check this
/// to see if the model called `FINAL()`.
#[derive(Default)]
struct FinalAnswer(Option<String>);

/// Buffer of console.log messages captured from JavaScript.
///
/// Stored in `OpState` as `ConsoleLogBuffer`.
#[derive(Default)]
struct ConsoleLogBuffer(Vec<String>);

/// Optional reference to the TypeScript checker pool, stored in `OpState`
/// so that `op_llm_query` can pass it to child RLM loops for typechecking.
#[derive(Clone)]
pub(crate) struct TsCheckerRef(pub(crate) Arc<TsCheckerPool>);

/// Flag indicating the durable task was suspended during JS execution.
///
/// Set by `op_tool_call_join` when `join_tool` returns `ControlFlow::Suspend`.
/// Checked by `run_execute_sync` to convert the JS termination error into
/// `TsError::Suspend` so that callers can propagate the suspension.
#[derive(Default)]
struct SuspendFlag(Option<ControlFlow>);

/// Flag indicating the durable task was terminated due to OOM.
///
/// The `Arc<AtomicBool>` is set in `add_near_heap_limit_callback` when the isolate is terminated
/// due to OOM, so that we can detect the termination reason on the other side of the deno_core boundary.
#[derive(Clone)]
struct OomFlag(Arc<AtomicBool>);

/// The 'magic' part of our deno-based tool calls. When we see a `ControlFlow::Suspend` error,
/// we set `SuspendFlag` and abort the current runtime. On the other side of the deno_core boundary
/// (where we start running the js code), we check for the flag via `take_suspend_flag` and continue
/// to propagate a `ControlFlow::Suspend` error.
///
/// In this way, a worker suspension can propagate all the way out of arbitrarily nested runtime loops,
/// where it eventually makes its way back to the parent tool call (e.g. `explore`), which suspends the worker.
///
/// When the worker is woken up, we'll re-run all of the code, but with an updated checkpoint cache,
/// allowing the task to make further progress. This relies on our javascript being deterministic,
/// which we enforce via the SES `lockdown` mechanism.
fn control_flow_to_js_error(control_flow: ControlFlow, state: &Rc<RefCell<OpState>>) -> JsErrorBox {
    match control_flow {
        cf @ ControlFlow::Suspend(_) => {
            state.borrow_mut().borrow_mut::<SuspendFlag>().0 = Some(cf);
            state
                .borrow_mut()
                .borrow::<IsolateHandle>()
                .terminate_execution();
            JsErrorBox::generic("UNREACHABLE - Tool call suspended".to_string())
        }
        other => JsErrorBox::generic(format!("Control flow error: {other:?}")),
    }
}

/// Maximum characters per `console.log` message before truncation.
const MAX_CONSOLE_LOG_CHARS: usize = 5_000;

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

/// Proof that an [`RlmPool`] permit has been acquired.
///
/// The inner constructor is private to this module, so only [`RlmPool::spawn_runtime`]
/// can create one. Functions that require a permit take `&RlmPermit` to enforce
/// at compile time that the caller went through the pool.
#[derive(Clone)]
pub struct RlmPermit {
    cancellation_token: CancellationToken,
    permit: Arc<OwnedSemaphorePermit>,
}

impl RlmPermit {
    fn new(permit: Arc<OwnedSemaphorePermit>) -> Self {
        Self {
            cancellation_token: CancellationToken::new(),
            permit,
        }
    }

    /// Create a permit backed by a fresh single-slot semaphore (tests only).
    ///
    /// # Panics
    ///
    /// Panics if the semaphore permit cannot be acquired (should not happen
    /// with a fresh single-slot semaphore).
    #[expect(clippy::expect_used)]
    pub fn for_test() -> Self {
        let sem = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = Arc::new(
            sem.try_acquire_owned()
                .expect("fresh single-slot semaphore should be acquirable"),
        );
        Self::new(permit)
    }

    /// Create a child permit whose cancellation token is derived from this one.
    ///
    /// Parent cancellation propagates down to the child, but the child being
    /// dropped (which cancels its token) does **not** propagate up to the parent.
    /// This prevents a completed child `JsRuntimeHandle` from prematurely
    /// aborting concurrent operations on the parent.
    pub(crate) fn child_permit(&self) -> Self {
        Self {
            cancellation_token: self.cancellation_token.child_token(),
            permit: self.permit.clone(),
        }
    }

    fn cancelled(&self) -> WaitForCancellationFuture<'_> {
        self.cancellation_token.cancelled()
    }

    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    /// Runs a future but cancels it if the RlmPermit is cancelled.
    ///
    /// Returns `Ok(result)` if the future completes before cancellation,
    /// or `Err(TsError::JsRuntime)` if the permit was cancelled.
    pub async fn run_with_cancellation<F, T>(&self, f: F) -> Result<T, TsError>
    where
        F: std::future::Future<Output = T>,
    {
        tokio::select! {
            result = f => Ok(result),
            () = self.cancelled() => {
                Err(TsError::JsRuntime {
                    message: "RLM execution cancelled".to_string(),
                })
            }
        }
    }
}

/// Result of executing a JS code block and atomically reading state.
pub struct ExecuteBlockResult {
    /// Whether execution succeeded or failed.
    /// The inner `Option<String>` is the final answer, if JS called `FINAL()`
    pub result: Result<Result<Option<String>, ControlFlow>, TsError>,
    /// Console log messages captured during execution.
    pub console_logs: Vec<String>,
}

/// Controls which globals are endowed into the SES compartment.
///
/// The RLM-specific data lives under [`RuntimeMode::Rlm`] so direct code
/// execution does not carry irrelevant placeholders like an empty `context`.
#[derive(Clone)]
pub enum RuntimeMode {
    /// Full RLM mode: `context`, `llm_query`, `llm_query_batched`,
    /// `FINAL`, `console`, `join_tool`, and `toolClient`.
    Rlm {
        /// The RLM-specific input source exposed to JavaScript and the prompt.
        input: RlmRuntimeInput,
        /// Recursion and execution metadata needed by `llm_query`.
        rlm_state: RlmRuntimeState,
    },
    /// Direct code execution mode: `console`, `join_tool`, and `toolClient`.
    CodeExecution,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeEndowment {
    Context,
    LlmQuery,
    LlmQueryBatched,
    Final,
    Console,
    JoinTool,
    ToolClient,
}

impl RuntimeMode {
    fn endowments(&self) -> &'static [RuntimeEndowment] {
        match self {
            Self::Rlm { .. } => &[
                RuntimeEndowment::Context,
                RuntimeEndowment::LlmQuery,
                RuntimeEndowment::LlmQueryBatched,
                RuntimeEndowment::Final,
                RuntimeEndowment::Console,
                RuntimeEndowment::JoinTool,
                RuntimeEndowment::ToolClient,
            ],
            Self::CodeExecution => &[
                RuntimeEndowment::Console,
                RuntimeEndowment::JoinTool,
                RuntimeEndowment::ToolClient,
            ],
        }
    }

    fn context(&self) -> Option<&str> {
        match self {
            Self::Rlm { input, .. } => Some(input.context_for_runtime()),
            Self::CodeExecution => None,
        }
    }

    fn tool_output(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Rlm { input, .. } => input.tool_output(),
            Self::CodeExecution => None,
        }
    }

    fn rlm_state(&self) -> Option<&RlmRuntimeState> {
        match self {
            Self::Rlm { rlm_state, .. } => Some(rlm_state),
            Self::CodeExecution => None,
        }
    }
}

/// Commands sent from [`JsRuntimeHandle`] to the pool thread.
enum RuntimeCommand {
    ExecuteBlock {
        name: String,
        code: String,
        reply: tokio::sync::oneshot::Sender<ExecuteBlockResult>,
        span: Span,
    },
}

/// Wrap user code for execution inside the SES Compartment.
///
/// The code is string-escaped and passed to `Compartment.evaluate()`.
/// Inside the compartment, it is wrapped in a strict-mode async IIFE so
/// that `let`/`const` declarations are scoped to the block and don't
/// cause redeclaration errors across iterations. `'use strict'` is
/// enforced by the compartment, but we include it explicitly for clarity.
/// Top-level `await` works because the IIFE is async. Users can still
/// share state across iterations via `globalThis` (the compartment's
/// own global, not the outer one).
fn wrap_user_code(code: &str) -> String {
    let escaped = serde_json::to_string(code).unwrap_or_else(|_| r#""""#.to_string());
    format!(
        "{{ \
           const __code = {escaped}; \
           const __expr = '(async () => {{ \"use strict\";\\n' + __code + '\\n}})()'; \
           globalThis.__t0_compartment.evaluate(__expr); \
         }}"
    )
}

/// Synchronous execute-and-run for use on pool threads.
///
/// Runs `execute_script` (blocking), cancels any pending termination, then
/// drives the event loop via `local_rt.block_on()`.
///
/// `local_rt` **must** be a `CurrentThread` tokio runtime because
/// `deno_unsync` (used internally by `#[op2(reentrant)]` async ops)
/// asserts `RuntimeFlavor::CurrentThread`.
struct ExecuteSyncResult {
    result: Result<Result<(), ControlFlow>, TsError>,
    oom_detected: bool,
}

fn run_execute_sync(
    runtime: &mut JsRuntime,
    name: &str,
    code: &str,
    local_rt: &tokio::runtime::Runtime,
) -> ExecuteSyncResult {
    // Enter the CurrentThread runtime context *before* execute_script.
    // V8 calls op function pointers synchronously during execute_script
    // (e.g. when JS invokes an async op like op_llm_query_batched).
    // deno_unsync asserts RuntimeFlavor::CurrentThread at that point,
    // so Handle::current() must already be the local runtime — not the
    // parent multi-threaded runtime inherited from spawn_blocking.
    let _guard = local_rt.enter();
    // Enter this span for the entire duration of the 'block_on' call, so that any futures spawned within it are instrumented with it.
    let _span_guard = tracing::info_span!("js_runtime_thread.event_loop", name = %name).entered();
    let wrapped = wrap_user_code(code);
    let result = match runtime.execute_script(name.to_string(), wrapped) {
        Ok(_) => local_rt.block_on(async {
            runtime
                .run_event_loop(PollEventLoopOptions::default())
                .await
                .map_err(|e| TsError::JsRuntime {
                    message: format!("Event loop error in {name}: {e}"),
                })
        }),
        Err(e) => Err(TsError::JsRuntime {
            message: format!("JS execution error in {name}: {e}"),
        }),
    };

    // Clear any pending termination so the runtime remains usable.
    // This must run on all paths — including after `execute_script` fails
    // due to `terminate_execution()` — otherwise the flag stays set and
    // the next command is immediately killed.
    runtime.v8_isolate().cancel_terminate_execution();

    if take_oom_flag(runtime) {
        return ExecuteSyncResult {
            result: Err(TsError::JsRuntime {
                message: "JS execution terminated due to out-of-memory error".to_string(),
            }),
            oom_detected: true,
        };
    }

    // Check if the JS termination was caused by a durable task suspension.
    // If so, replace the generic JS error with TsError::Suspend so callers
    // can propagate the suspension as ControlFlow::Suspend.
    if let Some(cf) = take_suspend_flag(runtime) {
        return ExecuteSyncResult {
            result: Ok(Err(cf)),
            oom_detected: false,
        };
    }

    ExecuteSyncResult {
        result: result.map(|()| Ok(())),
        oom_detected: false,
    }
}

/// A `Send + Sync` handle to a [`JsRuntime`] living on a dedicated OS thread.
///
/// Commands are sent over a channel; the pool thread executes them and replies
/// via oneshot channels. The V8 `IsolateHandle` allows the caller to terminate
/// execution on timeout without waiting for the thread.
///
/// Dropping the handle disconnects the command channel, causing the pool thread
/// to exit and releasing the semaphore permit.
pub struct JsRuntimeHandle {
    cmd_tx: Option<std::sync::mpsc::Sender<RuntimeCommand>>,
    isolate_handle: Option<deno_core::v8::IsolateHandle>,
    cancellation_token: Option<CancellationToken>,
    task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl JsRuntimeHandle {
    /// Typecheck and transpile TypeScript code, then execute the resulting JS.
    ///
    /// This is the primary entry point for running TypeScript. It calls
    /// Typecheck and transpile TypeScript code, then execute the resulting JS.
    ///
    /// If the operation exceeds `timeout`, the V8 isolate is terminated and
    /// this method returns immediately with an error. The pool thread recovers
    /// and remains available for subsequent commands.
    ///
    /// # Errors
    ///
    /// Returns `TsError::TypeCheck` if the TypeScript code has type errors.
    /// Returns `TsError::JsRuntime` if the script throws an exception or
    /// exceeds the timeout. Returns `TsError::PoolShutdown` if the pool
    /// thread has exited.
    #[tracing::instrument(name = "js_runtime_handle_execute_typescript_block", skip_all, fields(name = %name, timeout = ?timeout))]
    pub async fn execute_typescript_block(
        &self,
        checker: &TsCheckerPool,
        name: String,
        ambient_declarations: &str,
        typescript_code: &str,
        timeout: Duration,
    ) -> Result<ExecuteBlockResult, TsError> {
        let outcome = crate::ts_checker::prepare_typescript_block_outcome(
            checker,
            ambient_declarations,
            typescript_code,
            timeout,
        )
        .await?;
        let js_code = match outcome {
            crate::ts_checker::PreparedTypescriptBlock::Ready(js) => js,
            crate::ts_checker::PreparedTypescriptBlock::TypeError(diagnostics) => {
                return Err(TsError::TypeCheck {
                    message: diagnostics,
                });
            }
        };
        self.execute_js_block(name, &js_code, timeout).await
    }

    /// Execute a pre-transpiled JS code block, drive the event loop, and
    /// atomically read the final answer and console logs — one round-trip
    /// to the pool thread.
    #[tracing::instrument(name = "js_runtime_handle_execute_js_block", skip_all, fields(name = %name, timeout = ?timeout))]
    async fn execute_js_block(
        &self,
        name: String,
        code: &str,
        timeout: Duration,
    ) -> Result<ExecuteBlockResult, TsError> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.cmd_tx
            .as_ref()
            .ok_or(TsError::PoolShutdown)?
            .send(RuntimeCommand::ExecuteBlock {
                name,
                code: code.to_string(),
                reply: reply_tx,
                span: Span::current(),
            })
            .map_err(|_| TsError::PoolShutdown)?;

        match tokio::time::timeout(timeout, reply_rx).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(TsError::PoolShutdown),
            Err(_) => {
                // Timeout: terminate V8 and return immediately.
                // The pool thread will see the termination, call
                // cancel_terminate_execution(), and continue its loop.
                self.isolate_handle
                    .as_ref()
                    .ok_or(TsError::PoolShutdown)?
                    .terminate_execution();
                Err(TsError::JsRuntime {
                    message: "JS execution timed out".to_string(),
                })
            }
        }
    }

    /// Shut down the background runtime thread and wait for it to exit.
    ///
    /// This consumes the handle so callers cannot enqueue more work while
    /// shutdown is in progress. We explicitly drop the command sender before
    /// awaiting the blocking task so the runtime loop can observe channel
    /// closure and exit.
    pub async fn shutdown(mut self) -> Result<(), TsError> {
        if let Some(cancellation_token) = self.cancellation_token.take() {
            cancellation_token.cancel();
        }
        if let Some(isolate_handle) = self.isolate_handle.take() {
            isolate_handle.terminate_execution();
        }
        drop(self.cmd_tx.take());
        let task_handle = self.task_handle.take();

        if let Some(task_handle) = task_handle {
            task_handle.await.map_err(|e| TsError::JsRuntime {
                message: format!("Failed to wait for shutdown: {e}"),
            })?;
        }

        Ok(())
    }
}

impl Drop for JsRuntimeHandle {
    fn drop(&mut self) {
        if let Some(cancellation_token) = self.cancellation_token.take() {
            cancellation_token.cancel();
        }
        if let Some(isolate_handle) = self.isolate_handle.take() {
            isolate_handle.terminate_execution();
        }
    }
}

/// Semaphore-bounded pool for running RLM loops on dedicated OS threads.
///
/// Each [`run_rlm`](Self::run_rlm) call acquires a permit, spawns a
/// blocking task, and runs the RLM loop to completion (creates one or more `JsRuntime`s along the way).
/// The permit is released when the task finishes.
///
/// Recursive child runtimes (created inside ops like `op_llm_query`) use
/// [`spawn_child_runtime`] to get their own blocking task and [`JsRuntimeHandle`].
/// They do **not** acquire a semaphore permit, preventing deadlock.
#[derive(Clone, Debug)]
pub struct RlmPool {
    semaphore: Arc<tokio::sync::Semaphore>,
    oom_snapshot_config: Option<OomSnapshotConfig>,
}

impl RlmPool {
    pub fn oom_snapshot_config(&self) -> Option<&OomSnapshotConfig> {
        self.oom_snapshot_config.as_ref()
    }
}

pub use tensorzero_core::client::ToolContextHelper;

/// Constructs side info suitable for calling tools in our `ToolContext`,
/// using the provided value as the tool call id.
#[async_trait]
pub trait MakeSideInfo: Send + Sync {
    async fn make_side_info(&self, tool_call_id: String) -> Result<serde_json::Value, String>;
}

#[derive(Clone)]
pub struct ExposedTools {
    pub mode: Arc<ExposedToolMode>,
    pub tool_context_helper: Arc<Mutex<dyn ToolContextHelper>>,
    pub make_side_info: Arc<dyn MakeSideInfo>,
}

impl ExposedTools {
    pub fn exposed_tools_data(&self) -> &[ExposedToolData] {
        self.mode.exposed_tools_data()
    }

    /// Heartbeat: extends the durable task lease.
    /// Propagates errors so that cancellation can flow through.
    pub async fn heartbeat(&self) -> ToolResult<()> {
        self.tool_context_helper.lock().await.heartbeat().await
    }
}

/// Controls which *durable* tools are exposed to RLM typescript code
pub enum ExposedToolMode {
    /// Don't expose any tools
    None,
    /// Exposes only the provided tools
    Whitelist(Vec<ExposedToolData>),
}

impl ExposedToolMode {
    pub fn exposed_tools_data(&self) -> &[ExposedToolData] {
        match self {
            ExposedToolMode::None => &[],
            ExposedToolMode::Whitelist(tools) => tools,
        }
    }

    /// Constructs `ExposedToolMode::Whitelist` from pre-built tool data.
    pub fn whitelist(tool_data: Vec<ExposedToolData>) -> ExposedToolMode {
        ExposedToolMode::Whitelist(tool_data)
    }
}

impl RlmPool {
    /// Create a new pool with the given concurrency limit.
    ///
    /// # Errors
    ///
    /// Returns `TsError::InvalidConfig` if `max_concurrency` is 0.
    pub fn new(max_concurrency: usize) -> Result<Self, TsError> {
        if max_concurrency == 0 {
            return Err(TsError::InvalidConfig {
                message: "RlmPool max_concurrency must be at least 1".to_string(),
            });
        }
        Ok(Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrency)),
            oom_snapshot_config: None,
        })
    }

    /// Set the OOM snapshot config, enabling V8 heap snapshot uploads on OOM.
    #[must_use]
    pub fn with_oom_snapshot_config(mut self, config: OomSnapshotConfig) -> Self {
        self.oom_snapshot_config = Some(config);
        self
    }

    /// Spawn a direct code-execution runtime without entering the RLM loop.
    ///
    /// The resulting handle preserves REPL state across `execute_typescript_block` calls and
    /// exposes only the globals allowed in [`RuntimeMode::CodeExecution`].
    pub async fn spawn_code_runtime(
        &self,
        t0_client: Arc<dyn TensorZeroClient>,
        extra_inference_tags: ExtraInferenceTags,
        ts_checker: Arc<TsCheckerPool>,
        exposed_tools: Option<ExposedTools>,
    ) -> Result<JsRuntimeHandle, TsError> {
        self.spawn_runtime(RuntimeParams {
            t0_client,
            extra_inference_tags,
            mode: RuntimeMode::CodeExecution,
            ts_checker,
            exposed_tools,
            oom_snapshot_config: self.oom_snapshot_config.clone(),
        })
        .await
    }

    /// Spawn a new [`JsRuntime`] on a dedicated OS thread.
    ///
    /// Blocks until a semaphore permit is available, then spawns a blocking
    /// task and returns a handle. The runtime is ready for use when this returns.
    ///
    /// # Errors
    ///
    /// Returns `TsError::PoolShutdown` if the semaphore is closed, or
    /// `TsError::JsRuntime` if runtime creation fails.
    pub async fn spawn_runtime(&self, params: RuntimeParams) -> Result<JsRuntimeHandle, TsError> {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| TsError::PoolShutdown)?;

        let rlm_permit = RlmPermit::new(Arc::new(permit));
        spawn_runtime_handle(params, rlm_permit).await
    }
}

/// This represents the handle to the main multi-threaded runtime, which
/// we use to spawn the actual body of all our async deno ops.
/// See the comment on the `extension!` macro call in `ops.rs` for more details.
#[derive(Clone)]
struct MainRuntimeHandle(Handle);

/// Core runtime-spawn logic. Spawns a blocking task with a JsRuntime command loop.
async fn spawn_runtime_handle(
    params: RuntimeParams,
    rlm_permit: RlmPermit,
) -> Result<JsRuntimeHandle, TsError> {
    let RuntimeParams {
        t0_client,
        extra_inference_tags,
        mode,
        ts_checker,
        exposed_tools,
        oom_snapshot_config,
    } = params;
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<RuntimeCommand>();
    let (init_tx, init_rx) =
        tokio::sync::oneshot::channel::<Result<deno_core::v8::IsolateHandle, TsError>>();
    let cancellation_token = rlm_permit.cancellation_token();

    let task_handle = tokio::task::spawn_blocking(move || {
        // We run this *before* we build and enter our `new_current_thread` tokio runtime,
        // so that this is a reference to the main multi-threaded tokio runtime.
        let main_runtime_handle = MainRuntimeHandle(Handle::current());
        // Build a dedicated CurrentThread tokio runtime for this thread.
        // deno_unsync (used internally by `#[op2(reentrant)]` async ops)
        // asserts RuntimeFlavor::CurrentThread, so we cannot use the
        // caller's multi-threaded runtime handle here.
        let local_rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = init_tx.send(Err(TsError::JsRuntime {
                    message: format!("Failed to build current-thread tokio runtime for RLM: {e}"),
                }));
                return;
            }
        };

        // Hold the semaphore permit (if any) until this thread exits, so
        // the semaphore accurately reflects the number of live top-level
        // JsRuntime instances (including teardown).
        let mut runtime = match create_rlm_runtime(
            &mode,
            t0_client,
            extra_inference_tags,
            rlm_permit,
            ts_checker,
            exposed_tools,
            main_runtime_handle,
            oom_snapshot_config,
        ) {
            Ok(rt) => rt,
            Err(e) => {
                let _ = init_tx.send(Err(e));
                return;
            }
        };

        let handle = runtime.v8_isolate().thread_safe_handle();
        let _ = init_tx.send(Ok(handle));

        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                RuntimeCommand::ExecuteBlock {
                    name,
                    code,
                    reply,
                    span,
                } => {
                    let oom_detected = span.in_scope(|| {
                        let _guard =
                            tracing::info_span!("js_runtime_thread.execute_block", name = %name)
                                .entered();
                        let exec = run_execute_sync(&mut runtime, &name, &code, &local_rt);
                        let oom_detected = exec.oom_detected;
                        let final_answer = take_final_answer(&mut runtime);
                        let final_result = exec.result.map(|res| res.map(|()| final_answer));
                        let console_logs = take_console_logs(&mut runtime);
                        // Reply immediately so the caller is unblocked.
                        let _ = reply.send(ExecuteBlockResult {
                            result: final_result,
                            console_logs,
                        });
                        oom_detected
                    });
                    // After replying, capture a V8 heap snapshot.
                    // In the background, we'll upload the snapshot to S3
                    if oom_detected {
                        capture_and_upload_oom_snapshot(&mut runtime);
                    }
                }
            }
        }
    });

    let isolate_handle = init_rx.await.map_err(|_| TsError::PoolShutdown)??;

    Ok(JsRuntimeHandle {
        cmd_tx: Some(cmd_tx),
        isolate_handle: Some(isolate_handle),
        cancellation_token: Some(cancellation_token),
        task_handle: Some(task_handle),
    })
}

/// Spawn a child [`JsRuntimeHandle`] for recursive RLM calls.
///
/// Unlike [`RlmPool::spawn_runtime`], this does **not** acquire a semaphore
/// permit, preventing deadlock when a child call is made from within a
/// top-level RLM loop.
pub async fn spawn_child_runtime(
    params: RuntimeParams,
    rlm_permit: RlmPermit,
) -> Result<JsRuntimeHandle, TsError> {
    spawn_runtime_handle(params, rlm_permit).await
}
// ---------------------------------------------------------------------------
// Runtime creation
// ---------------------------------------------------------------------------

/// V8 heap limit for RLM isolates. Configurable via `RLM_JS_HEAP_LIMIT_MIB`.
/// Defaults to 100 MiB. We pass the limit to V8. If we get close to it
/// (as determined by our 'add_near_heap_limit_callback'), then we terminate
/// the isolate, which produces a nice Rust error on the other side.
static JS_HEAP_LIMIT: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("RLM_JS_HEAP_LIMIT_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100)
        * 1024
        * 1024
});

#[expect(clippy::too_many_arguments, reason = "threading OOM snapshot config")]
/// Create a new `JsRuntime` configured for RLM execution.
///
/// Restores a V8 snapshot with SES `lockdown()` already applied (all
/// ECMAScript intrinsics frozen). The `rlm_ext` extension is loaded at
/// restore time, which registers ops on the *unfrozen* `Deno.core.ops`
/// object and evaluates `init.js` (defining `__t0_*` bridge functions on
/// the outer `globalThis`). A SES `Compartment` isolates user code so
/// that only explicitly endowed globals are visible.
fn create_rlm_runtime(
    mode: &RuntimeMode,
    t0_client: Arc<dyn TensorZeroClient>,
    extra_inference_tags: ExtraInferenceTags,
    rlm_permit: RlmPermit,
    ts_checker: Arc<TsCheckerPool>,
    exposed_tools: Option<ExposedTools>,
    main_runtime_handle: MainRuntimeHandle,
    oom_snapshot_config: Option<OomSnapshotConfig>,
) -> Result<JsRuntime, TsError> {
    let mut runtime = JsRuntime::new(RuntimeOptions {
        extensions: vec![rlm_ext::init()],
        create_params: Some(v8::CreateParams::default().heap_limits(0, *JS_HEAP_LIMIT)),
        ..Default::default()
    });

    let isolate = runtime.v8_isolate().thread_safe_handle();
    let oom_flag = OomFlag(Arc::new(AtomicBool::new(false)));
    let oom_flag_clone = oom_flag.clone();

    runtime.add_near_heap_limit_callback(move |current_limit, _initial_limit| {
        oom_flag_clone.0.store(true, Ordering::Relaxed);
        isolate.terminate_execution();
        // Bump the limit so V8 can finish cleanup before termination takes effect
        current_limit * 2
    });

    // Load SES and call lockdown() to freeze all ECMAScript intrinsics.
    // This must happen AFTER extension initialization (which registers ops
    // on the unfrozen Deno.core.ops) but BEFORE injecting user-visible data.
    runtime
        .execute_script("<ses.umd.js>", SES_JS.to_string())
        .map_err(|e| TsError::JsRuntime {
            message: format!("Failed to load SES: {e}"),
        })?;
    runtime
        .execute_script("<ses_init.js>", SES_INIT_JS.to_string())
        .map_err(|e| TsError::JsRuntime {
            message: format!("Failed to initialize SES lockdown: {e}"),
        })?;

    // Build toolClient methods from exposed tools (before moving into OpState).
    let tool_client_methods: String = exposed_tools
        .as_ref()
        .map_or(&[] as &[ExposedToolData], |et| et.exposed_tools_data())
        .iter()
        .map(|tool| {
            let name = &tool.name;
            format!(
                "{name}: function(params) {{ \
                   return globalThis.__t0_tool_call_dispatch(\"{name}\", params); \
                 }}"
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    {
        let op_state = runtime.op_state();
        let mut state = op_state.borrow_mut();
        state.put(t0_client);
        state.put(extra_inference_tags);
        state.put(FinalAnswer::default());
        state.put(ConsoleLogBuffer::default());
        state.put(SuspendFlag::default());
        state.put(rlm_permit);
        state.put(oom_flag);
        state.put(TsCheckerRef(ts_checker));
        state.put(main_runtime_handle);
        if let Some(oom_snapshot_config) = oom_snapshot_config {
            state.put(oom_snapshot_config);
        }
        // We need to be able to terminate the runtime from within `control_flow_to_js_error`,
        // so we need an `IsolateHandle` available in our `OpState`.
        state.put(runtime.v8_isolate().thread_safe_handle());
        if let Some(rlm_state) = mode.rlm_state() {
            state.put(rlm_state.clone());
        }
        if let Some(exposed_tools) = exposed_tools {
            state.put(exposed_tools);
        }
    }

    if let Some(context) = mode.context() {
        // Inject context onto the outer globalThis (used by compartment setup below).
        let escaped_context = serde_json::to_string(context).map_err(|e| TsError::JsRuntime {
            message: format!("Failed to serialize context for JS injection: {e}"),
        })?;
        runtime
            .execute_script(
                "<rlm_context_injection>",
                format!("globalThis.__t0_context = {escaped_context};"),
            )
            .map_err(|e| TsError::JsRuntime {
                message: format!("Failed to inject context: {e}"),
            })?;

        // Inject `toolOutput` if explicitly provided by the caller.
        if let Some(tool_output) = mode.tool_output() {
            let serialized =
                serde_json::to_string(tool_output).map_err(|e| TsError::JsRuntime {
                    message: format!("Failed to serialize tool_output for JS injection: {e}"),
                })?;
            runtime
                .execute_script(
                    "<rlm_tool_output_injection>",
                    format!("{{ globalThis.__t0_toolOutput = {serialized}; }}"),
                )
                .map_err(|e| TsError::JsRuntime {
                    message: format!("Failed to inject toolOutput: {e}"),
                })?;
        }
    }

    // Create a SES Compartment with curated endowments. User code runs
    // inside this compartment and cannot see Deno, __t0_* helpers, or any
    // host objects — only what we explicitly pass in.
    let compartment_js =
        build_compartment_setup_script(mode, &tool_client_methods, mode.tool_output().is_some());
    runtime
        .execute_script("<rlm_compartment_setup>", compartment_js)
        .map_err(|e| TsError::JsRuntime {
            message: format!("Failed to create SES compartment: {e}"),
        })?;

    Ok(runtime)
}

fn build_compartment_setup_script(
    mode: &RuntimeMode,
    tool_client_methods: &str,
    has_tool_output: bool,
) -> String {
    let mut endowment_entries = Vec::new();

    if has_tool_output {
        endowment_entries.push("toolOutput: globalThis.__t0_toolOutput".to_string());
    }

    for endowment in mode.endowments() {
        match endowment {
            RuntimeEndowment::Context => {
                endowment_entries.push("context: globalThis.__t0_context".to_string());
            }
            RuntimeEndowment::LlmQuery => {
                endowment_entries.push("llm_query: globalThis.__t0_llm_query".to_string());
            }
            RuntimeEndowment::LlmQueryBatched => {
                endowment_entries
                    .push("llm_query_batched: globalThis.__t0_llm_query_batched".to_string());
            }
            RuntimeEndowment::Final => {
                endowment_entries.push("FINAL: globalThis.__t0_FINAL".to_string());
            }
            RuntimeEndowment::Console => {
                endowment_entries.push("console: globalThis.__t0_console".to_string());
            }
            RuntimeEndowment::JoinTool => {
                endowment_entries.push("join_tool: globalThis.__t0_tool_call_join".to_string());
            }
            RuntimeEndowment::ToolClient => {
                endowment_entries.push(format!("toolClient: {{ {tool_client_methods} }}"));
            }
        }
    }

    let mut script = format!(
        "{{ \
           const endowments = {{ {} }}; \
        ",
        endowment_entries.join(", ")
    );
    script.push_str(
        "globalThis.__t0_compartment = globalThis.__t0_create_rlm_compartment(endowments); \
         for (const k of Object.keys(endowments)) { \
           Object.defineProperty(globalThis.__t0_compartment.globalThis, k, {writable:false,configurable:false}); \
         } \
       }",
    );
    script
}

/// Read the final answer from the runtime's `OpState`, if one has been set.
fn take_final_answer(runtime: &mut JsRuntime) -> Option<String> {
    let op_state = runtime.op_state();
    let mut state = op_state.borrow_mut();
    state.borrow_mut::<FinalAnswer>().0.take()
}

/// Check and reset the OOM flag from the runtime's `OpState`.
///
/// Returns `true` if the isolate was terminated due to OOM.
fn take_oom_flag(runtime: &mut JsRuntime) -> bool {
    let op_state = runtime.op_state();
    let state = op_state.borrow();
    state.borrow::<OomFlag>().0.swap(false, Ordering::Relaxed)
}

/// Check and clear the suspend flag from the runtime's `OpState`.
///
/// Returns `true` if a tool call join triggered a task suspension during
/// the last execution block.
fn take_suspend_flag(runtime: &mut JsRuntime) -> Option<ControlFlow> {
    let op_state = runtime.op_state();
    let mut state = op_state.borrow_mut();
    state.borrow_mut::<SuspendFlag>().0.take()
}

/// Drain and return the console log buffer from the runtime's `OpState`.
fn take_console_logs(runtime: &mut JsRuntime) -> Vec<String> {
    let op_state = runtime.op_state();
    let mut state = op_state.borrow_mut();
    std::mem::take(&mut state.borrow_mut::<ConsoleLogBuffer>().0)
}

// ---------------------------------------------------------------------------
// OOM heap snapshot capture & upload
// ---------------------------------------------------------------------------

/// Capture a V8 heap snapshot from the runtime and spawn a background S3
/// upload on the main tokio runtime. Called on the pool thread after the
/// OOM error has already been sent back to the caller.
/// Note that while the snapshot capture will block further command processing on the pool thread,
/// it will *not* block the main RLM loop from continuing to make progress (e.g.
/// running a new code generation LLM inference with the OOM error from the previous iteration).
fn capture_and_upload_oom_snapshot(runtime: &mut JsRuntime) {
    let op_state = runtime.op_state();
    let state = op_state.borrow();
    let config = match state.try_borrow::<OomSnapshotConfig>() {
        Some(c) => c.clone(),
        None => return,
    };
    let main_handle = state.borrow::<MainRuntimeHandle>().clone();
    let org_id = state
        .borrow::<ExtraInferenceTags>()
        .0
        .get("organization_id")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    drop(state);

    let mut snapshot_data: Vec<u8> = Vec::new();
    runtime.v8_isolate().take_heap_snapshot(|chunk| {
        snapshot_data.extend_from_slice(chunk);
        true
    });

    if snapshot_data.is_empty() {
        tracing::warn!("V8 heap snapshot capture returned no data after OOM");
        return;
    }

    main_handle.0.spawn(async move {
        if let Err(e) = upload_oom_snapshot(&config, &org_id, snapshot_data).await {
            tracing::warn!("Failed to upload OOM heap snapshot to S3: {e}");
        }
    });
}

/// Upload the heap snapshot bytes to S3.
async fn upload_oom_snapshot(
    config: &OomSnapshotConfig,
    org_id: &str,
    data: Vec<u8>,
) -> Result<(), aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::put_object::PutObjectError>> {
    let id = uuid::Uuid::now_v7();
    let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown-host".to_string());
    let key = format!("{org_id}/{hostname}-{id}.heapsnapshot");
    tracing::info!(
        key,
        size = data.len(),
        "Captured V8 heap snapshot after OOM, uploading to S3"
    );

    let body = aws_sdk_s3::primitives::ByteStream::from(data);
    config
        .s3_client
        .put_object()
        .bucket(&config.bucket_name)
        .key(&key)
        .body(body)
        .send()
        .await?;

    tracing::info!(key, bucket = %config.bucket_name, "Uploaded OOM heap snapshot to S3");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_query::extract_text_from_response;
    use crate::tensorzero_client::MockTensorZeroClient;
    use tensorzero_core::client::ClientInferenceParams;
    use tensorzero_types::{
        ChatInferenceResponse, ContentBlockChatOutput, InferenceResponse, InputMessageContent,
        Usage,
    };

    const TEST_TIMEOUT: Duration = Duration::from_secs(30);

    async fn test_ts_checker() -> Arc<TsCheckerPool> {
        Arc::new(
            TsCheckerPool::new(1)
                .await
                .expect("failed to create TS checker pool"),
        )
    }

    fn test_extra_inference_tags() -> ExtraInferenceTags {
        ExtraInferenceTags::new(&crate::InferenceContext {
            session_id: Uuid::nil(),
            event_id: Uuid::nil(),
            organization_id: "test-org".to_string(),
            workspace_id: "test-ws".to_string(),
        })
    }

    fn test_rlm_state(
        _context: &str,
        depth: u32,
        max_depth: u32,
        execution_timeout_secs: u64,
    ) -> RlmRuntimeState {
        RlmRuntimeState {
            depth,
            max_depth,
            max_iterations: 10,
            episode_id: Uuid::nil(),
            instructions: "Summarize this data.".to_string(),
            execution_timeout_secs,
            rlm_query: None,
        }
    }

    fn first_inference_text(params: &ClientInferenceParams) -> String {
        params
            .input
            .messages
            .iter()
            .flat_map(|msg| msg.content.iter())
            .find_map(|content| match content {
                InputMessageContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    async fn test_runtime(context: &str) -> JsRuntime {
        test_runtime_with_tool_output(context, None).await
    }

    fn test_input(context: &str) -> RlmRuntimeInput {
        if context.is_empty() {
            RlmRuntimeInput::InstructionsOnly
        } else {
            RlmRuntimeInput::Context(context.to_string())
        }
    }

    async fn test_runtime_with_tool_output(
        context: &str,
        tool_output: Option<&serde_json::Value>,
    ) -> JsRuntime {
        test_runtime_with_client(context, tool_output, Arc::new(MockTensorZeroClient::new())).await
    }

    async fn test_runtime_with_client(
        context: &str,
        tool_output: Option<&serde_json::Value>,
        client: Arc<dyn TensorZeroClient>,
    ) -> JsRuntime {
        let rlm_state = test_rlm_state(context, 0, 3, 30);
        let mode = RuntimeMode::Rlm {
            input: tool_output
                .cloned()
                .map_or_else(|| test_input(context), RlmRuntimeInput::ToolOutput),
            rlm_state,
        };
        create_rlm_runtime(
            &mode,
            client,
            test_extra_inference_tags(),
            RlmPermit::for_test(),
            test_ts_checker().await,
            None,
            MainRuntimeHandle(Handle::current()),
            None,
        )
        .unwrap_or_else(|e| std::panic::panic_any(format!("Failed to create runtime: {e}")))
    }

    async fn test_code_execution_runtime(client: Arc<dyn TensorZeroClient>) -> JsRuntime {
        create_rlm_runtime(
            &RuntimeMode::CodeExecution,
            client,
            test_extra_inference_tags(),
            RlmPermit::for_test(),
            test_ts_checker().await,
            None,
            MainRuntimeHandle(Handle::current()),
            None,
        )
        .unwrap_or_else(|e| std::panic::panic_any(format!("Failed to create runtime: {e}")))
    }

    async fn test_runtime_at_max_depth(
        context: &str,
        client: Arc<dyn TensorZeroClient>,
    ) -> JsRuntime {
        test_runtime_at_max_depth_with_timeout(context, client, 30).await
    }

    async fn test_runtime_at_max_depth_with_timeout(
        context: &str,
        client: Arc<dyn TensorZeroClient>,
        execution_timeout_secs: u64,
    ) -> JsRuntime {
        let rlm_state = test_rlm_state(context, 3, 3, execution_timeout_secs);
        let mode = RuntimeMode::Rlm {
            input: test_input(context),
            rlm_state,
        };
        create_rlm_runtime(
            &mode,
            client,
            test_extra_inference_tags(),
            RlmPermit::for_test(),
            test_ts_checker().await,
            None,
            MainRuntimeHandle(Handle::current()),
            None,
        )
        .unwrap_or_else(|e| std::panic::panic_any(format!("Failed to create runtime: {e}")))
    }

    async fn execute_and_run(
        runtime: &mut JsRuntime,
        name: impl Into<String>,
        code: &str,
        timeout: Duration,
    ) -> Result<(), TsError> {
        let name = name.into();
        let isolate_handle = runtime.v8_isolate().thread_safe_handle();
        let timed_out = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let timed_out_watchdog = timed_out.clone();
        let (cancel_tx, cancel_rx) = std::sync::mpsc::channel::<()>();
        let watchdog_name = name.clone();
        let watchdog = std::thread::Builder::new()
            .name(format!("rlm-watchdog-{}", &name))
            .spawn(move || match cancel_rx.recv_timeout(timeout) {
                Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    timed_out_watchdog.store(true, std::sync::atomic::Ordering::SeqCst);
                    isolate_handle.terminate_execution();
                }
            })
            .map_err(|e| TsError::JsRuntime {
                message: format!("Failed to spawn watchdog thread: {e}"),
            })?;

        let wrapped = wrap_user_code(code);
        let script_result =
            runtime
                .execute_script(name.clone(), wrapped)
                .map_err(|e| TsError::JsRuntime {
                    message: format!("JS execution error in {name}: {e}"),
                });

        let result = match script_result {
            Ok(_) => runtime
                .run_event_loop(PollEventLoopOptions::default())
                .await
                .map_err(|e| TsError::JsRuntime {
                    message: format!("Event loop error in {name}: {e}"),
                }),
            Err(e) => Err(e),
        };

        let _ = cancel_tx.send(());
        let _ = watchdog.join();
        runtime.v8_isolate().cancel_terminate_execution();

        if timed_out.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(TsError::JsRuntime {
                message: format!("JS execution timed out in {watchdog_name}"),
            });
        }

        result
    }

    async fn eval_to_string(runtime: &mut JsRuntime, expr: &str) -> String {
        let code = format!("console.log(String({expr}));");
        execute_and_run(runtime, "<eval>", &code, TEST_TIMEOUT)
            .await
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        let logs = take_console_logs(runtime);
        logs.into_iter()
            .last()
            .unwrap_or_else(|| std::panic::panic_any("no log output"))
    }

    #[tokio::test]
    async fn test_context_injection() {
        let mut runtime = test_runtime("hello world").await;
        let value = eval_to_string(&mut runtime, "context").await;
        assert_eq!(value, "hello world");
    }

    #[tokio::test]
    async fn test_code_execution_mode_endows_only_execution_globals() {
        let mut runtime = test_code_execution_runtime(Arc::new(MockTensorZeroClient::new())).await;
        assert_eq!(
            eval_to_string(&mut runtime, "typeof context").await,
            "undefined"
        );
        assert_eq!(
            eval_to_string(&mut runtime, "typeof toolOutput").await,
            "undefined"
        );
        assert_eq!(
            eval_to_string(&mut runtime, "typeof FINAL").await,
            "undefined"
        );
        assert_eq!(
            eval_to_string(&mut runtime, "typeof llm_query").await,
            "undefined"
        );
        assert_eq!(
            eval_to_string(&mut runtime, "typeof llm_query_batched").await,
            "undefined"
        );
        assert_eq!(
            eval_to_string(&mut runtime, "typeof console").await,
            "object"
        );
        assert_eq!(
            eval_to_string(&mut runtime, "typeof join_tool").await,
            "function"
        );
        assert_eq!(
            eval_to_string(&mut runtime, "typeof toolClient").await,
            "object"
        );
    }

    #[tokio::test]
    async fn test_pool_spawn_code_runtime_uses_code_execution_mode() {
        let pool = RlmPool::new(1).expect("failed to create pool");
        let handle = pool
            .spawn_code_runtime(
                Arc::new(MockTensorZeroClient::new()),
                test_extra_inference_tags(),
                test_ts_checker().await,
                None,
            )
            .await
            .expect("spawn_code_runtime should succeed");
        let result = handle
            .execute_js_block(
                "<code_execution>".to_string(),
                "console.log(typeof context, typeof FINAL, typeof llm_query, typeof console, typeof join_tool, typeof toolClient);",
                TEST_TIMEOUT,
            )
            .await
            .expect("execute_js_block should succeed");
        assert_eq!(
            result.console_logs,
            vec!["undefined undefined undefined object function object"]
        );
        handle.shutdown().await.expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_context_with_special_chars() {
        let context = "line 1\nline 2\ttab\r\n\"quoted\" and \\backslash";
        let mut runtime = test_runtime(context).await;
        let value = eval_to_string(&mut runtime, "context").await;
        assert_eq!(value, context);
    }

    #[tokio::test]
    async fn test_tool_output_parsed_from_json_context() {
        let context = r#"{"items":["a","b"],"count":2}"#;
        let tool_output: serde_json::Value = serde_json::from_str(context).unwrap();
        let mut runtime = test_runtime_with_tool_output(context, Some(&tool_output)).await;
        let value = eval_to_string(&mut runtime, "toolOutput.count").await;
        assert_eq!(value, "2");
        let items = eval_to_string(&mut runtime, "toolOutput.items.length").await;
        assert_eq!(items, "2");
    }

    #[tokio::test]
    async fn test_tool_output_undefined_for_non_json_context() {
        let mut runtime = test_runtime_with_tool_output("not json at all", None).await;
        let value = eval_to_string(&mut runtime, "typeof toolOutput").await;
        assert_eq!(value, "undefined");
    }

    #[tokio::test]
    async fn test_tool_output_not_set_for_json_context_without_explicit_tool_output() {
        // Even though the context is valid JSON, toolOutput should not be set
        // when tool_output is None — verifying the old heuristic is gone.
        let context = r#"{"items":["a","b"],"count":2}"#;
        let mut runtime = test_runtime_with_tool_output(context, None).await;
        let value = eval_to_string(&mut runtime, "typeof toolOutput").await;
        assert_eq!(value, "undefined");
    }

    #[tokio::test]
    async fn test_final_stores_value() {
        let mut runtime = test_runtime("data").await;
        execute_and_run(&mut runtime, "<test>", "FINAL('the answer');", TEST_TIMEOUT)
            .await
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(
            take_final_answer(&mut runtime),
            Some("the answer".to_string())
        );
    }

    #[tokio::test]
    async fn test_final_with_object() {
        let mut runtime = test_runtime("data").await;
        execute_and_run(
            &mut runtime,
            "<test>",
            "FINAL({key: 'value', num: 42});",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        let answer = take_final_answer(&mut runtime);
        assert!(answer.is_some());
        let parsed: serde_json::Value = serde_json::from_str(&answer.unwrap_or_default())
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(parsed["key"], "value");
        assert_eq!(parsed["num"], 42);
    }

    #[tokio::test]
    async fn test_final_empty_string_ignored() {
        let mut runtime = test_runtime("data").await;
        execute_and_run(&mut runtime, "<test>", "FINAL('');", TEST_TIMEOUT)
            .await
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(take_final_answer(&mut runtime), None);
    }

    #[tokio::test]
    async fn test_console_log_captures_output() {
        let mut runtime = test_runtime("data").await;
        execute_and_run(
            &mut runtime,
            "<test>",
            "console.log('hello'); console.log('world', 42);",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        let logs = take_console_logs(&mut runtime);
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0], "hello");
        assert_eq!(logs[1], "world 42");
    }

    #[tokio::test]
    async fn test_console_log_truncates_large_output() {
        let mut runtime = test_runtime("data").await;
        // Generate a message larger than MAX_CONSOLE_LOG_CHARS (5000)
        let large_msg = "x".repeat(10_000);
        let code = format!("console.log('{large_msg}');");
        execute_and_run(&mut runtime, "<test>", &code, TEST_TIMEOUT)
            .await
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        let logs = take_console_logs(&mut runtime);
        assert_eq!(logs.len(), 1);
        assert!(
            logs[0].len() < 6_000,
            "Truncated log should be much shorter than original 10000 chars, got {}",
            logs[0].len()
        );
        assert!(
            logs[0].contains("... (truncated, 10000 chars total)"),
            "Should contain truncation suffix, got: {}",
            &logs[0][logs[0].len().saturating_sub(60)..]
        );
    }

    #[tokio::test]
    async fn test_console_log_truncates_using_character_count() {
        let mut runtime = test_runtime("data").await;
        let large_msg = "界".repeat(MAX_CONSOLE_LOG_CHARS + 1);
        let code = format!("console.log('{large_msg}');");
        execute_and_run(&mut runtime, "<test>", &code, TEST_TIMEOUT)
            .await
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));

        let logs = take_console_logs(&mut runtime);
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("truncated"));
        assert!(logs[0].contains(&format!("{} chars total", MAX_CONSOLE_LOG_CHARS + 1)));
    }

    #[tokio::test]
    async fn test_console_log_does_not_truncate_small_output() {
        let mut runtime = test_runtime("data").await;
        let small_msg = "y".repeat(100);
        let code = format!("console.log('{small_msg}');");
        execute_and_run(&mut runtime, "<test>", &code, TEST_TIMEOUT)
            .await
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        let logs = take_console_logs(&mut runtime);
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0], small_msg);
    }

    #[tokio::test]
    async fn test_js_syntax_error() {
        let mut runtime = test_runtime("data").await;
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "this is not valid javascript {{{",
            TEST_TIMEOUT,
        )
        .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("JS execution error")
        );
    }

    #[tokio::test]
    async fn test_js_runtime_error() {
        let mut runtime = test_runtime("data").await;
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "throw new Error('intentional');",
            TEST_TIMEOUT,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_infinite_loop_timeout() {
        let mut runtime = test_runtime("data").await;
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "while(true) {}",
            Duration::from_secs(2),
        )
        .await;
        assert!(
            result.is_err(),
            "Infinite loop should be terminated by timeout"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("timed out"),
            "Expected timeout error, got: {err}"
        );
        execute_and_run(
            &mut runtime,
            "<after_timeout>",
            "FINAL('recovered');",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("Runtime should recover: {e}")));
        assert_eq!(
            take_final_answer(&mut runtime),
            Some("recovered".to_string())
        );
    }

    #[test]
    fn test_pool_rejects_zero_concurrency() {
        let result = RlmPool::new(0);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("at least 1"),
            "Expected 'at least 1' in error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_llm_query_calls_inference() {
        let mut mock = MockTensorZeroClient::new();
        mock.expect_inference()
            .times(1)
            .returning(|_| Box::pin(async { Ok(mock_chat_response(&["LLM says hello"])) }));
        let client: Arc<dyn TensorZeroClient> = Arc::new(mock);
        let mut runtime = test_runtime_at_max_depth("data", client).await;
        execute_and_run(
            &mut runtime,
            "<test>",
            "llm_query('test prompt').then(r => { console.log(r); });",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        let logs = take_console_logs(&mut runtime);
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0], "LLM says hello");
    }

    #[tokio::test]
    async fn test_llm_query_batched_calls_inference() {
        let mut mock = MockTensorZeroClient::new();
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();
        mock.expect_inference().times(3).returning(move |_| {
            let n = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move { Ok(mock_chat_response(&[&format!("response_{n}")])) })
        });
        let client: Arc<dyn TensorZeroClient> = Arc::new(mock);
        let mut runtime = test_runtime_at_max_depth("data", client).await;
        execute_and_run(
            &mut runtime,
            "<test>",
            "llm_query_batched(['a', 'bb', 'ccc']).then(r => { console.log(JSON.stringify(r)); });",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        let logs = take_console_logs(&mut runtime);
        assert_eq!(logs.len(), 1);
        let parsed: Vec<String> = serde_json::from_str(&logs[0])
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(parsed.len(), 3);
    }

    #[tokio::test]
    async fn test_llm_query_timeout_is_enforced() {
        let mut mock = MockTensorZeroClient::new();
        mock.expect_inference().times(1).returning(|_| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(mock_chat_response(&["late response"]))
            })
        });
        let client: Arc<dyn TensorZeroClient> = Arc::new(mock);
        let mut runtime = test_runtime_at_max_depth_with_timeout("data", client, 1).await;
        let start = std::time::Instant::now();
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "llm_query('slow prompt').then(r => { console.log(r); });",
            TEST_TIMEOUT,
        )
        .await;
        assert!(result.is_err(), "llm_query should fail due to timeout");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("timed out"),
            "Expected timeout error, got: {err}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "Timeout should fire quickly"
        );
        execute_and_run(
            &mut runtime,
            "<after_llm_timeout>",
            "FINAL('recovered_after_llm_timeout');",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("Runtime should recover: {e}")));
        assert_eq!(
            take_final_answer(&mut runtime),
            Some("recovered_after_llm_timeout".to_string())
        );
    }

    #[tokio::test]
    async fn test_llm_query_batched_times_out_if_any_prompt_stalls() {
        let mut mock = MockTensorZeroClient::new();
        mock.expect_inference().times(3).returning(|params| {
            let text = first_inference_text(&params);
            Box::pin(async move {
                if text.contains("slow_batch_prompt") {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    Ok(mock_chat_response(&["late response"]))
                } else {
                    Ok(mock_chat_response(&["ok"]))
                }
            })
        });
        let client: Arc<dyn TensorZeroClient> = Arc::new(mock);
        let mut runtime = test_runtime_at_max_depth_with_timeout("data", client, 1).await;
        let start = std::time::Instant::now();
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "llm_query_batched(['fast_a', 'slow_batch_prompt', 'fast_b']).then(r => { console.log(JSON.stringify(r)); });",
            TEST_TIMEOUT,
        )
        .await;
        assert!(
            result.is_err(),
            "batched llm_query should fail if one prompt exceeds timeout"
        );
        assert!(result.unwrap_err().to_string().contains("timed out"));
        assert!(start.elapsed() < Duration::from_secs(3));
        execute_and_run(
            &mut runtime,
            "<after_batch_timeout>",
            "FINAL('batch_recovered');",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("Runtime should recover: {e}")));
        assert_eq!(
            take_final_answer(&mut runtime),
            Some("batch_recovered".to_string())
        );
    }

    #[tokio::test]
    async fn test_take_final_answer_clears() {
        let mut runtime = test_runtime("data").await;
        execute_and_run(&mut runtime, "<test>", "FINAL('first');", TEST_TIMEOUT)
            .await
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(take_final_answer(&mut runtime), Some("first".to_string()));
        assert_eq!(take_final_answer(&mut runtime), None);
    }

    #[tokio::test]
    async fn test_take_console_logs_clears() {
        let mut runtime = test_runtime("data").await;
        execute_and_run(&mut runtime, "<test>", "console.log('msg');", TEST_TIMEOUT)
            .await
            .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        let logs = take_console_logs(&mut runtime);
        assert_eq!(logs.len(), 1);
        let logs = take_console_logs(&mut runtime);
        assert!(logs.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_code_blocks_share_state() {
        let mut runtime = test_runtime("data").await;
        execute_and_run(
            &mut runtime,
            "<block_0>",
            "globalThis.counter = 0;",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        execute_and_run(
            &mut runtime,
            "<block_1>",
            "globalThis.counter += 10;",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        execute_and_run(
            &mut runtime,
            "<block_2>",
            "FINAL(String(globalThis.counter));",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(take_final_answer(&mut runtime), Some("10".to_string()));
    }

    #[tokio::test]
    async fn test_let_redeclaration_across_blocks_works() {
        // Each code block is wrapped in an async IIFE, so `let`/`const`
        // declarations are scoped and don't conflict across blocks.
        let tool_output = serde_json::json!([1, 2, 3]);
        let mut runtime = test_runtime_with_tool_output(r"[1,2,3]", Some(&tool_output)).await;
        execute_and_run(
            &mut runtime,
            "<iter_0>",
            "let items = toolOutput;\nconsole.log('len', items.length);",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        // Same `let items` in a new block — works because of IIFE wrapping
        execute_and_run(
            &mut runtime,
            "<iter_1>",
            "let items = toolOutput;\nFINAL(String(items.length));",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(take_final_answer(&mut runtime), Some("3".to_string()));
    }

    #[tokio::test]
    async fn test_globalthis_persists_across_blocks() {
        // globalThis assignments persist across IIFE-wrapped blocks.
        let tool_output = serde_json::json!([1, 2, 3]);
        let mut runtime = test_runtime_with_tool_output(r"[1,2,3]", Some(&tool_output)).await;
        execute_and_run(
            &mut runtime,
            "<iter_0>",
            "globalThis.myItems = toolOutput;",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        execute_and_run(
            &mut runtime,
            "<iter_1>",
            "FINAL(String(globalThis.myItems.length));",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(take_final_answer(&mut runtime), Some("3".to_string()));
    }

    #[tokio::test]
    async fn test_internal_globals_are_frozen() {
        // User code cannot overwrite internal globals like FINAL, console, etc.
        let tool_output = serde_json::json!({"x": 1});
        let mut runtime = test_runtime_with_tool_output(r#"{"x":1}"#, Some(&tool_output)).await;
        // Attempting to overwrite FINAL should throw (strict mode in IIFE)
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "FINAL = function() {};",
            TEST_TIMEOUT,
        )
        .await;
        assert!(result.is_err(), "overwriting FINAL should fail");

        // toolOutput should also be read-only
        let result = execute_and_run(
            &mut runtime,
            "<test2>",
            "toolOutput = 'hacked';",
            TEST_TIMEOUT,
        )
        .await;
        assert!(result.is_err(), "overwriting toolOutput should fail");

        // But the originals should still work
        execute_and_run(
            &mut runtime,
            "<test3>",
            "FINAL(String(toolOutput.x));",
            TEST_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| std::panic::panic_any(format!("{e}")));
        assert_eq!(take_final_answer(&mut runtime), Some("1".to_string()));
    }

    fn mock_chat_response(texts: &[&str]) -> InferenceResponse {
        use tensorzero_types::Text as T0Text;
        InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::nil(),
            episode_id: Uuid::nil(),
            variant_name: "test".to_string(),
            content: texts
                .iter()
                .map(|t| {
                    ContentBlockChatOutput::Text(T0Text {
                        text: (*t).to_string(),
                    })
                })
                .collect(),
            usage: Usage::default(),
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        })
    }

    #[tokio::test]
    async fn test_fetch_unavailable() {
        let mut runtime = test_runtime("data").await;
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "fetch('http://example.com').then(r => console.log(r));",
            TEST_TIMEOUT,
        )
        .await;
        assert!(
            result.is_err(),
            "fetch() should not be available in sandbox"
        );
    }

    #[tokio::test]
    async fn test_deno_namespace_hidden_by_compartment() {
        // Deno is not endowed, so it is completely invisible inside the Compartment.
        let mut runtime = test_runtime("data").await;
        let value = eval_to_string(&mut runtime, "typeof Deno").await;
        assert_eq!(value, "undefined");
    }

    #[tokio::test]
    async fn test_prototype_pollution_blocked() {
        // SES lockdown() freezes all intrinsics — prototype mutation should throw.
        let mut runtime = test_runtime("data").await;
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "Array.prototype.polluted = true;",
            TEST_TIMEOUT,
        )
        .await;
        assert!(
            result.is_err(),
            "Prototype pollution should be blocked by SES lockdown"
        );
    }

    #[tokio::test]
    async fn test_ops_inaccessible_from_compartment() {
        // Direct op access is impossible from inside the Compartment.
        let mut runtime = test_runtime("data").await;
        let result = execute_and_run(
            &mut runtime,
            "<test>",
            "Deno.core.ops.op_set_final('hacked');",
            TEST_TIMEOUT,
        )
        .await;
        assert!(
            result.is_err(),
            "Direct op access should fail inside Compartment"
        );
    }

    #[tokio::test]
    async fn test_internal_tz_globals_hidden() {
        // __t0_* bridge functions on the outer globalThis should not be visible.
        let mut runtime = test_runtime("data").await;
        let value = eval_to_string(&mut runtime, "typeof __t0_FINAL").await;
        assert_eq!(value, "undefined");
        let value = eval_to_string(&mut runtime, "typeof __t0_compartment").await;
        assert_eq!(value, "undefined");
        let value = eval_to_string(&mut runtime, "typeof __t0_create_rlm_compartment").await;
        assert_eq!(value, "undefined");
    }

    #[tokio::test]
    async fn test_unicode_context_injection() {
        let context = "Hello \u{4e16}\u{754c} \u{1f600} \u{0645}\u{0631}\u{062d}\u{0628}\u{0627}";
        let mut runtime = test_runtime(context).await;
        let value = eval_to_string(&mut runtime, "context").await;
        assert_eq!(value, context);
    }

    #[tokio::test]
    async fn test_unicode_context_char_length() {
        let context = "\u{4e16}\u{754c}";
        let mut runtime = test_runtime(context).await;
        let len = eval_to_string(&mut runtime, "context.length").await;
        assert_eq!(len, "2");
    }

    #[tokio::test]
    async fn test_unicode_emoji_surrogate_pair() {
        let context = "\u{1f600}";
        let mut runtime = test_runtime(context).await;
        let value = eval_to_string(&mut runtime, "context").await;
        assert_eq!(value, context);
        let len = eval_to_string(&mut runtime, "context.length").await;
        assert_eq!(len, "2");
    }

    #[tokio::test]
    async fn test_child_runtime_drop_does_not_cancel_parent() {
        let parent_permit = RlmPermit::for_test();
        let client: Arc<dyn TensorZeroClient> = Arc::new(MockTensorZeroClient::new());
        let rlm_state = test_rlm_state("child ctx", 1, 3, 30);
        let child_handle = spawn_child_runtime(
            RuntimeParams {
                t0_client: client,
                extra_inference_tags: test_extra_inference_tags(),
                mode: RuntimeMode::Rlm {
                    input: RlmRuntimeInput::Context("child context".to_string()),
                    rlm_state,
                },
                ts_checker: test_ts_checker().await,
                exposed_tools: None,
                oom_snapshot_config: None,
            },
            parent_permit.child_permit(),
        )
        .await
        .expect("spawn_child_runtime should succeed");
        child_handle
            .execute_js_block("<child>".to_string(), "FINAL('done');", TEST_TIMEOUT)
            .await
            .expect("child execute_js_block should succeed");
        drop(child_handle);
        let result = parent_permit
            .run_with_cancellation(async { "parent still alive" })
            .await;
        assert_eq!(
            result.expect("parent should not be cancelled"),
            "parent still alive"
        );
    }

    #[test]
    fn test_extract_text_single_block() {
        let response = mock_chat_response(&["hello world"]);
        let text = extract_text_from_response(&response);
        assert!(text.is_ok());
        assert_eq!(text.unwrap_or_default(), "hello world");
    }

    #[test]
    fn test_extract_text_multiple_blocks() {
        let response = mock_chat_response(&["first", "second"]);
        let text = extract_text_from_response(&response);
        assert!(text.is_ok());
        assert_eq!(text.unwrap_or_default(), "first\nsecond");
    }

    #[test]
    fn test_extract_text_no_text_blocks() {
        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::nil(),
            episode_id: Uuid::nil(),
            variant_name: "test".to_string(),
            content: vec![],
            usage: Usage::default(),
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        });
        let result = extract_text_from_response(&response);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no text blocks"));
    }

    #[test]
    fn test_extract_text_json_response() {
        use tensorzero_types::{JsonInferenceOutput, JsonInferenceResponse};
        let response = InferenceResponse::Json(JsonInferenceResponse {
            inference_id: Uuid::nil(),
            episode_id: Uuid::nil(),
            variant_name: "test".to_string(),
            output: JsonInferenceOutput {
                raw: Some("{\"key\":\"value\"}".to_string()),
                parsed: Some(serde_json::json!({"key": "value"})),
            },
            usage: Usage::default(),
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        });
        let result = extract_text_from_response(&response);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Json"));
    }
}
