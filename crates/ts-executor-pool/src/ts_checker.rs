//! TypeScript type-checking pool for RLM code blocks.
//!
//! Provides a pool of V8 runtimes with the TypeScript compiler preloaded
//! via a V8 snapshot created at build time. Each runtime can check and
//! transpile TypeScript code against ambient declarations, returning either
//! stripped JavaScript or type error diagnostics.

use tensorzero_ts_types::TsTypeBundle;

use super::error::TsError;
use std::fmt::Write;
use std::time::Duration;

/// Holds data extracted from the tool registry for a given exposed tool
pub struct ExposedToolData {
    pub name: String,
    pub param_type: TsTypeBundle,
    pub param_type_name: String,
    pub output_type: TsTypeBundle,
    pub output_type_name: String,
}

/// Pre-built V8 snapshot containing the TypeScript compiler and checker helper.
/// Created by build.rs at compile time. The path includes content hashes so
/// the snapshot is regenerated when either `typescript.js` or `ts_checker.js` changes.
const TS_CHECKER_SNAPSHOT: &[u8] = include_bytes!(env!("TS_CHECKER_SNAPSHOT_PATH"));

/// Static ambient declarations for the full RLM sandbox.
const STATIC_RLM_AMBIENT: &str = include_str!("js/rlm_ambient.d.ts");

/// Marker separating the shared ES ambient types from the RLM-specific globals.
const RLM_SANDBOX_GLOBALS_MARKER: &str = "// === RLM Sandbox Globals ===";

/// Ambient globals available in direct code-execution mode.
const CODE_EXECUTION_AMBIENT_GLOBALS: &str = r"
// === Code Execution Globals ===

/** Console for logging intermediate results. */
declare var console: {
  log(...args: unknown[]): void;
};

/**
 * Report a final value back to the Rust caller via
 * `ExecuteBlockResult::result`. Terminates execution; code after the call
 * does not run. Pair with a runtime check on the returned `Option<String>`.
 */
declare function FINAL(value: string): never;
";

/// Result of a successful typecheck: the transpiled JavaScript with types stripped.
#[derive(Debug)]
pub struct TranspileResult {
    /// JavaScript code with all TypeScript type annotations removed.
    pub js_code: String,
}

/// Result of a failed typecheck: formatted diagnostic messages.
#[derive(Debug)]
pub struct TypeCheckError {
    /// Human-readable TypeScript diagnostic messages.
    pub diagnostics: String,
}

/// Result of [`TsCheckerPool::check_and_transpile`].
pub type TypeCheckResult = Result<TranspileResult, TypeCheckError>;

/// Result of preparing a TypeScript block for execution.
#[derive(Debug, PartialEq, Eq)]
pub enum PreparedTypescriptBlock {
    Ready(String),
    TypeError(String),
}

/// Command sent from the public API to a checker worker thread.
enum CheckerCommand {
    Check {
        code: String,
        ambient: String,
        reply: tokio::sync::oneshot::Sender<Result<TypeCheckResult, TsError>>,
    },
}

/// Handle to a single V8 worker thread. This is the bb8 "connection".
pub struct TsCheckerWorker {
    sender: std::sync::mpsc::Sender<CheckerCommand>,
    broken: bool,
}

/// bb8 connection manager that spawns V8 worker threads.
struct TsCheckerManager;

impl bb8::ManageConnection for TsCheckerManager {
    type Connection = TsCheckerWorker;
    type Error = TsError;

    async fn connect(&self) -> Result<Self::Connection, Self::Error> {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<CheckerCommand>();
        let (init_tx, init_rx) = tokio::sync::oneshot::channel::<Result<(), TsError>>();

        std::thread::Builder::new()
            .name("ts-checker".to_string())
            .spawn(move || {
                // Create a V8 runtime from the pre-built snapshot
                let mut runtime = deno_core::JsRuntime::new(deno_core::RuntimeOptions {
                    startup_snapshot: Some(TS_CHECKER_SNAPSHOT),
                    ..Default::default()
                });

                // Verify the snapshot loaded correctly by checking for __t0_check_and_transpile
                if let Err(e) = runtime.execute_script(
                    "<verify>",
                    "if (typeof __t0_check_and_transpile !== 'function') throw new Error('snapshot missing __t0_check_and_transpile');".to_string(),
                ) {
                    let _ = init_tx.send(Err(TsError::JsRuntime {
                        message: format!("Snapshot verification failed: {e}"),
                    }));
                    return;
                }

                // Signal successful initialization
                let _ = init_tx.send(Ok(()));

                // Process check requests
                while let Ok(cmd) = cmd_rx.recv() {
                    match cmd {
                        CheckerCommand::Check {
                            code,
                            ambient,
                            reply,
                        } => {
                            let result = run_check(&mut runtime, &code, &ambient);
                            let _ = reply.send(result);
                        }
                    }
                }
            })
            .map_err(|e| TsError::JsRuntime {
                message: format!("Failed to spawn ts-checker thread: {e}"),
            })?;

        init_rx.await.map_err(|_| TsError::JsRuntime {
            message: "ts-checker thread dropped init sender without responding".to_string(),
        })??;

        Ok(TsCheckerWorker {
            sender: cmd_tx,
            broken: false,
        })
    }

    async fn is_valid(&self, _conn: &mut Self::Connection) -> Result<(), Self::Error> {
        // No-op: broken connections are detected via has_broken
        Ok(())
    }

    fn has_broken(&self, conn: &mut Self::Connection) -> bool {
        conn.broken
    }
}

/// A pool of V8 runtimes with the TypeScript compiler preloaded.
///
/// Uses bb8 to manage worker threads. Each worker loads a V8 snapshot at
/// startup containing `typescript.js` and `ts_checker.js`, providing
/// near-instant initialization.
pub struct TsCheckerPool {
    pool: bb8::Pool<TsCheckerManager>,
}

impl std::fmt::Debug for TsCheckerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.pool.state();
        f.debug_struct("TsCheckerPool")
            .field("size", &state.connections)
            .field("idle", &state.idle_connections)
            .finish_non_exhaustive()
    }
}

impl TsCheckerPool {
    /// Create a new pool with the given number of checker threads.
    ///
    /// Each thread creates a `deno_core::JsRuntime` from a pre-built V8
    /// snapshot containing the TypeScript compiler and waits for check requests.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidConfig`] if `pool_size == 0`, or
    /// [`TsError::JsRuntime`] if a worker thread fails to initialize.
    pub async fn new(pool_size: usize) -> Result<Self, TsError> {
        if pool_size == 0 {
            return Err(TsError::InvalidConfig {
                message: "TsCheckerPool pool_size must be greater than 0".to_string(),
            });
        }

        let pool_size_u32 = u32::try_from(pool_size).map_err(|_| TsError::InvalidConfig {
            message: format!("pool_size {pool_size} exceeds u32::MAX"),
        })?;

        let pool = bb8::Pool::builder()
            .max_size(pool_size_u32)
            .min_idle(Some(pool_size_u32))
            .max_lifetime(None)
            .idle_timeout(None)
            .test_on_check_out(false)
            .build(TsCheckerManager)
            .await
            .map_err(|e| TsError::JsRuntime {
                message: format!("Failed to build TsCheckerPool: {e}"),
            })?;

        Ok(Self { pool })
    }

    /// Typecheck TypeScript code against ambient declarations, returning
    /// transpiled JavaScript on success or diagnostics on failure.
    ///
    /// # Errors
    ///
    /// Returns `TsError::PoolShutdown` if the pool is unavailable.
    pub async fn check_and_transpile(
        &self,
        code: &str,
        ambient: &str,
    ) -> Result<TypeCheckResult, TsError> {
        let mut conn = self.pool.get().await.map_err(|_| TsError::PoolShutdown)?;

        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();

        if conn
            .sender
            .send(CheckerCommand::Check {
                code: code.to_string(),
                ambient: ambient.to_string(),
                reply: reply_tx,
            })
            .is_err()
        {
            conn.broken = true;
            return Err(TsError::PoolShutdown);
        }

        let result = reply_rx.await.map_err(|_| {
            conn.broken = true;
            TsError::PoolShutdown
        })?;

        // Mark the worker as broken on infrastructure errors (TsError) so
        // bb8 discards the potentially-corrupted V8 runtime.
        // TypeCheckError is a normal type-check failure — the worker is fine.
        if result.is_err() {
            conn.broken = true;
        }

        result
    }
}

/// Typecheck and transpile a block, preserving user-facing diagnostics so the
/// caller can decide whether to treat type errors as retryable or fatal.
pub async fn prepare_typescript_block_outcome(
    checker: &TsCheckerPool,
    ambient: &str,
    code: &str,
    timeout: Duration,
) -> Result<PreparedTypescriptBlock, TsError> {
    match tokio::time::timeout(timeout, checker.check_and_transpile(code, ambient)).await {
        Ok(Ok(Ok(transpiled))) => Ok(PreparedTypescriptBlock::Ready(transpiled.js_code)),
        Ok(Ok(Err(type_err))) => Ok(PreparedTypescriptBlock::TypeError(type_err.diagnostics)),
        Ok(Err(e)) => Err(TsError::TypeCheck {
            message: format!("TypeScript checker infrastructure error: {e}"),
        }),
        Err(_elapsed) => Err(TsError::TypeCheck {
            message: "TypeScript type check timed out".to_string(),
        }),
    }
}

/// Run the typecheck synchronously on a worker thread's runtime.
fn run_check(
    runtime: &mut deno_core::JsRuntime,
    code: &str,
    ambient: &str,
) -> Result<TypeCheckResult, TsError> {
    let escaped_code = serde_json::to_string(code).map_err(|e| TsError::JsRuntime {
        message: format!("Failed to serialize code for checker: {e}"),
    })?;
    let escaped_ambient = serde_json::to_string(ambient).map_err(|e| TsError::JsRuntime {
        message: format!("Failed to serialize ambient for checker: {e}"),
    })?;

    let script = format!("__t0_check_and_transpile({escaped_code}, {escaped_ambient})");

    let result = runtime
        .execute_script("<ts_check>", script)
        .map_err(|e| TsError::JsRuntime {
            message: format!("TypeScript checker script error: {e}"),
        })?;

    // Extract the string result from V8
    let result_str = {
        deno_core::scope!(scope, runtime);
        let local = result.open(scope);
        local.to_rust_string_lossy(scope)
    };

    // Parse the JSON result
    let parsed: serde_json::Value =
        serde_json::from_str(&result_str).map_err(|e| TsError::JsRuntime {
            message: format!("Failed to parse checker result: {e}"),
        })?;

    if parsed["ok"].as_bool() == Some(true) {
        let js = parsed["js"]
            .as_str()
            .ok_or_else(|| TsError::JsRuntime {
                message: "Checker returned ok=true but no js field".to_string(),
            })?
            .to_string();
        Ok(Ok(TranspileResult { js_code: js }))
    } else {
        let diagnostics = parsed["diagnostics"]
            .as_str()
            .unwrap_or("Unknown type error")
            .to_string();
        Ok(Err(TypeCheckError { diagnostics }))
    }
}

/// Build ambient declarations by combining tool-specific types with the
/// static RLM ambient declarations.
///
/// When `output_ts_type` is `Some`, the tool type bundle is prepended to
/// the static ambient file and a typed `toolOutput` declaration is appended.
/// When `None`, only the static ambient file is used.
pub fn build_ambient_declarations(output_ts_type: Option<&str>) -> String {
    match output_ts_type {
        Some(ts_type) => {
            let tool_output_decl = match extract_root_type_name(ts_type) {
                Some(name) => format!("declare var toolOutput: {name};"),
                None => "declare var toolOutput: any;".to_string(),
            };
            format!("{ts_type}\n\n{STATIC_RLM_AMBIENT}\n{tool_output_decl}\n")
        }
        None => {
            format!("{STATIC_RLM_AMBIENT}\ndeclare var toolOutput: any;\n")
        }
    }
}

/// Shared helper: writes the `TensorzeroToolHandle`, `join_tool`, `ToolClient` interface,
/// and `toolClient` global declaration.
fn write_tool_client_interface(tool_ts: &mut String, exposed_tools: &[ExposedToolData]) {
    tool_ts.push_str(
        "/** An opaque type produced by methods in `ToolClient`, which can be passed to `join_tool` to get the result */\n\
interface TensorzeroToolHandle<T> {}\n\
declare function join_tool<T>(handle: TensorzeroToolHandle<T>): Promise<T>;\n",
    );

    tool_ts.push_str("interface ToolClient {\n");
    for tool in exposed_tools {
        let tool_name = &tool.name;
        let tool_param = &tool.param_type_name;
        let tool_output = &tool.output_type_name;
        #[expect(clippy::expect_used)]
        writeln!(
            tool_ts,
            "{tool_name}(params: {tool_param}): Promise<TensorzeroToolHandle<{tool_output}>>;"
        )
        .expect("writing to String should not fail");
    }
    tool_ts.push_str("}\n");
    tool_ts.push_str("declare var toolClient: ToolClient;\n");
}

/// Build the full TypeScript declarations for `join_tool` and `toolClient`,
/// including type body definitions for each tool's parameters and output.
pub fn build_tool_client_declarations(exposed_tools: &[ExposedToolData]) -> String {
    let mut tool_ts = exposed_tools
        .iter()
        .map(|tool| format!("{}\n{}", tool.param_type.0, tool.output_type.0))
        .collect::<Vec<String>>()
        .join("\n");
    if !tool_ts.is_empty() {
        tool_ts.push('\n');
    }
    write_tool_client_interface(&mut tool_ts, exposed_tools);
    tool_ts
}

/// Build a slim `ToolClient` interface listing only method signatures (type names, no bodies).
///
/// Unlike [`build_tool_client_declarations`], this omits the full TypeScript type body
/// definitions for each tool's parameters and output. The LLM can use `describe_tool`
/// to discover those on demand.
pub fn build_tool_client_interface_only(exposed_tools: &[ExposedToolData]) -> String {
    let mut tool_ts = String::new();
    write_tool_client_interface(&mut tool_ts, exposed_tools);
    tool_ts
}

/// Build ambient declarations for direct code execution.
///
/// This mode exposes only the globals that exist in `RuntimeMode::CodeExecution`:
/// `console`, `join_tool`, and `toolClient`.
pub fn build_code_execution_ambient_declarations(exposed_tools: &[ExposedToolData]) -> String {
    let static_ambient = STATIC_RLM_AMBIENT
        .split_once(RLM_SANDBOX_GLOBALS_MARKER)
        .map_or(STATIC_RLM_AMBIENT, |(prefix, _)| prefix)
        .trim_end();
    let tool_client = build_tool_client_declarations(exposed_tools);
    format!("{static_ambient}\n\n{CODE_EXECUTION_AMBIENT_GLOBALS}\n{tool_client}")
}

/// Extract the root type name from a TypeScript type bundle string.
///
/// Bundles are topologically sorted (dependencies first, root type last),
/// so we find the last `type X =` or `interface X {` declaration.
/// Also handles `export type` and `export interface` prefixes.
fn extract_root_type_name(bundle: &str) -> Option<&str> {
    bundle.lines().rev().find_map(|line| {
        let trimmed = line.trim();
        // Strip optional `export` prefix
        let trimmed = trimmed.strip_prefix("export ").unwrap_or(trimmed);
        // Match `type X` or `interface X`
        let rest = trimmed
            .strip_prefix("type ")
            .or_else(|| trimmed.strip_prefix("interface "))?;
        // Extract the identifier: everything up to the first non-identifier char
        let name_end = rest
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        let name = &rest[..name_end];
        if name.is_empty() { None } else { Some(name) }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    static SHARED_POOL: tokio::sync::OnceCell<TsCheckerPool> = tokio::sync::OnceCell::const_new();

    async fn shared_pool() -> &'static TsCheckerPool {
        SHARED_POOL
            .get_or_init(|| async {
                TsCheckerPool::new(2)
                    .await
                    .unwrap_or_else(|e| panic!("Failed to create TsCheckerPool: {e}"))
            })
            .await
    }

    fn static_ambient() -> String {
        build_ambient_declarations(None)
    }

    fn fake_exposed_tools() -> Vec<ExposedToolData> {
        vec![ExposedToolData {
            name: "list_items".to_string(),
            param_type: tensorzero_ts_types::TsTypeBundle(
                "type ListItemsParams = { limit: number };",
            ),
            param_type_name: "ListItemsParams".to_string(),
            output_type: tensorzero_ts_types::TsTypeBundle("type ListItemsOutput = string[];"),
            output_type_name: "ListItemsOutput".to_string(),
        }]
    }

    #[tokio::test]
    async fn test_prepare_typescript_block_outcome_returns_type_error() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let outcome = prepare_typescript_block_outcome(
            pool,
            &ambient,
            "const total: number = 'bad';",
            Duration::from_secs(5),
        )
        .await
        .unwrap();

        match outcome {
            PreparedTypescriptBlock::TypeError(diagnostics) => {
                assert!(
                    diagnostics.contains("number"),
                    "Expected 'number' in diagnostics: {diagnostics}"
                );
            }
            other @ PreparedTypescriptBlock::Ready(_) => {
                panic!("Expected TypeError, got: {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn test_valid_ts_passes_typecheck() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let code = r"
            const x: number = 42;
            const y: string = x.toString();
            console.log(y);
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(result.is_ok(), "Valid TS should pass typecheck");
        let transpiled = result.unwrap();
        assert!(
            !transpiled.js_code.contains(": number"),
            "Types should be stripped"
        );
        assert!(
            transpiled.js_code.contains("const x"),
            "JS code should be preserved"
        );
    }

    #[tokio::test]
    async fn test_type_error_returns_diagnostics() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let code = "const x: number = \"not a number\";";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(result.is_err(), "Type error should fail typecheck");
        let err = result.unwrap_err();
        assert!(
            !err.diagnostics.is_empty(),
            "Diagnostics should be non-empty"
        );
    }

    #[tokio::test]
    async fn test_worker_not_broken_after_type_error() {
        // Use a pool of size 1 to ensure the same worker handles both requests.
        // A type error (TypeCheckError) is a normal result — the worker should
        // NOT be marked as broken and should be reusable for subsequent checks.
        let pool = TsCheckerPool::new(1).await.unwrap();
        let ambient = static_ambient();

        // First: cause a type error
        let bad_code = "const x: number = \"not a number\";";
        let result = pool.check_and_transpile(bad_code, &ambient).await.unwrap();
        assert!(result.is_err(), "Type error should fail typecheck");

        // Second: valid code should still work (same worker, not marked broken)
        let good_code = "const x: number = 42;";
        let result = pool.check_and_transpile(good_code, &ambient).await.unwrap();
        assert!(result.is_ok(), "Valid code should pass after type error");
    }

    #[tokio::test]
    async fn test_plain_js_passes_typecheck() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let code = r"
            var x = 42;
            var y = x.toString();
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(result.is_ok(), "Plain JS should pass typecheck");
    }

    #[tokio::test]
    async fn test_declare_var_for_cross_iteration_state() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let code = "\
            declare var myState: string[];\n\
            if (typeof myState === \"undefined\") {\n\
                myState = [];\n\
            }\n\
            myState.push(\"item\");\n\
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(result.is_ok(), "declare var pattern should pass typecheck");
    }

    #[tokio::test]
    async fn test_concurrent_checks() {
        let ambient = static_ambient();

        let mut handles = Vec::new();
        for i in 0..8 {
            let ambient = ambient.clone();
            #[expect(
                clippy::disallowed_methods,
                reason = "Test helper spawning concurrent checks"
            )]
            handles.push(tokio::spawn(async move {
                let pool = shared_pool().await;
                let code = format!("const v{i}: number = {i};");
                let result = pool.check_and_transpile(&code, &ambient).await.unwrap();
                assert!(result.is_ok(), "Concurrent check {i} should pass");
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_ambient_with_tool_types() {
        let pool = shared_pool().await;
        let tool_type = "type ToolOutput = { items: Array<{ id: string; name: string }> };";
        let ambient = build_ambient_declarations(Some(tool_type));

        let code = r"
            const data: ToolOutput = JSON.parse(context);
            const names: string[] = data.items.map((item) => item.name);
            console.log(names);
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(
            result.is_ok(),
            "Code using tool-specific types should pass: {:?}",
            result.err().map(|e| e.diagnostics)
        );
    }

    #[tokio::test]
    async fn test_tool_output_typed_access() {
        let pool = shared_pool().await;
        let tool_type = "type ToolOutput = { items: Array<{ id: string; name: string }> };";
        let ambient = build_ambient_declarations(Some(tool_type));

        // Use toolOutput directly — no JSON.parse needed
        let code = r"
            const names: string[] = toolOutput.items.map((item) => item.name);
            FINAL(names.join(','));
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(
            result.is_ok(),
            "Code using toolOutput directly should pass: {:?}",
            result.err().map(|e| e.diagnostics)
        );
    }

    #[tokio::test]
    async fn test_tool_output_wrong_property_fails() {
        let pool = shared_pool().await;
        let tool_type = "type ToolOutput = { items: Array<{ id: string; name: string }> };";
        let ambient = build_ambient_declarations(Some(tool_type));

        let code = r"
            const emails: string[] = toolOutput.items.map((item) => item.email);
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(
            result.is_err(),
            "Accessing non-existent property on toolOutput should fail typecheck"
        );
    }

    #[tokio::test]
    async fn test_wrong_property_access_fails() {
        let pool = shared_pool().await;
        let tool_type = "type ToolOutput = { items: Array<{ id: string; name: string }> };";
        let ambient = build_ambient_declarations(Some(tool_type));

        let code = r"
            const data: ToolOutput = JSON.parse(context);
            const emails: string[] = data.items.map((item) => item.email);
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(
            result.is_err(),
            "Accessing non-existent property should fail typecheck"
        );
        let err = result.unwrap_err();
        assert!(
            err.diagnostics.contains("email"),
            "Diagnostics should mention the bad property: {}",
            err.diagnostics
        );
    }

    #[tokio::test]
    async fn test_rlm_globals_available() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let code = "\
            const ctx: string = context;\n\
            const len: number = ctx.length;\n\
            console.log(len);\n\
            FINAL(\"done\");\n\
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(
            result.is_ok(),
            "RLM globals should be available: {:?}",
            result.err().map(|e| e.diagnostics)
        );
    }

    #[tokio::test]
    async fn test_primitive_value_constructors() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let code = r#"
            const s: string = String(42);
            const n: number = Number("3.14");
            const b: boolean = Boolean("");
            const isInt: boolean = Number.isInteger(42);
            const fromCode: string = String.fromCharCode(65);
            console.log(s, n, b, isInt, fromCode);
        "#;
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(
            result.is_ok(),
            "Primitive value constructors should work: {:?}",
            result.err().map(|e| e.diagnostics)
        );
    }

    #[tokio::test]
    async fn test_async_llm_query_types() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let code = "\
            async function process() {\n\
                const result: string = await llm_query(\"test prompt\");\n\
                const batch: string[] = await llm_query_batched([\"a\", \"b\"]);\n\
                FINAL(result);\n\
            }\n\
            process();\n\
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(
            result.is_ok(),
            "Async llm_query types should work: {:?}",
            result.err().map(|e| e.diagnostics)
        );
    }

    #[tokio::test]
    async fn test_top_level_await_passes_typecheck() {
        let pool = shared_pool().await;
        let ambient = static_ambient();
        let code = "\
            const result: string = await llm_query(\"test prompt\");\n\
            FINAL(result);\n\
        ";
        let result = pool.check_and_transpile(code, &ambient).await.unwrap();
        assert!(
            result.is_ok(),
            "Top-level await should pass typecheck: {:?}",
            result.err().map(|e| e.diagnostics)
        );
    }

    #[tokio::test]
    async fn test_pool_zero_size_errors() {
        let result = TsCheckerPool::new(0).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, TsError::InvalidConfig { .. }));
    }

    #[test]
    fn test_build_ambient_with_none() {
        let ambient = build_ambient_declarations(None);
        assert!(ambient.contains("declare var context: string"));
        assert!(ambient.contains("declare function FINAL"));
        assert!(ambient.contains("declare var toolOutput: any"));
    }

    #[test]
    fn test_build_ambient_with_some() {
        let ambient = build_ambient_declarations(Some("type Foo = { bar: number };"));
        assert!(ambient.contains("type Foo"));
        assert!(ambient.contains("declare var context: string"));
        // When output_ts_type is set, toolOutput gets the specific type
        assert!(ambient.contains("declare var toolOutput: Foo;"));
    }

    #[test]
    fn test_build_tool_client_declarations_includes_join_tool_and_methods() {
        let declarations = build_tool_client_declarations(&fake_exposed_tools());
        assert!(declarations.contains("declare function join_tool"));
        assert!(declarations.contains("interface ToolClient"));
        assert!(declarations.contains("list_items(params: ListItemsParams)"));
        assert!(declarations.contains("Promise<TensorzeroToolHandle<ListItemsOutput>>"));
    }

    #[test]
    fn test_code_execution_ambient_excludes_rlm_only_globals() {
        let ambient = build_code_execution_ambient_declarations(&fake_exposed_tools());
        assert!(!ambient.contains("declare var context: string"));
        assert!(ambient.contains("declare function FINAL"));
        assert!(!ambient.contains("declare function llm_query"));
        assert!(!ambient.contains("declare function llm_query_batched"));
        assert!(ambient.contains("declare var console"));
        assert!(ambient.contains("declare function join_tool"));
        assert!(ambient.contains("declare var toolClient"));
    }

    #[test]
    fn test_extract_root_type_name_single_type() {
        let bundle = "type ToolOutput = { items: string[] };";
        assert_eq!(extract_root_type_name(bundle), Some("ToolOutput"));
    }

    #[test]
    fn test_extract_root_type_name_multi_type_bundle() {
        let bundle = "\
            type Inner = { id: string };\n\
            type ToolOutput = { items: Inner[] };";
        assert_eq!(extract_root_type_name(bundle), Some("ToolOutput"));
    }

    #[test]
    fn test_extract_root_type_name_with_generic() {
        let bundle = "type ListInferencesResponse = Array<{ id: string; }>;";
        assert_eq!(
            extract_root_type_name(bundle),
            Some("ListInferencesResponse")
        );
    }

    #[test]
    fn test_extract_root_type_name_empty_string() {
        assert_eq!(extract_root_type_name(""), None);
    }

    #[test]
    fn test_extract_root_type_name_interface() {
        assert_eq!(
            extract_root_type_name("interface Foo { bar: number }"),
            Some("Foo")
        );
    }

    #[test]
    fn test_extract_root_type_name_export_type() {
        assert_eq!(
            extract_root_type_name("export type Foo = { bar: number };"),
            Some("Foo")
        );
    }

    #[test]
    fn test_extract_root_type_name_export_interface() {
        assert_eq!(
            extract_root_type_name("export interface Foo { bar: number }"),
            Some("Foo")
        );
    }

    #[test]
    fn test_extract_root_type_name_no_type_keyword() {
        assert_eq!(extract_root_type_name("const x = 42;"), None);
    }

    #[test]
    fn test_extract_root_type_name_with_export() {
        // If the bundle has 'type X =' lines, it should find the last one
        let bundle = "type A = string;\ntype B = number;";
        assert_eq!(extract_root_type_name(bundle), Some("B"));
    }

    #[test]
    fn test_build_tool_client_interface_only_excludes_type_bodies() {
        let tools = fake_exposed_tools();
        let result = build_tool_client_interface_only(&tools);

        // Should contain the ToolClient interface with method signatures
        assert!(result.contains("interface ToolClient"));
        assert!(result.contains("list_items(params: ListItemsParams)"));
        assert!(result.contains("TensorzeroToolHandle"));
        assert!(result.contains("join_tool"));
        assert!(result.contains("declare var toolClient: ToolClient"));

        // Should NOT contain the type body definitions
        assert!(
            !result.contains("type ListItemsParams = { limit: number }"),
            "interface-only output should not contain param type bodies"
        );
        assert!(
            !result.contains("type ListItemsOutput = string[]"),
            "interface-only output should not contain output type bodies"
        );

        // Should be reasonably short
        assert!(
            result.len() < 500,
            "interface-only output for 1 tool should be compact, got {} chars",
            result.len()
        );
    }

    #[test]
    fn test_build_tool_client_declarations_includes_type_bodies() {
        let tools = fake_exposed_tools();
        let full = build_tool_client_declarations(&tools);
        let slim = build_tool_client_interface_only(&tools);

        // Full version should contain the type bodies
        assert!(full.contains("type ListItemsParams = { limit: number }"));
        assert!(full.contains("type ListItemsOutput = string[]"));

        // Full version should also contain the interface
        assert!(full.contains("interface ToolClient"));

        // Full version should be strictly longer
        assert!(
            full.len() > slim.len(),
            "full declarations ({} chars) should be longer than interface-only ({} chars)",
            full.len(),
            slim.len()
        );
    }

    #[test]
    fn test_interface_only_size_scales_linearly() {
        let tools: Vec<ExposedToolData> = (0..5)
            .map(|i| ExposedToolData {
                name: format!("tool_{i}"),
                param_type: tensorzero_ts_types::TsTypeBundle(
                    "type VeryLongTypeName = { field_a: string; field_b: number; field_c: boolean; field_d: { nested: string }; };",
                ),
                param_type_name: format!("Tool{i}Params"),
                output_type: tensorzero_ts_types::TsTypeBundle(
                    "type VeryLongOutputType = { results: Array<{ id: string; value: number; metadata: Record<string, string> }>; };",
                ),
                output_type_name: format!("Tool{i}Output"),
            })
            .collect();

        let slim = build_tool_client_interface_only(&tools);
        let full = build_tool_client_declarations(&tools);

        // Slim version should be much smaller than full version
        assert!(
            slim.len() < full.len() / 2,
            "interface-only ({} chars) should be less than half of full ({} chars)",
            slim.len(),
            full.len()
        );

        // Slim version should not contain any type body definitions
        assert!(!slim.contains("VeryLongTypeName"));
        assert!(!slim.contains("VeryLongOutputType"));

        // But should reference the type names in method signatures
        assert!(slim.contains("Tool0Params"));
        assert!(slim.contains("Tool4Output"));
    }
}
