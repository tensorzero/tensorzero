//! TypeScript type checking tests.
//!
//! These tests verify that the TypeScript type checker correctly identifies type errors
//! and that valid code passes type checking.
//!
//! Run with: cargo test --test typescript_typecheck --features typescript
//!
//! Requires: `tsc` (TypeScript compiler) to be installed and in PATH.
//! Install with: `npm install -g typescript`

#![cfg(feature = "typescript")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use durable_tools::typescript::{
    CTX_TYPE_DEFINITIONS, DiagnosticSeverity, SubprocessTypeChecker, TypeCheckResult, TypeChecker,
    TypeScriptTool, TypeScriptToolError,
};
use std::sync::Arc;

// ============================================================================
// Helper to get the tsc type checker (fails if unavailable)
// ============================================================================

fn tsc_type_checker() -> Arc<dyn TypeChecker> {
    let status = std::process::Command::new("tsc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    assert!(
        status.is_ok() && status.unwrap().success(),
        "tsc (TypeScript compiler) is required for these tests. Install with: npm install -g typescript"
    );

    Arc::new(SubprocessTypeChecker::tsc())
}

// ============================================================================
// Type Checker Unit Tests
// ============================================================================

#[tokio::test]
async fn valid_typescript_code_passes_type_check() {
    let checker = tsc_type_checker();

    let code = r#"
        export default {
            name: "test_tool",
            async run(params: { message: string }, sideInfo: unknown) {
                const taskId: string = ctx.taskId();
                const episodeId: string = ctx.episodeId();
                return { echoed: params.message, taskId, episodeId };
            }
        };
    "#;

    let result = checker.check(code, CTX_TYPE_DEFINITIONS).await.unwrap();
    assert!(
        result.success,
        "Valid code should pass type check. Diagnostics: {}",
        result.format_diagnostics()
    );
}

#[tokio::test]
async fn type_error_in_ctx_method_return_type_fails() {
    let checker = tsc_type_checker();

    // ctx.taskId() returns string, but we're assigning to number
    let code = r#"
        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                const taskId: number = ctx.taskId();
                return { taskId };
            }
        };
    "#;

    let result = checker.check(code, CTX_TYPE_DEFINITIONS).await.unwrap();
    assert!(
        !result.success,
        "Code with type error should fail type check"
    );
    assert!(
        !result.diagnostics.is_empty(),
        "Should have at least one diagnostic"
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error),
        "Should have at least one error diagnostic"
    );
}

#[tokio::test]
async fn type_error_in_ctx_method_argument_fails() {
    let checker = tsc_type_checker();

    // ctx.sleepFor expects (string, number), but we're passing (number, string)
    let code = r#"
        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                await ctx.sleepFor(123, "not a number");
                return {};
            }
        };
    "#;

    let result = checker.check(code, CTX_TYPE_DEFINITIONS).await.unwrap();
    assert!(
        !result.success,
        "Code with type error should fail type check"
    );
}

#[tokio::test]
async fn valid_async_ctx_methods_pass_type_check() {
    let checker = tsc_type_checker();

    let code = r#"
        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                // Test async ctx methods with correct return types
                const randNum: number = await ctx.rand();
                const timestamp: string = await ctx.now();
                const uuid: string = await ctx.uuid7();

                await ctx.sleepFor("test_sleep", 1000);
                await ctx.emitEvent("test_event", { data: "test" });

                return { randNum, timestamp, uuid };
            }
        };
    "#;

    let result = checker.check(code, CTX_TYPE_DEFINITIONS).await.unwrap();
    assert!(
        result.success,
        "Valid async ctx method usage should pass type check. Diagnostics: {}",
        result.format_diagnostics()
    );
}

#[tokio::test]
async fn valid_call_tool_passes_type_check() {
    let checker = tsc_type_checker();

    let code = r#"
        interface SearchResult {
            results: string[];
        }

        export default {
            name: "test_tool",
            async run(params: { query: string }, sideInfo: unknown) {
                const result = await ctx.callTool<SearchResult>(
                    "search",
                    { q: params.query },
                    {}
                );
                return { results: result.results };
            }
        };
    "#;

    let result = checker.check(code, CTX_TYPE_DEFINITIONS).await.unwrap();
    assert!(
        result.success,
        "Valid callTool usage should pass type check. Diagnostics: {}",
        result.format_diagnostics()
    );
}

#[tokio::test]
async fn valid_spawn_and_join_tool_passes_type_check() {
    let checker = tsc_type_checker();

    let code = r#"
        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                // Spawn returns a handle ID (string)
                const handleId: string = await ctx.spawnTool("background_task", {}, {});

                // Join returns the result (typed via generic)
                const result = await ctx.joinTool<{ done: boolean }>(handleId);

                return { completed: result.done };
            }
        };
    "#;

    let result = checker.check(code, CTX_TYPE_DEFINITIONS).await.unwrap();
    assert!(
        result.success,
        "Valid spawn/join tool usage should pass type check. Diagnostics: {}",
        result.format_diagnostics()
    );
}

#[tokio::test]
async fn valid_await_event_passes_type_check() {
    let checker = tsc_type_checker();

    let code = r#"
        interface UserEvent {
            userId: string;
            action: string;
        }

        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                // awaitEvent with optional timeout
                const event = await ctx.awaitEvent<UserEvent>("user_action", 5000);
                return { userId: event.userId, action: event.action };
            }
        };
    "#;

    let result = checker.check(code, CTX_TYPE_DEFINITIONS).await.unwrap();
    assert!(
        result.success,
        "Valid awaitEvent usage should pass type check. Diagnostics: {}",
        result.format_diagnostics()
    );
}

#[tokio::test]
async fn undefined_variable_fails_type_check() {
    let checker = tsc_type_checker();

    let code = r#"
        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                return { value: undefinedVariable };
            }
        };
    "#;

    let result = checker.check(code, CTX_TYPE_DEFINITIONS).await.unwrap();
    assert!(
        !result.success,
        "Code with undefined variable should fail type check"
    );
}

// ============================================================================
// TypeScriptToolBuilder Type Checking Tests
// ============================================================================

#[tokio::test]
async fn build_checked_succeeds_with_valid_code() {
    let _ = tsc_type_checker(); // Ensure tsc is available

    let code = r#"
        export default {
            name: "test_tool",
            async run(params: { message: string }, sideInfo: unknown) {
                const taskId: string = ctx.taskId();
                return { echoed: params.message, taskId };
            }
        };
    "#;

    let result = TypeScriptTool::builder("test_tool")
        .typescript_code(code)
        .build_checked()
        .await;

    assert!(
        result.is_ok(),
        "build_checked should succeed with valid code. Error: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn build_checked_fails_with_type_error() {
    let _ = tsc_type_checker(); // Ensure tsc is available

    let code = r#"
        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                // Type error: assigning string to number
                const taskId: number = ctx.taskId();
                return { taskId };
            }
        };
    "#;

    let result = TypeScriptTool::builder("test_tool")
        .typescript_code(code)
        .build_checked()
        .await;

    assert!(result.is_err(), "build_checked should fail with type error");

    match result.unwrap_err() {
        TypeScriptToolError::TypeCheck(msg) => {
            assert!(
                !msg.is_empty(),
                "TypeCheck error should contain diagnostic message"
            );
        }
        other => panic!("Expected TypeCheck error, got: {:?}", other),
    }
}

#[tokio::test]
async fn build_with_diagnostics_returns_diagnostics() {
    let _ = tsc_type_checker(); // Ensure tsc is available

    let code = r#"
        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                // Type error: assigning string to number
                const episodeId: number = ctx.episodeId();
                return { episodeId };
            }
        };
    "#;

    let result = TypeScriptTool::builder("test_tool")
        .typescript_code(code)
        .build_with_diagnostics()
        .await;

    assert!(
        result.is_ok(),
        "build_with_diagnostics should return Ok even with type errors"
    );

    let (tool, type_check_result) = result.unwrap();
    assert!(
        !type_check_result.success,
        "Type check result should indicate failure"
    );
    assert!(
        !type_check_result.diagnostics.is_empty(),
        "Should have diagnostics for type error"
    );

    // The tool should still be built (transpilation should succeed)
    assert!(
        !tool.js_code().is_empty(),
        "Tool should have transpiled JS code"
    );
}

#[tokio::test]
async fn build_with_custom_type_checker() {
    // Create a custom type checker (also validates tsc is available)
    let checker = tsc_type_checker();

    let code = r#"
        export default {
            name: "test_tool",
            async run(params: { value: string }, sideInfo: unknown) {
                return { result: params.value };
            }
        };
    "#;

    let result = TypeScriptTool::builder("test_tool")
        .typescript_code(code)
        .type_checker(checker)
        .build_checked()
        .await;

    assert!(
        result.is_ok(),
        "build_checked with custom checker should succeed. Error: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn existing_build_method_unchanged() {
    // The sync build() method should still work without type checking
    let code = r#"
        export default {
            name: "test_tool",
            async run(params: unknown, sideInfo: unknown) {
                // This has a type error, but build() doesn't check types
                const taskId: number = ctx.taskId();
                return { taskId };
            }
        };
    "#;

    let result = TypeScriptTool::builder("test_tool")
        .typescript_code(code)
        .build();

    assert!(
        result.is_ok(),
        "build() should succeed even with type errors (no type checking). Error: {:?}",
        result.err()
    );
}

// ============================================================================
// TypeCheckResult Tests
// ============================================================================

#[test]
fn type_check_result_success_helper() {
    let result = TypeCheckResult::success();
    assert!(result.success, "success() should create successful result");
    assert!(
        result.diagnostics.is_empty(),
        "success() should have no diagnostics"
    );
}

#[test]
fn type_check_result_format_diagnostics() {
    use durable_tools::typescript::TypeCheckDiagnostic;

    let result = TypeCheckResult::failure(vec![
        TypeCheckDiagnostic {
            file: "tool.ts".to_string(),
            line: Some(5),
            column: Some(10),
            message: "Type 'string' is not assignable to type 'number'".to_string(),
            severity: DiagnosticSeverity::Error,
        },
        TypeCheckDiagnostic {
            file: "tool.ts".to_string(),
            line: Some(10),
            column: None,
            message: "Another error".to_string(),
            severity: DiagnosticSeverity::Error,
        },
    ]);

    let formatted = result.format_diagnostics();
    assert!(
        formatted.contains("tool.ts:5:10"),
        "Should include file:line:col"
    );
    assert!(
        formatted.contains("Type 'string'"),
        "Should include error message"
    );
    assert!(
        formatted.contains("Another error"),
        "Should include all diagnostics"
    );
}
