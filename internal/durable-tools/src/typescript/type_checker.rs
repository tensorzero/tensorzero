//! TypeScript type checking support.
//!
//! This module provides a trait-based abstraction for type checking TypeScript code
//! before execution. The default implementation uses a subprocess to run `tsc` or `deno check`.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use async_trait::async_trait;
use regex::Regex;
use tokio::process::Command;

/// Regex pattern for tsc format: file.ts(line,col): error TS1234: message
#[expect(clippy::expect_used)]
static TSC_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^([^(\n]+)\((\d+),(\d+)\):\s*(error|warning)\s+TS\d+:\s*(.+)$")
        .expect("TSC_PATTERN is a valid regex")
});

/// Regex pattern for deno format: file.ts:line:col - error TS1234: message
#[expect(clippy::expect_used)]
static DENO_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^([^:\n]+):(\d+):(\d+)\s*-\s*(error|warning)\s+TS\d+:\s*(.+)$")
        .expect("DENO_PATTERN is a valid regex")
});

use super::error::TypeScriptToolError;

/// The default type definitions for the `ctx` object.
pub const CTX_TYPE_DEFINITIONS: &str = include_str!("ctx.d.ts");

/// Result of type checking TypeScript code.
#[derive(Debug, Clone)]
pub struct TypeCheckResult {
    /// Whether type checking succeeded with no errors.
    pub success: bool,
    /// Diagnostic messages from the type checker.
    pub diagnostics: Vec<TypeCheckDiagnostic>,
}

impl TypeCheckResult {
    /// Create a successful result with no diagnostics.
    pub fn success() -> Self {
        Self {
            success: true,
            diagnostics: Vec::new(),
        }
    }

    /// Create a failed result with diagnostics.
    pub fn failure(diagnostics: Vec<TypeCheckDiagnostic>) -> Self {
        Self {
            success: false,
            diagnostics,
        }
    }

    /// Format diagnostics as a human-readable string.
    pub fn format_diagnostics(&self) -> String {
        self.diagnostics
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A single diagnostic message from the type checker.
#[derive(Debug, Clone)]
pub struct TypeCheckDiagnostic {
    /// The file where the diagnostic occurred.
    pub file: String,
    /// Line number (1-indexed), if available.
    pub line: Option<u32>,
    /// Column number (1-indexed), if available.
    pub column: Option<u32>,
    /// The diagnostic message.
    pub message: String,
    /// Severity of the diagnostic.
    pub severity: DiagnosticSeverity,
}

impl std::fmt::Display for TypeCheckDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.line, self.column) {
            (Some(line), Some(col)) => {
                write!(
                    f,
                    "{}:{}:{}: {}: {}",
                    self.file, line, col, self.severity, self.message
                )
            }
            (Some(line), None) => {
                write!(
                    f,
                    "{}:{}: {}: {}",
                    self.file, line, self.severity, self.message
                )
            }
            _ => {
                write!(f, "{}: {}: {}", self.file, self.severity, self.message)
            }
        }
    }
}

/// Severity level of a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// An error that prevents compilation.
    Error,
    /// A warning that doesn't prevent compilation.
    Warning,
}

impl std::fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticSeverity::Error => write!(f, "error"),
            DiagnosticSeverity::Warning => write!(f, "warning"),
        }
    }
}

/// Trait for type checking TypeScript code.
#[async_trait]
pub trait TypeChecker: Send + Sync {
    /// Check the given TypeScript code for type errors.
    ///
    /// # Arguments
    /// * `code` - The TypeScript code to check
    /// * `ctx_types` - Type definitions for the `ctx` object (use `CTX_TYPE_DEFINITIONS` for default)
    ///
    /// # Returns
    /// A `TypeCheckResult` containing success status and any diagnostics.
    async fn check(
        &self,
        code: &str,
        ctx_types: &str,
    ) -> Result<TypeCheckResult, TypeScriptToolError>;
}

/// Type checker that uses `tsc` or `deno check` via subprocess.
pub struct SubprocessTypeChecker {
    /// The command to run (e.g., "tsc" or "deno").
    command: String,
    /// Arguments to pass to the command.
    args: Vec<String>,
    /// Timeout for the type checking process.
    timeout: Duration,
}

impl SubprocessTypeChecker {
    /// Create a new type checker using `tsc`.
    ///
    /// This requires TypeScript to be installed and `tsc` to be in the PATH.
    pub fn tsc() -> Self {
        Self {
            command: "tsc".to_string(),
            args: vec![
                "--noEmit".to_string(),
                "--strict".to_string(),
                "--target".to_string(),
                "ES2020".to_string(),
                "--module".to_string(),
                "ESNext".to_string(),
                "--moduleResolution".to_string(),
                "node".to_string(),
            ],
            timeout: Duration::from_secs(30),
        }
    }

    /// Create a new type checker using `deno check`.
    ///
    /// This requires Deno to be installed and `deno` to be in the PATH.
    pub fn deno() -> Self {
        Self {
            command: "deno".to_string(),
            args: vec!["check".to_string()],
            timeout: Duration::from_secs(30),
        }
    }

    /// Create a new type checker with a custom command and arguments.
    pub fn custom(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            timeout: Duration::from_secs(30),
        }
    }

    /// Set the timeout for type checking.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set a custom path to the type checker binary.
    #[must_use]
    pub fn with_binary_path(mut self, path: impl Into<String>) -> Self {
        self.command = path.into();
        self
    }
}

#[async_trait]
impl TypeChecker for SubprocessTypeChecker {
    async fn check(
        &self,
        code: &str,
        ctx_types: &str,
    ) -> Result<TypeCheckResult, TypeScriptToolError> {
        // Create a temporary directory for the type checking
        let temp_dir = tempfile::tempdir().map_err(|e| {
            TypeScriptToolError::TypeCheck(format!("Failed to create temp directory: {e}"))
        })?;

        let tool_file = temp_dir.path().join("tool.ts");
        let ctx_file = temp_dir.path().join("ctx.d.ts");

        // Write the ctx.d.ts file
        tokio::fs::write(&ctx_file, ctx_types).await.map_err(|e| {
            TypeScriptToolError::TypeCheck(format!("Failed to write ctx.d.ts: {e}"))
        })?;

        // Write the tool code with a reference to ctx.d.ts
        let code_with_reference = format!("/// <reference path=\"ctx.d.ts\" />\n{code}");
        tokio::fs::write(&tool_file, &code_with_reference)
            .await
            .map_err(|e| TypeScriptToolError::TypeCheck(format!("Failed to write tool.ts: {e}")))?;

        // Run the type checker
        let result = self.run_type_checker(&tool_file).await?;

        Ok(result)
    }
}

impl SubprocessTypeChecker {
    async fn run_type_checker(
        &self,
        tool_file: &PathBuf,
    ) -> Result<TypeCheckResult, TypeScriptToolError> {
        let mut cmd = Command::new(&self.command);
        cmd.args(&self.args);
        cmd.arg(tool_file);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let child = cmd.spawn().map_err(|e| {
            TypeScriptToolError::TypeCheck(format!(
                "Failed to spawn type checker `{}`: {}",
                self.command, e
            ))
        })?;

        let output = tokio::time::timeout(self.timeout, child.wait_with_output())
            .await
            .map_err(|_| {
                TypeScriptToolError::TypeCheck(format!(
                    "Type checking timed out after {:?}",
                    self.timeout
                ))
            })?
            .map_err(|e| {
                TypeScriptToolError::TypeCheck(format!("Failed to wait for type checker: {e}"))
            })?;

        // Parse the output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined_output = format!("{stdout}{stderr}");

        if output.status.success() {
            Ok(TypeCheckResult::success())
        } else {
            let diagnostics = parse_tsc_output(&combined_output);
            if diagnostics.is_empty() {
                // If we couldn't parse any diagnostics but the command failed,
                // create a generic error diagnostic
                Ok(TypeCheckResult::failure(vec![TypeCheckDiagnostic {
                    file: "tool.ts".to_string(),
                    line: None,
                    column: None,
                    message: combined_output.trim().to_string(),
                    severity: DiagnosticSeverity::Error,
                }]))
            } else {
                Ok(TypeCheckResult::failure(diagnostics))
            }
        }
    }
}

/// Parse tsc/deno check output into diagnostics.
///
/// Both tsc and deno check use similar output formats:
/// `file.ts(line,col): error TS1234: message`
/// or
/// `file.ts:line:col - error TS1234: message`
fn parse_tsc_output(output: &str) -> Vec<TypeCheckDiagnostic> {
    let mut diagnostics = Vec::new();

    for cap in TSC_PATTERN.captures_iter(output) {
        let file = cap.get(1).map(|m| m.as_str()).unwrap_or("unknown");
        let line = cap.get(2).and_then(|m| m.as_str().parse().ok());
        let column = cap.get(3).and_then(|m| m.as_str().parse().ok());
        let severity = match cap.get(4).map(|m| m.as_str()) {
            Some("warning") => DiagnosticSeverity::Warning,
            _ => DiagnosticSeverity::Error,
        };
        let message = cap.get(5).map(|m| m.as_str()).unwrap_or("").to_string();

        // Normalize the file name to just show "tool.ts" instead of full path
        let file = normalize_filename(file);

        diagnostics.push(TypeCheckDiagnostic {
            file,
            line,
            column,
            message,
            severity,
        });
    }

    for cap in DENO_PATTERN.captures_iter(output) {
        let file = cap.get(1).map(|m| m.as_str()).unwrap_or("unknown");
        let line = cap.get(2).and_then(|m| m.as_str().parse().ok());
        let column = cap.get(3).and_then(|m| m.as_str().parse().ok());
        let severity = match cap.get(4).map(|m| m.as_str()) {
            Some("warning") => DiagnosticSeverity::Warning,
            _ => DiagnosticSeverity::Error,
        };
        let message = cap.get(5).map(|m| m.as_str()).unwrap_or("").to_string();

        // Normalize the file name
        let file = normalize_filename(file);

        diagnostics.push(TypeCheckDiagnostic {
            file,
            line,
            column,
            message,
            severity,
        });
    }

    diagnostics
}

/// Normalize a file path to just show the filename.
fn normalize_filename(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

/// Create a default type checker.
///
/// This tries to use `tsc` if available, otherwise falls back to `deno check`.
/// Returns `None` if neither is available.
pub fn default_type_checker() -> Option<Arc<dyn TypeChecker>> {
    // Try tsc first
    if std::process::Command::new("tsc")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
    {
        return Some(Arc::new(SubprocessTypeChecker::tsc()));
    }

    // Fall back to deno
    if std::process::Command::new("deno")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
    {
        return Some(Arc::new(SubprocessTypeChecker::deno()));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tsc_output() {
        let output = r#"
tool.ts(5,10): error TS2322: Type 'string' is not assignable to type 'number'.
tool.ts(10,5): error TS2345: Argument of type 'boolean' is not assignable to parameter of type 'string'.
"#;

        let diagnostics = parse_tsc_output(output);
        assert_eq!(diagnostics.len(), 2, "Should parse 2 diagnostics");

        assert_eq!(diagnostics[0].file, "tool.ts");
        assert_eq!(diagnostics[0].line, Some(5));
        assert_eq!(diagnostics[0].column, Some(10));
        assert!(diagnostics[0].message.contains("Type 'string'"));
        assert_eq!(diagnostics[0].severity, DiagnosticSeverity::Error);

        assert_eq!(diagnostics[1].line, Some(10));
        assert_eq!(diagnostics[1].column, Some(5));
    }

    #[test]
    fn test_parse_deno_output() {
        let output = r#"
tool.ts:5:10 - error TS2322: Type 'string' is not assignable to type 'number'.
"#;

        let diagnostics = parse_tsc_output(output);
        assert_eq!(diagnostics.len(), 1, "Should parse 1 diagnostic");

        assert_eq!(diagnostics[0].file, "tool.ts");
        assert_eq!(diagnostics[0].line, Some(5));
        assert_eq!(diagnostics[0].column, Some(10));
    }

    #[test]
    fn test_type_check_result_format() {
        let result = TypeCheckResult::failure(vec![TypeCheckDiagnostic {
            file: "tool.ts".to_string(),
            line: Some(5),
            column: Some(10),
            message: "Type error".to_string(),
            severity: DiagnosticSeverity::Error,
        }]);

        let formatted = result.format_diagnostics();
        assert!(
            formatted.contains("tool.ts:5:10"),
            "Should include file, line, and column"
        );
        assert!(formatted.contains("error"), "Should include severity");
        assert!(formatted.contains("Type error"), "Should include message");
    }

    #[test]
    fn test_normalize_filename() {
        assert_eq!(normalize_filename("/tmp/abc123/tool.ts"), "tool.ts");
        assert_eq!(normalize_filename("tool.ts"), "tool.ts");
        // Windows-style paths are only normalized correctly on Windows
        #[cfg(windows)]
        assert_eq!(normalize_filename("C:\\Users\\test\\tool.ts"), "tool.ts");
    }
}
