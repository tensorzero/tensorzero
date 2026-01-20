//! TypeScript to JavaScript transpilation.

use std::sync::Arc;

use deno_ast::{EmitOptions, MediaType, ParseParams, TranspileModuleOptions, TranspileOptions};

use super::error::TypeScriptToolError;

/// Transpile TypeScript code to JavaScript.
///
/// The transpiled code converts ESM `export default` to `globalThis.default`
/// assignment so it can be executed as a script (not a module) in the Deno runtime.
///
/// # Errors
///
/// Returns `TypeScriptToolError::Transpile` if the TypeScript code has syntax errors
/// or cannot be parsed.
pub fn transpile_typescript(code: &str) -> Result<String, TypeScriptToolError> {
    let specifier = deno_ast::ModuleSpecifier::parse("file:///tool.ts")
        .map_err(|e| TypeScriptToolError::Transpile(e.to_string()))?;

    let parsed = deno_ast::parse_module(ParseParams {
        specifier,
        text: Arc::from(code),
        media_type: MediaType::TypeScript,
        capture_tokens: false,
        scope_analysis: false,
        maybe_syntax: None,
    })
    .map_err(|e| TypeScriptToolError::Transpile(e.to_string()))?;

    let transpiled = parsed
        .transpile(
            &TranspileOptions::default(),
            &TranspileModuleOptions::default(),
            &EmitOptions::default(),
        )
        .map_err(|e| TypeScriptToolError::Transpile(e.to_string()))?;

    let js_code = transpiled.into_source().text;

    // Convert ESM `export default` to `globalThis.default` assignment.
    // This is necessary because we execute the code as a script, not a module.
    // The ESM syntax `export default X` is not valid in script context.
    // Note: deno_ast's ModuleKind::Cjs doesn't actually transform the syntax,
    // it only affects import resolution, so we need to do this manually.
    let js_code = convert_esm_default_export(&js_code);

    Ok(js_code)
}

/// Convert ESM `export default` syntax to `globalThis.default` assignment.
///
/// This handles the common patterns:
/// - `export default { ... };` → `globalThis.default = { ... };`
/// - `export default X;` → `globalThis.default = X;`
fn convert_esm_default_export(code: &str) -> String {
    code.replace("export default ", "globalThis.default = ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpile_simple_typescript() {
        let ts_code = r"
            const x: number = 42;
            export default { async run() { return { value: x }; } };
        ";

        let js_code = transpile_typescript(ts_code).expect("Transpilation should succeed");

        // Should not contain TypeScript type annotations
        assert!(
            !js_code.contains(": number"),
            "Type annotations should be stripped"
        );
        // Should contain the actual code
        assert!(js_code.contains("42"), "Literal values should be preserved");
        // Should convert ESM export to globalThis assignment for script execution
        assert!(
            js_code.contains("globalThis.default ="),
            "ESM export should be converted to globalThis.default"
        );
        assert!(
            !js_code.contains("export default"),
            "ESM export syntax should be removed"
        );
    }

    #[test]
    fn test_transpile_with_interface() {
        let ts_code = r"
            interface Params {
                query: string;
                limit?: number;
            }

            export default {
                async run(params: Params) {
                    return { query: params.query };
                }
            };
        ";

        let js_code = transpile_typescript(ts_code).expect("Transpilation should succeed");

        // Interfaces should be stripped
        assert!(!js_code.contains("interface"));
        // Type annotations should be stripped
        assert!(!js_code.contains(": Params"));
    }

    #[test]
    fn test_transpile_syntax_error() {
        let ts_code = r"
            const x: number = {;  // Syntax error
        ";

        let result = transpile_typescript(ts_code);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpile_with_async_await() {
        let ts_code = r#"
            async function fetchData<T>(url: string): Promise<T> {
                return {} as T;
            }

            export default {
                async run() {
                    const data = await fetchData<string>("test");
                    return { data };
                }
            };
        "#;

        let js_code = transpile_typescript(ts_code).expect("Transpilation should succeed");

        // Should preserve async/await
        assert!(js_code.contains("async"));
        assert!(js_code.contains("await"));
        // Should strip generics
        assert!(!js_code.contains("<T>"));
    }

    #[test]
    fn test_transpile_with_class() {
        let ts_code = r#"
            class DataProcessor {
                private data: string[] = [];

                add(item: string): void {
                    this.data.push(item);
                }

                getAll(): string[] {
                    return [...this.data];
                }
            }

            export default {
                async run() {
                    const processor = new DataProcessor();
                    processor.add("test");
                    return { data: processor.getAll() };
                }
            };
        "#;

        let js_code = transpile_typescript(ts_code).expect("Transpilation should succeed");

        // Should preserve class structure
        assert!(js_code.contains("class DataProcessor"));
        // Should strip type annotations
        assert!(!js_code.contains(": string[]"));
        assert!(!js_code.contains(": void"));
    }
}
