// # MiniJinja Template Dependency Analysis
//
// This module performs **static analysis** of MiniJinja templates to discover template
// dependencies through `include`, `import`, `from`, and `extends` statements.
//
// ## Three-Phase Analysis
//
// 1. **Parse**: MiniJinja's `machinery::parse` converts template source into an AST
// 2. **Traverse**: `LoadCollector` walks the AST to find template load statements
// 3. **Analyze**: `analyse_expr` extracts template names from expressions
//
// ## Static Analysis Results
//
// - **Static** : `"template.html"` → knows exact template name
// - **Dynamic** : `template_var` → cannot resolve (variable, function call, etc.)
// - **Partial** : `["known.html", var]` → knows some values, not all
//
// The analysis is conservative: when uncertain, expressions are marked dynamic.
// Conditional expressions like `"a.html" if x else "b.html"` analyze both branches.

use minijinja::machinery::{self, ast, Span, WhitespaceConfig};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Value;
use minijinja::{Environment, Error};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;

use crate::error::{AnalysisError, DynamicLoadLocation, LoadKind};

/// Outcome of analysing an expression used for template loading.
#[derive(Debug, Clone)]
struct LoadValue {
    /// Static template names discovered in the expression.
    known: Vec<String>,
    /// `true` if `known` represents the full set of possible values.
    complete: bool,
    /// Optional explanation when the expression cannot be fully resolved.
    reason: Option<&'static str>,
    /// Span information (if incomplete).
    span: Option<Span>,
}

impl LoadValue {
    fn empty_static() -> Self {
        LoadValue {
            known: Vec::new(),
            complete: true,
            reason: None,
            span: None,
        }
    }

    fn fully_static(values: Vec<String>) -> Self {
        let mut rv = LoadValue::empty_static();
        for value in values {
            rv.push_known(value);
        }
        rv
    }

    fn fully_dynamic(reason: &'static str, span: Span) -> Self {
        LoadValue {
            known: Vec::new(),
            complete: false,
            reason: Some(reason),
            span: Some(span),
        }
    }

    fn push_known(&mut self, value: String) {
        if !self.known.iter().any(|existing| existing == &value) {
            self.known.push(value);
        }
    }

    fn extend(&mut self, other: LoadValue) {
        for value in other.known {
            self.push_known(value);
        }
        if !other.complete {
            self.complete = false;
            if self.reason.is_none() {
                self.reason = other.reason;
            }
            if self.span.is_none() {
                self.span = other.span;
            }
        }
    }
}

/// Captures a single load site inside a template.
#[derive(Debug, Clone)]
struct TemplateLoad {
    kind: LoadKind,
    value: LoadValue,
}

/// Extracts the source code text corresponding to a span for error reporting.
///
/// This is used to show the user exactly what code caused a dynamic load error.
fn extract_source_quote(source: &str, span: Span) -> String {
    let start_offset = span.start_offset as usize;
    let end_offset = span.end_offset as usize;
    let end = end_offset.min(source.len());
    source[start_offset..end].to_string()
}

/// Parses template source and collects template loads with explicit parser configuration.
///
/// This is the low-level parsing function that:
/// 1. Calls MiniJinja's `machinery::parse` to convert source text into an AST
/// 2. Creates a `LoadCollector` visitor to traverse the AST
/// 3. Returns all discovered template load operations
///
/// The syntax and whitespace configurations control how MiniJinja parses the template
/// (e.g., custom delimiters, whitespace handling).
fn collect_template_loads_with_config(
    source: &str,
    name: &str,
    syntax_config: SyntaxConfig,
    whitespace_config: WhitespaceConfig,
) -> Result<Vec<TemplateLoad>, Error> {
    // Parse the template source into an AST
    let ast = machinery::parse(source, name, syntax_config, whitespace_config)?;

    // Visit the AST to collect template loads
    let mut collector = LoadCollector::default();
    collector.visit_stmt(&ast);

    Ok(collector.loads)
}

/// AST visitor that collects all template load operations from a parsed MiniJinja template.
///
/// This struct implements the visitor pattern to walk through the MiniJinja Abstract Syntax Tree
/// (AST) and identify all statements that load other templates. The visitor recursively descends
/// through the AST, examining each statement node and extracting template load operations.
///
/// # MiniJinja AST Structure
///
/// MiniJinja templates are parsed into an AST consisting of two primary node types:
/// - **Statements (`ast::Stmt`)**: Control flow and structural elements (loops, conditionals, blocks, etc.)
/// - **Expressions (`ast::Expr`)**: Values and computations (variables, constants, operations, etc.)
///
/// Template loading occurs at the statement level via:
/// - `{% import "template.html" as name %}` → `ast::Stmt::Import`
/// - `{% from "template.html" import macro %}` → `ast::Stmt::FromImport`
/// - `{% extends "base.html" %}` → `ast::Stmt::Extends`
/// - `{% include "partial.html" %}` → `ast::Stmt::Include`
///
/// Each of these statements contains an expression that specifies the template name.
/// This visitor identifies these statements and analyzes their expressions to extract
/// the template names.
#[derive(Default)]
struct LoadCollector {
    /// Accumulated list of template load operations discovered during AST traversal.
    loads: Vec<TemplateLoad>,
}

impl LoadCollector {
    /// Visits a single AST statement node and processes any template loads it contains.
    ///
    /// This method implements the core of the visitor pattern, using pattern matching to
    /// handle each type of statement in the MiniJinja AST. The method either:
    /// 1. Records template load operations (import, from, extends, include)
    /// 2. Recursively visits child statements for container nodes
    /// 3. Ignores statements that don't affect template loading
    ///
    /// # AST Node Categories
    ///
    /// ## Container Nodes (recursive descent)
    /// These nodes contain child statements that must be visited:
    /// - `Template`: Root node containing top-level statements
    /// - `ForLoop`: Loop body and else body must both be visited
    /// - `IfCond`: True and false branches must both be visited
    /// - `WithBlock`, `SetBlock`, `AutoEscape`, `FilterBlock`: Single body to visit
    /// - `Block`: Template inheritance block body
    /// - `Macro`, `CallBlock`: Macro definition bodies
    ///
    /// ## Template Load Nodes (extraction)
    /// These nodes reference other templates and are the primary targets:
    /// - `Import`: `{% import "template" as name %}` - loads entire template as namespace
    /// - `FromImport`: `{% from "template" import name %}` - imports specific items
    /// - `Extends`: `{% extends "base" %}` - template inheritance
    /// - `Include`: `{% include "partial" %}` - inline template inclusion
    ///
    /// ## Leaf Nodes (ignored)
    /// These nodes don't contain child statements or template references:
    /// - `EmitRaw`, `EmitExpr`: Output statements
    /// - `Set`, `Do`: Variable assignment and expression evaluation
    /// - `Continue`, `Break`: Loop control flow
    fn visit_stmt<'a>(&mut self, stmt: &ast::Stmt<'a>) {
        match stmt {
            // Root template node - visit all top-level statements
            ast::Stmt::Template(template) => self.visit_stmts(&template.children),

            // For loops have two statement lists: body and else body
            // Both must be visited because template loads can appear in either
            ast::Stmt::ForLoop(for_loop) => {
                self.visit_stmts(&for_loop.body);
                self.visit_stmts(&for_loop.else_body);
            }

            // If statements have two branches that must both be visited
            // Even if a branch is never executed at runtime, we need to know
            // about all possible template loads for static analysis
            ast::Stmt::IfCond(if_cond) => {
                self.visit_stmts(&if_cond.true_body);
                self.visit_stmts(&if_cond.false_body);
            }

            // Single-body container nodes - visit their children
            ast::Stmt::WithBlock(with_block) => self.visit_stmts(&with_block.body),
            ast::Stmt::SetBlock(set_block) => self.visit_stmts(&set_block.body),
            ast::Stmt::AutoEscape(auto_escape) => self.visit_stmts(&auto_escape.body),
            ast::Stmt::FilterBlock(filter_block) => self.visit_stmts(&filter_block.body),

            // TEMPLATE LOAD STATEMENTS - These are what we're looking for!

            // {% import "template.html" as namespace %}
            // Loads an entire template and binds it to a namespace variable
            ast::Stmt::Import(import_stmt) => {
                self.record_load(LoadKind::Import, &import_stmt.expr);
            }

            // {% from "template.html" import macro1, macro2 %}
            // Imports specific macros or variables from another template
            ast::Stmt::FromImport(from_import) => {
                self.record_load(LoadKind::FromImport, &from_import.expr);
            }

            // {% extends "base.html" %}
            // Template inheritance - this template extends a parent template
            ast::Stmt::Extends(extends) => {
                self.record_load(LoadKind::Extends, &extends.name);
            }

            // {% include "partial.html" %}
            // Inline inclusion of another template
            // Note: can have ignore_missing flag for optional includes
            ast::Stmt::Include(include) => {
                self.record_load(
                    LoadKind::Include {
                        ignore_missing: include.ignore_missing,
                    },
                    &include.name,
                );
            }

            // Template inheritance and macro blocks - visit their bodies
            ast::Stmt::Block(block) => self.visit_stmts(&block.body),
            ast::Stmt::Macro(macro_stmt) => self.visit_stmts(&macro_stmt.body),
            ast::Stmt::CallBlock(call_block) => {
                self.visit_stmts(&call_block.macro_decl.body);
            }

            // Leaf nodes that don't affect template loading - no action needed
            ast::Stmt::EmitRaw(_) | ast::Stmt::EmitExpr(_) => {}
            ast::Stmt::Set(_) | ast::Stmt::Do(_) => {}
            ast::Stmt::Continue(_) | ast::Stmt::Break(_) => {}
        }
    }

    /// Helper to visit multiple statements sequentially.
    fn visit_stmts<'a>(&mut self, stmts: &[ast::Stmt<'a>]) {
        for stmt in stmts {
            self.visit_stmt(stmt);
        }
    }

    /// Records a template load operation by analyzing its expression.
    ///
    /// This is called when we encounter an import, from, extends, or include statement.
    /// The expression is analyzed to determine which template(s) it refers to.
    fn record_load(&mut self, kind: LoadKind, expr: &ast::Expr<'_>) {
        let value = analyse_expr(expr);
        self.loads.push(TemplateLoad { kind, value });
    }
}

/// Analyzes an expression to determine which template names it could resolve to.
///
/// This function performs **static analysis** on MiniJinja expressions to extract template names.
/// The goal is to determine all possible template names an expression could evaluate to at runtime,
/// without actually executing the template.
///
/// # Analysis Strategy
///
/// We classify expressions into three categories:
///
/// ## 1. Fully Static (complete=true)
/// Expressions that resolve to a known, fixed set of string values at compile time.
/// These are safe for static analysis.
///
/// **Examples:**
/// - `"template.html"` → single known value
/// - `["a.html", "b.html"]` → multiple known values
/// - `"a.html" if true else "b.html"` → both branches analyzed, both values extracted
///
/// ## 2. Partially Static (complete=false, known=non-empty)
/// Expressions where we know *some* possible values, but not all.
///
/// **Examples:**
/// - `"a.html" if condition else template_var` → we know "a.html" but not what template_var is
/// - `["known.html", some_variable]` → we know "known.html" but not the variable value
///
/// ## 3. Fully Dynamic (complete=false, known=empty)
/// Expressions that cannot be resolved without runtime information.
///
/// **Examples:**
/// - `template_var` → variable reference (could be anything)
/// - `get_template()` → function call (result unknown)
/// - `config.template_name` → attribute access (value unknown)
/// - `templates[index]` → computed access (index unknown)
///
/// # Expression Type Handling
///
/// The function uses pattern matching to handle each expression type from the MiniJinja AST:
///
/// - **Const**: Constants like strings, numbers, or lists. We attempt to extract string values
///   using constant folding. Numbers and other non-string constants are considered dynamic.
///
/// - **List**: List literals like `["a.html", "b.html"]`. We recursively analyze each element
///   and aggregate the results. If any element is dynamic, the entire list is partially complete.
///
/// - **IfExpr**: Ternary expressions like `"a.html" if condition else "b.html"`. We analyze
///   both branches and combine their results. If there's no else branch (e.g., `"a.html" if condition`),
///   we still extract the static template name from the true branch for dependency analysis.
///
/// - **Other**: All other expression types (Var, Call, GetAttr, BinOp, etc.) are conservatively
///   marked as fully dynamic since we cannot determine their values without runtime execution.
///
/// # Return Value
///
/// Returns a [`LoadValue`] containing:
/// - `known`: All statically-known template names extracted from the expression
/// - `complete`: Whether `known` represents the complete set of possible values
/// - `reason`: Human-readable explanation if the expression is dynamic (e.g., "variable", "call")
/// - `span`: Source location information for error reporting
fn analyse_expr(expr: &ast::Expr<'_>) -> LoadValue {
    match expr {
        // CONSTANT EXPRESSIONS: {% include "template.html" %}
        // These are the ideal case - direct string literals or constant lists.
        // We use constant folding to extract the actual string values.
        ast::Expr::Const(constant) => match strings_from_value(&constant.value) {
            Some(values) => LoadValue::fully_static(values),
            None => {
                // Constant exists but isn't a string or list of strings
                // Example: {% include 42 %} (invalid but we catch it)
                LoadValue::fully_dynamic("non-string constant", expr.span())
            }
        },

        // LIST EXPRESSIONS: {% include ["a.html", "b.html", some_var] %}
        // We recursively analyze each item in the list and aggregate the results.
        // If any item is dynamic, the overall result is incomplete.
        ast::Expr::List(list) => {
            let mut aggregate = LoadValue::empty_static();
            for item in &list.items {
                let item_value = analyse_expr(item);
                aggregate.extend(item_value);
            }
            // If we accumulated dynamic items without a span, use the list's overall span
            if !aggregate.complete && aggregate.span.is_none() {
                aggregate.span = Some(expr.span());
            }
            aggregate
        }

        // CONDITIONAL EXPRESSIONS: {% include "a.html" if cond else "b.html" %}
        // We must analyze BOTH branches because either could execute at runtime.
        // For static analysis, we assume both possibilities must be accounted for.
        ast::Expr::IfExpr(if_expr) => {
            let mut aggregate = LoadValue::empty_static();

            // Analyze the true branch
            aggregate.extend(analyse_expr(&if_expr.true_expr));

            // Analyze the false branch if it exists
            if let Some(false_expr) = &if_expr.false_expr {
                aggregate.extend(analyse_expr(false_expr));
            }
            // Note: If there's no else clause (e.g., {% include "template.html" if condition %}),
            // we still extract the static template name from the true branch. The template may
            // not be loaded at runtime if the condition is false, but we know which template
            // WOULD be loaded if the condition is true, allowing static dependency analysis.
            aggregate
        }

        // ALL OTHER EXPRESSIONS: Variables, function calls, operations, etc.
        // These cannot be statically resolved, so we mark them as fully dynamic.
        //
        // Examples:
        // - ast::Expr::Var → {% include template_var %} (variable reference)
        // - ast::Expr::Call → {% include get_template() %} (function call)
        // - ast::Expr::GetAttr → {% include config.template %} (attribute access)
        // - ast::Expr::GetItem → {% include templates[0] %} (subscript)
        // - ast::Expr::BinOp → {% include "base" ~ ".html" %} (string concatenation)
        // - ast::Expr::Filter → {% include name|default("x.html") %} (filter application)
        //
        // Note: expr.description() provides a user-friendly name like "variable" or "call"
        other => LoadValue::fully_dynamic(other.description(), other.span()),
    }
}

/// Extracts string values from a MiniJinja [`Value`] through constant folding.
///
/// This function performs **constant folding** - the process of evaluating constant expressions
/// at compile time rather than runtime. For template loading, we need to extract string values
/// from MiniJinja's runtime value representation.
///
/// # MiniJinja Value System
///
/// MiniJinja uses a dynamic [`Value`] type that can represent various runtime types:
/// - Strings: `Value::from("template.html")`
/// - Numbers: `Value::from(42)`
/// - Lists: `Value::from(vec!["a.html", "b.html"])`
/// - Maps: `Value::from_object(...)`
/// - And more: booleans, None, objects, etc.
///
/// This function attempts to extract strings from these values, supporting both:
/// 1. Direct string values
/// 2. Lists/sequences of strings (including nested lists)
///
/// # Algorithm
///
/// The function uses a recursive strategy:
///
/// 1. **Base case - String**: If the value is a string, return it as a single-element vector
/// 2. **Recursive case - Iterable**: If the value is iterable (list, tuple, etc.):
///    - Recursively extract strings from each element
///    - Flatten and deduplicate the results
/// 3. **Failure case**: If the value is neither a string nor an iterable, return `None`
///
/// # Deduplication
///
/// The function automatically deduplicates string values to avoid redundant template loads.
/// For example, `["a.html", "b.html", "a.html"]` returns just `["a.html", "b.html"]`.
///
/// # Return Value
///
/// - `Some(Vec<String>)`: Successfully extracted string values (deduplicated)
/// - `None`: Value cannot be converted to strings (wrong type or mixed types)
fn strings_from_value(value: &Value) -> Option<Vec<String>> {
    // Base case: value is a direct string
    if let Some(as_str) = value.as_str() {
        return Some(vec![as_str.to_owned()]);
    }

    // Recursive case: value is iterable (list, tuple, etc.)
    // try_iter() returns Ok(iterator) for sequences, Err for non-iterables
    let Ok(iter) = value.try_iter() else {
        return None; // Not a string and not iterable → cannot extract strings
    };

    let mut values = Vec::new();
    for item in iter {
        // Recursively extract strings from this item
        // If any item fails to produce strings, the entire operation fails (? operator)
        let nested = strings_from_value(&item)?;

        // Add each extracted string, deduplicating as we go
        for nested_value in nested {
            if !values.iter().any(|existing| existing == &nested_value) {
                values.push(nested_value);
            }
        }
    }
    Some(values)
}

/// Recursively collects all template paths referenced by a root template.
///
/// This function performs a breadth-first traversal of the template dependency graph,
/// starting from the specified root template. It analyzes each template's AST to identify
/// all `include`, `import`, `from`, and `extends` statements, extracting the referenced
/// template names.
///
/// # Parameters
///
/// - `env`: A MiniJinja [`Environment`] containing the templates to analyze. Templates must
///   be accessible via the environment's loader or previously added with `add_template()`.
/// - `template_name`: The name of the root template to start analysis from.
///
/// # Returns
///
/// - `Ok(HashSet<PathBuf>)`: A set of all discovered template paths, including the root template.
///   Each path corresponds to a template name as a [`PathBuf`].
///   Note: these may not exist in the file system. This can be valid if:
///      * the paths are intended to be conditionally loaded and the condition is false
///      * there is an array of templates to load and at least one other template in the array exists
/// - `Err(AnalysisError)`: An error if parsing fails or if any dynamic template loads are detected.
///
/// # Errors
///
/// Returns [`AnalysisError::ParseError`] if any template cannot be parsed by MiniJinja.
///
/// Returns [`AnalysisError::DynamicLoadsFound`] if any template contains template loads that
/// cannot be statically resolved, such as:
/// - Variable references: `{% include template_var %}`
/// - Conditionals without else: `{% include 'file.html' if condition %}`
/// - Function calls or complex expressions: `{% include get_template() %}`
///
/// # Circular Dependencies
///
/// The function handles circular dependencies correctly by tracking visited templates.
/// If template A includes B and B includes A, both will be included in the result set
/// without causing infinite recursion.
///
/// # Example
///
/// ```rust
/// use minijinja::Environment;
/// use minijinja_utils::collect_all_template_paths;
///
/// let mut env = Environment::new();
/// env.add_template("base.html", "{% block content %}{% endblock %}").unwrap();
/// env.add_template("page.html", "{% extends 'base.html' %}{% block content %}Hello{% endblock %}").unwrap();
///
/// let paths = collect_all_template_paths(&env, "page.html").unwrap();
/// assert_eq!(paths.len(), 2); // page.html and base.html
/// ```
///
/// # Example with Dynamic Load Error
///
/// ```rust
/// use minijinja::Environment;
/// use minijinja_utils::{collect_all_template_paths, AnalysisError};
///
/// let mut env = Environment::new();
/// env.add_template("dynamic.html", "{% include template_var %}").unwrap();
///
/// match collect_all_template_paths(&env, "dynamic.html") {
///     Err(AnalysisError::DynamicLoadsFound(locations)) => {
///         assert_eq!(locations[0].reason, "variable");
///         println!("Dynamic load at {}:{}", locations[0].line, locations[0].column);
///     }
///     _ => panic!("Expected dynamic load error"),
/// }
/// ```
pub fn collect_all_template_paths(
    env: &Environment<'_>,
    template_name: &str,
) -> Result<HashSet<PathBuf>, AnalysisError> {
    let mut visited = HashSet::new();
    let mut to_visit = VecDeque::new();
    let mut all_paths = HashSet::new();
    let mut dynamic_loads = Vec::new();

    to_visit.push_back(template_name.to_string());

    // Extract configuration from environment
    let whitespace_config = WhitespaceConfig {
        keep_trailing_newline: env.keep_trailing_newline(),
        trim_blocks: env.trim_blocks(),
        lstrip_blocks: env.lstrip_blocks(),
    };

    while let Some(current_template) = to_visit.pop_front() {
        if visited.contains(&current_template) {
            continue;
        }
        visited.insert(current_template.clone());

        // Add the template name as a path
        all_paths.insert(PathBuf::from(&current_template));

        // Try to get the template from the environment
        // If it doesn't exist, skip it (it may be missing or have an unsafe path)
        let source = match env.get_template(&current_template) {
            Ok(template) => template.source().to_string(),
            Err(e) => {
                // Template doesn't exist in environment - skip analysis of this template
                // The missing template will be handled later (e.g., at render time)
                tracing::warn!(
                    "Could not load referenced template `{}` from environment: {}. Skipping recursive analysis.",
                    current_template,
                    e
                );
                continue;
            }
        };

        // Analyze the template for loads
        let loads = collect_template_loads_with_config(
            &source,
            &current_template,
            Default::default(), // Use default syntax config
            whitespace_config,
        )?;

        for load in loads {
            if !load.value.complete {
                // Found a dynamic load - record it
                let span = load.value.span.unwrap_or(Span {
                    start_line: 1,
                    start_col: 1,
                    start_offset: 0,
                    end_line: 1,
                    end_col: 1,
                    end_offset: 0,
                });
                let source_quote = extract_source_quote(&source, span);

                dynamic_loads.push(DynamicLoadLocation {
                    template_name: current_template.clone(),
                    line: span.start_line as usize,
                    column: span.start_col as usize,
                    span: (span.start_offset as usize, span.end_offset as usize),
                    source_quote,
                    reason: load.value.reason.unwrap_or("unknown").to_string(),
                    load_kind: load.kind,
                });
            }

            // Add known templates to visit queue
            for known_template in load.value.known {
                if !visited.contains(&known_template) {
                    to_visit.push_back(known_template);
                }
            }
        }
    }

    if !dynamic_loads.is_empty() {
        return Err(AnalysisError::DynamicLoadsFound(dynamic_loads));
    }

    Ok(all_paths)
}
