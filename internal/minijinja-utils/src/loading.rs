use minijinja::machinery::{self, ast, Span, WhitespaceConfig};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Value;
use minijinja::{Environment, Error};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;

use crate::error::{AnalysisError, DynamicLoadLocation, LoadKind};

/// Span information extracted from minijinja AST.
#[derive(Debug, Clone, Copy)]
struct SpanInfo {
    line: usize,
    column: usize,
    start_offset: usize,
    end_offset: usize,
}

impl SpanInfo {
    fn from_ast_span(span: Span) -> Self {
        SpanInfo {
            line: span.start_line as usize,
            column: span.start_col as usize,
            start_offset: span.start_offset as usize,
            end_offset: span.end_offset as usize,
        }
    }
}

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
    span: Option<SpanInfo>,
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

    fn fully_dynamic(reason: &'static str, span: SpanInfo) -> Self {
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

    fn mark_incomplete(&mut self, reason: &'static str, span: SpanInfo) {
        self.complete = false;
        if self.reason.is_none() {
            self.reason = Some(reason);
        }
        if self.span.is_none() {
            self.span = Some(span);
        }
    }
}

/// Captures a single load site inside a template.
#[derive(Debug, Clone)]
struct TemplateLoad {
    kind: LoadKind,
    value: LoadValue,
}

/// Extract source quote from span.
fn extract_source_quote(source: &str, span: SpanInfo) -> String {
    let end = span.end_offset.min(source.len());
    source[span.start_offset..end].to_string()
}

/// Analyse template source with an explicit configuration.
fn collect_template_loads_with_config(
    source: &str,
    name: &str,
    syntax_config: SyntaxConfig,
    whitespace_config: WhitespaceConfig,
) -> Result<Vec<TemplateLoad>, Error> {
    let ast = machinery::parse(source, name, syntax_config, whitespace_config)?;
    let mut collector = LoadCollector::default();
    collector.visit_stmt(&ast);
    Ok(collector.loads)
}

/// Analyse a template registered in an environment using the environment settings.
fn collect_template_loads_from_env(
    env: &Environment<'_>,
    template_name: &str,
) -> Result<Vec<TemplateLoad>, Error> {
    let template = env.get_template(template_name)?;
    let compiled = machinery::get_compiled_template(&template);

    let whitespace_config = WhitespaceConfig {
        keep_trailing_newline: env.keep_trailing_newline(),
        trim_blocks: env.trim_blocks(),
        lstrip_blocks: env.lstrip_blocks(),
    };

    collect_template_loads_with_config(
        template.source(),
        template.name(),
        compiled.syntax_config.clone(),
        whitespace_config,
    )
}

#[derive(Default)]
struct LoadCollector {
    loads: Vec<TemplateLoad>,
}

impl LoadCollector {
    fn visit_stmt<'a>(&mut self, stmt: &ast::Stmt<'a>) {
        match stmt {
            ast::Stmt::Template(template) => self.visit_stmts(&template.children),
            ast::Stmt::ForLoop(for_loop) => {
                self.visit_stmts(&for_loop.body);
                self.visit_stmts(&for_loop.else_body);
            }
            ast::Stmt::IfCond(if_cond) => {
                self.visit_stmts(&if_cond.true_body);
                self.visit_stmts(&if_cond.false_body);
            }
            ast::Stmt::WithBlock(with_block) => self.visit_stmts(&with_block.body),
            ast::Stmt::SetBlock(set_block) => self.visit_stmts(&set_block.body),
            ast::Stmt::AutoEscape(auto_escape) => self.visit_stmts(&auto_escape.body),
            ast::Stmt::FilterBlock(filter_block) => self.visit_stmts(&filter_block.body),
            ast::Stmt::Import(import_stmt) => {
                self.record_load(LoadKind::Import, &import_stmt.expr);
            }
            ast::Stmt::FromImport(from_import) => {
                self.record_load(LoadKind::FromImport, &from_import.expr);
            }
            ast::Stmt::Extends(extends) => {
                self.record_load(LoadKind::Extends, &extends.name);
            }
            ast::Stmt::Include(include) => {
                self.record_load(
                    LoadKind::Include {
                        ignore_missing: include.ignore_missing,
                    },
                    &include.name,
                );
            }
            ast::Stmt::Block(block) => self.visit_stmts(&block.body),
            ast::Stmt::Macro(macro_stmt) => self.visit_stmts(&macro_stmt.body),
            ast::Stmt::CallBlock(call_block) => {
                self.visit_stmts(&call_block.macro_decl.body);
            }
            ast::Stmt::EmitRaw(_) | ast::Stmt::EmitExpr(_) => {}
            ast::Stmt::Set(_) | ast::Stmt::Do(_) => {}
            ast::Stmt::Continue(_) | ast::Stmt::Break(_) => {}
        }
    }

    fn visit_stmts<'a>(&mut self, stmts: &[ast::Stmt<'a>]) {
        for stmt in stmts {
            self.visit_stmt(stmt);
        }
    }

    fn record_load(&mut self, kind: LoadKind, expr: &ast::Expr<'_>) {
        let value = analyse_expr(expr);
        self.loads.push(TemplateLoad { kind, value });
    }
}

fn analyse_expr(expr: &ast::Expr<'_>) -> LoadValue {
    match expr {
        ast::Expr::Const(constant) => match strings_from_value(&constant.value) {
            Some(values) => LoadValue::fully_static(values),
            None => LoadValue::fully_dynamic(
                "non-string constant",
                SpanInfo::from_ast_span(expr.span()),
            ),
        },
        ast::Expr::List(list) => {
            let mut aggregate = LoadValue::empty_static();
            for item in &list.items {
                let item_value = analyse_expr(item);
                aggregate.extend(item_value);
            }
            // Use the overall list span if incomplete
            if !aggregate.complete && aggregate.span.is_none() {
                aggregate.span = Some(SpanInfo::from_ast_span(expr.span()));
            }
            aggregate
        }
        ast::Expr::IfExpr(if_expr) => {
            let mut aggregate = LoadValue::empty_static();
            aggregate.extend(analyse_expr(&if_expr.true_expr));
            if let Some(false_expr) = &if_expr.false_expr {
                aggregate.extend(analyse_expr(false_expr));
            } else {
                aggregate.mark_incomplete(
                    "conditional without else",
                    SpanInfo::from_ast_span(expr.span()),
                );
            }
            aggregate
        }
        other => {
            LoadValue::fully_dynamic(other.description(), SpanInfo::from_ast_span(other.span()))
        }
    }
}

fn strings_from_value(value: &Value) -> Option<Vec<String>> {
    if let Some(as_str) = value.as_str() {
        return Some(vec![as_str.to_owned()]);
    }

    let Ok(iter) = value.try_iter() else {
        return None;
    };

    let mut values = Vec::new();
    for item in iter {
        let nested = strings_from_value(&item)?;
        for nested_value in nested {
            if !values.iter().any(|existing| existing == &nested_value) {
                values.push(nested_value);
            }
        }
    }
    Some(values)
}

/// Recursively collect all template paths from an environment starting from a root template.
/// Returns an error if any dynamic loads are found or if parsing fails.
pub fn collect_all_template_paths(
    env: &Environment<'_>,
    template_name: &str,
) -> Result<HashSet<PathBuf>, AnalysisError> {
    let mut visited = HashSet::new();
    let mut to_visit = VecDeque::new();
    let mut all_paths = HashSet::new();
    let mut dynamic_loads = Vec::new();

    to_visit.push_back(template_name.to_string());

    while let Some(current_template) = to_visit.pop_front() {
        if visited.contains(&current_template) {
            continue;
        }
        visited.insert(current_template.clone());

        // Add the template name as a path
        all_paths.insert(PathBuf::from(&current_template));

        // Get the template to access its source for error reporting
        let template = env.get_template(&current_template)?;

        // Analyze the template for loads
        let loads = collect_template_loads_from_env(env, &current_template)?;

        for load in loads {
            if !load.value.complete {
                // Found a dynamic load - record it
                let source = template.source();
                let span = load.value.span.unwrap_or(SpanInfo {
                    line: 1,
                    column: 1,
                    start_offset: 0,
                    end_offset: 0,
                });
                let source_quote = extract_source_quote(source, span);

                dynamic_loads.push(DynamicLoadLocation {
                    template_name: current_template.clone(),
                    line: span.line,
                    column: span.column,
                    span: (span.start_offset, span.end_offset),
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
