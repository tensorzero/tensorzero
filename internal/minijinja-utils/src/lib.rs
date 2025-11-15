use minijinja::machinery::{self, ast, WhitespaceConfig};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Value;
use minijinja::{Environment, Error};

/// Describes the template loading statement that triggered analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadKind {
    Include { ignore_missing: bool },
    Import,
    FromImport,
    Extends,
}

/// Outcome of analysing an expression used for template loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadValue {
    /// Static template names discovered in the expression.
    pub known: Vec<String>,
    /// `true` if `known` represents the full set of possible values.
    pub complete: bool,
    /// Optional explanation when the expression cannot be fully resolved.
    pub reason: Option<&'static str>,
}

impl LoadValue {
    fn empty_static() -> Self {
        LoadValue {
            known: Vec::new(),
            complete: true,
            reason: None,
        }
    }

    fn fully_static(values: Vec<String>) -> Self {
        let mut rv = LoadValue::empty_static();
        for value in values {
            rv.push_known(value);
        }
        rv
    }

    fn fully_dynamic(reason: &'static str) -> Self {
        LoadValue {
            known: Vec::new(),
            complete: false,
            reason: Some(reason),
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
        }
    }

    fn mark_incomplete(&mut self, reason: &'static str) {
        self.complete = false;
        if self.reason.is_none() {
            self.reason = Some(reason);
        }
    }
}

/// Captures a single load site inside a template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateLoad {
    pub kind: LoadKind,
    pub value: LoadValue,
}

/// Analyse template source with default MiniJinja configuration.
pub fn collect_template_loads(source: &str, name: &str) -> Result<Vec<TemplateLoad>, Error> {
    collect_template_loads_with_config(source, name, SyntaxConfig, WhitespaceConfig::default())
}

/// Analyse template source with an explicit configuration.
pub fn collect_template_loads_with_config(
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
pub fn collect_template_loads_from_env(
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
            None => LoadValue::fully_dynamic("non-string constant"),
        },
        ast::Expr::List(list) => {
            let mut aggregate = LoadValue::empty_static();
            for item in &list.items {
                let item_value = analyse_expr(item);
                aggregate.extend(item_value);
            }
            aggregate
        }
        ast::Expr::IfExpr(if_expr) => {
            let mut aggregate = LoadValue::empty_static();
            aggregate.extend(analyse_expr(&if_expr.true_expr));
            if let Some(false_expr) = &if_expr.false_expr {
                aggregate.extend(analyse_expr(false_expr));
            } else {
                aggregate.mark_incomplete("conditional without else");
            }
            aggregate
        }
        other => LoadValue::fully_dynamic(other.description()),
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::PathBuf;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn static_load_value(values: &[&str]) -> LoadValue {
        LoadValue {
            known: values.iter().map(|s| s.to_string()).collect(),
            complete: true,
            reason: None,
        }
    }

    fn partial_load_value(values: &[&str], reason: &'static str) -> LoadValue {
        LoadValue {
            known: values.iter().map(|s| s.to_string()).collect(),
            complete: false,
            reason: Some(reason),
        }
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let mut path = std::env::temp_dir();
            let unique = format!(
                "minijinja-test-{}-{}",
                process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            path.push(unique);
            fs::create_dir_all(&path).unwrap();
            TestDir { path }
        }

        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn include_static_string() {
        let loads = collect_template_loads("{% include 'greeting.html' %}", "inline").unwrap();
        assert_eq!(
            loads,
            vec![TemplateLoad {
                kind: LoadKind::Include {
                    ignore_missing: false,
                },
                value: static_load_value(&["greeting.html"]),
            }]
        );
    }

    #[test]
    fn include_ignore_missing_and_list() {
        let loads = collect_template_loads(
            "{% include ['first.html', 'second.html'] ignore missing %}",
            "inline",
        )
        .unwrap();
        assert_eq!(
            loads,
            vec![TemplateLoad {
                kind: LoadKind::Include {
                    ignore_missing: true,
                },
                value: static_load_value(&["first.html", "second.html"]),
            }]
        );
    }

    #[test]
    fn include_with_dynamic_branch() {
        let loads =
            collect_template_loads("{% include ['first.html', helper_name] %}", "inline").unwrap();
        assert_eq!(
            loads,
            vec![TemplateLoad {
                kind: LoadKind::Include {
                    ignore_missing: false,
                },
                value: partial_load_value(&["first.html"], "variable"),
            }]
        );
    }

    #[test]
    fn import_and_from_import() {
        let loads = collect_template_loads(
            "{% import 'macros.html' as macros %}{% from 'helpers.html' import util %}",
            "inline",
        )
        .unwrap();
        assert_eq!(
            loads,
            vec![
                TemplateLoad {
                    kind: LoadKind::Import,
                    value: static_load_value(&["macros.html"]),
                },
                TemplateLoad {
                    kind: LoadKind::FromImport,
                    value: static_load_value(&["helpers.html"]),
                },
            ]
        );
    }

    #[test]
    fn extends_and_include_with_conditional() {
        let loads = collect_template_loads(
            "{% extends 'base.html' %}{% include 'a.html' if flag else 'b.html' %}",
            "inline",
        )
        .unwrap();
        assert_eq!(
            loads,
            vec![
                TemplateLoad {
                    kind: LoadKind::Extends,
                    value: static_load_value(&["base.html"]),
                },
                TemplateLoad {
                    kind: LoadKind::Include {
                        ignore_missing: false,
                    },
                    value: static_load_value(&["a.html", "b.html"]),
                },
            ]
        );
    }

    #[test]
    fn dynamic_include() {
        let loads = collect_template_loads("{% include template_name %}", "inline").unwrap();
        assert_eq!(
            loads,
            vec![TemplateLoad {
                kind: LoadKind::Include {
                    ignore_missing: false,
                },
                value: partial_load_value(&[], "variable"),
            }]
        );
    }

    #[test]
    fn collects_from_environment_with_path_loader() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("child.html"),
            "{% extends 'layout.html' %}{% include ['partial.html', helper] %}",
        )
        .unwrap();
        fs::write(temp_dir.path().join("layout.html"), "layout body").unwrap();
        fs::write(temp_dir.path().join("partial.html"), "partial body").unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let loads = collect_template_loads_from_env(&env, "child.html").unwrap();
        assert_eq!(
            loads,
            vec![
                TemplateLoad {
                    kind: LoadKind::Extends,
                    value: static_load_value(&["layout.html"]),
                },
                TemplateLoad {
                    kind: LoadKind::Include {
                        ignore_missing: false,
                    },
                    value: partial_load_value(&["partial.html"], "variable"),
                },
            ]
        );
    }
}
