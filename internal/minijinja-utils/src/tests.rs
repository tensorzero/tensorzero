use super::*;
use minijinja::Environment;
use std::path::PathBuf;

#[test]
fn test_simple_static_includes() {
    let mut env = Environment::new();
    let main_source = "{% include 'header.html' %}Content{% include 'footer.html' %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("header.html", "Header").unwrap();
    env.add_template("footer.html", "Footer").unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("header.html")));
    assert!(paths.contains(&PathBuf::from("footer.html")));
}

#[test]
fn test_nested_includes() {
    let mut env = Environment::new();
    let main_source = "{% include 'partial.html' %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("partial.html", "{% include 'nested.html' %}")
        .unwrap();
    env.add_template("nested.html", "Nested content").unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("partial.html")));
    assert!(paths.contains(&PathBuf::from("nested.html")));
}

#[test]
fn test_extends_and_blocks() {
    let mut env = Environment::new();
    env.add_template("base.html", "{% block content %}{% endblock %}")
        .unwrap();
    let child_source = "{% extends 'base.html' %}{% block content %}Child content{% endblock %}";
    env.add_template("child.html", child_source).unwrap();

    let paths = collect_all_template_paths(&env, "child.html", child_source).unwrap();

    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&PathBuf::from("child.html")));
    assert!(paths.contains(&PathBuf::from("base.html")));
}

#[test]
fn test_import_statements() {
    let mut env = Environment::new();
    env.add_template("macros.html", "{% macro test() %}Macro{% endmacro %}")
        .unwrap();
    let main_source = "{% import 'macros.html' as m %}{{ m.test() }}";
    env.add_template("main.html", main_source).unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("macros.html")));
}

#[test]
fn test_dynamic_include_error() {
    let mut env = Environment::new();
    let main_source = "{% include template_name %}";
    env.add_template("main.html", main_source).unwrap();

    let result = collect_all_template_paths(&env, "main.html", main_source);

    assert!(result.is_err());
    match result {
        Err(AnalysisError::DynamicLoadsFound(locations)) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].template_name, "main.html");
            assert_eq!(locations[0].reason, "variable");
            assert!(matches!(locations[0].load_kind, LoadKind::Include { .. }));
            assert!(locations[0].source_quote.contains("template_name"));
        }
        _ => panic!("Expected DynamicLoadsFound error"),
    }
}

#[test]
fn test_conditional_without_else_extracts_static_name() {
    let mut env = Environment::new();
    let main_source = "{% include 'static.html' if condition %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("static.html", "Static content").unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("static.html")));
}

#[test]
fn test_static_list_includes() {
    let mut env = Environment::new();
    let main_source = "{% include ['first.html', 'second.html'] %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("first.html", "First").unwrap();
    env.add_template("second.html", "Second").unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("first.html")));
    assert!(paths.contains(&PathBuf::from("second.html")));
}

#[test]
fn test_mixed_list_with_dynamic_error() {
    let mut env = Environment::new();
    let main_source = "{% include ['static.html', dynamic_var] %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("static.html", "Static content").unwrap();

    let result = collect_all_template_paths(&env, "main.html", main_source);

    assert!(result.is_err());
    match result {
        Err(AnalysisError::DynamicLoadsFound(locations)) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].reason, "variable");
        }
        Err(e) => panic!("Expected DynamicLoadsFound error, got: {e:?}"),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

#[test]
fn test_error_contains_line_and_column() {
    let mut env = Environment::new();
    let main_source = "Line 1\nLine 2\n{% include variable %}";
    env.add_template("main.html", main_source).unwrap();

    let result = collect_all_template_paths(&env, "main.html", main_source);

    assert!(result.is_err());
    match result {
        Err(AnalysisError::DynamicLoadsFound(locations)) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].line, 3);
            assert!(locations[0].column > 0);
        }
        _ => panic!("Expected DynamicLoadsFound error"),
    }
}

#[test]
fn test_circular_dependency_handled() {
    let mut env = Environment::new();
    let a_source = "{% include 'b.html' %}";
    env.add_template("a.html", a_source).unwrap();
    env.add_template("b.html", "{% include 'a.html' %}")
        .unwrap();

    let paths = collect_all_template_paths(&env, "a.html", a_source).unwrap();

    // Should handle circular dependency without infinite loop
    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&PathBuf::from("a.html")));
    assert!(paths.contains(&PathBuf::from("b.html")));
}

#[test]
fn test_multiple_conditionals_without_else() {
    let mut env = Environment::new();
    let main_source =
        "{% include 'header.html' if show_header %}{% include 'footer.html' if show_footer %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("header.html", "Header").unwrap();
    env.add_template("footer.html", "Footer").unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("header.html")));
    assert!(paths.contains(&PathBuf::from("footer.html")));
}

#[test]
fn test_nested_conditional_without_else() {
    let mut env = Environment::new();
    let main_source = "{% include 'partial.html' if condition %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("partial.html", "{% include 'nested.html' %}")
        .unwrap();
    env.add_template("nested.html", "Content").unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("partial.html")));
    assert!(paths.contains(&PathBuf::from("nested.html")));
}

#[test]
fn test_conditional_with_dynamic_name_still_fails() {
    let mut env = Environment::new();
    let main_source = "{% include template_var if condition %}";
    env.add_template("main.html", main_source).unwrap();

    let result = collect_all_template_paths(&env, "main.html", main_source);

    assert!(result.is_err());
    match result {
        Err(AnalysisError::DynamicLoadsFound(locations)) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].reason, "variable");
        }
        _ => panic!("Expected DynamicLoadsFound error for variable"),
    }
}

#[test]
fn test_conditional_if_else_extracts_both_branches() {
    let mut env = Environment::new();
    let main_source = "{% include 'option_a.html' if condition else 'option_b.html' %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("option_a.html", "Option A").unwrap();
    env.add_template("option_b.html", "Option B").unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("option_a.html")));
    assert!(paths.contains(&PathBuf::from("option_b.html")));
}

#[test]
fn test_mixed_conditional_and_list() {
    let mut env = Environment::new();
    let main_source = "{% include ['a.html', 'b.html'] if condition %}";
    env.add_template("main.html", main_source).unwrap();
    env.add_template("a.html", "A").unwrap();
    env.add_template("b.html", "B").unwrap();

    let paths = collect_all_template_paths(&env, "main.html", main_source).unwrap();

    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&PathBuf::from("main.html")));
    assert!(paths.contains(&PathBuf::from("a.html")));
    assert!(paths.contains(&PathBuf::from("b.html")));
}
