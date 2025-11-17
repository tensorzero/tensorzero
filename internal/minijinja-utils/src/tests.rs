use crate::{collect_all_template_paths, AnalysisError};
use minijinja::Environment;
use std::path::PathBuf;

fn set_contains<P: Into<PathBuf>>(set: &std::collections::HashSet<PathBuf>, p: P) -> bool {
    let want = p.into();
    set.iter().any(|x| x == &want)
}

#[test]
fn collects_static_dependencies() {
    let mut env = Environment::new();
    env.add_template("main.html", "{% include 'header.html' %}Body").unwrap();
    env.add_template("header.html", "Header").unwrap();

    let paths = collect_all_template_paths(&env, "main.html").expect("should collect paths");
    // Should include both the root and the included template.
    assert!(set_contains(&paths, "main.html"));
    assert!(set_contains(&paths, "header.html"));
    assert_eq!(paths.len(), 2);
}

#[test]
fn reports_dynamic_loads() {
    let mut env = Environment::new();
    env.add_template("dynamic.html", "{% include template_var %}").unwrap();

    let err = collect_all_template_paths(&env, "dynamic.html")
        .expect_err("should error on dynamic loads");

    match err {
        AnalysisError::DynamicLoadsFound(locations) => {
            assert!(!locations.is_empty(), "should report at least one location");
            // Basic sanity checks on the first reported location.
            let loc = &locations[0];
            assert_eq!(loc.template_name, "dynamic.html");
            assert!(
                loc.load_kind.to_string().contains("include"),
                "should be an include statement"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
