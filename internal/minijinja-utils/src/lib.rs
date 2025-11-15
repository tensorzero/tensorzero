mod error;
mod loading;

pub use error::{AnalysisError, DynamicLoadLocation, LoadKind};
pub use loading::collect_all_template_paths;

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::Environment;
    use std::fs;
    use std::path::PathBuf;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

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
    fn test_simple_static_includes() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("main.html"),
            "{% include 'header.html' %}Content{% include 'footer.html' %}",
        )
        .unwrap();
        fs::write(temp_dir.path().join("header.html"), "Header").unwrap();
        fs::write(temp_dir.path().join("footer.html"), "Footer").unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let paths = collect_all_template_paths(&env, "main.html").unwrap();

        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&PathBuf::from("main.html")));
        assert!(paths.contains(&PathBuf::from("header.html")));
        assert!(paths.contains(&PathBuf::from("footer.html")));
    }

    #[test]
    fn test_nested_includes() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("main.html"),
            "{% include 'partial.html' %}",
        )
        .unwrap();
        fs::write(
            temp_dir.path().join("partial.html"),
            "{% include 'nested.html' %}",
        )
        .unwrap();
        fs::write(temp_dir.path().join("nested.html"), "Nested content").unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let paths = collect_all_template_paths(&env, "main.html").unwrap();

        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&PathBuf::from("main.html")));
        assert!(paths.contains(&PathBuf::from("partial.html")));
        assert!(paths.contains(&PathBuf::from("nested.html")));
    }

    #[test]
    fn test_extends_and_blocks() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("base.html"),
            "{% block content %}{% endblock %}",
        )
        .unwrap();
        fs::write(
            temp_dir.path().join("child.html"),
            "{% extends 'base.html' %}{% block content %}Child content{% endblock %}",
        )
        .unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let paths = collect_all_template_paths(&env, "child.html").unwrap();

        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&PathBuf::from("child.html")));
        assert!(paths.contains(&PathBuf::from("base.html")));
    }

    #[test]
    fn test_import_statements() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("macros.html"),
            "{% macro test() %}Macro{% endmacro %}",
        )
        .unwrap();
        fs::write(
            temp_dir.path().join("main.html"),
            "{% import 'macros.html' as m %}{{ m.test() }}",
        )
        .unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let paths = collect_all_template_paths(&env, "main.html").unwrap();

        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&PathBuf::from("main.html")));
        assert!(paths.contains(&PathBuf::from("macros.html")));
    }

    #[test]
    fn test_dynamic_include_error() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("main.html"),
            "{% include template_name %}",
        )
        .unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let result = collect_all_template_paths(&env, "main.html");

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
    fn test_conditional_without_else_error() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("main.html"),
            "{% include 'static.html' if condition %}",
        )
        .unwrap();
        fs::write(temp_dir.path().join("static.html"), "Static content").unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let result = collect_all_template_paths(&env, "main.html");

        assert!(result.is_err());
        match result {
            Err(AnalysisError::DynamicLoadsFound(locations)) => {
                assert_eq!(locations.len(), 1);
                assert_eq!(locations[0].reason, "conditional without else");
            }
            Err(e) => panic!("Expected DynamicLoadsFound error, got: {e:?}"),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_static_list_includes() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("main.html"),
            "{% include ['first.html', 'second.html'] %}",
        )
        .unwrap();
        fs::write(temp_dir.path().join("first.html"), "First").unwrap();
        fs::write(temp_dir.path().join("second.html"), "Second").unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let paths = collect_all_template_paths(&env, "main.html").unwrap();

        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&PathBuf::from("main.html")));
        assert!(paths.contains(&PathBuf::from("first.html")));
        assert!(paths.contains(&PathBuf::from("second.html")));
    }

    #[test]
    fn test_mixed_list_with_dynamic_error() {
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("main.html"),
            "{% include ['static.html', dynamic_var] %}",
        )
        .unwrap();
        fs::write(temp_dir.path().join("static.html"), "Static content").unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let result = collect_all_template_paths(&env, "main.html");

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
        let temp_dir = TestDir::new();
        fs::write(
            temp_dir.path().join("main.html"),
            "Line 1\nLine 2\n{% include variable %}",
        )
        .unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let result = collect_all_template_paths(&env, "main.html");

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
        let temp_dir = TestDir::new();
        fs::write(temp_dir.path().join("a.html"), "{% include 'b.html' %}").unwrap();
        fs::write(temp_dir.path().join("b.html"), "{% include 'a.html' %}").unwrap();

        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(temp_dir.path()));

        let paths = collect_all_template_paths(&env, "a.html").unwrap();

        // Should handle circular dependency without infinite loop
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&PathBuf::from("a.html")));
        assert!(paths.contains(&PathBuf::from("b.html")));
    }
}
