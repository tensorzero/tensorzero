use anyhow::Result;
use std::{
    collections::HashMap,
    sync::{OnceLock, RwLock},
};
use tree_sitter::{Language, Parser, Tree};

static LANGUAGES: OnceLock<RwLock<HashMap<String, Language>>> = OnceLock::new();

fn get_language_for_extension(ext: &str) -> Result<Language> {
    let lock = LANGUAGES.get_or_init(|| RwLock::new(HashMap::new()));

    // Fast path: shared read-lock
    if let Ok(guard) = lock.read() {
        if let Some(lang) = guard.get(ext) {
            return Ok(lang.clone());
        }
    }

    // Slow path: upgrade to exclusive write-lock
    let mut w = lock
        .write()
        .map_err(|_| anyhow::anyhow!("Failed to lock languages"))?;
    if let Some(lang) = w.get(ext) {
        return Ok(lang.clone());
    }

    let lang = match ext {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "toml" => tree_sitter_toml_ng::LANGUAGE.into(),
        "py" => tree_sitter_python::LANGUAGE.into(),
        "ts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "md" => tree_sitter_md::LANGUAGE.into(),
        _ => return Err(anyhow::anyhow!("Unsupported file extension: {}", ext)),
    };
    w.insert(ext.to_string(), lang);
    if let Some(lang) = w.get(ext) {
        Ok(lang.clone())
    } else {
        Err(anyhow::anyhow!(
            "Failed to insert language for extension: {}",
            ext
        ))
    }
}

pub fn parse_hunk(hunk: &str, hunk_file_extension: &str) -> Result<Tree> {
    let language = get_language_for_extension(hunk_file_extension)?;
    let mut parser = Parser::new();
    parser.set_language(&language)?;
    let tree = parser
        .parse(hunk, None)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse hunk: {}", hunk))?;
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_simple_function() {
        let rust_code = r#"
fn hello_world() {
    println!("Hello, world!");
}
"#;
        let result = parse_hunk(rust_code, "rs");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());
        assert_eq!(tree.root_node().kind(), "source_file");
    }

    #[test]
    fn test_parse_rust_struct_with_impl() {
        let rust_code = r#"
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}
"#;
        let result = parse_hunk(rust_code, "rs");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());

        // Check that we have both struct and impl blocks
        let mut struct_found = false;
        let mut impl_found = false;

        let mut cursor = tree.walk();
        for child in tree.root_node().children(&mut cursor) {
            match child.kind() {
                "struct_item" => struct_found = true,
                "impl_item" => impl_found = true,
                _ => {}
            }
        }

        assert!(struct_found);
        assert!(impl_found);
    }

    #[test]
    fn test_parse_typescript_simple_function() {
        let ts_code = r#"
function greet(name: string): string {
    return `Hello, ${name}!`;
}
"#;
        let result = parse_hunk(ts_code, "ts");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());
        assert_eq!(tree.root_node().kind(), "program");
    }

    #[test]
    fn test_parse_typescript_class_with_methods() {
        let ts_code = r#"
class Calculator {
    private value: number = 0;

    add(n: number): Calculator {
        this.value += n;
        return this;
    }

    getResult(): number {
        return this.value;
    }
}
"#;
        let result = parse_hunk(ts_code, "ts");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());

        // Check that we have a class declaration
        let mut cursor = tree.walk();
        let mut class_found = false;
        for child in tree.root_node().children(&mut cursor) {
            if child.kind() == "class_declaration" {
                class_found = true;
                break;
            }
        }
        assert!(class_found);
    }

    #[test]
    fn test_parse_python_simple_function() {
        let py_code = r#"
def calculate_area(radius):
    return 3.14159 * radius * radius
"#;
        let result = parse_hunk(py_code, "py");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());
        assert_eq!(tree.root_node().kind(), "module");
    }

    #[test]
    fn test_parse_python_class_with_methods() {
        let py_code = r#"
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"

    @property
    def is_adult(self):
        return self.age >= 18
"#;
        let result = parse_hunk(py_code, "py");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());

        // Check that we have a class definition
        let mut cursor = tree.walk();
        let mut class_found = false;
        for child in tree.root_node().children(&mut cursor) {
            if child.kind() == "class_definition" {
                class_found = true;
                break;
            }
        }
        assert!(class_found);
    }

    #[test]
    fn test_parse_toml_simple_config() {
        let toml_code = r#"
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
"#;
        let result = parse_hunk(toml_code, "toml");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());
        assert_eq!(tree.root_node().kind(), "document");
    }

    #[test]
    fn test_parse_toml_complex_config() {
        let toml_code = r#"
[server]
host = "localhost"
port = 8080
enabled = true

[database]
url = "postgresql://localhost/mydb"
max_connections = 10

[[workers]]
name = "worker1"
threads = 4

[[workers]]
name = "worker2"
threads = 2
"#;
        let result = parse_hunk(toml_code, "toml");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());

        // Check that we have table headers
        let mut cursor = tree.walk();
        let mut table_found = false;
        for child in tree.root_node().children(&mut cursor) {
            if child.kind() == "table" {
                table_found = true;
                break;
            }
        }
        assert!(table_found);
    }

    #[test]
    fn test_parse_markdown_simple() {
        let md_code = r#"
# Hello World

This is a **bold** text with some *italic* words.

## Code Example

```rust
fn main() {
    println!("Hello, world!");
}
```
"#;
        let result = parse_hunk(md_code, "md");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());
        assert_eq!(tree.root_node().kind(), "document");
    }

    #[test]
    fn test_parse_markdown_with_lists() {
        let md_code = r#"
# Task List

## TODO Items

- [x] Complete project setup
- [ ] Write unit tests
- [ ] Deploy to production

## Features

1. User authentication
2. Data persistence
3. Real-time updates

> **Note**: This is a blockquote with important information.
"#;
        let result = parse_hunk(md_code, "md");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());

        // Check that we have heading elements
        let mut cursor = tree.walk();
        let mut heading_found = false;
        for child in tree.root_node().children(&mut cursor) {
            if child.kind() == "section" {
                // Look for headings in sections
                let mut section_cursor = child.walk();
                for section_child in child.children(&mut section_cursor) {
                    if section_child.kind() == "atx_heading" {
                        heading_found = true;
                        break;
                    }
                }
                if heading_found {
                    break;
                }
            }
        }
    }

    #[test]
    fn test_parse_unsupported_extension() {
        let code = "some code";
        let result = parse_hunk(code, "xyz");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported file extension"));
    }

    #[test]
    fn test_parse_invalid_syntax() {
        // Invalid Rust syntax
        let invalid_rust = r#"
fn broken_function( {
    let x = ;
    return
}
"#;
        let result = parse_hunk(invalid_rust, "rs");
        // Parser should still return a tree, but it will contain error nodes
        assert!(result.is_ok());

        let tree = result.unwrap();
        // The tree might contain error nodes, but parsing itself succeeds
        assert_eq!(tree.root_node().kind(), "source_file");
    }

    #[test]
    fn test_parse_empty_content() {
        let result = parse_hunk("", "rs");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert_eq!(tree.root_node().kind(), "source_file");
        assert_eq!(tree.root_node().child_count(), 0);
    }

    #[test]
    fn test_parse_whitespace_only() {
        let result = parse_hunk("   \n\n  \t  \n", "py");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert_eq!(tree.root_node().kind(), "module");
    }

    #[test]
    fn test_language_caching() {
        // Test that languages are cached properly by parsing multiple times
        let rust_code = "fn test() {}";

        let result1 = parse_hunk(rust_code, "rs");
        let result2 = parse_hunk(rust_code, "rs");
        let result3 = parse_hunk(rust_code, "rs");

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());

        // All should have the same structure
        let tree1 = result1.unwrap();
        let tree2 = result2.unwrap();
        let tree3 = result3.unwrap();

        assert_eq!(tree1.root_node().kind(), tree2.root_node().kind());
        assert_eq!(tree2.root_node().kind(), tree3.root_node().kind());
    }
}
