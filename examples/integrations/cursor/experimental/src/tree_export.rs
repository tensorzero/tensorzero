#![expect(clippy::print_stdout)]

use anyhow::Result;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use tree_sitter::{Node, Tree};

use crate::parsing::parse_hunk;

#[derive(Debug)]
pub struct TreeExportInfo {
    pub name: String,
    pub language: String,
    pub description: String,
    pub source_code: String,
    pub tree: Tree,
}

/// Convert a tree-sitter node to a JSON structure compatible with D3.js
pub fn node_to_json(node: Node, source: &[u8]) -> Value {
    let text = node.utf8_text(source).unwrap_or("");

    let mut children = Vec::new();
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        children.push(node_to_json(child, source));
    }

    json!({
        "name": node.kind(),
        "text": if node.child_count() == 0 { text } else { "" },
        "full_text": text,
        "start_position": {
            "row": node.start_position().row,
            "column": node.start_position().column
        },
        "end_position": {
            "row": node.end_position().row,
            "column": node.end_position().column
        },
        "is_leaf": node.child_count() == 0,
        "child_count": node.child_count(),
        "children": children
    })
}

/// Export tree information to JSON format
pub fn export_tree_to_json(tree_info: &TreeExportInfo) -> Result<Value> {
    let root_json = node_to_json(tree_info.tree.root_node(), tree_info.source_code.as_bytes());

    Ok(json!({
        "metadata": {
            "name": tree_info.name,
            "language": tree_info.language,
            "description": tree_info.description,
            "source_length": tree_info.source_code.len(),
            "node_count": count_nodes(tree_info.tree.root_node())
        },
        "source_code": tree_info.source_code,
        "tree": root_json
    }))
}

/// Count total nodes in the tree
fn count_nodes(node: Node) -> usize {
    let mut count = 1; // Count current node
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        count += count_nodes(child);
    }

    count
}

/// Get all tree export information from parsing tests
pub fn get_all_test_trees() -> Result<Vec<TreeExportInfo>> {
    let trees = vec![
        TreeExportInfo {
            name: "rust_simple_function".to_string(),
            language: "rust".to_string(),
            description: "Simple Rust function with println! macro".to_string(),
            source_code: r#"fn hello_world() {
    println!("Hello, world!");
}"#
            .to_string(),
            tree: parse_hunk(
                r#"fn hello_world() {
    println!("Hello, world!");
}"#,
                "rs",
            )?,
        },
        TreeExportInfo {
            name: "rust_struct_with_impl".to_string(),
            language: "rust".to_string(),
            description: "Rust struct with implementation block".to_string(),
            source_code: r#"struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}"#
            .to_string(),
            tree: parse_hunk(
                r#"struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}"#,
                "rs",
            )?,
        },
        // TypeScript examples
        TreeExportInfo {
            name: "typescript_simple_function".to_string(),
            language: "typescript".to_string(),
            description: "TypeScript function with template string".to_string(),
            source_code: r#"function greet(name: string): string {
    return `Hello, ${name}!`;
}"#
            .to_string(),
            tree: parse_hunk(
                r#"function greet(name: string): string {
    return `Hello, ${name}!`;
}"#,
                "ts",
            )?,
        },
        TreeExportInfo {
            name: "typescript_class_with_methods".to_string(),
            language: "typescript".to_string(),
            description: "TypeScript class with private field and methods".to_string(),
            source_code: r#"class Calculator {
    private value: number = 0;

    add(n: number): Calculator {
        this.value += n;
        return this;
    }

    getResult(): number {
        return this.value;
    }
}"#
            .to_string(),
            tree: parse_hunk(
                r#"class Calculator {
    private value: number = 0;

    add(n: number): Calculator {
        this.value += n;
        return this;
    }

    getResult(): number {
        return this.value;
    }
}"#,
                "ts",
            )?,
        },
        TreeExportInfo {
            name: "python_simple_function".to_string(),
            language: "python".to_string(),
            description: "Simple Python function with arithmetic".to_string(),
            source_code: r#"def calculate_area(radius):
    return 3.14159 * radius * radius"#
                .to_string(),
            tree: parse_hunk(
                r#"def calculate_area(radius):
    return 3.14159 * radius * radius"#,
                "py",
            )?,
        },
        TreeExportInfo {
            name: "python_class_with_methods".to_string(),
            language: "python".to_string(),
            description: "Python class with methods and property decorator".to_string(),
            source_code: r#"class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"

    @property
    def is_adult(self):
        return self.age >= 18"#
                .to_string(),
            tree: parse_hunk(
                r#"class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"

    @property
    def is_adult(self):
        return self.age >= 18"#,
                "py",
            )?,
        },
        TreeExportInfo {
            name: "toml_simple_config".to_string(),
            language: "toml".to_string(),
            description: "Simple TOML package configuration".to_string(),
            source_code: r#"
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
"#
            .to_string(),
            tree: parse_hunk(
                r#"
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
"#,
                "toml",
            )?,
        },
        TreeExportInfo {
            name: "toml_complex_config".to_string(),
            language: "toml".to_string(),
            description: "Complex TOML configuration with arrays".to_string(),
            source_code: r#"
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
"#
            .to_string(),
            tree: parse_hunk(
                r#"
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
"#,
                "toml",
            )?,
        },
        // Markdown examples
        TreeExportInfo {
            name: "markdown_simple".to_string(),
            language: "markdown".to_string(),
            description: "Simple Markdown with headings and code block".to_string(),
            source_code: r#"
# Hello World

This is a **bold** text with some *italic* words.

## Code Example

```rust
fn main() {
    println!("Hello, world!");
}
```
"#
            .to_string(),
            tree: parse_hunk(
                r#"
# Hello World

This is a **bold** text with some *italic* words.

## Code Example

```rust
fn main() {
    println!("Hello, world!");
}
```
"#,
                "md",
            )?,
        },
        TreeExportInfo {
            name: "markdown_with_lists".to_string(),
            language: "markdown".to_string(),
            description: "Markdown with task lists and blockquotes".to_string(),
            source_code: r#"
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
"#
            .to_string(),
            tree: parse_hunk(
                r#"
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
"#,
                "md",
            )?,
        },
        TreeExportInfo {
            name: "rust_complex_function".to_string(),
            language: "rust".to_string(),
            description: "Complex Rust function with pattern matching".to_string(),
            source_code: r#"fn check_value(x: i32) -> &'static str {
    if x > 0 {
        "positive"
    } else if x < 0 {
        "negative"
    } else {
        "zero"
    }
}"#
            .to_string(),
            tree: parse_hunk(
                r#"fn check_value(x: i32) -> &'static str {
    if x > 0 {
        "positive"
    } else if x < 0 {
        "negative"
    } else {
        "zero"
    }
}"#,
                "rs",
            )?,
        },
        TreeExportInfo {
            name: "rust_loop_example".to_string(),
            language: "rust".to_string(),
            description: "Rust for loop with range".to_string(),
            source_code: r#"fn print_numbers() {
    for i in 0..5 {
        println!("{}", i);
    }
}"#
            .to_string(),
            tree: parse_hunk(
                r#"fn print_numbers() {
    for i in 0..5 {
        println!("{}", i);
    }
}"#,
                "rs",
            )?,
        },
        TreeExportInfo {
            name: "rust_data_structures".to_string(),
            language: "rust".to_string(),
            description: "Rust function using Vec and iteration".to_string(),
            source_code: r#"fn process_data() {
    let data = vec![1, 2, 3, 4, 5];
    for item in data {
        println!("{}", item);
    }
}"#
            .to_string(),
            tree: parse_hunk(
                r#"fn process_data() {
    let data = vec![1, 2, 3, 4, 5];
    for item in data {
        println!("{}", item);
    }
}"#,
                "rs",
            )?,
        },
        TreeExportInfo {
            name: "rust_error_handling".to_string(),
            language: "rust".to_string(),
            description: "Rust function with Result return type".to_string(),
            source_code: r#"fn read_file() -> Result<String, std::io::Error> {
    std::fs::read_to_string("file.txt")
}"#
            .to_string(),
            tree: parse_hunk(
                r#"fn read_file() -> Result<String, std::io::Error> {
    std::fs::read_to_string("file.txt")
}"#,
                "rs",
            )?,
        },
    ];

    Ok(trees)
}

/// Export all trees to JSON files in the specified directory
pub fn export_all_trees(output_dir: &Path) -> Result<()> {
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir)?;

    let trees = get_all_test_trees()?;
    let mut index = Vec::new();

    for tree_info in &trees {
        let json_data = export_tree_to_json(tree_info)?;
        let filename = format!("{}.json", tree_info.name);
        let filepath = output_dir.join(&filename);

        // Write individual tree file
        fs::write(&filepath, serde_json::to_string_pretty(&json_data)?)?;

        // Add to index
        index.push(json!({
            "name": tree_info.name,
            "language": tree_info.language,
            "description": tree_info.description,
            "filename": filename,
            "node_count": count_nodes(tree_info.tree.root_node()),
            "source_length": tree_info.source_code.len()
        }));
    }

    // Write index file
    let index_data = json!({
        "trees": index,
        "total_count": trees.len(),
        "generated_at": chrono::Utc::now().to_rfc3339()
    });

    fs::write(
        output_dir.join("index.json"),
        serde_json::to_string_pretty(&index_data)?,
    )?;

    println!("Exported {} trees to {}", trees.len(), output_dir.display());

    Ok(())
}
