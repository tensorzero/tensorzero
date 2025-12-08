use anyhow::Result;
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, RwLock},
};
use tree_sitter::{Language, Parser, Tree};

use crate::clickhouse::InferenceInfo;
use crate::util::NormalizedInferenceTreeInfo;

/// Container for inference data with associated parsed trees
#[derive(Debug)]
pub struct InferenceWithTrees {
    /// Vector of normalized inference tree information containing parsed ASTs
    pub trees: Vec<NormalizedInferenceTreeInfo>,
    /// Shared reference to the inference metadata and results
    pub inference: Arc<InferenceInfo>,
}

impl InferenceWithTrees {
    /// Create a new container with inference trees and metadata
    pub fn new(trees: Vec<NormalizedInferenceTreeInfo>, inference: Arc<InferenceInfo>) -> Self {
        Self { trees, inference }
    }

    /// Get the inference ID
    pub fn id(&self) -> &uuid::Uuid {
        &self.inference.id
    }

    /// Get the number of parsed trees
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Check if this inference has any parsed trees
    pub fn has_trees(&self) -> bool {
        !self.trees.is_empty()
    }
}

impl From<(Vec<NormalizedInferenceTreeInfo>, Arc<InferenceInfo>)> for InferenceWithTrees {
    fn from((trees, inference): (Vec<NormalizedInferenceTreeInfo>, Arc<InferenceInfo>)) -> Self {
        Self::new(trees, inference)
    }
}

impl From<InferenceWithTrees> for (Vec<NormalizedInferenceTreeInfo>, Arc<InferenceInfo>) {
    fn from(val: InferenceWithTrees) -> Self {
        (val.trees, val.inference)
    }
}

static LANGUAGES: OnceLock<RwLock<HashMap<String, Language>>> = OnceLock::new();

fn get_language_for_extension(ext: &str) -> Result<Language> {
    let lock = LANGUAGES.get_or_init(|| RwLock::new(HashMap::new()));

    // Fast path: shared read-lock
    if let Ok(guard) = lock.read()
        && let Some(lang) = guard.get(ext)
    {
        return Ok(lang.clone());
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
        _ => return Err(anyhow::anyhow!("Unsupported file extension: {ext}")),
    };
    w.insert(ext.to_string(), lang);
    if let Some(lang) = w.get(ext) {
        Ok(lang.clone())
    } else {
        Err(anyhow::anyhow!(
            "Failed to insert language for extension: {ext}"
        ))
    }
}

pub fn parse_hunk(hunk: &str, hunk_file_extension: &str) -> Result<Tree> {
    let language = get_language_for_extension(hunk_file_extension)?;
    let mut parser = Parser::new();
    parser.set_language(&language)?;
    let tree = parser
        .parse(hunk, None)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse hunk: {hunk}"))?;
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Node;

    #[derive(Debug, Clone)]
    struct ExpectedNode {
        kind: &'static str,
        field_name: Option<&'static str>,
        text: Option<&'static str>,
        children: Vec<ExpectedNode>,
    }

    impl ExpectedNode {
        fn new(kind: &'static str) -> Self {
            Self {
                kind,
                field_name: None,
                text: None,
                children: Vec::new(),
            }
        }

        fn with_field(mut self, field_name: &'static str) -> Self {
            self.field_name = Some(field_name);
            self
        }

        fn with_text(mut self, text: &'static str) -> Self {
            self.text = Some(text);
            self
        }

        fn with_children(mut self, children: Vec<ExpectedNode>) -> Self {
            self.children = children;
            self
        }
    }

    fn assert_tree_structure(node: Node, expected: &ExpectedNode, source: &str) {
        assert_eq!(
            node.kind(),
            expected.kind,
            "Expected node kind '{}', got '{}' at position {}",
            expected.kind,
            node.kind(),
            node.start_position()
        );

        if let Some(expected_text) = expected.text {
            let actual_text = node.utf8_text(source.as_bytes()).unwrap();
            assert_eq!(
                actual_text.trim(),
                expected_text.trim(),
                "Expected text '{}', got '{}' for node kind '{}'",
                expected_text.trim(),
                actual_text.trim(),
                node.kind()
            );
        }

        if !expected.children.is_empty() {
            let mut cursor = node.walk();
            let actual_children: Vec<Node> = node.children(&mut cursor).collect();

            assert_eq!(
                actual_children.len(),
                expected.children.len(),
                "Expected {} children for node '{}', got {}. Actual children: {:?}",
                expected.children.len(),
                node.kind(),
                actual_children.len(),
                actual_children.iter().map(Node::kind).collect::<Vec<_>>()
            );

            for (actual_child, expected_child) in
                actual_children.iter().zip(expected.children.iter())
            {
                assert_tree_structure(*actual_child, expected_child, source);
            }
        }
    }

    #[test]
    fn test_parse_rust_simple_function() {
        let rust_code = r#"fn hello_world() {
    println!("Hello, world!");
}"#;
        let result = parse_hunk(rust_code, "rs");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());
        assert_eq!(tree.root_node().kind(), "source_file");

        let expected = ExpectedNode::new("source_file").with_children(vec![
            ExpectedNode::new("function_item").with_children(vec![
                ExpectedNode::new("fn").with_text("fn"),
                ExpectedNode::new("identifier").with_text("hello_world"),
                ExpectedNode::new("parameters").with_children(vec![
                    ExpectedNode::new("(").with_text("("),
                    ExpectedNode::new(")").with_text(")"),
                ]),
                ExpectedNode::new("block").with_children(vec![
                    ExpectedNode::new("{").with_text("{"),
                    ExpectedNode::new("expression_statement").with_children(vec![
                        ExpectedNode::new("macro_invocation").with_children(vec![
                            ExpectedNode::new("identifier").with_text("println"),
                            ExpectedNode::new("!").with_text("!"),
                            ExpectedNode::new("token_tree").with_children(vec![
                                ExpectedNode::new("(").with_text("("),
                                ExpectedNode::new("string_literal").with_children(vec![
                                    ExpectedNode::new("\"").with_text("\""),
                                    ExpectedNode::new("string_content").with_text("Hello, world!"),
                                    ExpectedNode::new("\"").with_text("\""),
                                ]),
                                ExpectedNode::new(")").with_text(")"),
                            ]),
                        ]),
                        ExpectedNode::new(";").with_text(";"),
                    ]),
                    ExpectedNode::new("}").with_text("}"),
                ]),
            ]),
        ]);

        assert_tree_structure(tree.root_node(), &expected, rust_code);
    }

    #[test]
    fn test_parse_rust_struct_with_impl() {
        let rust_code = r"struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}";
        let result = parse_hunk(rust_code, "rs");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());

        let expected = ExpectedNode::new("source_file").with_children(vec![
            ExpectedNode::new("struct_item").with_children(vec![
                ExpectedNode::new("struct").with_field("struct"),
                ExpectedNode::new("type_identifier")
                    .with_field("name")
                    .with_text("Point"),
                ExpectedNode::new("field_declaration_list")
                    .with_field("body")
                    .with_children(vec![
                        ExpectedNode::new("{"),
                        ExpectedNode::new("field_declaration").with_children(vec![
                            ExpectedNode::new("field_identifier")
                                .with_field("name")
                                .with_text("x"),
                            ExpectedNode::new(":").with_field(":"),
                            ExpectedNode::new("primitive_type")
                                .with_field("type")
                                .with_text("i32"),
                        ]),
                        ExpectedNode::new(","),
                        ExpectedNode::new("field_declaration").with_children(vec![
                            ExpectedNode::new("field_identifier")
                                .with_field("name")
                                .with_text("y"),
                            ExpectedNode::new(":").with_field(":"),
                            ExpectedNode::new("primitive_type")
                                .with_field("type")
                                .with_text("i32"),
                        ]),
                        ExpectedNode::new(","),
                        ExpectedNode::new("}"),
                    ]),
            ]),
            ExpectedNode::new("impl_item").with_children(vec![
                ExpectedNode::new("impl").with_field("impl"),
                ExpectedNode::new("type_identifier")
                    .with_field("type")
                    .with_text("Point"),
                ExpectedNode::new("declaration_list")
                    .with_field("body")
                    .with_children(vec![
                        ExpectedNode::new("{"),
                        ExpectedNode::new("function_item").with_children(vec![
                            ExpectedNode::new("fn").with_field("fn"),
                            ExpectedNode::new("identifier")
                                .with_field("name")
                                .with_text("new"),
                            ExpectedNode::new("parameters")
                                .with_field("parameters")
                                .with_children(vec![
                                    ExpectedNode::new("("),
                                    ExpectedNode::new("parameter").with_children(vec![
                                        ExpectedNode::new("identifier")
                                            .with_field("pattern")
                                            .with_text("x"),
                                        ExpectedNode::new(":").with_field(":"),
                                        ExpectedNode::new("primitive_type")
                                            .with_field("type")
                                            .with_text("i32"),
                                    ]),
                                    ExpectedNode::new(","),
                                    ExpectedNode::new("parameter").with_children(vec![
                                        ExpectedNode::new("identifier")
                                            .with_field("pattern")
                                            .with_text("y"),
                                        ExpectedNode::new(":").with_field(":"),
                                        ExpectedNode::new("primitive_type")
                                            .with_field("type")
                                            .with_text("i32"),
                                    ]),
                                    ExpectedNode::new(")"),
                                ]),
                            ExpectedNode::new("->").with_field("->"),
                            ExpectedNode::new("type_identifier")
                                .with_field("return_type")
                                .with_text("Self"),
                            ExpectedNode::new("block")
                                .with_field("body")
                                .with_children(vec![
                                    ExpectedNode::new("{"),
                                    ExpectedNode::new("struct_expression").with_children(vec![
                                        ExpectedNode::new("type_identifier")
                                            .with_field("name")
                                            .with_text("Self"),
                                        ExpectedNode::new("field_initializer_list")
                                            .with_field("body")
                                            .with_children(vec![
                                                ExpectedNode::new("{"),
                                                ExpectedNode::new("shorthand_field_initializer")
                                                    .with_children(vec![
                                                        ExpectedNode::new("identifier")
                                                            .with_text("x"),
                                                    ]),
                                                ExpectedNode::new(","),
                                                ExpectedNode::new("shorthand_field_initializer")
                                                    .with_children(vec![
                                                        ExpectedNode::new("identifier")
                                                            .with_text("y"),
                                                    ]),
                                                ExpectedNode::new("}"),
                                            ]),
                                    ]),
                                    ExpectedNode::new("}"),
                                ]),
                        ]),
                        ExpectedNode::new("}"),
                    ]),
            ]),
        ]);

        assert_tree_structure(tree.root_node(), &expected, rust_code);
    }

    #[test]
    fn test_parse_typescript_simple_function() {
        let ts_code = r"function greet(name: string): string {
    return `Hello, ${name}!`;
}";
        let result = parse_hunk(ts_code, "ts");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());
        assert_eq!(tree.root_node().kind(), "program");

        let expected = ExpectedNode::new("program").with_children(vec![
            ExpectedNode::new("function_declaration").with_children(vec![
                ExpectedNode::new("function").with_text("function"),
                ExpectedNode::new("identifier").with_text("greet"),
                ExpectedNode::new("formal_parameters").with_children(vec![
                    ExpectedNode::new("(").with_text("("),
                    ExpectedNode::new("required_parameter").with_children(vec![
                        ExpectedNode::new("identifier").with_text("name"),
                        ExpectedNode::new("type_annotation").with_children(vec![
                            ExpectedNode::new(":").with_text(":"),
                            ExpectedNode::new("predefined_type").with_children(vec![
                                ExpectedNode::new("string").with_text("string"),
                            ]),
                        ]),
                    ]),
                    ExpectedNode::new(")").with_text(")"),
                ]),
                ExpectedNode::new("type_annotation").with_children(vec![
                    ExpectedNode::new(":").with_text(":"),
                    ExpectedNode::new("predefined_type")
                        .with_children(vec![ExpectedNode::new("string").with_text("string")]),
                ]),
                ExpectedNode::new("statement_block").with_children(vec![
                    ExpectedNode::new("{").with_text("{"),
                    ExpectedNode::new("return_statement").with_children(vec![
                        ExpectedNode::new("return").with_text("return"),
                        ExpectedNode::new("template_string").with_children(vec![
                            ExpectedNode::new("`").with_text("`"),
                            ExpectedNode::new("string_fragment").with_text("Hello, "),
                            ExpectedNode::new("template_substitution").with_children(vec![
                                ExpectedNode::new("${").with_text("${"),
                                ExpectedNode::new("identifier").with_text("name"),
                                ExpectedNode::new("}").with_text("}"),
                            ]),
                            ExpectedNode::new("string_fragment").with_text("!"),
                            ExpectedNode::new("`").with_text("`"),
                        ]),
                        ExpectedNode::new(";").with_text(";"),
                    ]),
                    ExpectedNode::new("}").with_text("}"),
                ]),
            ]),
        ]);

        assert_tree_structure(tree.root_node(), &expected, ts_code);
    }

    #[test]
    fn test_parse_typescript_class_with_methods() {
        let ts_code = r"class Calculator {
    private value: number = 0;

    add(n: number): Calculator {
        this.value += n;
        return this;
    }

    getResult(): number {
        return this.value;
    }
}";
        let result = parse_hunk(ts_code, "ts");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());

        let expected = ExpectedNode::new("program").with_children(vec![
            ExpectedNode::new("class_declaration").with_children(vec![
                ExpectedNode::new("class"),
                ExpectedNode::new("type_identifier").with_text("Calculator"),
                ExpectedNode::new("class_body").with_children(vec![
                    ExpectedNode::new("{"),
                    ExpectedNode::new("public_field_definition").with_children(vec![
                        ExpectedNode::new("accessibility_modifier")
                            .with_children(vec![ExpectedNode::new("private").with_text("private")]),
                        ExpectedNode::new("property_identifier").with_text("value"),
                        ExpectedNode::new("type_annotation").with_children(vec![
                            ExpectedNode::new(":").with_text(":"),
                            ExpectedNode::new("predefined_type").with_children(vec![
                                ExpectedNode::new("number").with_text("number"),
                            ]),
                        ]),
                        ExpectedNode::new("=").with_text("="),
                        ExpectedNode::new("number").with_text("0"),
                    ]),
                    ExpectedNode::new(";").with_text(";"),
                    ExpectedNode::new("method_definition").with_children(vec![
                        ExpectedNode::new("property_identifier").with_text("add"),
                        ExpectedNode::new("formal_parameters").with_children(vec![
                            ExpectedNode::new("("),
                            ExpectedNode::new("required_parameter").with_children(vec![
                                ExpectedNode::new("identifier").with_text("n"),
                                ExpectedNode::new("type_annotation")
                                    .with_field("type")
                                    .with_children(vec![
                                        ExpectedNode::new(":"),
                                        ExpectedNode::new("predefined_type").with_text("number"),
                                    ]),
                            ]),
                            ExpectedNode::new(")"),
                        ]),
                        ExpectedNode::new("type_annotation")
                            .with_field("return_type")
                            .with_children(vec![
                                ExpectedNode::new(":"),
                                ExpectedNode::new("type_identifier").with_text("Calculator"),
                            ]),
                        ExpectedNode::new("statement_block")
                            .with_field("body")
                            .with_children(vec![
                                ExpectedNode::new("{"),
                                ExpectedNode::new("expression_statement").with_children(vec![
                                    ExpectedNode::new("augmented_assignment_expression")
                                        .with_children(vec![
                                            ExpectedNode::new("member_expression")
                                                .with_field("left")
                                                .with_children(vec![
                                                    ExpectedNode::new("this").with_field("object"),
                                                    ExpectedNode::new(".").with_field("."),
                                                    ExpectedNode::new("property_identifier")
                                                        .with_field("property")
                                                        .with_text("value"),
                                                ]),
                                            ExpectedNode::new("+=").with_field("operator"),
                                            ExpectedNode::new("identifier")
                                                .with_field("right")
                                                .with_text("n"),
                                        ]),
                                    ExpectedNode::new(";"),
                                ]),
                                ExpectedNode::new("return_statement").with_children(vec![
                                    ExpectedNode::new("return"),
                                    ExpectedNode::new("this"),
                                    ExpectedNode::new(";"),
                                ]),
                                ExpectedNode::new("}"),
                            ]),
                    ]),
                    ExpectedNode::new("method_definition").with_children(vec![
                        ExpectedNode::new("property_identifier")
                            .with_field("name")
                            .with_text("getResult"),
                        ExpectedNode::new("formal_parameters")
                            .with_field("parameters")
                            .with_children(vec![ExpectedNode::new("("), ExpectedNode::new(")")]),
                        ExpectedNode::new("type_annotation")
                            .with_field("return_type")
                            .with_children(vec![
                                ExpectedNode::new(":"),
                                ExpectedNode::new("predefined_type").with_text("number"),
                            ]),
                        ExpectedNode::new("statement_block")
                            .with_field("body")
                            .with_children(vec![
                                ExpectedNode::new("{"),
                                ExpectedNode::new("return_statement").with_children(vec![
                                    ExpectedNode::new("return"),
                                    ExpectedNode::new("member_expression").with_children(vec![
                                        ExpectedNode::new("this").with_field("object"),
                                        ExpectedNode::new(".").with_field("."),
                                        ExpectedNode::new("property_identifier")
                                            .with_field("property")
                                            .with_text("value"),
                                    ]),
                                    ExpectedNode::new(";"),
                                ]),
                                ExpectedNode::new("}"),
                            ]),
                    ]),
                    ExpectedNode::new("}"),
                ]),
            ]),
        ]);

        assert_tree_structure(tree.root_node(), &expected, ts_code);
    }

    #[test]
    fn test_parse_python_simple_function() {
        let py_code = r"def calculate_area(radius):
    return 3.14159 * radius * radius";
        let result = parse_hunk(py_code, "py");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());
        assert_eq!(tree.root_node().kind(), "module");

        let expected = ExpectedNode::new("module").with_children(vec![
            ExpectedNode::new("function_definition").with_children(vec![
                ExpectedNode::new("def").with_field("def"),
                ExpectedNode::new("identifier")
                    .with_field("name")
                    .with_text("calculate_area"),
                ExpectedNode::new("parameters")
                    .with_field("parameters")
                    .with_children(vec![
                        ExpectedNode::new("("),
                        ExpectedNode::new("identifier").with_text("radius"),
                        ExpectedNode::new(")"),
                    ]),
                ExpectedNode::new(":").with_field(":"),
                ExpectedNode::new("block")
                    .with_field("body")
                    .with_children(vec![ExpectedNode::new("return_statement").with_children(
                        vec![
                            ExpectedNode::new("return"),
                            ExpectedNode::new("binary_operator").with_children(vec![
                            ExpectedNode::new("binary_operator")
                                .with_field("left")
                                .with_children(vec![
                                    ExpectedNode::new("float")
                                        .with_field("left")
                                        .with_text("3.14159"),
                                    ExpectedNode::new("*").with_field("operator"),
                                    ExpectedNode::new("identifier")
                                        .with_field("right")
                                        .with_text("radius"),
                                ]),
                            ExpectedNode::new("*").with_field("operator"),
                            ExpectedNode::new("identifier")
                                .with_field("right")
                                .with_text("radius"),
                        ]),
                        ],
                    )]),
            ]),
        ]);

        assert_tree_structure(tree.root_node(), &expected, py_code);
    }

    #[test]
    fn test_parse_python_class_with_methods() {
        let py_code = r#"class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"

    @property
    def is_adult(self):
        return self.age >= 18"#;
        let result = parse_hunk(py_code, "py");
        assert!(result.is_ok());

        let tree = result.unwrap();
        assert!(!tree.root_node().is_error());

        let expected = ExpectedNode::new("module").with_children(vec![
            ExpectedNode::new("class_definition").with_children(vec![
                ExpectedNode::new("class").with_field("class"),
                ExpectedNode::new("identifier")
                    .with_field("name")
                    .with_text("Person"),
                ExpectedNode::new(":").with_field(":"),
                ExpectedNode::new("block")
                    .with_field("body")
                    .with_children(vec![
                        ExpectedNode::new("function_definition").with_children(vec![
                            ExpectedNode::new("def").with_field("def"),
                            ExpectedNode::new("identifier")
                                .with_field("name")
                                .with_text("__init__"),
                            ExpectedNode::new("parameters")
                                .with_field("parameters")
                                .with_children(vec![
                                    ExpectedNode::new("("),
                                    ExpectedNode::new("identifier").with_text("self"),
                                    ExpectedNode::new(","),
                                    ExpectedNode::new("identifier").with_text("name"),
                                    ExpectedNode::new(","),
                                    ExpectedNode::new("identifier").with_text("age"),
                                    ExpectedNode::new(")"),
                                ]),
                            ExpectedNode::new(":").with_field(":"),
                            ExpectedNode::new("block")
                                .with_field("body")
                                .with_children(vec![
                                    ExpectedNode::new("expression_statement").with_children(vec![
                                        ExpectedNode::new("assignment").with_children(vec![
                                            ExpectedNode::new("attribute").with_children(vec![
                                                ExpectedNode::new("identifier").with_text("self"),
                                                ExpectedNode::new(".").with_text("."),
                                                ExpectedNode::new("identifier").with_text("name"),
                                            ]),
                                            ExpectedNode::new("=").with_text("="),
                                            ExpectedNode::new("identifier").with_text("name"),
                                        ]),
                                    ]),
                                    ExpectedNode::new("expression_statement").with_children(vec![
                                        ExpectedNode::new("assignment").with_children(vec![
                                            ExpectedNode::new("attribute").with_children(vec![
                                                ExpectedNode::new("identifier").with_text("self"),
                                                ExpectedNode::new(".").with_text("."),
                                                ExpectedNode::new("identifier").with_text("age"),
                                            ]),
                                            ExpectedNode::new("=").with_text("="),
                                            ExpectedNode::new("identifier").with_text("age"),
                                        ]),
                                    ]),
                                ]),
                        ]),
                        ExpectedNode::new("function_definition").with_children(vec![
                            ExpectedNode::new("def").with_field("def"),
                            ExpectedNode::new("identifier")
                                .with_field("name")
                                .with_text("introduce"),
                            ExpectedNode::new("parameters")
                                .with_field("parameters")
                                .with_children(vec![
                                    ExpectedNode::new("("),
                                    ExpectedNode::new("identifier").with_text("self"),
                                    ExpectedNode::new(")"),
                                ]),
                            ExpectedNode::new(":").with_field(":"),
                            ExpectedNode::new("block")
                                .with_field("body")
                                .with_children(vec![
                                    ExpectedNode::new("return_statement").with_children(vec![
                                        ExpectedNode::new("return"),
                                        ExpectedNode::new("string").with_children(vec![
                                            ExpectedNode::new("string_start").with_text("f\""),
                                            ExpectedNode::new("string_content")
                                                .with_text("Hi, I'm "),
                                            ExpectedNode::new("interpolation").with_children(vec![
                                                ExpectedNode::new("{"),
                                                ExpectedNode::new("attribute").with_children(vec![
                                                    ExpectedNode::new("identifier")
                                                        .with_field("object")
                                                        .with_text("self"),
                                                    ExpectedNode::new(".").with_field("."),
                                                    ExpectedNode::new("identifier")
                                                        .with_field("attribute")
                                                        .with_text("name"),
                                                ]),
                                                ExpectedNode::new("}"),
                                            ]),
                                            ExpectedNode::new("string_content")
                                                .with_text(" and I'm "),
                                            ExpectedNode::new("interpolation").with_children(vec![
                                                ExpectedNode::new("{"),
                                                ExpectedNode::new("attribute").with_children(vec![
                                                    ExpectedNode::new("identifier")
                                                        .with_field("object")
                                                        .with_text("self"),
                                                    ExpectedNode::new(".").with_field("."),
                                                    ExpectedNode::new("identifier")
                                                        .with_field("attribute")
                                                        .with_text("age"),
                                                ]),
                                                ExpectedNode::new("}"),
                                            ]),
                                            ExpectedNode::new("string_content")
                                                .with_text(" years old"),
                                            ExpectedNode::new("string_end").with_text("\""),
                                        ]),
                                    ]),
                                ]),
                        ]),
                        ExpectedNode::new("decorated_definition").with_children(vec![
                            ExpectedNode::new("decorator").with_children(vec![
                                ExpectedNode::new("@"),
                                ExpectedNode::new("identifier").with_text("property"),
                            ]),
                            ExpectedNode::new("function_definition")
                                .with_field("definition")
                                .with_children(vec![
                                    ExpectedNode::new("def").with_field("def"),
                                    ExpectedNode::new("identifier")
                                        .with_field("name")
                                        .with_text("is_adult"),
                                    ExpectedNode::new("parameters")
                                        .with_field("parameters")
                                        .with_children(vec![
                                            ExpectedNode::new("("),
                                            ExpectedNode::new("identifier").with_text("self"),
                                            ExpectedNode::new(")"),
                                        ]),
                                    ExpectedNode::new(":").with_field(":"),
                                    ExpectedNode::new("block").with_field("body").with_children(
                                        vec![ExpectedNode::new("return_statement").with_children(
                                            vec![
                                                ExpectedNode::new("return"),
                                                ExpectedNode::new("comparison_operator")
                                                    .with_children(vec![
                                                        ExpectedNode::new("attribute")
                                                            .with_field("left")
                                                            .with_children(vec![
                                                                ExpectedNode::new("identifier")
                                                                    .with_field("object")
                                                                    .with_text("self"),
                                                                ExpectedNode::new(".")
                                                                    .with_field("."),
                                                                ExpectedNode::new("identifier")
                                                                    .with_field("attribute")
                                                                    .with_text("age"),
                                                            ]),
                                                        ExpectedNode::new(">=")
                                                            .with_field("operators"),
                                                        ExpectedNode::new("integer")
                                                            .with_field("right")
                                                            .with_text("18"),
                                                    ]),
                                            ],
                                        )],
                                    ),
                                ]),
                        ]),
                    ]),
            ]),
        ]);

        assert_tree_structure(tree.root_node(), &expected, py_code);
    }

    #[test]
    fn test_parse_toml_simple_config() {
        let toml_code = r#"
[package]
name = "my-app"
version = "0.1.0"
edition.workspace = true

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
        let md_code = r"
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
";
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
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported file extension")
        );
    }

    #[test]
    fn test_parse_invalid_syntax() {
        // Invalid Rust syntax
        let invalid_rust = r"
fn broken_function( {
    let x = ;
    return
}
";
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

    // Integration tests for tree edit distance functionality
    mod edit_distance_integration_tests {
        use super::*;
        use crate::ted::minimum_ted;

        // Helper function to calculate edit distance between two code snippets
        fn calculate_edit_distance(
            source_code: &str,
            target_code: &str,
            file_extension: &str,
        ) -> crate::ted::TedInfo {
            let source_tree =
                parse_hunk(source_code, file_extension).expect("Failed to parse source code");
            let target_tree =
                parse_hunk(target_code, file_extension).expect("Failed to parse target code");

            minimum_ted(
                &source_tree.root_node(),
                source_code.as_bytes(),
                &target_tree.root_node(),
                target_code.as_bytes(),
            )
        }

        // Helper function to assert exact edit distance
        fn assert_edit_distance(
            source_code: &str,
            target_code: &str,
            file_extension: &str,
            expected_distance: u64,
        ) {
            let result = calculate_edit_distance(source_code, target_code, file_extension);
            assert_eq!(
                result.min_ted, expected_distance,
                "Expected distance {} but got {}.\nSource:\n{}\nTarget:\n{}\nTED ratio: {}",
                expected_distance, result.min_ted, source_code, target_code, result.ted_ratio
            );
        }

        #[test]
        fn test_identical_code_zero_distance() {
            let code = r#"fn hello_world() {
    println!("Hello, world!");
}"#;
            assert_edit_distance(code, code, "rs", 0);
            let result = calculate_edit_distance(code, code, "rs");
            assert!(
                (result.ted_ratio - 1.0).abs() < f64::EPSILON,
                "Identical code should have TED ratio of 1.0, got {}",
                result.ted_ratio
            );
        }

        #[test]
        fn test_identical_complex_code_zero_distance() {
            let code = r"struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}";
            assert_edit_distance(code, code, "rs", 0);
            let result = calculate_edit_distance(code, code, "rs");
            assert!(
                (result.ted_ratio - 1.0).abs() < f64::EPSILON,
                "Identical complex code should have TED ratio of 1.0, got {}",
                result.ted_ratio
            );
        }

        #[test]
        fn test_whitespace_differences() {
            let source = r#"fn hello() {
    println!("hello");
}"#;
            let target = r#"fn hello() {
        println!("hello");
    }"#;
            // Whitespace should not affect tree structure significantly
            assert_edit_distance(source, target, "rs", 0);
            let result = calculate_edit_distance(source, target, "rs");
            assert!(
                (result.ted_ratio - 1.0).abs() < f64::EPSILON,
                "Whitespace differences should not affect TED ratio, got {}",
                result.ted_ratio
            );
        }

        #[test]
        fn test_single_statement_insertion() {
            let source = r"fn test() {
}";
            let target = r#"fn test() {
    println!("Hello");
}"#;
            // Adding one statement should have moderate distance
            assert_edit_distance(source, target, "rs", 7);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 7.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Single statement insertion should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_single_statement_deletion() {
            let source = r#"fn test() {
    println!("Hello");
}"#;
            let target = r"fn test() {
}";
            // Removing one statement should have moderate distance
            assert_edit_distance(source, target, "rs", 12);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 12.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Single statement deletion should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_variable_name_change() {
            let source = r#"fn test() {
    let x = 5;
    println!("{}", x);
}"#;
            let target = r#"fn test() {
    let y = 5;
    println!("{}", y);
}"#;
            // Variable rename should have low distance
            assert_edit_distance(source, target, "rs", 2);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 2.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Variable name change should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_literal_value_change() {
            let source = r"fn test() {
    let x = 5;
}";
            let target = r"fn test() {
    let x = 10;
}";
            // Changing literal value should be distance 1
            assert_edit_distance(source, target, "rs", 1);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 1.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Literal value change should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_function_parameter_addition() {
            let source = r#"fn greet() {
    println!("Hello");
}"#;
            let target = r#"fn greet(name: &str) {
    println!("Hello, {}", name);
}"#;
            // Adding parameter and using it should have moderate distance
            assert_edit_distance(source, target, "rs", 9);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 9.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Function parameter addition should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_function_name_change() {
            let source = r#"fn hello() {
    println!("Hello");
}"#;
            let target = r#"fn greet() {
    println!("Hello");
}"#;
            // Function name change should be distance 1
            assert_edit_distance(source, target, "rs", 1);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 1.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Function name change should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_extract_variable_refactoring() {
            let source = r#"fn calculate() {
    let result = 2 + 3 * 4;
    println!("{}", result);
}"#;
            let target = r#"fn calculate() {
    let temp = 3 * 4;
    let result = 2 + temp;
    println!("{}", result);
}"#;
            // Extract variable should have moderate distance
            assert_edit_distance(source, target, "rs", 13);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 13.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Extract variable refactoring should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_inline_variable_refactoring() {
            let source = r#"fn calculate() {
    let temp = 3 * 4;
    let result = 2 + temp;
    println!("{}", result);
}"#;
            let target = r#"fn calculate() {
    let result = 2 + 3 * 4;
    println!("{}", result);
}"#;
            // Inline variable should have moderate distance
            assert_edit_distance(source, target, "rs", 13);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 13.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Inline variable refactoring should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_control_flow_change_if_to_match() {
            let source = r#"fn check_value(x: i32) -> &'static str {
    if x > 0 {
        "positive"
    } else if x < 0 {
        "negative"
    } else {
        "zero"
    }
}"#;
            let target = r#"fn check_value(x: i32) -> &'static str {
    match x {
        x if x > 0 => "positive",
        x if x < 0 => "negative",
        _ => "zero",
    }
}"#;
            // Control flow change should have higher distance
            assert_edit_distance(source, target, "rs", 26);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 26.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Control flow change should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_loop_type_change() {
            let source = r#"fn print_numbers() {
    for i in 0..5 {
        println!("{}", i);
    }
}"#;
            let target = r#"fn print_numbers() {
    let mut i = 0;
    while i < 5 {
        println!("{}", i);
        i += 1;
    }
}"#;
            // Loop type change should have moderate to high distance
            assert_edit_distance(source, target, "rs", 20);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 20.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Loop type change should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_data_structure_change() {
            let source = r#"fn process_data() {
    let data = vec![1, 2, 3, 4, 5];
    for item in data {
        println!("{}", item);
    }
}"#;
            let target = r#"fn process_data() {
    let data = [1, 2, 3, 4, 5];
    for item in data {
        println!("{}", item);
    }
}"#;
            // Vec to array should have low distance
            assert_edit_distance(source, target, "rs", 4);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 4.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Data structure change should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_method_call_vs_function_call() {
            let source = r#"fn process() {
    let s = String::from("hello");
    let len = s.len();
    println!("{}", len);
}"#;
            let target = r#"fn process() {
    let s = String::from("hello");
    let len = str::len(&s);
    println!("{}", len);
}"#;
            // Method vs function call should have low distance
            assert_edit_distance(source, target, "rs", 7);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 7.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Method vs function call should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_error_handling_addition() {
            let source = r#"fn read_file() -> String {
    std::fs::read_to_string("file.txt").unwrap()
}"#;
            let target = r#"fn read_file() -> Result<String, std::io::Error> {
    std::fs::read_to_string("file.txt")
}"#;
            // Adding proper error handling should have moderate distance
            assert_edit_distance(source, target, "rs", 16);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 16.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Error handling addition should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_completely_different_functions() {
            let source = r"fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}";
            let target = r"fn quicksort(arr: &mut [i32]) {
    if arr.len() <= 1 {
        return;
    }
    let pivot = partition(arr);
    let (left, right) = arr.split_at_mut(pivot);
    quicksort(left);
    quicksort(&mut right[1..]);
}";
            // Completely different functions should have high distance
            assert_edit_distance(source, target, "rs", 47);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 47.0 / result.size as f64; // 1.0 - 47.0/60.0 = 0.2167
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Completely different functions should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_struct_to_enum_change() {
            let source = r"struct Color {
    r: u8,
    g: u8,
    b: u8,
}";
            let target = r"enum Color {
    Red,
    Green,
    Blue,
    Rgb(u8, u8, u8),
}";
            // Struct to enum should have high distance
            assert_edit_distance(source, target, "rs", 17);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 17.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Struct to enum change should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_impl_block_addition() {
            let source = r"struct Point {
    x: i32,
    y: i32,
}";
            let target = r"struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}";
            // Adding impl block should have low distance (found minimum in subtree)
            assert_edit_distance(source, target, "rs", 1);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 1.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Impl block addition should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_pattern_matching_expansion() {
            let source = r#"fn handle_option(opt: Option<i32>) {
    if let Some(value) = opt {
        println!("Got: {}", value);
    }
}"#;
            let target = r#"fn handle_option(opt: Option<i32>) {
    match opt {
        Some(value) if value > 0 => println!("Positive: {}", value),
        Some(value) => println!("Non-positive: {}", value),
        None => println!("No value"),
    }
}"#;
            // Pattern matching expansion should have high distance
            assert_edit_distance(source, target, "rs", 32);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 32.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Pattern matching expansion should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_closure_vs_function() {
            let source = r#"fn process_numbers() {
    let numbers = vec![1, 2, 3, 4, 5];
    let doubled: Vec<i32> = numbers.iter().map(|x| x * 2).collect();
    println!("{:?}", doubled);
}"#;
            let target = r#"fn double(x: &i32) -> i32 {
    x * 2
}

fn process_numbers() {
    let numbers = vec![1, 2, 3, 4, 5];
    let doubled: Vec<i32> = numbers.iter().map(double).collect();
    println!("{:?}", doubled);
}"#;
            // Closure to named function should have moderate distance
            assert_edit_distance(source, target, "rs", 10);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 10.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Closure vs function should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_iterator_vs_loop() {
            let source = r"fn sum_squares(numbers: &[i32]) -> i32 {
    let mut sum = 0;
    for num in numbers {
        sum += num * num;
    }
    sum
}";
            let target = r"fn sum_squares(numbers: &[i32]) -> i32 {
    numbers.iter().map(|x| x * x).sum()
}";
            // Iterator vs manual loop should have high distance
            assert_edit_distance(source, target, "rs", 35);
            let result = calculate_edit_distance(source, target, "rs");
            let expected_ratio = 1.0 - 35.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Iterator vs loop should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        // TypeScript tests for different patterns
        #[test]
        fn test_typescript_class_to_function() {
            let source = r"class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}";
            let target = r"function add(a: number, b: number): number {
    return a + b;
}";
            // Class to function should have moderate distance
            assert_edit_distance(source, target, "ts", 8);
            let result = calculate_edit_distance(source, target, "ts");
            let expected_ratio = 1.0 - 8.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "TypeScript class to function should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_typescript_arrow_vs_regular_function() {
            let source = r"function greet(name: string): string {
    return `Hello, ${name}!`;
}";
            let target = r"const greet = (name: string): string => {
    return `Hello, ${name}!`;
};";
            // Arrow vs regular function should have low to moderate distance
            assert_edit_distance(source, target, "ts", 5);
            let result = calculate_edit_distance(source, target, "ts");
            let expected_ratio = 1.0 - 5.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "TypeScript arrow vs regular function should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        // Python tests for different patterns
        #[test]
        fn test_python_class_vs_function() {
            let source = r"class Calculator:
    def add(self, a, b):
        return a + b";
            let target = r"def add(a, b):
    return a + b";
            // Class method to function should have moderate distance
            assert_edit_distance(source, target, "py", 7);
            let result = calculate_edit_distance(source, target, "py");
            let expected_ratio = 1.0 - 7.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Python class vs function should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }

        #[test]
        fn test_python_list_comprehension_vs_loop() {
            let source = r"def square_numbers(numbers):
    result = []
    for num in numbers:
        result.append(num ** 2)
    return result";
            let target = r"def square_numbers(numbers):
    return [num ** 2 for num in numbers]";
            // List comprehension vs loop should have high distance
            assert_edit_distance(source, target, "py", 29);
            let result = calculate_edit_distance(source, target, "py");
            let expected_ratio = 1.0 - 29.0 / result.size as f64;
            assert!(
                (result.ted_ratio - expected_ratio).abs() < f64::EPSILON,
                "Python list comprehension vs loop should have TED ratio {}, got {}",
                expected_ratio,
                result.ted_ratio
            );
        }
    }

    // Tests for InferenceWithTrees struct
    mod inference_with_trees_tests {
        use tensorzero_core::inference::types::StoredInput;

        use super::*;
        use std::path::PathBuf;

        #[test]
        fn test_inference_with_trees_basic_functionality() {
            // Create mock parsed tree data
            let trees = vec![NormalizedInferenceTreeInfo {
                paths: vec![PathBuf::from("test.rs")],
                tree: parse_hunk("fn test() {}", "rs").unwrap(),
                src: "fn test() {}".as_bytes().to_vec(),
            }];

            // Create a mock inference (we'll use the existing data from the codebase)
            // Rather than construct InferenceInfo manually, let's just test the struct operations
            let inference = Arc::new(InferenceInfo {
                id: uuid::Uuid::nil(), // Use nil UUID for testing
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: vec![],
            });

            let container = InferenceWithTrees::new(trees, inference.clone());

            assert_eq!(container.tree_count(), 1);
            assert!(container.has_trees());
            assert_eq!(container.id(), &inference.id);
        }

        #[test]
        fn test_inference_with_trees_empty() {
            let trees = vec![];
            let inference = Arc::new(InferenceInfo {
                id: uuid::Uuid::nil(),
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: vec![],
            });

            let container = InferenceWithTrees::new(trees, inference.clone());

            assert_eq!(container.tree_count(), 0);
            assert!(!container.has_trees());
            assert_eq!(container.id(), &inference.id);
        }

        #[test]
        fn test_inference_with_trees_from_tuple() {
            let trees = vec![];
            let inference = Arc::new(InferenceInfo {
                id: uuid::Uuid::nil(),
                input: StoredInput {
                    system: None,
                    messages: vec![],
                },
                output: vec![],
            });

            let tuple = (trees, inference.clone());
            let container: InferenceWithTrees = tuple.into();

            assert_eq!(container.tree_count(), 0);
            assert!(!container.has_trees());
            assert_eq!(container.id(), &inference.id);
        }
    }
}
