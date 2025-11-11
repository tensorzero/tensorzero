"""
Unit tests for generate_schema_types.py

Run with: pytest test_generate_schema_types.py -v
"""

from typing import Any

from generate_schema_types import (
    extract_exported_names_from_schema,
    extract_names_from_schema_recursive,
    rewrite_refs_recursive,
)


class TestExtractExportedNamesFromSchema:
    """Tests for extract_exported_names_from_schema function"""

    def test_empty_schema(self):
        """Test with empty schema"""
        schema = {}
        result = extract_exported_names_from_schema(schema)  # pyright: ignore[reportUnknownArgumentType]
        assert result == []

    def test_basic_definitions(self):
        """Test extracting basic definition names"""
        schema = {"$defs": {"TypeA": {"type": "string"}, "TypeB": {"type": "number"}, "TypeC": {"type": "boolean"}}}

        result = extract_exported_names_from_schema(schema)

        assert result == ["TypeA", "TypeB", "TypeC"]

    def test_extracts_titles(self):
        """Test that titles are extracted"""
        schema = {"$defs": {"inner_name": {"title": "PublicName", "type": "object"}}}

        result = extract_exported_names_from_schema(schema)

        assert "inner_name" in result
        assert "PublicName" in result

    def test_skips_ref_only_titles(self):
        """Test that titles on $ref-only schemas are skipped"""
        schema = {
            "$defs": {"Wrapper": {"title": "ShouldBeSkipped", "$ref": "#/$defs/Inner"}, "Inner": {"type": "object"}}
        }

        result = extract_exported_names_from_schema(schema)

        assert "Wrapper" in result
        assert "Inner" in result
        assert "ShouldBeSkipped" not in result

    def test_oneof_with_inline_schemas(self):
        """Test extracting names from oneOf with inline schemas"""
        schema = {
            "$defs": {
                "Union": {
                    "oneOf": [
                        {"title": "VariantA", "type": "object", "properties": {"type": {"const": "a"}}},
                        {"title": "VariantB", "type": "object", "properties": {"type": {"const": "b"}}},
                    ]
                }
            }
        }

        result = extract_exported_names_from_schema(schema)

        assert "Union" in result
        assert "VariantA" in result
        assert "VariantB" in result

    def test_oneof_with_refs_only(self):
        """Test that oneOf with only $ref pointers doesn't create phantom types"""
        schema = {
            "$defs": {
                "Union": {
                    "oneOf": [
                        {"title": "PhantomA", "$ref": "#/$defs/TypeA"},
                        {"title": "PhantomB", "$ref": "#/$defs/TypeB"},
                    ]
                },
                "TypeA": {"type": "object"},
                "TypeB": {"type": "object"},
            }
        }

        result = extract_exported_names_from_schema(schema)

        assert "Union" in result
        assert "TypeA" in result
        assert "TypeB" in result
        # Phantom types should NOT be extracted
        assert "PhantomA" not in result
        assert "PhantomB" not in result

    def test_allof_composition(self):
        """Test extracting names from allOf with inline schemas"""
        schema = {
            "$defs": {
                "Extended": {
                    "allOf": [
                        {"$ref": "#/$defs/Base"},
                        {"title": "Extension", "type": "object", "properties": {"extra": {"type": "string"}}},
                    ]
                },
                "Base": {"type": "object"},
            }
        }

        result = extract_exported_names_from_schema(schema)

        assert "Extended" in result
        assert "Base" in result
        assert "Extension" in result

    def test_nested_properties(self):
        """Test extracting names from nested property schemas"""
        schema = {
            "$defs": {
                "Container": {"type": "object", "properties": {"nested": {"title": "NestedType", "type": "object"}}}
            }
        }

        result = extract_exported_names_from_schema(schema)

        assert "Container" in result
        assert "NestedType" in result

    def test_sorted_output(self):
        """Test that output is sorted alphabetically"""
        schema = {"$defs": {"Zebra": {"type": "string"}, "Apple": {"type": "string"}, "Mango": {"type": "string"}}}

        result = extract_exported_names_from_schema(schema)

        assert result == ["Apple", "Mango", "Zebra"]


class TestExtractNamesFromSchemaRecursive:
    """Tests for extract_names_from_schema_recursive helper function"""

    def test_extracts_title(self):
        """Test basic title extraction"""
        schema = {"title": "MyType", "type": "object"}
        names: set[str] = set()

        extract_names_from_schema_recursive(schema, names)

        assert "MyType" in names

    def test_skips_title_with_ref(self):
        """Test that titles on $ref schemas are skipped"""
        schema = {"title": "SkipMe", "$ref": "#/$defs/Other"}
        names: set[str] = set()

        extract_names_from_schema_recursive(schema, names)

        assert "SkipMe" not in names

    def test_handles_non_dict(self):
        """Test that non-dict input is handled gracefully"""
        names: set[str] = set()

        extract_names_from_schema_recursive("not a dict", names)
        extract_names_from_schema_recursive(None, names)
        extract_names_from_schema_recursive(123, names)

        assert len(names) == 0

    def test_processes_array_items(self):
        """Test that array item schemas are processed"""
        schema = {"type": "array", "items": {"title": "ItemType", "type": "object"}}
        names: set[str] = set()

        extract_names_from_schema_recursive(schema, names)

        assert "ItemType" in names


class TestRewriteRefsRecursive:
    """Test cases for the rewrite_refs_recursive function."""

    def test_simple_root_ref(self):
        """Test rewriting a simple $ref: # to $ref: #/$defs/SchemaName."""
        schema = {"$ref": "#"}
        result = rewrite_refs_recursive(schema, "MySchema")
        assert result == {"$ref": "#/$defs/MySchema"}

    def test_nested_root_ref_in_properties(self):
        """Test rewriting $ref: # nested in properties."""
        schema = {"type": "object", "properties": {"field": {"$ref": "#"}}}
        result = rewrite_refs_recursive(schema, "RecursiveType")
        expected = {"type": "object", "properties": {"field": {"$ref": "#/$defs/RecursiveType"}}}
        assert result == expected

    def test_root_ref_in_array_items(self):
        """Test rewriting $ref: # in array items."""
        schema = {"type": "array", "items": {"$ref": "#"}}
        result = rewrite_refs_recursive(schema, "ListType")
        expected = {"type": "array", "items": {"$ref": "#/$defs/ListType"}}
        assert result == expected

    def test_multiple_root_refs(self):
        """Test rewriting multiple $ref: # in the same schema."""
        schema = {"oneOf": [{"$ref": "#"}, {"$ref": "#"}]}
        result = rewrite_refs_recursive(schema, "UnionType")
        expected = {"oneOf": [{"$ref": "#/$defs/UnionType"}, {"$ref": "#/$defs/UnionType"}]}
        assert result == expected

    def test_preserves_other_refs(self):
        """Test that non-root $ref values are preserved unchanged."""
        schema = {
            "properties": {
                "self": {"$ref": "#"},
                "other": {"$ref": "#/$defs/OtherType"},
                "deep": {"$ref": "#/$defs/deeply/nested/Type"},
            }
        }
        result = rewrite_refs_recursive(schema, "MyType")
        expected = {
            "properties": {
                "self": {"$ref": "#/$defs/MyType"},
                "other": {"$ref": "#/$defs/OtherType"},
                "deep": {"$ref": "#/$defs/deeply/nested/Type"},
            }
        }
        assert result == expected

    def test_deeply_nested_refs(self):
        """Test rewriting deeply nested $ref: # references."""
        schema = {
            "oneOf": [
                {"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#"}}}},
                {"type": "object", "properties": {"child": {"$ref": "#"}}},
            ]
        }
        result = rewrite_refs_recursive(schema, "TreeNode")
        expected = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/TreeNode"}}},
                },
                {"type": "object", "properties": {"child": {"$ref": "#/$defs/TreeNode"}}},
            ]
        }
        assert result == expected

    def test_no_refs(self):
        """Test that schemas without $ref are unchanged."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        result = rewrite_refs_recursive(schema, "Person")
        assert result == schema

    def test_empty_dict(self):
        """Test that empty dict is unchanged."""
        schema: dict[str, Any] = {}
        result = rewrite_refs_recursive(schema, "Empty")
        assert result == {}

    def test_empty_list(self):
        """Test that empty list is unchanged."""
        schema: list[Any] = []
        result = rewrite_refs_recursive(schema, "Empty")
        assert result == []

    def test_primitives_unchanged(self):
        """Test that primitive values are unchanged."""
        assert rewrite_refs_recursive("string", "Schema") == "string"
        assert rewrite_refs_recursive(123, "Schema") == 123
        assert rewrite_refs_recursive(True, "Schema") is True
        assert rewrite_refs_recursive(None, "Schema") is None
        assert rewrite_refs_recursive(3.14, "Schema") == 3.14

    def test_list_of_dicts_with_refs(self):
        """Test rewriting refs in a list of dictionaries."""
        schema = [{"$ref": "#"}, {"$ref": "#/$defs/Other"}, {"type": "string"}]
        result = rewrite_refs_recursive(schema, "Mixed")
        expected = [{"$ref": "#/$defs/Mixed"}, {"$ref": "#/$defs/Other"}, {"type": "string"}]
        assert result == expected

    def test_real_world_inference_filter(self):
        """Test with a real-world example similar to InferenceFilter."""
        schema = {
            "oneOf": [
                {
                    "title": "AndFilter",
                    "type": "object",
                    "properties": {
                        "children": {"type": "array", "items": {"$ref": "#"}},
                        "type": {"type": "string", "const": "and"},
                    },
                    "required": ["type", "children"],
                },
                {
                    "title": "OrFilter",
                    "type": "object",
                    "properties": {
                        "children": {"type": "array", "items": {"$ref": "#"}},
                        "type": {"type": "string", "const": "or"},
                    },
                    "required": ["type", "children"],
                },
                {
                    "title": "NotFilter",
                    "type": "object",
                    "properties": {"child": {"$ref": "#"}, "type": {"type": "string", "const": "not"}},
                    "required": ["type", "child"],
                },
            ]
        }

        result = rewrite_refs_recursive(schema, "InferenceFilter")

        # Check that all # refs were rewritten
        assert result["oneOf"][0]["properties"]["children"]["items"]["$ref"] == "#/$defs/InferenceFilter"
        assert result["oneOf"][1]["properties"]["children"]["items"]["$ref"] == "#/$defs/InferenceFilter"
        assert result["oneOf"][2]["properties"]["child"]["$ref"] == "#/$defs/InferenceFilter"

    def test_allof_with_refs(self):
        """Test rewriting refs in allOf composition."""
        schema = {"allOf": [{"$ref": "#/$defs/Base"}, {"properties": {"recursive": {"$ref": "#"}}}]}
        result = rewrite_refs_recursive(schema, "Derived")
        expected = {"allOf": [{"$ref": "#/$defs/Base"}, {"properties": {"recursive": {"$ref": "#/$defs/Derived"}}}]}
        assert result == expected

    def test_anyof_with_refs(self):
        """Test rewriting refs in anyOf."""
        schema = {"anyOf": [{"$ref": "#"}, {"type": "null"}]}
        result = rewrite_refs_recursive(schema, "Optional")
        expected = {"anyOf": [{"$ref": "#/$defs/Optional"}, {"type": "null"}]}
        assert result == expected

    def test_does_not_modify_original(self):
        """Test that the original schema is not modified (returns new object)."""
        original = {"properties": {"self": {"$ref": "#"}}}
        original_copy = {"properties": {"self": {"$ref": "#"}}}

        result = rewrite_refs_recursive(original, "Test")

        # Original should be unchanged
        assert original == original_copy
        # Result should have rewritten refs
        assert result["properties"]["self"]["$ref"] == "#/$defs/Test"
