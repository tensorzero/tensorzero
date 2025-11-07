"""
Unit tests for generate_schema_types.py

Run with: pytest test_generate_schema_types.py -v
"""

import pytest
from generate_schema_types import (
    transform_ref_properties_to_allof,
    extract_exported_names_from_schema,
    extract_names_from_schema_recursive,
)


class TestTransformRefPropertiesToAllof:
    """Tests for transform_ref_properties_to_allof function"""

    def test_basic_transformation(self):
        """Test basic transformation of properties + $ref to allOf"""
        schema = {
            "type": "object",
            "properties": {"type": {"const": "foo"}},
            "$ref": "#/$defs/Inner",
            "required": ["type"]
        }

        result = transform_ref_properties_to_allof(schema)

        assert "allOf" in result
        assert len(result["allOf"]) == 2
        assert result["allOf"][0] == {
            "type": "object",
            "properties": {"type": {"const": "foo"}},
            "required": ["type"]
        }
        assert result["allOf"][1] == {"$ref": "#/$defs/Inner"}

    def test_preserves_metadata(self):
        """Test that title and description are preserved at top level"""
        schema = {
            "title": "MyType",
            "description": "A test type",
            "type": "object",
            "properties": {"type": {"const": "foo"}},
            "$ref": "#/$defs/Inner",
            "required": ["type"]
        }

        result = transform_ref_properties_to_allof(schema)

        assert result["title"] == "MyType"
        assert result["description"] == "A test type"
        assert "allOf" in result

    def test_no_transformation_without_ref(self):
        """Test that schemas without $ref are not transformed"""
        schema = {
            "type": "object",
            "properties": {"type": {"const": "foo"}},
            "required": ["type"]
        }

        result = transform_ref_properties_to_allof(schema)

        assert "allOf" not in result
        assert result == schema

    def test_no_transformation_without_properties(self):
        """Test that schemas without properties are not transformed"""
        schema = {
            "type": "object",
            "$ref": "#/$defs/Inner"
        }

        result = transform_ref_properties_to_allof(schema)

        assert "allOf" not in result
        assert result == schema

    def test_recursive_transformation(self):
        """Test that nested schemas are transformed recursively"""
        schema = {
            "type": "object",
            "$defs": {
                "Nested": {
                    "type": "object",
                    "properties": {"type": {"const": "nested"}},
                    "$ref": "#/$defs/Base",
                    "required": ["type"]
                }
            }
        }

        result = transform_ref_properties_to_allof(schema)

        assert "allOf" in result["$defs"]["Nested"]
        assert len(result["$defs"]["Nested"]["allOf"]) == 2

    def test_transforms_in_oneof(self):
        """Test that transformation works inside oneOf arrays"""
        schema = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"type": {"const": "foo"}},
                    "$ref": "#/$defs/Base",
                    "required": ["type"]
                },
                {
                    "type": "object",
                    "properties": {"type": {"const": "bar"}},
                    "$ref": "#/$defs/Base",
                    "required": ["type"]
                }
            ]
        }

        result = transform_ref_properties_to_allof(schema)

        assert "allOf" in result["oneOf"][0]
        assert "allOf" in result["oneOf"][1]

    def test_invalid_input(self):
        """Test that non-dict input raises ValueError"""
        with pytest.raises(ValueError, match="Schema is not a dictionary"):
            transform_ref_properties_to_allof("not a dict")


class TestExtractExportedNamesFromSchema:
    """Tests for extract_exported_names_from_schema function"""

    def test_empty_schema(self):
        """Test with empty schema"""
        schema = {}
        result = extract_exported_names_from_schema(schema)
        assert result == []

    def test_basic_definitions(self):
        """Test extracting basic definition names"""
        schema = {
            "$defs": {
                "TypeA": {"type": "string"},
                "TypeB": {"type": "number"},
                "TypeC": {"type": "boolean"}
            }
        }

        result = extract_exported_names_from_schema(schema)

        assert result == ["TypeA", "TypeB", "TypeC"]

    def test_extracts_titles(self):
        """Test that titles are extracted"""
        schema = {
            "$defs": {
                "inner_name": {
                    "title": "PublicName",
                    "type": "object"
                }
            }
        }

        result = extract_exported_names_from_schema(schema)

        assert "inner_name" in result
        assert "PublicName" in result

    def test_skips_ref_only_titles(self):
        """Test that titles on $ref-only schemas are skipped"""
        schema = {
            "$defs": {
                "Wrapper": {
                    "title": "ShouldBeSkipped",
                    "$ref": "#/$defs/Inner"
                },
                "Inner": {
                    "type": "object"
                }
            }
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
                        {
                            "title": "VariantA",
                            "type": "object",
                            "properties": {"type": {"const": "a"}}
                        },
                        {
                            "title": "VariantB",
                            "type": "object",
                            "properties": {"type": {"const": "b"}}
                        }
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
                        {"title": "PhantomB", "$ref": "#/$defs/TypeB"}
                    ]
                },
                "TypeA": {"type": "object"},
                "TypeB": {"type": "object"}
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
                        {
                            "title": "Extension",
                            "type": "object",
                            "properties": {"extra": {"type": "string"}}
                        }
                    ]
                },
                "Base": {"type": "object"}
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
                "Container": {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "title": "NestedType",
                            "type": "object"
                        }
                    }
                }
            }
        }

        result = extract_exported_names_from_schema(schema)

        assert "Container" in result
        assert "NestedType" in result

    def test_sorted_output(self):
        """Test that output is sorted alphabetically"""
        schema = {
            "$defs": {
                "Zebra": {"type": "string"},
                "Apple": {"type": "string"},
                "Mango": {"type": "string"}
            }
        }

        result = extract_exported_names_from_schema(schema)

        assert result == ["Apple", "Mango", "Zebra"]


class TestExtractNamesFromSchemaRecursive:
    """Tests for extract_names_from_schema_recursive helper function"""

    def test_extracts_title(self):
        """Test basic title extraction"""
        schema = {"title": "MyType", "type": "object"}
        names = set()

        extract_names_from_schema_recursive(schema, names)

        assert "MyType" in names

    def test_skips_title_with_ref(self):
        """Test that titles on $ref schemas are skipped"""
        schema = {"title": "SkipMe", "$ref": "#/$defs/Other"}
        names = set()

        extract_names_from_schema_recursive(schema, names)

        assert "SkipMe" not in names

    def test_handles_non_dict(self):
        """Test that non-dict input is handled gracefully"""
        names = set()

        extract_names_from_schema_recursive("not a dict", names)
        extract_names_from_schema_recursive(None, names)
        extract_names_from_schema_recursive(123, names)

        assert len(names) == 0

    def test_processes_array_items(self):
        """Test that array item schemas are processed"""
        schema = {
            "type": "array",
            "items": {
                "title": "ItemType",
                "type": "object"
            }
        }
        names = set()

        extract_names_from_schema_recursive(schema, names)

        assert "ItemType" in names
