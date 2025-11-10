"""
Unit tests for generate_schema_types.py

Run with: pytest test_generate_schema_types.py -v
"""

from generate_schema_types import (
    extract_exported_names_from_schema,
    extract_names_from_schema_recursive,
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
