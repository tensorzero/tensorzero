#!/usr/bin/env python3
"""
Test DatapointMetadataUpdate with UNSET sentinel value.

This test verifies that the UNSET sentinel correctly handles three states:
1. UNSET (field omitted from JSON)
2. None (field explicitly null in JSON)
3. value (field has a value in JSON)
"""

import json
from dataclasses import asdict

from tensorzero.generated_types import UNSET, _UnsetType, DatapointMetadataUpdate


def serialize_for_api(obj) -> dict:
    """
    Serialize a dataclass to dict, omitting fields with UNSET value.

    This mimics how the API client should serialize requests.
    """
    result = {}
    for key, value in asdict(obj).items():
        if not isinstance(value, _UnsetType):
            result[key] = value
    return result


def test_unset_field():
    """Test that UNSET fields are omitted from JSON."""
    print("Test 1: UNSET field (omitted)")
    print("-" * 70)

    # Create with default (UNSET)
    update = DatapointMetadataUpdate()

    # Verify the field is UNSET
    assert isinstance(update.name, _UnsetType), "Default should be UNSET"
    assert update.name is UNSET, "Default should be the UNSET singleton"
    print(f"✓ Field value is UNSET: {update.name}")

    # Serialize
    serialized = serialize_for_api(update)
    json_str = json.dumps(serialized)

    # Verify 'name' key is not in the JSON
    assert 'name' not in serialized, "UNSET field should be omitted"
    assert 'name' not in json_str, "UNSET field should not appear in JSON string"
    print(f"✓ Serialized JSON: {json_str}")
    print(f"✓ Key 'name' is absent from JSON")
    print()


def test_explicit_none():
    """Test that explicit None is serialized as null."""
    print("Test 2: Explicit None (null in JSON)")
    print("-" * 70)

    # Create with explicit None
    update = DatapointMetadataUpdate(name=None)

    # Verify the field is None
    assert update.name is None, "Field should be None"
    print(f"✓ Field value is None: {update.name}")

    # Serialize
    serialized = serialize_for_api(update)
    json_str = json.dumps(serialized)

    # Verify 'name' key is in the JSON with null value
    assert 'name' in serialized, "None field should be present"
    assert serialized['name'] is None, "None field should be null"
    assert json_str == '{"name": null}', f"Expected null in JSON, got: {json_str}"
    print(f"✓ Serialized JSON: {json_str}")
    print(f"✓ Key 'name' is present with null value")
    print()


def test_explicit_value():
    """Test that a value is serialized correctly."""
    print("Test 3: Explicit value")
    print("-" * 70)

    # Create with a value
    update = DatapointMetadataUpdate(name="my-datapoint")

    # Verify the field has the value
    assert update.name == "my-datapoint", "Field should have the value"
    print(f"✓ Field value: {update.name}")

    # Serialize
    serialized = serialize_for_api(update)
    json_str = json.dumps(serialized)

    # Verify 'name' key is in the JSON with the value
    assert 'name' in serialized, "Value field should be present"
    assert serialized['name'] == "my-datapoint", "Value field should have the value"
    assert json_str == '{"name": "my-datapoint"}', f"Expected value in JSON, got: {json_str}"
    print(f"✓ Serialized JSON: {json_str}")
    print(f"✓ Key 'name' is present with value")
    print()


def test_modification_scenarios():
    """Test realistic modification scenarios."""
    print("Test 4: Realistic modification scenarios")
    print("-" * 70)

    # Scenario 1: Don't change the name
    print("Scenario 1: Don't change the name")
    update1 = DatapointMetadataUpdate()
    json1 = json.dumps(serialize_for_api(update1))
    print(f"  Request: {json1}")
    print(f"  → Server will not modify the name field")
    print()

    # Scenario 2: Clear the name
    print("Scenario 2: Clear the name (set to null)")
    update2 = DatapointMetadataUpdate(name=None)
    json2 = json.dumps(serialize_for_api(update2))
    print(f"  Request: {json2}")
    print(f"  → Server will set the name field to null")
    print()

    # Scenario 3: Set a new name
    print("Scenario 3: Set a new name")
    update3 = DatapointMetadataUpdate(name="new-name")
    json3 = json.dumps(serialize_for_api(update3))
    print(f"  Request: {json3}")
    print(f"  → Server will set the name field to 'new-name'")
    print()


def test_type_checking():
    """Test runtime type checking."""
    print("Test 5: Runtime type checking")
    print("-" * 70)

    update_unset = DatapointMetadataUpdate()
    update_none = DatapointMetadataUpdate(name=None)
    update_value = DatapointMetadataUpdate(name="test")

    # Check types
    print(f"isinstance(update_unset.name, _UnsetType): {isinstance(update_unset.name, _UnsetType)}")
    print(f"update_unset.name is UNSET: {update_unset.name is UNSET}")
    print(f"update_none.name is None: {update_none.name is None}")
    print(f"isinstance(update_value.name, str): {isinstance(update_value.name, str)}")

    assert isinstance(update_unset.name, _UnsetType)
    assert update_unset.name is UNSET
    assert update_none.name is None
    assert isinstance(update_value.name, str)

    print("✓ All type checks passed")
    print()


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing DatapointMetadataUpdate with UNSET Sentinel")
    print("=" * 70)
    print()

    try:
        test_unset_field()
        test_explicit_none()
        test_explicit_value()
        test_modification_scenarios()
        test_type_checking()

        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  - UNSET: Field omitted from JSON (don't change)")
        print("  - None: Field set to null in JSON (clear value)")
        print("  - Value: Field set to value in JSON (update value)")
        print()
        print("This correctly implements Rust's Option<Option<T>> semantics!")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
