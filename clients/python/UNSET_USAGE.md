# UNSET Sentinel Usage

The generated types include an `UNSET` sentinel value to distinguish between omitted fields and null values in API requests. This corresponds to Rust's `Option<Option<T>>` type.

## Fields with UNSET Support

The following fields support the `UNSET` sentinel:
- `UpdateChatDatapointRequest.tool_params`
- `UpdateJsonDatapointRequest.output`
- `DatapointMetadataUpdate.name`

## Usage Examples

### 1. Omit a field (don't change existing value)

```python
from tensorzero.types_generated import DatapointMetadataUpdate, UNSET

# Don't change the name field
update = DatapointMetadataUpdate(name=UNSET)
```

### 2. Set a field to null

```python
# Explicitly set name to null
update = DatapointMetadataUpdate(name=None)
```

### 3. Set a field to a value

```python
# Set name to a specific value
update = DatapointMetadataUpdate(name="My Datapoint")
```

## JSON Serialization

When serializing to JSON:
- `UNSET` → field is omitted from JSON
- `None` → field is set to `null` in JSON
- `value` → field is set to the value in JSON

Example JSON outputs:

```python
# Case 1: UNSET
DatapointMetadataUpdate(name=UNSET)
# → {}

# Case 2: None
DatapointMetadataUpdate(name=None)
# → {"name": null}

# Case 3: Value
DatapointMetadataUpdate(name="My Datapoint")
# → {"name": "My Datapoint"}
```

## Implementation Note

The `UNSET` sentinel is automatically generated during the type generation process and corresponds to fields in Rust that use `#[serde(deserialize_with = "deserialize_double_option")]`.
