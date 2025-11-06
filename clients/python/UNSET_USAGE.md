# UNSET Sentinel Usage

The generated types include an `UNSET` sentinel value to distinguish between omitted fields and null values in API requests. This corresponds to Rust's `Option<Option<T>>` type.

## How It Works

1. **Rust Side**: Fields that use `Option<Option<T>>` are marked with `#[serde(deserialize_with = "deserialize_double_option")]`
2. **OpenAPI Generation**: The OpenAPI schema is post-processed to add `x-double-option: true` to these fields
3. **Python Generation**: The post-processor reads the OpenAPI schema and generates Python types with UNSET sentinel for marked fields

## Fields with UNSET Support

Fields marked with `x-double-option: true` in the OpenAPI schema automatically get UNSET support:
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

The `UNSET` sentinel is automatically generated during the type generation process. The build pipeline:

1. Generates OpenAPI schema from Rust types using `utoipa`
2. Post-processes the schema to add `x-double-option: true` to fields with `deserialize_double_option`
3. Generates Python dataclasses from the OpenAPI schema using `datamodel-code-generator` with:
   - Custom templates that check for the `x-double-option` extension
   - Custom file header that defines the UNSET class
4. Formats the output with `black`

To add UNSET support to a new field:
1. Use `Option<Option<T>>` in Rust
2. Add `#[serde(deserialize_with = "deserialize_double_option")]`
3. Add `#[derive(utoipa::ToSchema)]` to the struct
4. Add the field to the `DOUBLE_OPTION_FIELDS` set in `add_double_option_extensions.py`
5. Run `build-openapi-bindings.sh` to regenerate types

The custom template (`templates/dataclass.jinja2`) checks for `field.extras.get('double_option')` and generates `| _UnsetType = UNSET` for marked fields.
