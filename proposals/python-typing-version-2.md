# Python Typing Version 2: Expose `update_datapoints` via PyO3

## Overview

This proposal outlines the implementation plan to expose the `update_datapoints` method from the Rust client to the Python client, with full type safety support for Pyright.

## Background

The Rust client has an `update_datapoints` method that allows updating existing datapoints in a dataset. This method is not currently exposed to the Python client. We want to expose it in a way that:

1. Maintains full type safety with Pyright
2. Exposes types directly via PyO3 as PyClasses
3. Handles the semantic complexity of `Option<Option<T>>` for nullable fields
4. Creates new datapoint versions (returns new IDs) rather than updating in-place

## Current Rust API

### Method Signature

```rust
async fn update_datapoints(
    &self,
    dataset_name: String,
    datapoints: Vec<UpdateDatapointRequest>,
) -> Result<UpdateDatapointsResponse, TensorZeroError>;
```

### Key Types

```rust
pub struct UpdateDatapointsResponse {
    pub ids: Vec<Uuid>,  // Newly generated IDs for updated datapoints
}

pub enum UpdateDatapointRequest {
    Chat(UpdateChatDatapointRequest),
    Json(UpdateJsonDatapointRequest),
}

pub struct UpdateChatDatapointRequest {
    pub id: Uuid,
    pub input: Option<Input>,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub tool_params: Option<Option<DynamicToolParams>>,
    pub tags: Option<HashMap<String, String>>,
    pub metadata: Option<DatapointMetadataUpdate>,
}

pub struct UpdateJsonDatapointRequest {
    pub id: Uuid,
    pub input: Option<Input>,
    pub output: Option<Option<JsonDatapointOutputUpdate>>,
    pub output_schema: Option<Value>,
    pub tags: Option<HashMap<String, String>>,
    pub metadata: Option<DatapointMetadataUpdate>,
}

pub struct JsonDatapointOutputUpdate {
    pub raw: String,
}

pub struct DatapointMetadataUpdate {
    pub name: Option<Option<String>>,
}
```

### Semantics of `Option<Option<T>>`

The double-Option pattern has special semantics:
- `None` = leave the field unchanged
- `Some(None)` = set the field to null
- `Some(Some(value))` = set the field to the specified value

## Proposed Python API

### Wrapper Classes for Nullable Fields

Create sentinel classes in `clients/python/tensorzero/types.py`:

```python
class Unchanged:
    """Sentinel to indicate a field should remain unchanged during update."""
    pass

class Null:
    """Sentinel to explicitly set a field to null during update."""
    pass
```

### PyO3 Class Exports

Export the following types as PyClasses in `clients/python/src/lib.rs`:

1. **DatapointMetadataUpdate**
   ```python
   @dataclass
   class DatapointMetadataUpdate:
       name: Union[Unchanged, Null, str]
   ```

2. **JsonDatapointOutputUpdate**
   ```python
   @dataclass
   class JsonDatapointOutputUpdate:
       raw: str
   ```

3. **ChatDatapointUpdate**
   ```python
   @dataclass
   class ChatDatapointUpdate:
       id: UUID
       input: Union[Unchanged, InferenceInput]
       output: Union[Unchanged, List[ContentBlock]]
       tool_params: Union[Unchanged, Null, Dict[str, Any]]
       tags: Union[Unchanged, Dict[str, str]]
       metadata: Union[Unchanged, DatapointMetadataUpdate]
   ```

4. **JsonDatapointUpdate**
   ```python
   @dataclass
   class JsonDatapointUpdate:
       id: UUID
       input: Union[Unchanged, InferenceInput]
       output: Union[Unchanged, Null, JsonDatapointOutputUpdate]
       output_schema: Union[Unchanged, Any]
       tags: Union[Unchanged, Dict[str, str]]
       metadata: Union[Unchanged, DatapointMetadataUpdate]
   ```

### Method Signatures

```python
# Sync version
def update_datapoints(
    self,
    *,
    dataset_name: str,
    datapoints: Sequence[Union[ChatDatapointUpdate, JsonDatapointUpdate]],
) -> List[UUID]:
    """
    Update datapoints in a dataset.

    This creates NEW datapoint versions with new IDs rather than updating in-place.
    The returned UUIDs are the new datapoint IDs, not the original ones.

    Args:
        dataset_name: Name of the dataset
        datapoints: List of datapoint updates

    Returns:
        List of new UUIDs for the created datapoint versions
    """

# Async version
async def update_datapoints(
    self,
    *,
    dataset_name: str,
    datapoints: Sequence[Union[ChatDatapointUpdate, JsonDatapointUpdate]],
) -> List[UUID]:
    """
    Update datapoints in a dataset.

    This creates NEW datapoint versions with new IDs rather than updating in-place.
    The returned UUIDs are the new datapoint IDs, not the original ones.

    Args:
        dataset_name: Name of the dataset
        datapoints: List of datapoint updates

    Returns:
        List of new UUIDs for the created datapoint versions
    """
```

## Implementation Plan

### 1. Create Sentinel Classes

**File:** `clients/python/tensorzero/types.py`

Add the `Unchanged` and `Null` sentinel classes.

### 2. Export PyO3 Classes

**File:** `clients/python/src/lib.rs`

- Add `#[pyclass]` declarations for:
  - `DatapointMetadataUpdate`
  - `JsonDatapointOutputUpdate`
  - `ChatDatapointUpdate`
  - `JsonDatapointUpdate`

- Implement custom deserialization logic to convert:
  - Python `Unchanged()` → Rust `None`
  - Python `Null()` → Rust `Some(None)`
  - Python value → Rust `Some(Some(value))`

### 3. Add Python Bindings

**File:** `clients/python/src/lib.rs`

Add two method implementations:

1. **Sync version** in `impl TensorZeroGateway` (around line 1000):
   - Accept `dataset_name: String` and `datapoints: Vec<Bound<'_, PyAny>>`
   - Deserialize Python objects to `Vec<UpdateDatapointRequest>`
   - Call `client.update_datapoints(dataset_name, datapoints)`
   - Use `tokio_block_on_without_gil` for async execution
   - Convert returned UUIDs to Python UUID objects
   - Return as `Py<PyList>`

2. **Async version** in `impl AsyncTensorZeroGateway` (around line 1800):
   - Same signature and logic
   - Use `pyo3_async_runtimes::tokio::future_into_py` for async execution

### 4. Add Type Stubs

**File:** `clients/python/tensorzero/tensorzero.pyi`

- Add method signatures for both sync and async versions
- Add type stubs for all new classes
- Export new types in `__all__` list
- Include comprehensive docstrings emphasizing the "new IDs" behavior

### 5. Write Tests

**File:** `clients/python/tests/test_datapoints.py`

Add test functions:

1. **test_update_chat_datapoint_sync**:
   - Create a chat datapoint
   - Update it with new output
   - Verify new ID is returned (different from original)
   - Optionally verify the update worked

2. **test_update_chat_datapoint_async**:
   - Same as sync but with async client

3. **test_update_with_unchanged**:
   - Create datapoint with multiple fields
   - Update with some fields as `Unchanged()`
   - Verify unchanged fields remain the same

4. **test_update_with_null**:
   - Update a field to null using `Null()`
   - Verify field is set to null

5. **test_update_json_datapoint**:
   - Test updating JSON datapoints
   - Verify output schema updates work

## Type Safety Considerations

### PyO3 Class Design

By exposing types as PyClasses with `#[pyclass]` and `#[pymethods]`, we get:

1. **Runtime type checking**: PyO3 validates types at the Rust boundary
2. **IDE support**: Pyright can see the class definitions
3. **Clear API**: Users construct typed objects rather than dicts
4. **Consistent patterns**: Matches how other types are exposed in the client

### Sentinel Pattern

The `Unchanged` and `Null` sentinels provide:

1. **Explicit intent**: Clear distinction between "don't change", "set to null", and "set to value"
2. **Type safety**: Can be included in Union types for Pyright
3. **User-friendly**: Simple to use (`Unchanged()` or `Null()`)
4. **No magic values**: No confusion with `None` or other Python values

## Testing Strategy

### Unit Tests

- Test all update scenarios (unchanged, null, value updates)
- Test both Chat and JSON datapoint types
- Test both sync and async clients
- Verify new IDs are returned

### Type Checking

- Run Pyright on test files to verify type safety
- Ensure no type errors in the stub file

### Integration Tests

- Create datapoint, update it, fetch it back
- Verify update semantics match Rust client behavior

## Migration Path

This is a new feature, so no migration is needed. Users who want to update datapoints can start using this method immediately.

## Alternatives Considered

### Alternative 1: Use `UNSET` sentinel constant

Instead of wrapper classes, use a module-level constant:

```python
UNSET = object()
```

**Pros:**
- Simpler (no class to instantiate)
- Common Python pattern

**Cons:**
- Less explicit in type signatures
- Harder to distinguish in deserialization logic
- Can't have multiple sentinel types (UNSET vs NULL)

**Decision:** Rejected. We need two sentinels (Unchanged and Null), and classes are more explicit.

### Alternative 2: Simplify to single Option

Remove the `Option<Option<T>>` pattern and treat `None` as "leave unchanged":

**Pros:**
- Simpler API
- No sentinel classes needed

**Cons:**
- Can't explicitly set fields to null
- Inconsistent with Rust API semantics
- Loses important functionality

**Decision:** Rejected. The ability to set fields to null is important for the update API.

### Alternative 3: Use inline dicts

Instead of exposing PyClasses, accept plain Python dicts:

**Pros:**
- Less boilerplate
- More Pythonic

**Cons:**
- No type safety
- Harder to validate at Rust boundary
- Inconsistent with project goal of "making typing happy"

**Decision:** Rejected. We want full type safety via PyO3 classes.

## Success Criteria

1. All tests pass (unit and integration)
2. Pyright reports no type errors in test files
3. Sync and async versions work correctly
4. New IDs are correctly returned
5. Sentinel values (Unchanged, Null) work as expected
6. Documentation clearly explains the "new IDs" behavior

## Timeline

This can be implemented in a single session:

1. Write the test (15 min)
2. Add sentinel classes (5 min)
3. Export PyO3 classes (30 min)
4. Add Python bindings (30 min)
5. Add type stubs (15 min)
6. Run tests and fix issues (15 min)

**Total:** ~2 hours
