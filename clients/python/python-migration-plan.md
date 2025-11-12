# Python Type Migration Plan: PyO3 Classes → Generated Dataclasses

## Overview

This document tracks the migration of PyO3 classes to auto-generated Python dataclasses using JSON Schema generation. The goal is to reduce PyO3 boilerplate and maintain a single source of truth in Rust.

## Migration Strategy

### For Simple Enums/Data Types

1. **Add JSON Schema generation in Rust**:
   - Add `JsonSchema` to derives
   - Add `#[export_schema]` attribute
   - Add `#[schemars(...)]` attributes for customization
   - Import: `use schemars::JsonSchema;` and `use tensorzero_derive::export_schema;`

2. **Remove PyO3 class registration**:
   - Remove `#[pyclass]` attribute (or `#[cfg_attr(feature = "pyo3", pyclass(...))]`)
   - Remove `m.add_class::<Type>()?;` from `clients/python/src/lib.rs`
   - Remove unused imports

3. **Generate Python types**:
   - Run: `python clients/python/generate_schema_types.py`
   - Types auto-generated in `clients/python/tensorzero/generated_types.py`

4. **Update Python imports**:
   - Import from `.generated_types` in `__init__.py`
   - Remove manual type definitions from `types.py`
   - Update `__all__` exports

5. **Rebuild and test**:
   - `cd clients/python && uv run maturin develop --release`
   - Verify imports and type structure

## Status Tracking

### ✅ Phase 1: Completed Migrations

#### OptimizationJobStatus
- **Status**: ✅ Fully Migrated
- **Type**: Simple enum
- **Commit Date**: 2025-01-12
- **Changes**:
  - Added `JsonSchema` derive and `#[export_schema]` attribute
  - Removed `#[cfg_attr(feature = "pyo3", pyclass(str, eq))]`
  - Removed from `m.add_class::<OptimizationJobStatus>()?` registration
  - Generated type: `Literal["Pending", "Completed", "Failed"]`
  - Updated imports to use `generated_types`
- **Files Modified**:
  - `tensorzero-core/src/optimization/mod.rs`
  - `clients/python/src/lib.rs`
  - `clients/python/tensorzero/__init__.py`
  - `clients/python/tensorzero/types.py` (removed manual definition)
- **Notes**:
  - `OptimizationJobInfoPyClass.get_status()` returns `&str` (correct - wrapper stays PyO3)
  - Type is auto-generated from schema and matches manual definition

---

### 🔄 Phase 1: In Progress

#### ResolvedInput & ResolvedInputMessage
- **Status**: 🔄 Pending
- **Type**: Simple data structures
- **Priority**: High (used internally)
- **Estimated Complexity**: Low
- **Location**: `tensorzero-core/src/inference/types/resolved_input.rs`
- **Current Usage**: Internal representation of resolved input messages
- **Migration Plan**:
  1. Add `JsonSchema` derives
  2. Add `#[export_schema]` attributes
  3. Remove PyO3 class registrations
  4. Update schema generation
  5. Update Python imports

#### Datapoint (LegacyDatapoint)
- **Status**: 🔄 Pending
- **Type**: Enum with Chat/Json variants
- **Priority**: High (marked as legacy, safe to migrate)
- **Estimated Complexity**: Medium
- **Location**: `tensorzero-core/src/endpoints/datasets/legacy.rs`
- **Current Usage**: Returned from `get_datapoint`, `list_datapoints_legacy`
- **Migration Plan**:
  1. Add `JsonSchema` derives to enum and variants
  2. Add `#[export_schema]` attribute
  3. Remove `#[pyclass(str, name = "LegacyDatapoint")]`
  4. Convert return types to use `convert_response_to_python`
  5. Update schema generation
  6. Update Python imports

---

### 📋 Phase 2: Planned

#### OptimizationJobInfoPyClass
- **Status**: 📋 Planned
- **Type**: Wrapper class with getters
- **Priority**: Medium
- **Estimated Complexity**: Medium
- **Notes**: Wraps `OptimizationJobInfo` enum, provides property access
- **Migration Plan**: Convert to generated dataclass with dacite

#### OptimizationJobHandle
- **Status**: 📋 Planned
- **Type**: Enum with serialization logic
- **Priority**: Medium
- **Estimated Complexity**: Medium
- **Notes**: Has `to_base64_urlencoded()` and `from_base64_urlencoded()` methods
- **Migration Plan**: Generate type, implement serialization in Python or keep methods in Rust

#### Uninitialized*Config Classes (6 types)
- **Status**: 📋 Planned
- **Type**: Configuration structs with validation
- **Priority**: Low
- **Estimated Complexity**: Medium
- **Types**:
  - `UninitializedOpenAISFTConfig`
  - `UninitializedOpenAIRFTConfig`
  - `UninitializedFireworksSFTConfig`
  - `UninitializedDiclOptimizationConfig`
  - `UninitializedGCPVertexGeminiSFTConfig`
  - `UninitializedTogetherSFTConfig`
- **Notes**: Have `#[new]` constructors with validation - keep validation in Rust
- **Migration Plan**: Generate types, validation happens when configs are used

---

### 🚫 Not Migrating (Keep as PyO3)

#### EvaluationJobHandler & AsyncEvaluationJobHandler
- **Reason**: Complex stateful behavior with iterator protocols
- **Details**:
  - Implements `__iter__`/`__next__` and `__aiter__`/`__anext__`
  - Maintains `Mutex<Receiver<EvaluationUpdate>>` for streaming
  - Has stateful accumulation (`evaluation_infos`, `evaluation_errors`)
  - Too complex for dataclass + dacite approach

#### Config Wrapper Classes
- **Reason**: Runtime wrappers around Arc pointers
- **Types**:
  - `ConfigPyClass`
  - `FunctionsConfigPyClass`
  - `FunctionConfigChatPyClass`
  - `FunctionConfigJsonPyClass`
  - `VariantsConfigPyClass`
  - `ChatCompletionConfigPyClass`
  - `BestOfNSamplingConfigPyClass`
  - `DiclConfigPyClass`
  - `MixtureOfNConfigPyClass`
  - `ChainOfThoughtConfigPyClass`
- **Details**: Wrap `Arc<Config>` and provide controlled access patterns

---

## Reference: Complete Migration Pattern

### Rust Side

```rust
// Before
#[cfg_attr(feature = "pyo3", pyclass(str, eq))]
pub enum MyType {
    VariantA,
    VariantB,
}

// After
use schemars::JsonSchema;
use tensorzero_derive::export_schema;

#[derive(JsonSchema, ...)]
#[schemars(rename_all = "PascalCase")]
#[export_schema]
pub enum MyType {
    VariantA,
    VariantB,
}
```

### Python Side

```python
# Before (types.py)
MyType = Literal["VariantA", "VariantB"]

# After (auto-generated in generated_types.py)
MyType = Literal["VariantA", "VariantB"]

# __init__.py
from .generated_types import MyType  # Instead of .tensorzero or .types
```

### Lib Registration

```rust
// Remove from clients/python/src/lib.rs
m.add_class::<MyType>()?;  // DELETE THIS LINE
```

## Benefits of Migration

1. **Single Source of Truth**: Schema defined in Rust, Python types auto-generated
2. **Reduced Boilerplate**: No PyO3 `#[pyclass]` and `#[pymethods]` needed
3. **Type Safety**: Generated types match Rust types exactly
4. **Maintainability**: Schema changes automatically propagate to Python
5. **Consistency**: Uses same generation pipeline as all other types
6. **Better IDE Support**: Generated Python types have proper type hints

## Testing Checklist

For each migrated type:

- [ ] Compiles without errors in Rust
- [ ] Schema generation succeeds
- [ ] Type appears in `generated_types.py`
- [ ] Can import from `tensorzero` module
- [ ] Type structure matches expectations (Literal, dataclass, etc.)
- [ ] Existing tests pass (if applicable)
- [ ] Python type checkers accept the type (pyright, mypy)

## Common Issues

1. **Import not found**: Regenerate schemas with `python generate_schema_types.py`
2. **Type mismatch**: Check `#[schemars(...)]` attributes for rename/tag configuration
3. **Compilation error**: Ensure `use schemars::JsonSchema;` is imported
4. **Union type issues**: dacite handles Union types automatically, verify with tests

## Next Steps

1. Complete Phase 1 migrations (ResolvedInput, Datapoint)
2. Run comprehensive test suite
3. Evaluate Phase 2 migrations based on Phase 1 learnings
4. Document any patterns or gotchas discovered
5. Consider scripting the migration process for remaining types

---

Last Updated: 2025-01-12
