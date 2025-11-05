# Type Stub Generation

This directory contains a script to generate type stubs for the `tensorzero.tensorzero` Rust extension module.

## Generating Stubs

To regenerate the type stubs:

```bash
uv run python generate_stubs.py > tensorzero/tensorzero.pyi
```

## Validation Results

✅ **Pyright**: 0 errors, 0 warnings
✅ **Stubtest**: 0 errors - All stubs match runtime exactly!
✅ **Runtime Tests**: All passing

```bash
# Type checking with pyright
uv run pyright tensorzero/tensorzero.pyi

# Runtime validation with stubtest
uv run stubtest tensorzero.tensorzero
```

## Type Annotations

The generated stubs use **proper type annotations** instead of `Any` wherever possible:

- **UUID types**: `UUID` for IDs (inference_id, episode_id, datapoint_id, etc.)
- **String types**: `str` for names, templates, messages, etc.
- **Boolean types**: `bool` for flags (verbose_errors, async_setup, dryrun, etc.)
- **Integer types**: `int` for limits, offsets, counts, etc.
- **Float types**: `float` for timeouts, values, etc.
- **List types**: `List[UUID]`, `List[str]`, `List[Dict[str, Any]]`, etc.
- **Dict types**: `Dict[str, str]` for tags, `Dict[str, Any]` for inputs/outputs
- **Config types**: Specific config classes like `FunctionsConfig`, `VariantsConfig` for config properties
- **Union types**: Proper unions for parameters that accept multiple types
- **Optional types**: `Optional[T]` for parameters with defaults

Only ~132 remaining `Any` types (down from 580+), primarily for:
- Complex nested structures where runtime type is truly dynamic
- PyO3 method return types that need inference context
- Configuration internals

## Implementation Notes

The stub generator (`generate_stubs.py`) handles several PyO3 patterns:

1. **Tagged Enums with Constructors**: Types like `Datapoint` with `Chat` and `Json` variants that take parameters
2. **Simple Enums**: Types like `OptimizationJobStatus` with `Pending`, `Completed`, `Failed` variants as instances
3. **Hybrid Types**: Types like `StoredInference` that are tagged enums but also have properties
4. **Gateway Classes**: Sync and async gateways with proper `@classmethod` decorators for builders
5. **Properties**: PyO3 `getset_descriptor` attributes mapped to `@property` methods with correct types

The generator inspects the runtime module to ensure accuracy and automatically handles:
- Keyword-only parameters with proper types
- Default values
- Async methods and coroutine return types
- Context manager protocols (`__enter__`, `__exit__`, `__aenter__`, `__aexit__`)
- UUID types throughout the API
- Proper return types for gateway methods (e.g., `CreateDatapointsResponse`, `List[Datapoint]`)
- Config property return types (e.g., `Config.functions -> FunctionsConfig`, `variants -> VariantsConfig`)
