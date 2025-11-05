# Type Stub Generation

This directory contains a script to generate type stubs for the `tensorzero.tensorzero` Rust extension module.

## Generating Stubs

To regenerate the type stubs:

```bash
uv run python generate_stubs.py > tensorzero/tensorzero.pyi
```

## Validation

The generated stubs pass type checking and runtime validation:

```bash
# Type checking with pyright
uv run pyright tensorzero/tensorzero.pyi

# Runtime validation with stubtest  
uv run stubtest tensorzero.tensorzero
```

## Known Limitation

There is 1 expected stubtest error:

- `CreateDatapointsFromInferenceOutputSource.None` - The `None` variant cannot be represented in Python type stubs since `None` is a reserved keyword. The variant exists at runtime and is documented in the class docstring.

## Implementation Notes

The stub generator (`generate_stubs.py`) handles several PyO3 patterns:

1. **Tagged Enums with Constructors**: Types like `Datapoint` with `Chat` and `Json` variants that take parameters
2. **Simple Enums**: Types like `OptimizationJobStatus` with `Pending`, `Completed`, `Failed` variants as instances
3. **Hybrid Types**: Types like `StoredInference` that are tagged enums but also have properties
4. **Gateway Classes**: Sync and async gateways with proper `@classmethod` decorators for builders
5. **Properties**: PyO3 `getset_descriptor` attributes mapped to `@property` methods

The generator inspects the runtime module to ensure accuracy and automatically handles:
- Keyword-only parameters
- Default values
- Async methods and coroutine return types
- Context manager protocols (`__enter__`, `__exit__`, `__aenter__`, `__aexit__`)
