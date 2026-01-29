# Developer Notes

> [!IMPORTANT]
>
> These notes are for developers who are working on the Python client itself, not for developers using the client for their own projects.

## Local Installation

We recommend using [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
uv run maturin develop --uv
uv run python
```

## Running tests

First, build the client with `uv run maturin develop --uv --features e2e_tests`.

Integration tests can be run with `./test.sh` (this requires the same setup as `cargo test-e2e` - see `CONTRIBUTING.md`)

This also runs all tests for OpenAI compatibility in Python.

## Naming

There are several different names in use in this client:

- `python` - this is the name of the top-level directory for the Python client implementation.
- `tensorzero-python` - this is the rust _crate_ name, so that we get sensible output from running Cargo
- `tensorzero` - this is the name of the Python package (python code can use `import tensorzero`)
- `tensorzero_rust` - this is the (locally-renamed) Rust client package, which avoids conflicts with pyo3-generated code.

## Generating Python dataclasses from JSON Schema

For pure value types (mostly used in APIs), we generate them from Rust, via JSON Schema.

```
Rust Types (with annotations)
    ↓ (cargo test export_schema)
JSON Schemas (in clients/schemas/)
    ↓ (python generate_schema_types.py)
Python Dataclasses (in clients/python/tensorzero/generated_types/generated_types.py)
```

## (WIP) Customizing generated types

There are a few important ways we should customize the JSON Schemas generated from Rust:

### Naming tagged enum variants

For Rust enums (union types), add a title to each variant that holds values if the enum is **tagged** in Serde representation. Do not add this to **untagged** enums.

```rust
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    #[schemars(title = "ContentBlockText")]
    Text({ text: String })
    #[schemars(title = "ContentBlockImage")]
    Image({ data: String, url: String })
}
```

**Rationale:** By default JSON Schema doesn't name the structured enum variants, so for this struct:

```rust
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text({ text: String })
    Image({ data: String, url: String })
}
```

The generated python by default is:

```python
@dataclass
class ContentBlock1:
    text: str
    type: Literal["text"]

@dataclass
class ContentBlock2:
    data: str
    url: str
    type: Literal["image"]

ContentBlock = ContentBlock1 | ContentBlock2
```

This is bad for Python consumers who construct the enum variants (`ContentBlock1` instead of `ContentBlockText`) directly.

However, for untagged enums, the underlying type is directly included as a `$ref` and generated as one of the union types, so the title is useless.

### Explicitly tagging "double option" fields

For any fields typed `Option<Option<T>>`, add this annotation:

```rust
struct DatapointMetadataUpdate {
    #[schemars(extend("x-double-option" = true))]
    name: Option<Option<String>>,
}
```

**Rationale**: In Rust, we use `Option<Option<T>>` in some update operations to support HTTP PATCH semantics:

- Missing value means "do not modify" (`None`)
- `null` means "set to empty" (`Some(None)`)
- `value` means "set to `value`" (`Some(Some(value))`)

JSON Schema represents this as a nullable field that's not required. However, by default, the generated Python uses `T | None` and doesn't distinguish `null` and missing values.

To represent the semantic difference, we generate custom sentinel values in dataclasses to represent this, so:

```rust
struct DatapointMetadataUpdate {
    #[schemars(extend("x-double-option" = true))]
    name: Option<Option<String>>,
}
```

The generated python is:

```python
@dataclass
class DatapointMetadataUpdate:
    name: str | None | OmitType = OMIT
```
