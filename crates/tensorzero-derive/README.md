# tensorzero-derive

This is an internal crate containing derive macros using by `tensorzero-core`

Current macros:

## `#[derive(TensorZeroDeserialize)]`

This is a drop-in replacement for `#[derive(serde::Deserialize)]` (only for enums). This uses `serde_path_to_error` to produce better error messages when deserializing non-externally-tagged enums (e.g. `#[serde(tag = "type")]`).

See 'tests/deserialize.rs' for an example

## `#[export_schema]`

This macro generates tests for exporting JSON schemas (generated with `schemars`), similar to `ts-rs`.

Usage:

1. **In Rust:** Add the derive macros

```rust
use tensorzero_derive::export_schema;

#[derive(JsonSchema, Serialize, Deserialize)]
#[export_schema]
pub struct NewType {
    pub field: String,
}
```

2. **Run schema generation:** `cargo test export_schema`. This exports the schema to `REPOSITORY_ROOT/clients/schema`.
