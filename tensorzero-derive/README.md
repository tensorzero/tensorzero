# tensorzero-derive

This is an internal crate containing derive macros using by `tensorzero-internal`

Current macros:

## `#[derive(TensorZeroDeserialize)]`

This is a drop-in replacement for `#[derive(serde::Deserialize)]` (only for enums). This uses `serde_path_to_error` to produce better error messages when deserializing non-externally-tagged enums (e.g. `#[serde(tag = "type")]`).

See 'tests/deserialize.rs' for an example
