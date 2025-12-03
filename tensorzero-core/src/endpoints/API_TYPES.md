# A note about API types

We currently use Rust as the source of truth for API types.

## Patch semantics

For APIs that update data in our database, we use patch semantics where the request represents a partial update, so users are not forced to
specify every data field. We also allow users to clear a field in many cases.

In Rust, we represent these types as `Option<Option<T>>`s, and use these custom annotations to support Python and TypeScript:

```rust
pub struct UpdateChatDatapointRequest {
    // ... omitted ...

    /// Chat datapoint output. If omitted, it will be left unchanged. If specified as `null`, it will be set to
    /// `null`. Otherwise, it will overwrite the existing output (and can be an empty array).
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "Chat datapoint output.

If omitted (which uses the default value `UNSET`), it will be left unchanged. If set to `None`, it will be cleared.
Otherwise, it will overwrite the existing output (and can be an empty list).")]
    pub output: Option<Option<Vec<ContentBlockChatOutput>>>,

    // ... omitted ...
}
```

The 3 semantics of this field are represented as follows in each language:

- Semantic: replace the field with a given value.
  - JSON: the key is set to the given value.
  - Python: the field is set to the given value.
  - Rust: the field is set to `Some(Some(value))`.
- Semantic: clear the field.
  - JSON: the key is set to `null`.
  - Python: the field is set to `None`.
  - Rust: the field is set to `Some(None)`.
- Semantic: leave as-is, do not update the field.
  - JSON: the key is omitted in the object.
  - Python: the field is not set (and takes the default value `UNSET`). We have serialization logic that removes fields with the default value `UNSET`.
  - Rust: the field is set to `None`, or use `..Default::default()` if supported.
