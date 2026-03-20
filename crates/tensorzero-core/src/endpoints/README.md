# Endpoint API types

Rust is the source of truth for API types.

## OpenAPI / SDK naming guidance

When modeling request or response unions for `utoipa`, prefer reusable named schemas over inline object variants.

### Preferred pattern

Use discriminated enums whose object-shaped variants wrap named structs:

```rust
#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct ToolOutcomeRejected {
    pub reason: String,
}

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolOutcome {
    Success(AutopilotToolResult),
    Rejected(ToolOutcomeRejected),
    Missing,
}
```

This gives `utoipa` a named `ToSchema` type to reference via `$ref`, which tends to produce cleaner SDKs.

### Avoid when possible

Avoid struct-like enum variants for public API types when the variant is object-shaped:

```rust
pub enum ToolOutcome {
    Rejected { reason: String },
}
```

Those variants are typically emitted as inline `oneOf` / `allOf` members, which many SDK generators rename to generic names like `UnionMember1` or `ParamsUnionMember2`.

### Practical rule of thumb

For public API enums derived with `utoipa::ToSchema`:

- keep the enum discriminated with `#[serde(tag = "type")]`
- if a variant carries an object payload, extract that payload into a named `struct`
- derive `utoipa::ToSchema` on the payload struct
- optionally use `#[schema(title = "...")]` on enum variants for stable branch names
- optionally use `#[schema(as = path::to::Name)]` on structs if you need a stable component name

### Why this matters

Named payload structs let `utoipa` emit reusable component schemas and `$ref`s instead of anonymous inline objects. That improves:

- generated SDK type names
- schema reuse and composability
- readability of the generated OpenAPI

## Patch semantics

For APIs that update data in our database, we use patch semantics where the request represents a partial update, so users are not forced to specify every data field. We also allow users to clear a field in many cases.

In Rust, we represent these types as `Option<Option<T>>`s, and use these custom annotations to support Python and TypeScript:

```rust
pub struct UpdateChatDatapointRequest {
    // ... omitted ...

    /// Chat datapoint output. If omitted, it will be left unchanged. If specified as `null`, it will be set to
    /// `null`. Otherwise, it will overwrite the existing output (and can be an empty array).
    #[serde(default, deserialize_with = "deserialize_double_option")]
    #[schemars(extend("x-double-option" = true), description = "Chat datapoint output.

If omitted (which uses the default value `OMIT`), it will be left unchanged. If set to `None`, it will be cleared.
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
  - Python: the field is not set (and takes the default value `OMIT`). We have serialization logic that removes fields with the default value `OMIT`.
  - Rust: the field is set to `None`, or use `..Default::default()` if supported.
