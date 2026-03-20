# Endpoint API type guidance

- Rust is the source of truth for endpoint API types.
- For public `utoipa::ToSchema` unions, prefer named payload structs over inline object variants.
- Keep discriminated enums, but model object variants as `Variant(NamedStruct)` instead of `Variant { ... }`.
- This helps `utoipa` emit `$ref`-based component schemas and improves SDK naming ergonomics.
- Use `README.md` in this directory for the fuller rationale and the patch-semantics guidance.
