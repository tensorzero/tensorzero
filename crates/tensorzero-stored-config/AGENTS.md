# Stored Config Types (`tensorzero-stored-config`)

This crate contains types for database-persisted configs and the Postgres migrations for config tables. These types are distinct from the `Uninitialized*` types used for TOML config loading in `tensorzero-core`.

## Key rules

### Never reuse `Uninitialized*` types inside stored types

Even if a type has no `ResolvedTomlPathData`, create a parallel `Stored*` type. The TOML types use `#[serde(deny_unknown_fields)]` which breaks forward compatibility. Stored types and TOML types evolve independently — decoupling them now avoids forced schema bumps later.

### No `#[serde(deny_unknown_fields)]`

Stored types must NOT use `deny_unknown_fields`. This allows additive field additions (new `Option<T>` fields) without a schema version bump. Old readers safely ignore unknown fields.

### No `#[serde(default)]`

Stored types must NOT use `#[serde(default)]`. The write path always writes the complete canonical form. On the read side:

- `Option<T>` fields: missing keys deserialize to `None` naturally.
- Non-`Option` collections (`Vec`, `BTreeMap`) and structs with defaults (`RetryConfig`): wrap in `Option<T>`. `None` means "this row predates the field"; `Some(vec![])` means "explicitly empty." The conversion to `Uninitialized*` maps `None` to the appropriate default.

### No `#[serde(flatten)]`

Stored types must NEVER use `#[serde(flatten)]`. Flatten has known edge cases with tagged enums and makes the JSON shape implicit rather than explicit. Instead:

- If flattening an enum into a struct (e.g., variant config + timeouts), nest the enum under an explicit key (e.g., `variant: StoredLLMJudgeVariantConfig`).
- If one struct wraps another (e.g., chain-of-thought wrapping chat completion), nest it under an explicit field name (e.g., `inner: StoredLLMJudgeChatCompletionVariantConfig`). Do not duplicate fields.

### No custom serializers/deserializers

Stored types must use only standard `#[derive(Serialize, Deserialize)]`. No `TensorZeroDeserialize` (because we don't care about error messages in this crate), no custom `impl Deserialize`, no `deserialize_with`. This keeps the serialization behavior trivially verifiable from the type definition alone.

Because stored types use only standard serde derives with no `flatten` or custom (de)serializers, avoid exhaustive serde-shape tests that only restate the type definition. Small serde round-trip smoke tests are acceptable when they guard representative stored configs or a regression in the persisted shape.

### No raw `serde_json::Value` unless the database shape is map-like

Do not serialize raw `serde_json::Value` in stored types unless the data is intentionally represented as a `HashMap`-like structure in the database. Prefer explicit stored structs and enums so the persisted JSON shape remains typed, reviewable, and forward-compatible.

### Use `#[serde_with::skip_serializing_none]` on all structs

Use `#[serde_with::skip_serializing_none]` on all stored structs to automatically apply `#[serde(skip_serializing_if = "Option::is_none")]` to every `Option<T>` field. This keeps JSONB compact and avoids two representations for "absent" (missing key vs. `null`). Using the struct-level attribute instead of per-field annotations prevents forgetting it on new fields.

### `PartialEq` on all stored types

Derive `PartialEq` on all stored types for use in conversion tests and assertions.

### No dependency on `tensorzero-core`

This crate must not depend on `tensorzero-core` (it would be circular). Types from core that appear in stored configs (e.g., `ExtraBodyConfig`, `ExtraHeadersConfig`) have parallel stored equivalents in this crate. Types from `tensorzero-types` (e.g., `JsonMode`, `ServiceTier`) can be used directly. We should soon move all config types in here.

### No `HashMap`

In multiple places we hash the content of a config to get a value for dedupe purposes. This does not work with HashMaps, which do not guarantee element ordering. Use BTreeMap to guarantee that entries are sorted by key.

### Schema revision policy

Each config table has a `schema_revision` column. Each stored config type defines its own `SCHEMA_REVISION` constant.

**Additive changes** (new `Option<T>` fields) do NOT require a revision bump — old readers ignore unknown fields, new readers get `None` for missing fields.

**Breaking changes** REQUIRE a revision bump. A change is breaking if an old reader cannot correctly deserialize a row written by a new writer. Examples:

- Renaming or removing a field
- Changing enum variant tag strings or discriminator values
- Changing nesting structure (e.g., moving a field into a sub-object)
- Promoting an `Option<T>` to required (after backfill)

Never change a field's type or semantics; always add a new field (safe), and remove the old one (breaking).

When bumping a revision, follow the expand-and-contract cycle: the new code must read both old and new revisions, write the old revision during the overlap period, then contract to the new revision once old writers are drained.

**New enum variants** do NOT require a revision bump. Old rows never contain the new variant, so old readers are unaffected. However, new variants require a two-phase deploy:

1. **Phase 1 (expand):** Deploy code that can _read_ the new variant but does not yet _write_ it.
2. **Phase 2 (write):** Deploy code that writes the new variant. All running readers can now handle it.

## Migrations

SQL migrations for config tables live in `src/postgres/migrations/`. The migration runner in `tensorzero-core` calls `tensorzero_stored_config::postgres::make_migrator()` to apply them. Migrations use a dedicated tracking table (`tensorzero_stored_config__sqlx_migrations`).
