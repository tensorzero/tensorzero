# Rust

- Use `cargo check` for quick verification, restrict further (e.g. `cargo check --package tensorzero-core`) if appropriate. For complex changes, you might want to run `cargo check --all-targets --all-features`. Test suite compilation is slow.
- If you update Rust types or functions used in TypeScript, regenerate bindings with `pnpm build-bindings` (from root), then rebuild the NAPI bindings with `pnpm --filter=tensorzero-node build`. Run `cargo check` first to catch compilation errors.
- If you change a signature of a struct, function, and so on, use `grep` to find all instances in the codebase. For example, search for `StructName {` when updating struct fields.
- Place crate imports at the top of the file or module using `use crate::...`. Avoid imports inside functions or tests. Avoid long inline crate paths.
- Run tests with `cargo nextest`.
- Once you're done with your work, make sure to:
  - Run `cargo fmt`.
  - Run `cargo clippy --all-targets --all-features -- -D warnings` to catch warnings and errors.
  - Run unit tests with `cargo test-unit-fast` which uses `nextest` under the hood.
- When writing tests, key assertions should include a custom message stating the expected behavior.
- Use `#[expect(clippy::...)]` instead of `#[allow(clippy::...)]`.
- For internally-tagged enums (`#[serde(tag = "...")]`) without lifetimes, use `TensorZeroDeserialize` instead of `Deserialize` for better error messages via `serde_path_to_error`.

## For APIs

- Use `_` instead of `-` in API routes.
- Prefer using `#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]` for ts-rs exports.
- For any `Option` types visible from the frontend, include `#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]` and `#[serde(skip_serializing_if = "Option::is_none")]` so `None` values are not returned over the wire. In very rare cases we may decide do return `null`s, but in general we want to omit them.
- Some tests make HTTP requests to the gateway; to start the gateway, you can run `cargo run-e2e`. (This gateway has dependencies on some docker containers, and it's appropriate to ask the user to run `docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up`.)
- We use RFC 3339 as the standard format for datetime.

## The responsibility between API handlers and database interfaces

- API handler will be a thin function that handles properties injected by Axum and calls a function to perform business logic.
- Business logic layer will generate all data that TensorZero is responsible for (e.g. UUIDs for new datapoints, `staled_at` timestamps).
- Database layer (ClickHouse and/or Postgres) will insert data as-is into the backing database, with the only exception of `updated_at` timestamps which we insert by calling native functions in the database.

## For Postgres (sqlx)

- **Do not use `format!` for SQL queries.** Use `sqlx::QueryBuilder` for dynamic queries.
  - Use `.push()` for trusted SQL fragments (table names, SQL keywords).
  - Use `.push_bind()` for user-provided values (prevents SQL injection, handles types).
  - Use `.build_query_scalar()` for scalar results, `.build()` for row results.
- **Prefer `sqlx::query!` for static queries** (queries where only values change, not structure). This provides compile-time verification and typed field access (`row.field_name` instead of `row.get("field_name")`).
  - Use `QueryBuilder` only when the query structure is dynamic (e.g., optional WHERE clauses, dynamic table names, conditional JOINs, pagination with optional before/after).
  - For columns that sqlx infers as nullable but are guaranteed non-null by your query logic, use type overrides: `SELECT column as "column!"` to get a non-optional type.
  - For aggregates that should be non-null, use the same pattern: `SELECT COUNT(*)::BIGINT as "total!"`.
- After adding or modifying `sqlx::query!` / `sqlx::query_as!` / `sqlx::query_scalar!` macros, run `cargo sqlx prepare --workspace -- --all-features --all-targets` to regenerate the query cache. This requires a running Postgres database with up-to-date migrations. The generated `.sqlx` directory must be committed to version control.
- Prefer "Postgres" instead of "PostgreSQL" in comments, error messages, docs, etc.

# Python Dependencies

We use `uv` to manage Python dependencies.

# Type generation for TypeScript

We use `ts-rs` and `n-api` for TypeScript-Rust interoperability.

- To generate TypeScript type definitions from Rust types, run `pnpm build-bindings`. Then, rebuild `tensorzero-node` with `pnpm -r build`. The generated type definitions will live in `internal/tensorzero-node/lib/bindings/`.
- To generate implementations for `n-api` functions to be called in TypeScript, and package types in `internal/tensorzero-node` for UI, run `pnpm --filter=tensorzero-node run build`.
- Remember to run `pnpm -r typecheck` to make sure TypeScript and Rust implementations agree on types. Prefer to maintain all types in Rust.

# CI/CD

- Most GitHub Actions workflows run on Unix only, but some also run on Windows and macOS. For workflows that run on multiple operating systems, ensure any bash scripts are compatible with all three platforms. You can check which OS a workflow uses by looking at the `runs-on` field. Setting `shell: bash` in the job definition is often sufficient.

# Misc

- `CONTRIBUTING.md` has additional context on working on this codebase.
- Prefer backticks (`) instead of ticks (') to wrap technical terms in comments, error messages, READMEs, etc.
