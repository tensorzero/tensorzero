# Postgres Query Guidelines

## Row deserialization

Use `sqlx::FromRow` (derived or manually implemented) for deserializing query results into structs. Use `build_query_as::<T>()` with `QueryBuilder` or `sqlx::query_as!` for static queries to leverage typed deserialization.

Avoid manual `row.get("column")` extraction when `FromRow` can be used instead.

## Static vs dynamic queries

- Use `sqlx::query!` / `sqlx::query_as!` / `sqlx::query_scalar!` for **static queries** where only bind values change but the query structure is fixed. This provides compile-time verification.
- Use `QueryBuilder` only when the **query structure is dynamic** (e.g., dynamic table names, conditional WHERE clauses, conditional JOINs).
