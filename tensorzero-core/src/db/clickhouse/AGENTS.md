# ClickHouse UUID Ordering

ClickHouse does not preserve chronological ordering for UUIDv7 values when you sort/order by a `UUID` column directly.

For tables where a UUIDv7 value is part of the ordering key, store it as `UInt128` (for example, `run_id_uint`) instead of `UUID`.

When reading/writing:

- Convert `UUID` -> `UInt128` on write with `toUInt128({id:UUID})`.
- Convert `UInt128` -> `UUID` on read with `uint_to_uuid(id_uint)`.
- Order by the `UInt128` column directly.
