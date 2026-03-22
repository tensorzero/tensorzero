"""Postgres COPY-based export/import via asyncpg.

Used by the snapshot system to persist and restore Postgres state
(all tables as binary COPY files) across eval runs.
"""

import logging
import re
from datetime import date, timedelta
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

EXTENSION = ".pgcopy"

# Schemas to export — public (eval bookkeeping) + tensorzero (inference/feedback data)
EXPORTABLE_SCHEMAS = ("public", "tensorzero")

# Tables that shouldn't be imported (gateway recreates them on startup).
# Importing these would overwrite the running gateway's runtime state
# (e.g. config snapshot hash, deployment ID) and break config writes.
SKIP_TABLES = {
    ("public", "_sqlx_migrations"),
    ("public", "__durable_sqlx_migrations"),
    ("public", "tensorzero_auth__sqlx_migrations"),
    ("tensorzero", "config_snapshots"),
    ("tensorzero", "deployment_id"),
}


async def list_tables(postgres_url: str) -> list[tuple[str, str]]:
    """List all exportable tables in the Postgres database.

    Returns (schema_name, table_name) tuples. Excludes partitioned parent
    tables (relkind = 'p') since COPY doesn't work on them — their leaf
    partitions (relkind = 'r') are included instead.

    Args:
        postgres_url: Full Postgres connection URL.

    Returns:
        Sorted list of (schema_name, table_name) tuples.
    """
    conn = await asyncpg.connect(postgres_url)
    try:
        rows = await conn.fetch(
            """
            SELECT n.nspname AS schema_name, c.relname AS table_name
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = ANY($1)
              AND c.relkind = 'r'
            ORDER BY n.nspname, c.relname
            """,
            list(EXPORTABLE_SCHEMAS),
        )
    finally:
        await conn.close()

    return [(row["schema_name"], row["table_name"]) for row in rows]


def _file_stem(schema: str, table: str) -> str:
    """Encode schema.table as a filename stem."""
    return f"{schema}.{table}"


def _parse_file_stem(stem: str) -> tuple[str, str]:
    """Parse a filename stem into (schema, table).

    Handles both new format (schema.table) and old format (table only,
    assumed to be public schema for backward compatibility).
    """
    parts = stem.split(".", 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    # Old-format file without schema prefix — assume public
    return ("public", stem)


async def export_table(
    postgres_url: str,
    schema_name: str,
    table_name: str,
    output_path: Path,
) -> Path:
    """Export a single Postgres table in binary COPY format.

    Args:
        postgres_url: Full Postgres connection URL.
        schema_name: Schema containing the table.
        table_name: Name of the table to export.
        output_path: Path to write the file.

    Returns:
        The output path written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await asyncpg.connect(postgres_url)
    try:
        result = await conn.copy_from_table(
            table_name,
            output=str(output_path),
            format="binary",
            schema_name=schema_name,
        )
    finally:
        await conn.close()

    size = output_path.stat().st_size
    logger.debug(
        "Exported %s.%s → %s (%d bytes, %s)",
        schema_name,
        table_name,
        output_path,
        size,
        result,
    )
    return output_path


async def import_table(
    postgres_url: str,
    schema_name: str,
    table_name: str,
    data_path: Path,
) -> None:
    """Import a binary COPY file into a Postgres table.

    The table must already exist (created by gateway migrations).

    Args:
        postgres_url: Full Postgres connection URL.
        schema_name: Schema containing the table.
        table_name: Name of the table to import into.
        data_path: Path to the binary COPY file to import.
    """
    size = data_path.stat().st_size
    if size == 0:
        logger.debug("Skipping empty file for %s.%s", schema_name, table_name)
        return

    conn = await asyncpg.connect(postgres_url)
    try:
        result = await conn.copy_to_table(
            table_name,
            source=str(data_path),
            format="binary",
            schema_name=schema_name,
        )
    finally:
        await conn.close()

    logger.debug(
        "Imported %s → %s.%s (%d bytes, %s)",
        data_path,
        schema_name,
        table_name,
        size,
        result,
    )


async def export_all_tables(
    postgres_url: str,
    output_dir: Path,
) -> list[Path]:
    """Export all tables in a Postgres database as binary COPY files.

    Args:
        postgres_url: Full Postgres connection URL.
        output_dir: Directory to write files into.

    Returns:
        List of paths to the exported files.
    """
    tables = await list_tables(postgres_url)
    logger.info("Exporting %d Postgres tables to %s", len(tables), output_dir)

    paths: list[Path] = []
    for schema, table in tables:
        path = output_dir / f"{_file_stem(schema, table)}{EXTENSION}"
        await export_table(postgres_url, schema, table, path)
        paths.append(path)

    return paths


_DAILY_PARTITION_RE = re.compile(r"^(.+)_(\d{4})_(\d{2})_(\d{2})$")
_MONTHLY_PARTITION_RE = re.compile(r"^(.+)_(\d{4})_(\d{2})$")


async def _create_missing_partitions(
    postgres_url: str,
    missing_tables: list[tuple[str, str]],
) -> set[tuple[str, str]]:
    """Create date-partition tables that the snapshot needs but the gateway didn't create.

    The gateway only creates partitions for the current week (daily) or current
    4 months (monthly). Snapshots taken on a different day may reference
    partitions outside that window. This function parses the date from the
    table name and creates the partition attached to its parent.

    Returns the set of (schema, table) pairs that were successfully created.
    """
    if not missing_tables:
        return set()

    created: set[tuple[str, str]] = set()
    conn = await asyncpg.connect(postgres_url)
    try:
        for schema, table in missing_tables:
            daily = _DAILY_PARTITION_RE.match(table)
            monthly = _MONTHLY_PARTITION_RE.match(table) if not daily else None

            if daily:
                parent_name = daily.group(1)
                y, m, d = int(daily.group(2)), int(daily.group(3)), int(daily.group(4))
                start = date(y, m, d)
                end = start + timedelta(days=1)
            elif monthly:
                parent_name = monthly.group(1)
                y, m = int(monthly.group(2)), int(monthly.group(3))
                start = date(y, m, 1)
                end = date(y + (1 if m == 12 else 0), (m % 12) + 1, 1)
            else:
                continue

            # Verify the parent is a partitioned table
            is_partitioned = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = $1 AND c.relname = $2 AND c.relkind = 'p'
                )
                """,
                schema,
                parent_name,
            )
            if not is_partitioned:
                continue

            await conn.execute(
                f'CREATE TABLE "{schema}"."{table}" '
                f'PARTITION OF "{schema}"."{parent_name}" '
                f"FOR VALUES FROM ('{start}') TO ('{end}')"
            )
            created.add((schema, table))
            logger.debug("Created missing partition %s.%s", schema, table)
    finally:
        await conn.close()

    if created:
        logger.info("Created %d missing date partitions", len(created))
    return created


async def import_all_tables(
    postgres_url: str,
    data_dir: Path,
) -> None:
    """Import all snapshot data files from a directory into Postgres.

    Filenames encode the target schema and table: ``{schema}.{table}.pgcopy``.
    Old-format files without a schema prefix (``{table}.pgcopy``) are assumed
    to belong to the ``public`` schema for backward compatibility.

    Missing date-partition tables (from snapshots taken on a different day)
    are created automatically. Existing tables are truncated before import
    to avoid conflicts with rows inserted by gateway migrations.

    Args:
        postgres_url: Full Postgres connection URL.
        data_dir: Directory containing .pgcopy files.
    """
    pgcopy_files = sorted(data_dir.glob(f"*{EXTENSION}"))

    if not pgcopy_files:
        native_files = sorted(data_dir.glob("*.native"))
        if native_files:
            logger.warning(
                "Found .native (ClickHouse) files but no %s files in %s — "
                "this snapshot was created before the Postgres migration",
                EXTENSION,
                data_dir,
            )
        else:
            logger.warning("No snapshot data files found in %s", data_dir)
        return

    # Parse schema.table from each filename, skip migration tables
    import_items: list[tuple[str, str, Path]] = []
    for f in pgcopy_files:
        schema, table = _parse_file_stem(f.stem)
        if (schema, table) in SKIP_TABLES:
            continue
        import_items.append((schema, table, f))

    # Discover which tables exist in the target DB
    existing_tables = set(await list_tables(postgres_url))

    # Create any missing date partitions the snapshot needs
    missing = [(s, t) for s, t, _ in import_items if (s, t) not in existing_tables]
    created = await _create_missing_partitions(postgres_url, missing)
    existing_tables |= created

    # Filter to importable tables, skip any that still don't exist
    valid_items: list[tuple[str, str, Path]] = []
    skipped = 0
    for schema, table, df in import_items:
        if (schema, table) not in existing_tables:
            logger.debug(
                "Skipping %s.%s — table does not exist in target DB", schema, table
            )
            skipped += 1
            continue
        valid_items.append((schema, table, df))

    if skipped:
        logger.info("Skipped %d snapshot files for non-existent tables", skipped)

    logger.info("Importing %d pgcopy files from %s", len(valid_items), data_dir)

    # Truncate all target tables before importing to avoid conflicts with
    # rows inserted by gateway migrations (e.g. resource_bucket singleton,
    # deployment_id).
    if valid_items:
        conn = await asyncpg.connect(postgres_url)
        try:
            for schema, table, _df in valid_items:
                await conn.execute(f'TRUNCATE "{schema}"."{table}" CASCADE')
            logger.debug("Truncated %d tables before import", len(valid_items))
        finally:
            await conn.close()

    for schema, table, df in valid_items:
        await import_table(postgres_url, schema, table, df)
