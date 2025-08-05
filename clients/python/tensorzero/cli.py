"""TensorZero type-generation CLI.

This script assists with **type-safe** usage of the TensorZero configuration
system.  It performs two independent generation steps given a `tensorzero.toml`
file:

1.  Collect **function names and variant identifiers** and exposes them as
    strongly-typed symbols (``typing.Literal`` alias *and* an :class:`enum.Enum`
    for IDE auto-completion).
2.  Inspect every ``system_schema`` and ``user_schema`` reference inside the
    configuration, generating corresponding **Pydantic models** via
    ``datamodel-code-generator`` so that prompt arguments can be validated at
    runtime and coerced by static type-checkers.

The command is intentionally lightweight – it has no external runtime
dependencies besides ``toml`` (already required by the core TensorZero Python
client) and, optionally, ``datamodel-code-generator`` when schema generation is
requested.

Example
-------

```bash
$ tensorzero-gen path/to/tensorzero.toml --output ./generated
# ↳ generated/function_identifiers.py
# ↳ generated/schemas/<schema>_model.py
```

The resulting files can be *imported* in downstream projects or used directly
as stub packages for static analysis tools such as **mypy** or **pyright**.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List

import toml

try:
    # ``datamodel-code-generator`` is an *optional* dependency – only import
    # lazily when we actually need it.
    from datamodel_code_generator import (  # type: ignore
        DataModelType,
        InputFileType,
        generate,
    )

    _HAS_DM_CODEGEN = True
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    _HAS_DM_CODEGEN = False


def _python_identifier(raw: str) -> str:
    """Sanitise a string so that it becomes a *valid* Python identifier."""

    ident = re.sub(r"\W|^(?=\d)", "_", raw)
    # Ensure we do not accidentally create an empty identifier.
    return ident or "_"


def _collect_function_identifiers(conf: dict) -> List[str]:
    """Return fully-qualified *function/variant* identifiers from TOML config."""

    functions: dict = conf.get("functions", {})
    identifiers: list[str] = []

    for func_name, func_cfg in functions.items():
        # Baseline variant (implicit).
        identifiers.append(f"tensorzero::function_name::{func_name}")

        for variant_name in func_cfg.get("variants", {}).keys():
            identifiers.append(
                f"tensorzero::function_name::{func_name}::variant_name::{variant_name}"
            )

    return identifiers


def _render_identifier_module(identifiers: Iterable[str]) -> str:
    """Generate *Python source* containing Literal + Enum definitions."""

    identifiers = list(sorted(set(identifiers)))

    # Literal alias.
    literal_block = "\n".join(f'    "{ident}",' for ident in identifiers)
    literal_src = f"""from typing import Literal

FunctionIdentifier = Literal[
{literal_block}
]
"""

    # Enum – create *stable* member names (function name + optional variant).
    enum_members: list[str] = []
    for ident in identifiers:
        parts = ident.split("::")
        # Format: tensorzero::function_name::<FUNC>[::variant_name::<VARIANT>]
        func_name_raw = parts[2] if len(parts) >= 3 else "function"
        func_name = _python_identifier(func_name_raw.upper())
        if len(parts) >= 5 and parts[3] == "variant_name":
            variant_raw = parts[4]
            variant_name = _python_identifier(variant_raw.upper())
            enum_member_name = f"{func_name}_{variant_name}"
        else:
            enum_member_name = func_name
        enum_members.append(f'    {enum_member_name} = "{ident}"')

    enum_src = "\n".join(
        [
            "from enum import Enum",
            "",
            "class FunctionIdentifierEnum(str, Enum):",
            *enum_members,
            "",
        ]
    )

    return "\n".join([literal_src, enum_src])


def _generate_schema_models(
    *,
    schema_paths: Iterable[Path],
    output_dir: Path,
    base_dir: Path,
    target_python_version: str = "3.11",
) -> None:
    """Invoke *datamodel-code-generator* programmatically for each schema.

    The generated model modules mirror the directory structure of the original
    JSON schema files relative to *base_dir*.  For example, given
    ``prompts/build/system_schema.json`` the resulting module will be written to::

        <output_dir>/prompts/build/system_schema_model.py

    All intermediate directories are created as proper Python packages
    (i.e. with ``__init__.py`` files) so that the generated models can be
    imported via dotted paths.
    """

    if not _HAS_DM_CODEGEN:
        missing = (
            "datamodel-code-generator>=0.30 is required for schema generation; "
            "install it with: pip install 'datamodel-code-generator[http]'"
        )
        raise RuntimeError(missing)

    from tempfile import TemporaryDirectory

    for schema_path in schema_paths:
        if not schema_path.exists():
            print(f"[WARN] schema file not found: {schema_path}", file=sys.stderr)
            continue

        # Determine the destination directory relative to ``base_dir``.  If the
        # schema lives outside *base_dir* we simply place it at the root of
        # *output_dir*.
        try:
            rel_path = schema_path.relative_to(base_dir)
        except ValueError:
            rel_path = Path(schema_path.name)

        rel_dir_parts = rel_path.parts[:-1]  # exclude file name
        dest_dir_parts = [_python_identifier(part) for part in rel_dir_parts]
        dest_dir = (
            output_dir.joinpath(*dest_dir_parts) if dest_dir_parts else output_dir
        )

        # Ensure destination directory exists and is a Python package.
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Touch __init__.py for each package level so modules are importable.
        pkg_path = output_dir
        (pkg_path / "__init__.py").touch(exist_ok=True)
        for part in dest_dir_parts:
            pkg_path = pkg_path / part
            (pkg_path / "__init__.py").touch(exist_ok=True)

        module_name = _python_identifier(schema_path.stem + "_model")
        output_file = dest_dir / f"{module_name}.py"

        # Read once to avoid repeated IO inside the generator when given str.
        schema_str = schema_path.read_text(encoding="utf-8")

        # ``generate`` writes directly to the provided Path; we use a temporary
        # file first so that we can *always* overwrite (generate() will error if
        # multiple modules are produced but a *file* Path is supplied).
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "model.py"
            generate(
                schema_str,
                input_file_type=InputFileType.JsonSchema,
                input_filename=schema_path.name,
                output=tmp_path,
                output_model_type=DataModelType.PydanticV2BaseModel,
                formatters=[],  # skip black/isort to avoid python-version issues
            )
            # Write to final destination.
            output_file.write_text(
                tmp_path.read_text(encoding="utf-8"), encoding="utf-8"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tensorzero-gen",
        description="Generate typed helpers from a tensorzero.toml configuration.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to tensorzero.toml file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./generated"),
        help="Directory to write generated modules (default: ./generated).",
    )
    parser.add_argument(
        "--skip-schemas",
        action="store_true",
        help="Only generate function identifier types – skip schema models.",
    )
    args = parser.parse_args(argv)

    cfg_path: Path = args.config.expanduser().resolve()
    output_dir: Path = args.output.expanduser().resolve()

    if not cfg_path.exists():
        parser.error(f"configuration file not found: {cfg_path}")

    cfg_dict = toml.load(cfg_path)

    # ---------------------------------------------------------------------
    # 1. Function identifier types
    # ---------------------------------------------------------------------
    identifiers = _collect_function_identifiers(cfg_dict)
    identifier_module_src = _render_identifier_module(identifiers)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "__init__.py").touch(exist_ok=True)
    types_file = output_dir / "function_identifiers.py"
    types_file.write_text(identifier_module_src, encoding="utf-8")
    print(f"[OK] wrote function identifier types → {types_file}")

    # ---------------------------------------------------------------------
    # 2. Schema → Pydantic models
    # ---------------------------------------------------------------------
    if not args.skip_schemas:
        schema_rel_paths: set[str] = set()
        for func_cfg in cfg_dict.get("functions", {}).values():
            if "system_schema" in func_cfg:
                schema_rel_paths.add(func_cfg["system_schema"])
            if "user_schema" in func_cfg:
                schema_rel_paths.add(func_cfg["user_schema"])
        # Variants may override?  (currently unlikely – but be safe.)
        for func_cfg in cfg_dict.get("functions", {}).values():
            for variant_cfg in func_cfg.get("variants", {}).values():
                if "system_schema" in variant_cfg:
                    schema_rel_paths.add(variant_cfg["system_schema"])
                if "user_schema" in variant_cfg:
                    schema_rel_paths.add(variant_cfg["user_schema"])

        resolved_paths = [cfg_path.parent / Path(p) for p in schema_rel_paths]
        if resolved_paths:
            schema_out_dir = output_dir / "schemas"
            _generate_schema_models(
                schema_paths=resolved_paths,
                output_dir=schema_out_dir,
                base_dir=cfg_path.parent,
            )
            print(
                f"[OK] generated {len(resolved_paths)} pydantic model module(s) → {schema_out_dir}"
            )
        else:
            print("[INFO] no schema files declared – skipping model generation")

    print("[DONE] Generation complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
