"""Generate TensorZero configuration from an llmgym environment.


We build the TensorZeroConfig object directly (rather than going through
TensorZeroAgent) because the agent's __init__ starts an embedded gateway
that requires postgres_url, but the Path-3 code path in llmgym doesn't
forward it.  Writing config files doesn't need a running gateway.
"""

import logging
from pathlib import Path
from typing import Optional

import llmgym
import toml
from llmgym.agents.tensorzero.configs import (
    ChatCompletionConfig,
    TensorZeroConfig,
    TensorZeroFunctionConfigChat,
    TensorZeroFunctionConfigJson,
    TensorZeroFunctionConfigs,
    VariantConfigs,
)
from llmgym.types import FunctionConfigChat, MetricConfigs

logger = logging.getLogger(__name__)


def _format_function_name(name: str) -> str:
    """Sanitise a function name for TensorZero (replace hyphens with underscores)."""
    return name.replace("-", "_")


def parse_env_config(
    env_name: str,
    task_split: Optional[str] = None,
    env_config_extra: Optional[dict] = None,
) -> tuple[str, dict]:
    """Parse environment name and extract configuration for verifiers_v0 environments.

    Args:
        env_name: Name of the llmgym environment, potentially with verifiers_v0 format
                 (e.g., "verifiers_v0::tau2-bench" or "verifiers_v0::tau2-bench::math")
        task_split: Optional dataset split (e.g., "train", "test") to pass to the
                    environment constructor.
        env_config_extra: Optional dict of extra params to pass to llmgym.make().

    Returns:
        Tuple of (parsed_env_name, env_config).
    """
    env_config: dict = {}

    # Merge extra config first so env-specific logic can override
    if env_config_extra:
        env_config.update(env_config_extra)

    if env_name.startswith("verifiers_v0"):
        parts = env_name.split("::")
        if len(parts) < 2:
            raise ValueError(
                f"{env_name} is not a properly formatted verifiers environment"
            )
        if len(parts) == 2:
            env_name, verifiers_env_id = parts
            env_config["env_id"] = verifiers_env_id
        elif len(parts) == 3:
            if "tau2-bench" not in parts[1]:
                raise ValueError(
                    f"domain specification is only supported for tau2-bench, got {parts[1]}"
                )
            env_name, verifiers_env_id, domain = parts
            env_config["env_id"] = verifiers_env_id
            env_config["verifiers_env_args"] = {"domain": domain}

    if task_split is not None:
        if env_name == "harbor_v0":
            # Harbor uses "split" instead of "task_split"
            env_config["split"] = task_split
        else:
            env_config["task_split"] = task_split

    return env_name, env_config


def write_initial_env_config(
    env_name: str,
    output_path: str = ".",
    model_name: str = "openai::gpt-4o-mini",
    function_description: Optional[str] = None,
    env_config_extra: Optional[dict] = None,
) -> Path:
    """Write TensorZero configuration for a given llmgym environment.

    Args:
        env_name: Name of the llmgym environment to configure.
        output_path: Directory where the config folder will be created.
        model_name: The model to use for inference.
        function_description: Optional description to add to each function.
        env_config_extra: Optional dict of extra params for llmgym.make().

    Returns:
        Path to the created config directory.
    """
    parsed_env_name, env_config = parse_env_config(
        env_name, env_config_extra=env_config_extra
    )

    env = llmgym.make(parsed_env_name, config=env_config)

    # Build TensorZeroFunctionConfigs from the llmgym function configs,
    # mirroring the logic in TensorZeroAgent.__init__ (Path 3).
    functions = TensorZeroFunctionConfigs()
    variant_name = "initial"

    for function_name, function_config in env.functions.items():
        # Prefix with env_name:: to match TensorZeroAgent.format_function_name()
        fn_name = f"{parsed_env_name}::{_format_function_name(function_name)}"
        variants = VariantConfigs()
        variants[variant_name] = ChatCompletionConfig(
            name=variant_name,
            function_name=fn_name,
            model=model_name,
            system_template=function_config.example_system_template,
            user_template=function_config.example_user_template,
            assistant_template=function_config.example_assistant_template,
        )

        if isinstance(function_config, FunctionConfigChat):
            chat_kwargs: dict = dict(
                name=fn_name,
                variants=variants,
                system_schema=function_config.system_schema,
                user_schema=function_config.user_schema,
                assistant_schema=function_config.assistant_schema,
            )
            if function_config.tools_available:
                chat_kwargs["tools"] = function_config.tools_available
            functions[fn_name] = TensorZeroFunctionConfigChat(**chat_kwargs)
        else:
            functions[fn_name] = TensorZeroFunctionConfigJson(
                name=fn_name,
                variants=variants,
                system_schema=function_config.system_schema,
                user_schema=function_config.user_schema,
                assistant_schema=function_config.assistant_schema,
                output_schema=function_config.output_schema,
            )

    tz_config = TensorZeroConfig(
        functions=functions,
        metrics=env.metrics if env.metrics is not None else MetricConfigs(),
        tools=env.tools,
    )

    config_dir = tz_config.write(output_path)

    # Post-process the generated TOML
    toml_path = config_dir / "tensorzero.toml"
    config_data = toml.load(toml_path)

    # Enable Postgres observability backend
    config_data.setdefault("gateway", {}).setdefault("observability", {})["backend"] = "postgres"

    # Insert function description if provided
    if function_description:
        for fn_name in config_data.get("functions", {}):
            config_data["functions"][fn_name]["description"] = function_description

    with open(toml_path, "w") as f:
        toml.dump(config_data, f)
    logger.info("Post-processed config at %s", toml_path)

    logger.info("Configuration written to: %s", config_dir)
    return config_dir
