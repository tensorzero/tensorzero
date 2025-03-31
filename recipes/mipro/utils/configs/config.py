import json
import tempfile
from pathlib import Path
from typing import Optional, TypeVar

import toml
from pydantic import BaseModel

from .functions import FunctionConfigs, FunctionConfigType
from .gateway import GatewayConfig
from .metrics import MetricConfigs
from .tools import ToolConfigs

T = TypeVar("T", bound=BaseModel)


class TensorZeroConfig(BaseModel):
    """
    Configuration for TensorZero.
    """

    functions: FunctionConfigs
    metrics: Optional[MetricConfigs] = None
    tools: Optional[ToolConfigs] = None
    gateway: Optional[GatewayConfig] = None

    def write(self, base_dir: Optional[Path] = None) -> Path:
        if base_dir is None:
            base_temp_dir: Path = Path(tempfile.mkdtemp(prefix="mipro_"))
        else:
            base_temp_dir: Path = Path(base_dir)
            base_temp_dir.mkdir(exist_ok=True, parents=True)

        # The top-level config folder
        config_dir: Path = base_temp_dir / "config"
        config_dir.mkdir(exist_ok=True)

        # 1. Create the `functions` subdirectory and populate
        functions_dir = config_dir / "functions"
        functions_dir.mkdir(exist_ok=True)
        self.functions.write(functions_dir)

        # 2. Create the `tools` subdirectory and populate
        if self.tools:
            tools_dir = config_dir / "tools"
            tools_dir.mkdir(exist_ok=True)
            self.write_tools(tools_dir)

        # 3. Create the `tensorzero.toml` file
        config_dict = self.model_dump()
        for function_name, function_config in config_dict["functions"].items():
            del function_config["config_dir"]
            for variant_name, variant_config in function_config["variants"].items():
                del variant_config["config_dir"]
                if function_config["type"] == FunctionConfigType.CHAT:
                    del config_dict["functions"][function_name]["variants"][
                        variant_name
                    ]["json_mode"]
        if self.tools:
            for _tool_name, tool_config in config_dict["tools"].items():
                del tool_config["config_dir"]
        toml_file = toml.dumps(config_dict)
        with (config_dir / "tensorzero.toml").open("w", encoding="utf-8") as f:
            f.write(toml_file)

        return config_dir

    def write_tools(self, tools_dir: Path):
        if self.tools:
            for tool_name, tool_config in self.tools:
                tool_path = tools_dir / f"{tool_name}.json"
                schema_dict = {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "description": tool_config.description,
                    "properties": tool_config.parameters["properties"],
                    "required": tool_config.parameters["required"],
                    "additionalProperties": tool_config.parameters[
                        "additionalProperties"
                    ],
                }
                # Write out the tool JSON schema
                with tool_path.open("w", encoding="utf-8") as f:
                    json.dump(schema_dict, f, indent=2)
