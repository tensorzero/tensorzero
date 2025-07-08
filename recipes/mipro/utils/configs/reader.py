from pathlib import Path

import toml

from .config import TensorZeroConfig


def load_config(config_dir: str) -> TensorZeroConfig:
    with open(Path(config_dir) / "tensorzero.toml", "r") as f:
        data = toml.load(f)
    for k, v in data.items():
        if k == "functions":
            for function_name, function_dict in v.items():
                function_dict["config_dir"] = config_dir
                for _variant_name, variant_dict in function_dict["variants"].items():
                    variant_dict["function_name"] = function_name
                    variant_dict["config_dir"] = config_dir
        if k == "tools":
            for tool_name, tool_dict in v.items():
                tool_dict["name"] = tool_name
                tool_dict["config_dir"] = config_dir
    ## TODO: Handle models

    return TensorZeroConfig(**data)
