"""
TensorZero + OpenAI Agents SDK Integration

This module provides seamless integration between TensorZero and the OpenAI Agents SDK,
allowing developers to use TensorZero's production-grade LLM infrastructure with the
intuitive agent abstractions of the Agents SDK.

Usage:
    pip install tensorzero[agents]

    import tensorzero.agents as tz_agents
    await tz_agents.setup_tensorzero_agents("config/tensorzero.toml")

    # Now use normal Agents SDK code with TensorZero features automatically enabled
"""

import toml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

try:
    from agents import Agent, set_default_openai_client
    from openai import AsyncOpenAI

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

from . import patch_openai_client


class TensorZeroAgentsError(Exception):
    """Base exception for TensorZero Agents integration errors."""

    pass


class TensorZeroConfigParser:
    """Parses TensorZero configuration to extract templated functions and tools."""

    def __init__(self, config_path: Union[str, Path]):
        """Initialize with path to tensorzero.toml config file."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise TensorZeroAgentsError(f"TensorZero config not found: {config_path}")

        self.config = toml.load(self.config_path)
        self.templated_functions: Dict[str, Dict[str, Any]] = {}
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._parse_config()

    def _parse_config(self):
        """Parse the configuration to identify templated functions and tools."""
        # Parse functions with templates
        functions = self.config.get("functions", {})
        for func_name, func_config in functions.items():
            variants = func_config.get("variants", {})
            for variant_name, variant_config in variants.items():
                # Check if this variant has a system template
                if "system_template" in variant_config:
                    template_path = variant_config["system_template"]
                    full_path = self.config_path.parent / template_path

                    if full_path.exists():
                        template_content = full_path.read_text()

                        # Detect template variables (simple heuristic: look for {{ }} patterns)
                        import re

                        variables = re.findall(r"\{\{\s*(\w+)\s*\}\}", template_content)

                        self.templated_functions[f"{func_name}::{variant_name}"] = {
                            "function_name": func_name,
                            "variant_name": variant_name,
                            "template_path": str(template_path),
                            "template_content": template_content,
                            "variables": list(set(variables)),
                            "model": variant_config.get("model", ""),
                        }

        # Parse available tools
        tools = self.config.get("tools", {})
        for tool_name, tool_config in tools.items():
            # For file-based tool configs, resolve the path
            if isinstance(tool_config, str) and tool_config.endswith(".json"):
                tool_path = self.config_path.parent / tool_config
                if tool_path.exists():
                    tool_data = json.loads(tool_path.read_text())
                    self.available_tools[tool_name] = tool_data
            elif isinstance(tool_config, dict):
                self.available_tools[tool_name] = tool_config


class TensorZeroMessageConverter:
    """Converts normal OpenAI messages to TensorZero templated format when needed."""

    def __init__(self, config_parser: TensorZeroConfigParser):
        self.config_parser = config_parser

    def should_convert_message(self, model: str) -> bool:
        """Check if this model requires template variable conversion."""
        # Check if the model matches a templated function
        if model.startswith("tensorzero::function_name::"):
            func_variant = model.replace("tensorzero::function_name::", "")
            return func_variant in self.config_parser.templated_functions
        return False

    def convert_messages(
        self, messages: List[Dict[str, Any]], model: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Convert messages to TensorZero template format if needed."""
        if not self.should_convert_message(model):
            return messages

        func_variant = model.replace("tensorzero::function_name::", "")
        template_info = self.config_parser.templated_functions.get(func_variant)

        if not template_info or not template_info["variables"]:
            return messages

        # Extract template variables from kwargs or message content
        template_vars = {}

        # Try to extract variables from kwargs first
        for var in template_info["variables"]:
            if var in kwargs:
                template_vars[var] = kwargs[var]

        # If we have template variables, convert the user message
        if template_vars:
            converted_messages = []
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    # Convert to TensorZero template format
                    converted_msg = {
                        "role": "user",
                        "content": [
                            {"type": "text", "tensorzero::arguments": template_vars}
                        ],
                    }
                    converted_messages.append(converted_msg)
                else:
                    converted_messages.append(msg)
            return converted_messages

        return messages


class TensorZeroAgentsClient:
    """Enhanced OpenAI client that automatically handles TensorZero templating."""

    def __init__(self, config_parser: TensorZeroConfigParser, base_client: AsyncOpenAI):
        self.config_parser = config_parser
        self.message_converter = TensorZeroMessageConverter(config_parser)
        self.base_client = base_client

    async def chat_completions_create(self, **kwargs):
        """Override chat completions to handle template conversion."""
        # Extract model and messages
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])

        # Convert messages if this is a templated function
        if self.message_converter.should_convert_message(model):
            converted_messages = self.message_converter.convert_messages(
                messages, model, **kwargs
            )
            kwargs["messages"] = converted_messages

        # Call the base client
        return await self.base_client.chat.completions.create(**kwargs)


# Global state for the integration
_tensorzero_config: Optional[TensorZeroConfigParser] = None
_original_client: Optional[AsyncOpenAI] = None


async def setup_tensorzero_agents(
    config_path: Union[str, Path],
    base_url: str = "http://localhost:3000",
    api_key: str = "your-api-key",
    **client_kwargs,
) -> TensorZeroConfigParser:
    """
    Set up TensorZero + Agents SDK integration.

    This function:
    1. Parses your tensorzero.toml config to detect templated functions
    2. Patches the OpenAI client to route through TensorZero
    3. Enables automatic template variable detection and conversion
    4. Sets up the default client for the Agents SDK

    Args:
        config_path: Path to your tensorzero.toml configuration file
        base_url: TensorZero gateway URL (default: http://localhost:3000)
        api_key: API key for TensorZero (default: "your-api-key")
        **client_kwargs: Additional arguments for the OpenAI client

    Returns:
        TensorZeroConfigParser: Parsed configuration for inspection

    Raises:
        TensorZeroAgentsError: If agents SDK is not installed or config is invalid
    """
    if not AGENTS_AVAILABLE:
        raise TensorZeroAgentsError(
            "OpenAI Agents SDK not available. Install with: pip install tensorzero[agents]"
        )

    global _tensorzero_config, _original_client

    # Parse TensorZero configuration
    _tensorzero_config = TensorZeroConfigParser(config_path)

    # Create and patch OpenAI client
    client = AsyncOpenAI(
        base_url=f"{base_url}/openai/v1", api_key=api_key, **client_kwargs
    )

    # Apply TensorZero patching
    _original_client = client
    patched_client = patch_openai_client(client)

    # Create enhanced client with template detection
    enhanced_client = TensorZeroAgentsClient(_tensorzero_config, patched_client)

    # Set as default client for Agents SDK
    set_default_openai_client(enhanced_client.base_client)

    print(f"✅ TensorZero Agents integration setup complete!")
    print(f"   📝 Loaded config: {config_path}")
    print(
        f"   🔧 Found {len(_tensorzero_config.templated_functions)} templated functions"
    )
    print(f"   🛠️  Found {len(_tensorzero_config.available_tools)} tools")
    print(f"   🌐 Gateway URL: {base_url}")

    return _tensorzero_config


def get_config() -> Optional[TensorZeroConfigParser]:
    """Get the current TensorZero configuration."""
    return _tensorzero_config


def list_templated_functions() -> Dict[str, Dict[str, Any]]:
    """List all templated functions discovered in the configuration."""
    if _tensorzero_config is None:
        return {}
    return _tensorzero_config.templated_functions


def list_available_tools() -> Dict[str, Dict[str, Any]]:
    """List all tools available in the TensorZero configuration."""
    if _tensorzero_config is None:
        return {}
    return _tensorzero_config.available_tools


def create_agent_from_tensorzero_function(
    function_name: str,
    variant_name: str = "baseline",
    agent_name: Optional[str] = None,
    **agent_kwargs,
) -> "Agent":
    """
    Create an Agent from a TensorZero function configuration.

    Args:
        function_name: Name of the TensorZero function
        variant_name: Variant of the function to use (default: "baseline")
        agent_name: Name for the agent (default: function_name)
        **agent_kwargs: Additional arguments for Agent constructor

    Returns:
        Agent: Configured agent using the TensorZero function

    Raises:
        TensorZeroAgentsError: If function not found or agents not set up
    """
    if not AGENTS_AVAILABLE:
        raise TensorZeroAgentsError("OpenAI Agents SDK not available")

    if _tensorzero_config is None:
        raise TensorZeroAgentsError(
            "TensorZero agents not set up. Call setup_tensorzero_agents() first."
        )

    func_variant = f"{function_name}::{variant_name}"
    if func_variant not in _tensorzero_config.templated_functions:
        raise TensorZeroAgentsError(f"Templated function not found: {func_variant}")

    template_info = _tensorzero_config.templated_functions[func_variant]

    return Agent(
        name=agent_name or function_name,
        model=f"tensorzero::function_name::{func_variant}",
        instructions=template_info["template_content"],
        **agent_kwargs,
    )


# Convenience exports
__all__ = [
    "setup_tensorzero_agents",
    "get_config",
    "list_templated_functions",
    "list_available_tools",
    "create_agent_from_tensorzero_function",
    "TensorZeroAgentsError",
    "TensorZeroConfigParser",
]
