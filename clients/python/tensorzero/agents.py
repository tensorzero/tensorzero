from . import patch_openai_client, TensorZeroGateway, Config as TensorZeroConfig

from typing import Any, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager
import contextvars
import toml
from pathlib import Path


"""
High level approach:
1. Patch the OpenAI client to use the TensorZero gateway
    - This is easy
    - Validate by patching and running normal agents SDK code
    - This routes requests through the gateway, but tracing in clickhouse won't work
2. Define agents as functions in the config file
    - We could also parse tools from the config file but not sure what the value add is
        - TBD, this might be necessary for tracing to work
    - This works by just simply passing the tensorzero:function_name::variant_name format to the model parameter in agents
    - However, we don't get episode_id for all the tool calls, so we can't trace them in clickhouse
3. To get system template prompts:
    - We need to parse the config file -> we can get this from the gateway
    - We then need to intercept the request and add the system template to the request

"""


try:
    import agents
    from openai import AsyncOpenAI
    from openai._compat import cached_property
    from openai.resources.chat import AsyncChat
    from openai.resources.chat.completions import AsyncCompletions

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


# Global context variables for tracking state within the context manager
_tensorzero_context: contextvars.ContextVar[Optional["TensorZeroAgentsContext"]] = (
    contextvars.ContextVar("tensorzero_context", default=None)
)


class TensorZeroAgentsError(Exception):
    pass


class TensorZeroAgentsContext(BaseModel):
    """Context object that holds TensorZero state during the patched session."""

    original_client: Optional[AsyncOpenAI] = None
    patched_client: Optional[AsyncOpenAI] = None
    gateway_url: Optional[str] = None
    episode_id: Optional[str] = None
    config: Optional[TensorZeroConfig] = None
    function_variant_to_system_template_and_schema: dict[str, tuple[str, str]] = {}

    class Config:
        arbitrary_types_allowed = True


class TensorZeroAgentsCompletions(AsyncCompletions):
    def __init__(self, client: AsyncOpenAI):
        super().__init__(client)

    async def create(self, **kwargs):
        """Override completions to handle template conversion."""
        tz_context = _tensorzero_context.get()
        assert tz_context is not None, (
            "TensorZeroAgentsCompletions must be called within a with_tensorzero_agents_patched context"
        )

        if tz_context.episode_id:
            # Inject episode_id or other context as needed
            extra_body = kwargs.get("extra_body", {})
            extra_body["tensorzero::episode_id"] = tz_context.episode_id
            kwargs["extra_body"] = extra_body

        # Inject system template if it exists
        if (
            kwargs.get("model")
            in tz_context.function_variant_to_system_template_and_schema
        ):
            system_template, system_schema = (
                tz_context.function_variant_to_system_template_and_schema[
                    kwargs["model"]
                ]
            )
            if system_template:
                messages = kwargs.get("messages", [])
                arguments = {}
                if system_schema:
                    metadata: dict[str, Any] = kwargs.get("metadata", {})
                    system_schema_properties: dict[str, Any] = system_schema.get(
                        "properties", {}
                    )
                    for (
                        property_name,
                        property_schema,
                    ) in system_schema_properties.items():
                        provided_value = metadata.get(
                            f"tensorzero::arguments::{property_name}"
                        )
                        arguments[property_name] = provided_value
                    content_object = [
                        {"type": "text", "tensorzero::arguments": arguments}
                    ]
                else:
                    content_object = system_template
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = content_object
                else:
                    messages.insert(0, {"role": "system", "content": content_object})
                kwargs["messages"] = messages

        # Call the base client
        ret = await super().create(**kwargs)
        model_extra = ret.model_extra
        if model_extra and model_extra.get("episode_id"):
            tz_context.episode_id = model_extra["episode_id"]

        return ret


class TensorZeroAgentsChat(AsyncChat):
    @cached_property
    def completions(self) -> TensorZeroAgentsCompletions:
        return TensorZeroAgentsCompletions(self._client)


class TensorZeroAgentsClient(AsyncOpenAI):
    def __init__(
        self,
        base_url: Optional[str] = None,
    ):
        super().__init__(base_url=base_url)

    @cached_property
    def chat(self) -> TensorZeroAgentsChat:
        return TensorZeroAgentsChat(self)


@asynccontextmanager
async def with_tensorzero_agents_patched(
    config_path: str,
    *,
    clickhouse_url: Optional[str] = None,
    gateway_url: Optional[str] = None,
    episode_id: Optional[str] = None,
):
    """
    Context manager for TensorZero integration with OpenAI Agents SDK.

    This context manager:
    1. Configures the Agents SDK to use Chat Completions API (TensorZero compatible)
    2. Sets up OpenAI client patching for TensorZero gateway
    3. Parses TensorZero config for template detection
    4. Patches agents classes for context management
    5. Properly cleans up all patches on exit

    Args:
        config_path: Path to tensorzero.toml config file
        clickhouse_url: Optional ClickHouse URL override
        gateway_url: Optional gateway URL (if None, uses embedded gateway)
        episode_id: Optional episode ID for tracing

    Usage:
        async with with_tensorzero_agents_patched("config/tensorzero.toml") as tz_context:
            # Use normal Agents SDK code with TensorZero features automatically enabled
            agent = Agent(model="tensorzero::function_name::my_function")
            await agent.run("Hello!")
    """
    if not AGENTS_AVAILABLE:
        raise TensorZeroAgentsError(
            "OpenAI Agents SDK not available. Install with: pip install tensorzero[agents]"
        )

    # Configure Agents SDK to use Chat Completions API instead of Responses API
    # This is required because TensorZero only implements /openai/v1/chat/completions
    agents.set_default_openai_api("chat_completions")

    # Create context object
    tz_context = TensorZeroAgentsContext(gateway_url=gateway_url, episode_id=episode_id)

    # Set up context variables
    context_token = _tensorzero_context.set(tz_context)

    # TODO: Factor this out to helper function/class
    # Maybe hacky approach: Instead of parsing the config file, we build a throwaway embedded gateway and leverage the existing config parsing
    tmp_gateway = TensorZeroGateway.build_embedded(
        config_file=config_path, clickhouse_url=clickhouse_url
    )
    config = tmp_gateway.experimental_get_config()
    tz_context.config = config

    # functions isn't iterable right now, so we still need to parse the toml just to get the keys
    # TODO: This could be an API improvement, but seems out of scope for this PR
    toml_parsed_config = toml.load(Path(config_path))
    toml_parsed_functions: dict[str, dict] = toml_parsed_config.get("functions", {})
    embedded_functions = config.functions

    function_variant_to_system_template_and_schema: dict[str, tuple[str, str]] = {}

    for function_name, toml_parsed_function in toml_parsed_functions.items():
        embedded_function_config = embedded_functions[function_name]
        system_schema = embedded_function_config.system_schema
        embedded_variants = embedded_function_config.variants
        toml_parsed_variants: dict[str, Any] = toml_parsed_function.get("variants", {})
        for variant_name, toml_parsed_variant in toml_parsed_variants.items():
            embedded_variant_config = embedded_variants[variant_name]
            system_template = embedded_variant_config.system_template
            function_variant_to_system_template_and_schema[
                f"tensorzero::{function_name}::{variant_name}"
            ] = (system_template, system_schema)
            if variant_name == "baseline":
                function_variant_to_system_template_and_schema[
                    f"tensorzero::{function_name}"
                ] = (system_template, system_schema)

    tz_context.function_variant_to_system_template_and_schema = (
        function_variant_to_system_template_and_schema
    )
    try:
        # Set up OpenAI client patching
        if gateway_url:
            # Use external HTTP gateway
            original_client = TensorZeroAgentsClient(
                base_url=f"{gateway_url}/openai/v1"
            )
            patched_client = await patch_openai_client(
                original_client, config_file=config_path, clickhouse_url=clickhouse_url
            )
            tz_context.original_client = original_client
            tz_context.patched_client = patched_client
            agents.set_default_openai_client(patched_client)
        else:
            # Use embedded gateway with patching
            original_client = TensorZeroAgentsClient()
            patched_client = await patch_openai_client(
                original_client,
                config_file=config_path,
                clickhouse_url=clickhouse_url,
                async_setup=True,
            )
            tz_context.original_client = original_client
            tz_context.patched_client = patched_client
            agents.set_default_openai_client(patched_client)
        yield tz_context
    except Exception as e:
        raise e

    finally:
        _tensorzero_context.reset(context_token)

        if tz_context.original_client:
            agents.set_default_openai_client(tz_context.original_client)


# Convenience exports
__all__ = [
    "with_tensorzero_agents_patched",
    "TensorZeroAgentsError",
    "TensorZeroAgentsContext",
]
