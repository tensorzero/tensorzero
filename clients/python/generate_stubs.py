#!/usr/bin/env python3
"""Generate type stubs for tensorzero.tensorzero module.

This script inspects the runtime module to generate accurate type stubs.
Run: uv run python generate_stubs.py > tensorzero/tensorzero.pyi
"""

import inspect
import tensorzero.tensorzero as tz

# Type mappings for known properties and parameters
TYPE_HINTS = {
    # String types
    "function_name": "str",
    "dataset_name": "str",
    "raw": "str",
    "metric_name": "str",
    "gateway_url": "str",
    "config_file": "str",
    "clickhouse_url": "str",
    "postgres_url": "str",
    "variant_name": "str",
    "model_name": "str",
    "evaluation_name": "str",
    "message": "str",
    "name": "str",
    "role": "str",
    "content": "str",
    "type": "str",
    "system_template": "str",
    "user_template": "str",
    "assistant_template": "str",
    "model": "str",
    "account_id": "str",
    "base_url": "str",
    "inference_cache": "str",
    "order_by": "str",
    "tool_choice": "str",
    "api_key": "str",
    "timestamp": "str",  # ISO timestamp
    "estimated_finish": "str",  # ISO timestamp
    "system": "str",
    "embedding_model": "str",
    # Boolean types
    "verbose_errors": "bool",
    "async_setup": "bool",
    "is_custom": "bool",
    "dryrun": "bool",
    "internal": "bool",
    "stream": "bool",
    "parallel_tool_calls": "bool",
    "include_original_response": "bool",
    "append_to_existing_variants": "bool",
    # Integer types
    "limit": "int",
    "offset": "int",
    "page_size": "int",
    "concurrency": "int",
    "num_deleted_datapoints": "int",
    "k": "int",
    "batch_size": "int",
    "max_concurrency": "int",
    "dimensions": "int",
    # Float types
    "timeout": "float",
    "value": "float",
    # UUID types
    "id": "UUID",
    "inference_id": "UUID",
    "episode_id": "UUID",
    "datapoint_id": "UUID",
    "run_id": "UUID",
    # List types
    "ids": "List[UUID]",
    "datapoints": "List[Dict[str, Any]]",
    "messages": "List[Dict[str, Any]]",
    "variants": "List[str]",
    "allowed_tools": "List[str]",
    "additional_tools": "List[Dict[str, Any]]",
    "provider_tools": "List[Dict[str, Any]]",
    "dispreferred_outputs": "List[Dict[str, Any]]",
    # Dict types
    "input": "Dict[str, Any]",
    "output": "Dict[str, Any]",
    "params": "Dict[str, Any]",
    "credentials": "Dict[str, Any]",
    "metadata": "Dict[str, Any]",
    "filters": "Dict[str, Any]",
    "tags": "Dict[str, str]",
    "cache_options": "Dict[str, Any]",
    "extra_body": "Dict[str, Any]",
    "extra_headers": "Dict[str, str]",
    "output_schema": "Dict[str, Any]",
    "system_schema": "Dict[str, Any]",
    "user_schema": "Dict[str, Any]",
    "assistant_schema": "Dict[str, Any]",
    "filter": "Dict[str, Any]",
    "tool_params": "Dict[str, Any]",
    "dynamic_tool_params": "Dict[str, Any]",
    "otlp_traces_extra_headers": "Dict[str, str]",
    "internal_dynamic_variant_config": "Dict[str, Any]",
    "run_info": "Dict[str, Any]",
    "stored_input": "Dict[str, Any]",
    "stored_output": "Dict[str, Any]",
    # Special types
    "status": "OptimizationJobStatus",
    "job_handle": "OptimizationJobHandle",
    "optimization_config": "Union[OpenAISFTConfig, OpenAIRFTConfig, FireworksSFTConfig, TogetherSFTConfig, GCPVertexGeminiSFTConfig, DICLConfig, DICLOptimizationConfig]",
    "output_source": "str",  # String literal for output source
    "requests": "List[Union[UpdateChatDatapointRequest, UpdateJsonDatapointRequest]]",
}

# Property return type overrides - maps (class_name, property_name) -> return_type
# Use this for properties that return specific config classes instead of Dict[str, Any]
PROPERTY_RETURN_TYPES = {
    ("Config", "functions"): "FunctionsConfig",
    ("FunctionConfigChat", "variants"): "VariantsConfig",
    ("FunctionConfigJson", "variants"): "VariantsConfig",
}


def get_type_hint(name: str, has_default: bool = False) -> str:
    """Get type hint for a parameter or property name."""
    hint = TYPE_HINTS.get(name, "Any")

    # Special handling for optional parameters
    if has_default and hint != "Any" and not hint.startswith("Optional[") and not hint.startswith("Union["):
        hint = f"Optional[{hint}]"

    return hint


def format_param(param_name: str, param: inspect.Parameter) -> str:
    """Format a parameter for stub file."""
    has_default = param.default != inspect.Parameter.empty
    type_hint = get_type_hint(param_name, has_default)
    param_str = f"{param_name}: {type_hint}"

    if has_default:
        if param.default is None:
            param_str += " = None"
        elif isinstance(param.default, bool):
            param_str += f" = {param.default}"
        elif isinstance(param.default, str):
            param_str += f" = {param.default!r}"
        elif isinstance(param.default, (int, float)):
            param_str += f" = {param.default}"
        else:
            param_str += " = ..."

    return param_str


def get_init_signature(cls) -> str:
    """Get __init__ signature for a class."""
    try:
        sig = inspect.signature(cls.__init__)
        params = []
        keyword_only_started = False

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            if param.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_only_started:
                params.append("*")
                keyword_only_started = True

            params.append(format_param(param_name, param))

        param_str = ", ".join(params)
        return f"    def __init__(self, {param_str}) -> None: ..."
    except (ValueError, TypeError):
        return "    def __init__(self, *args: Any, **kwargs: Any) -> None: ..."


def get_properties(cls) -> list:
    """Get property names from a class."""
    properties = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(cls, name)
            attr_static = inspect.getattr_static(cls, name)
            # Check if it's a getset_descriptor (PyO3 property)
            if type(attr_static).__name__ == "getset_descriptor" and not callable(attr):
                properties.append(name)
        except AttributeError:
            pass
    return sorted(properties)


def get_methods(cls) -> list:
    """Get method names from a class."""
    methods = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(cls, name)
            attr_static = inspect.getattr_static(cls, name)
            # Check if it's a method_descriptor (PyO3 method)
            if type(attr_static).__name__ == "method_descriptor" and callable(attr):
                methods.append(name)
        except AttributeError:
            pass
    return sorted(methods)


def is_enum_variant(cls, name: str) -> bool:
    """Check if an attribute is an enum variant (type constructor or instance)."""
    try:
        attr = getattr(cls, name)

        # Pattern 1: Enum variant as instance (e.g., OptimizationJobStatus.Pending)
        # These are not callable and are instances of the enum class
        if not callable(attr) and type(attr) == cls:
            return True

        # Pattern 2: Tagged enum variant as constructor (e.g., Datapoint.Chat)
        # Check if it's a type (enum variant constructor)
        if type(attr).__name__ == "type":
            return True

        # Pattern 3: Check signature - enum variants have single _0 parameter
        if callable(attr):
            sig = inspect.signature(attr)
            params = list(sig.parameters.keys())
            return len(params) == 1 and params[0].startswith("_")

        return False
    except (ValueError, TypeError, AttributeError):
        return False


def get_enum_variants(cls) -> list:
    """Get enum variant names from a PyO3 enum class."""
    variants = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if is_enum_variant(cls, name):
            variants.append(name)
    return variants


def format_property(name: str, class_name: str = None) -> str:
    """Format a property with proper type hint."""
    # Check for class-specific property return type override
    if class_name and (class_name, name) in PROPERTY_RETURN_TYPES:
        type_hint = PROPERTY_RETURN_TYPES[(class_name, name)]
    else:
        type_hint = get_type_hint(name)
    return f"    @property\n    def {name}(self) -> {type_hint}: ..."


# Get runtime __all__
runtime_all = tz.__all__ if hasattr(tz, "__all__") else []

# Print header
print("# Type stubs for tensorzero.tensorzero")
print("# Auto-generated - DO NOT EDIT MANUALLY")
print("# Run: uv run python generate_stubs.py > tensorzero/tensorzero.pyi")
print()
print("from typing import Any, Coroutine, Dict, List, Optional, Type, Union")
print("from uuid import UUID")
print("from typing_extensions import final")
print()
print(f"__all__ = {runtime_all!r}")
print()

# ============================================================================
# Dataset V1 API Types
# ============================================================================
print("# ============================================================================")
print("# Dataset V1 API Types")
print("# ============================================================================")
print()

# JsonDatapointOutputUpdate
print("@final")
print("class JsonDatapointOutputUpdate:")
print('    """Update for JSON datapoint output."""')
print("    def __init__(self, raw: str, *args: Any) -> None: ...")
print("    @property")
print("    def raw(self) -> str: ...")
print()

# DatapointMetadataUpdate
print("@final")
print("class DatapointMetadataUpdate:")
print('    """Update for datapoint metadata."""')
print("    def __init__(self, *args: Any, name: Optional[str], **kwargs: Any) -> None: ...")
print("    @property")
print("    def name(self) -> Optional[str]: ...")
print()

# CreateChatDatapointRequest
print("@final")
print("class CreateChatDatapointRequest:")
print('    """Request to create a chat datapoint."""')
print("    def __init__(")
print("        self,")
print("        function_name: str,")
print("        input: Dict[str, Any],")
print("        episode_id: Optional[UUID] = None,")
print("        output: Optional[Dict[str, Any]] = None,")
print("        dynamic_tool_params: Optional[Dict[str, Any]] = None,")
print("        tags: Optional[Dict[str, str]] = None,")
print("        name: Optional[str] = None,")
print("        *args: Any,")
print("        **kwargs: Any,")
print("    ) -> None: ...")
for prop in ["function_name", "episode_id", "input", "output", "dynamic_tool_params", "tags", "name"]:
    print(format_property(prop))
print()

# CreateJsonDatapointRequest
print("@final")
print("class CreateJsonDatapointRequest:")
print('    """Request to create a JSON datapoint."""')
print("    def __init__(")
print("        self,")
print("        function_name: str,")
print("        input: Dict[str, Any],")
print("        episode_id: Optional[UUID] = None,")
print("        output: Optional[JsonDatapointOutputUpdate] = None,")
print("        output_schema: Optional[Dict[str, Any]] = None,")
print("        tags: Optional[Dict[str, str]] = None,")
print("        name: Optional[str] = None,")
print("        *args: Any,")
print("        **kwargs: Any,")
print("    ) -> None: ...")
for prop in ["function_name", "episode_id", "input", "output", "output_schema", "tags", "name"]:
    print(format_property(prop))
print()

# UpdateChatDatapointRequest
print("@final")
print("class UpdateChatDatapointRequest:")
print('    """Request to update a chat datapoint."""')
print("    def __init__(self, *args: Any, **kwargs: Any) -> None: ...")
for prop in ["id", "input", "output", "tool_params", "tags", "metadata"]:
    print(format_property(prop))
print()

# UpdateJsonDatapointRequest
print("@final")
print("class UpdateJsonDatapointRequest:")
print('    """Request to update a JSON datapoint."""')
print("    def __init__(self, *args: Any, **kwargs: Any) -> None: ...")
for prop in ["id", "input", "output", "output_schema", "tags", "metadata"]:
    print(format_property(prop))
print()

# UpdateDatapointMetadataRequest
print("@final")
print("class UpdateDatapointMetadataRequest:")
print('    """Request to update datapoint metadata."""')
print("    def __init__(self, id: UUID, metadata: DatapointMetadataUpdate, *args: Any, **kwargs: Any) -> None: ...")
print("    @property")
print("    def id(self) -> UUID: ...")
print("    @property")
print("    def metadata(self) -> DatapointMetadataUpdate: ...")
print()

# Wrapper request types
for cls_name, prop_name, prop_type in [
    ("CreateDatapointsRequest", "datapoints", "List[Union[CreateChatDatapointRequest, CreateJsonDatapointRequest]]"),
    ("UpdateDatapointsRequest", "datapoints", "List[Union[UpdateChatDatapointRequest, UpdateJsonDatapointRequest]]"),
    ("UpdateDatapointsMetadataRequest", "datapoints", "List[UpdateDatapointMetadataRequest]"),
    ("GetDatapointsRequest", "ids", "List[UUID]"),
    ("DeleteDatapointsRequest", "ids", "List[UUID]"),
]:
    print("@final")
    print(f"class {cls_name}:")
    print(f'    """{cls_name}"""')
    print(f"    def __init__(self, {prop_name}: {prop_type}, *args: Any, **kwargs: Any) -> None: ...")
    print(f"    @property")
    print(f"    def {prop_name}(self) -> {prop_type}: ...")
    print()

# Response types
print("@final")
print("class CreateDatapointsResponse:")
print('    """Response from creating datapoints."""')
print("    @property")
print("    def ids(self) -> List[UUID]: ...")
print()

print("@final")
print("class UpdateDatapointsResponse:")
print('    """Response from updating datapoints."""')
print("    @property")
print("    def ids(self) -> List[UUID]: ...")
print()

print("@final")
print("class GetDatapointsResponse:")
print('    """Response containing retrieved datapoints."""')
print("    @property")
print('    def datapoints(self) -> List["Datapoint"]: ...')
print()

print("@final")
print("class DeleteDatapointsResponse:")
print('    """Response from deleting datapoints."""')
print("    @property")
print("    def num_deleted_datapoints(self) -> int: ...")
print()

# ListDatapointsRequest
print("@final")
print("class ListDatapointsRequest:")
print('    """Request to list datapoints."""')
print("    def __init__(")
print("        self,")
print("        function_name: Optional[str] = None,")
print("        limit: Optional[int] = None,")
print("        page_size: Optional[int] = None,")
print("        offset: Optional[int] = None,")
print("        filter: Optional[Dict[str, Any]] = None,")
print("        *args: Any,")
print("        **kwargs: Any,")
print("    ) -> None: ...")
for prop in ["function_name", "limit", "page_size", "offset", "filter"]:
    print(format_property(prop))
print()

# CreateDatapointsFromInferenceRequest
print("@final")
print("class CreateDatapointsFromInferenceRequest:")
print('    """Request to create datapoints from inferences."""')
print("    def __init__(")
print("        self,")
print("        params: Dict[str, Any],")
print('        output_source: Optional["CreateDatapointsFromInferenceOutputSource"] = None,')
print("        *args: Any,")
print("        **kwargs: Any,")
print("    ) -> None: ...")
print("    @property")
print("    def params(self) -> Dict[str, Any]: ...")
print("    @property")
print('    def output_source(self) -> Optional["CreateDatapointsFromInferenceOutputSource"]: ...')
print()

# CreateDatapointsFromInferenceOutputSource - enum
cls = tz.CreateDatapointsFromInferenceOutputSource
variants = get_enum_variants(cls)
print("@final")
print("class CreateDatapointsFromInferenceOutputSource:")
print('    """Output source for creating datapoints from inferences."""')
for variant in variants:
    print(f'    {variant}: Type["CreateDatapointsFromInferenceOutputSource"]')
print("    def __init__(self, *args: Any) -> None: ...")
print()

# Datapoint - tagged enum with properties
cls = tz.Datapoint
variants = get_enum_variants(cls)
properties = get_properties(cls)
print("@final")
print("class Datapoint:")
print('    """A datapoint - tagged enum with Chat and Json variants."""')
for variant in variants:
    print(f'    {variant}: Type["Datapoint"]')
for prop in properties:
    print(format_property(prop))
print()

# ============================================================================
# Configuration Types
# ============================================================================
print("# ============================================================================")
print("# Configuration Types")
print("# ============================================================================")
print()

for cls_name in [
    "Config",
    "FunctionsConfig",
    "FunctionConfigChat",
    "FunctionConfigJson",
    "VariantsConfig",
    "ChatCompletionConfig",
    "BestOfNSamplingConfig",
    "MixtureOfNConfig",
    "ChainOfThoughtConfig",
]:
    cls = getattr(tz, cls_name)
    properties = get_properties(cls)
    print("@final")
    print(f"class {cls_name}:")
    print(f'    """{cls_name}"""')
    if properties:
        for prop in properties:
            print(format_property(prop, cls_name))
    else:
        print("    ...")
    print()

# ============================================================================
# Gateway Types
# ============================================================================
print("# ============================================================================")
print("# Gateway Types")
print("# ============================================================================")
print()

print("class BaseTensorZeroGateway:")
print('    """Base class for TensorZero gateways."""')
print("    def experimental_get_config(self) -> Config: ...")
print()

# TensorZeroGateway
gateway_cls = tz.TensorZeroGateway
print("@final")
print("class TensorZeroGateway(BaseTensorZeroGateway):")
print('    """A synchronous client for a TensorZero gateway."""')
print('    def __enter__(self) -> "TensorZeroGateway": ...')
print("    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...")
print("    @classmethod")
print(
    '    def build_embedded(cls, *, config_file: Optional[str] = None, clickhouse_url: Optional[str] = None, postgres_url: Optional[str] = None, timeout: Optional[float] = None) -> "TensorZeroGateway": ...'
)
print("    @classmethod")
print(
    '    def build_http(cls, *, gateway_url: str, timeout: Optional[float] = None, verbose_errors: bool = False, api_key: Optional[str] = None) -> "TensorZeroGateway": ...'
)

# Generate method stubs for TensorZeroGateway
for method in sorted(
    [name for name in dir(gateway_cls) if not name.startswith("_") and name not in ["build_embedded", "build_http"]]
):
    attr = getattr(gateway_cls, method)
    if not callable(attr):
        continue
    try:
        sig = inspect.signature(attr)
        params = []
        kw_started = False
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            if p.kind == inspect.Parameter.KEYWORD_ONLY and not kw_started:
                params.append("*")
                kw_started = True
            params.append(format_param(pname, p))
        param_str = ", ".join(params)

        # Determine return type based on method name
        if "datapoints" in method or "datapoint" in method:
            if method.startswith("create"):
                return_type = "CreateDatapointsResponse"
            elif method.startswith("update"):
                return_type = "UpdateDatapointsResponse"
            elif method.startswith("get"):
                return_type = "Union[Datapoint, GetDatapointsResponse]"
            elif method.startswith("delete"):
                return_type = "DeleteDatapointsResponse"
            elif method == "list_datapoints":
                return_type = "List[Datapoint]"
            else:
                return_type = "Any"
        elif "optimization" in method:
            if "launch" in method:
                return_type = "OptimizationJobHandle"
            elif "poll" in method:
                return_type = "OptimizationJobInfo"
            else:
                return_type = "Any"
        elif "evaluation" in method:
            if "run" in method and "episode" not in method:
                return_type = "EvaluationJobHandler"
            else:
                return_type = "Any"
        elif method == "inference":
            return_type = "Dict[str, Any]"
        else:
            return_type = "Any"

        print(f"    def {method}(self, {param_str}) -> {return_type}: ...")
    except:
        print(f"    def {method}(self, *args: Any, **kwargs: Any) -> Any: ...")
print()

# AsyncTensorZeroGateway
gateway_cls = tz.AsyncTensorZeroGateway
print("@final")
print("class AsyncTensorZeroGateway(BaseTensorZeroGateway):")
print('    """An async client for a TensorZero gateway."""')
print('    async def __aenter__(self) -> "AsyncTensorZeroGateway": ...')
print("    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...")
print("    @classmethod")
print(
    '    def build_embedded(cls, *, config_file: Optional[str] = None, clickhouse_url: Optional[str] = None, postgres_url: Optional[str] = None, timeout: Optional[float] = None, async_setup: bool = True) -> Coroutine[Any, Any, "AsyncTensorZeroGateway"]: ...'
)
print("    @classmethod")
print(
    '    def build_http(cls, *, gateway_url: str, timeout: Optional[float] = None, verbose_errors: bool = False, async_setup: bool = True, api_key: Optional[str] = None) -> Coroutine[Any, Any, "AsyncTensorZeroGateway"]: ...'
)

# Generate method stubs for AsyncTensorZeroGateway
non_async_methods = ["close", "experimental_get_config"]
for method in sorted(
    [name for name in dir(gateway_cls) if not name.startswith("_") and name not in ["build_embedded", "build_http"]]
):
    attr = getattr(gateway_cls, method)
    if not callable(attr):
        continue

    is_async = method not in non_async_methods

    try:
        sig = inspect.signature(attr)
        params = []
        kw_started = False
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            if p.kind == inspect.Parameter.KEYWORD_ONLY and not kw_started:
                params.append("*")
                kw_started = True
            params.append(format_param(pname, p))
        param_str = ", ".join(params)

        # Determine base return type
        if "datapoints" in method or "datapoint" in method:
            if method.startswith("create"):
                base_return = "CreateDatapointsResponse"
            elif method.startswith("update"):
                base_return = "UpdateDatapointsResponse"
            elif method.startswith("get"):
                base_return = "Union[Datapoint, GetDatapointsResponse]"
            elif method.startswith("delete"):
                base_return = "DeleteDatapointsResponse"
            elif method == "list_datapoints":
                base_return = "List[Datapoint]"
            else:
                base_return = "Any"
        elif "optimization" in method:
            if "launch" in method:
                base_return = "OptimizationJobHandle"
            elif "poll" in method:
                base_return = "OptimizationJobInfo"
            else:
                base_return = "Any"
        elif "evaluation" in method:
            if "run" in method and "episode" not in method:
                base_return = "AsyncEvaluationJobHandler"
            else:
                base_return = "Any"
        elif method == "inference":
            base_return = "Dict[str, Any]"
        elif method == "experimental_get_config":
            base_return = "Config"
        else:
            base_return = "Any"

        return_type = f"Coroutine[Any, Any, {base_return}]" if is_async else base_return
        print(f"    def {method}(self, {param_str}) -> {return_type}: ...")
    except:
        return_type = "Coroutine[Any, Any, Any]" if is_async else "Any"
        print(f"    def {method}(self, *args: Any, **kwargs: Any) -> {return_type}: ...")
print()

# LocalHttpGateway
cls = tz.LocalHttpGateway
properties = get_properties(cls)
print("@final")
print("class LocalHttpGateway(BaseTensorZeroGateway):")
print('    """LocalHttpGateway"""')
for prop in properties:
    print(format_property(prop))
print("    def close(self) -> None: ...")
print()

# ============================================================================
# Inference Types
# ============================================================================
print("# ============================================================================")
print("# Inference Types")
print("# ============================================================================")
print()

# StoredInference - tagged enum with properties
cls = tz.StoredInference
variants = get_enum_variants(cls)
properties = get_properties(cls)
print("@final")
print("class StoredInference:")
print('    """StoredInference - tagged enum."""')
for variant in variants:
    print(f'    {variant}: Type["StoredInference"]')
for prop in properties:
    print(format_property(prop))
print()

# RenderedSample
cls = tz.RenderedSample
properties = get_properties(cls)
print("@final")
print("class RenderedSample:")
print('    """RenderedSample"""')
for prop in properties:
    print(format_property(prop))
print()

# ResolvedInput
cls = tz.ResolvedInput
properties = get_properties(cls)
print("@final")
print("class ResolvedInput:")
print('    """ResolvedInput"""')
for prop in properties:
    print(format_property(prop))
print()

# ResolvedInputMessage
cls = tz.ResolvedInputMessage
properties = get_properties(cls)
print("@final")
print("class ResolvedInputMessage:")
print('    """ResolvedInputMessage"""')
for prop in properties:
    print(format_property(prop))
print()

# ============================================================================
# Optimization Types
# ============================================================================
print("# ============================================================================")
print("# Optimization Types")
print("# ============================================================================")
print()

# OptimizationJobHandle - tagged enum
cls = tz.OptimizationJobHandle
variants = get_enum_variants(cls)
print("@final")
print("class OptimizationJobHandle:")
print('    """OptimizationJobHandle - tagged enum."""')
for variant in variants:
    print(f'    {variant}: Type["OptimizationJobHandle"]')
print()

# OptimizationJobInfo
cls = tz.OptimizationJobInfo
properties = get_properties(cls)
print("@final")
print("class OptimizationJobInfo:")
print('    """OptimizationJobInfo"""')
for prop in properties:
    print(format_property(prop))
print()

# OptimizationJobStatus - enum
cls = tz.OptimizationJobStatus
variants = get_enum_variants(cls)
print("@final")
print("class OptimizationJobStatus:")
print('    """OptimizationJobStatus - enum."""')
for variant in variants:
    print(f'    {variant}: Type["OptimizationJobStatus"]')
print()

# ============================================================================
# Evaluation Types
# ============================================================================
print("# ============================================================================")
print("# Evaluation Types")
print("# ============================================================================")
print()

for cls_name in ["EvaluationJobHandler", "AsyncEvaluationJobHandler"]:
    cls = getattr(tz, cls_name)
    properties = get_properties(cls)
    methods = get_methods(cls)

    print("@final")
    print(f"class {cls_name}:")
    print(f'    """{cls_name}"""')
    for prop in properties:
        print(format_property(prop))
    for method in methods:
        print(f"    def {method}(self) -> Dict[str, Any]: ...")
    print()

# ============================================================================
# Optimization Config Types
# ============================================================================
print("# ============================================================================")
print("# Optimization Config Types")
print("# ============================================================================")
print()

for cls_name in [
    "OpenAISFTConfig",
    "OpenAIRFTConfig",
    "FireworksSFTConfig",
    "TogetherSFTConfig",
    "GCPVertexGeminiSFTConfig",
    "DICLConfig",
    "DICLOptimizationConfig",
]:
    cls = getattr(tz, cls_name)
    print("@final")
    print(f"class {cls_name}:")
    print(f'    """{cls_name}"""')
    print(get_init_signature(cls))
    if hasattr(cls, "__deprecated__"):
        print(f"    __deprecated__: str")
    print()

# ============================================================================
# Internal Functions
# ============================================================================
print("# ============================================================================")
print("# Internal Functions")
print("# ============================================================================")
print()
print(
    "def _start_http_gateway(*, config_file: str, clickhouse_url: str, postgres_url: str, async_setup: bool) -> Any: ..."
)
