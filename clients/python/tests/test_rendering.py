from uuid import UUID

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    FileBase64,
    JsonInferenceOutput,
    StoredInference,
    TensorZeroGateway,
    Text,
    Thought,
    Tool,
    ToolCall,
    ToolParams,
    ToolResult,
    UnknownContentBlock,
)
from tensorzero.util import uuid7


def test_sync_render_inferences_success(embedded_sync_client: TensorZeroGateway):
    rendered_inferences = embedded_sync_client.experimental_render_inferences(
        stored_inferences=[
            StoredInference(
                type="chat",
                function_name="basic_test",
                variant_name="default",
                input={
                    "system": {"assistant_name": "foo"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "thought", "text": "hmmm"},
                                {"type": "text", "value": "bar"},
                                {
                                    "type": "tool_call",
                                    "id": "123",
                                    "arguments": {"foo": "bar"},
                                    "name": "test_tool",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "value": "Hello world"},
                                {
                                    "type": "tool_result",
                                    "id": "123",
                                    "name": "test_tool",
                                    "result": "test",
                                },
                                {"type": "unknown", "data": [{"woo": "hoo"}]},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": {"mime_type": "image/png"},
                                    "storage_path": {
                                        "kind": {
                                            "type": "s3_compatible",
                                            "bucket_name": "tensorzero-e2e-test-images",
                                            "region": "us-east-1",
                                            "prefix": "",
                                        },
                                        "path": "observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
                                    },
                                }
                            ],
                        },
                    ],
                },
                output=[Text(text="Hello world")],
                episode_id=uuid7(),
                inference_id=uuid7(),
                tool_params=ToolParams(
                    tools_available=[
                        Tool(
                            name="test",
                            description="test",
                            parameters={"foo": "bar"},
                            strict=False,
                        )
                    ],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
                output_schema=None,
            ),
            StoredInference(
                type="json",
                function_name="json_success",
                variant_name="dummy",
                input={
                    "system": {"assistant_name": "Dr. Mehta"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "value": {"country": "Japan"}}
                            ],
                        },
                    ],
                },
                output=JsonInferenceOutput(
                    parsed={"answer": "Tokyo"}, raw='{"answer": "Tokyo"}'
                ),
                episode_id=uuid7(),
                inference_id=uuid7(),
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
                tool_params=None,
            ),
        ],
        variants={"basic_test": "test", "json_success": "test"},
    )
    assert len(rendered_inferences) == 2
    chat_inference = rendered_inferences[0]

    assert chat_inference.function_name == "basic_test"
    assert chat_inference.variant_name == "default"
    input = chat_inference.input
    # Test that templating actually happens here.
    assert input.system == "You are a helpful and friendly assistant named foo"
    messages = input.messages
    assert len(messages) == 3
    message = messages[0]
    assert message.role == "user"
    content = message.content
    assert len(content) == 3
    assert isinstance(content[0], Thought)
    assert content[0].type == "thought"
    assert content[0].text == "hmmm"
    assert isinstance(content[1], Text)
    assert content[1].type == "text"
    assert content[1].text == "bar"
    assert isinstance(content[2], ToolCall)
    assert content[2].type == "tool_call"
    assert content[2].id == "123"
    assert content[2].arguments == '{"foo":"bar"}'
    assert content[2].name == "test_tool"
    message = messages[1]
    assert message.role == "assistant"
    content = message.content
    assert len(content) == 3
    assert isinstance(content[0], Text)
    assert content[0].type == "text"
    assert content[0].text == "Hello world"
    assert isinstance(content[1], ToolResult)
    assert content[1].type == "tool_result"
    assert content[1].id == "123"
    assert content[1].name == "test_tool"
    assert content[1].result == "test"
    assert isinstance(content[2], UnknownContentBlock)
    assert content[2].type == "unknown"
    assert content[2].data == [{"woo": "hoo"}]
    output = rendered_inferences[0].output

    message = messages[2]
    assert message.role == "user"
    content = message.content
    assert len(content) == 1
    assert isinstance(content[0], FileBase64)
    assert content[0].type == "file"
    assert content[0].mime_type == "image/png"
    assert len(content[0].data) > 1000

    assert isinstance(output, list)
    assert len(output) == 1
    assert isinstance(output[0], Text)
    assert output[0].type == "text"
    assert output[0].text == "Hello world"
    assert isinstance(rendered_inferences[0].episode_id, UUID)
    assert isinstance(rendered_inferences[0].inference_id, UUID)
    tool_params = rendered_inferences[0].tool_params
    assert tool_params is not None
    tools_available = tool_params.tools_available
    assert len(tools_available) == 1
    tool = tools_available[0]
    assert tool.name == "test"
    assert tool.description == "test"
    assert tool.parameters == {"foo": "bar"}
    assert not tool.strict
    # Not implemented yet
    # TODO: test this
    # assert tool_params.tool_choice == "auto"
    assert not tool_params.parallel_tool_calls
    assert rendered_inferences[0].output_schema is None

    json_inference = rendered_inferences[1]
    assert json_inference.function_name == "json_success"
    assert json_inference.variant_name == "dummy"
    input = json_inference.input
    # templating happens here
    assert (
        input.system
        == """You are a helpful and friendly assistant named Dr. Mehta.

Please answer the questions in a JSON with key "answer".

Do not include any other text than the JSON object. Do not include "```json" or "```" or anything else.

Example Response:

{
    "answer": "42"
}"""
    )
    messages = input.messages
    assert len(messages) == 1
    message = messages[0]
    assert message.role == "user"
    content = message.content
    assert len(content) == 1
    assert isinstance(content[0], Text)
    assert content[0].type == "text"
    # templating happens here
    assert content[0].text == "What is the name of the capital city of Japan?"
    output = json_inference.output
    assert json_inference.output_schema == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    assert isinstance(json_inference.episode_id, UUID)
    assert isinstance(json_inference.inference_id, UUID)
    assert json_inference.tool_params is None
    assert json_inference.output_schema == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }


def test_sync_render_inferences_nonexistent_function(
    embedded_sync_client: TensorZeroGateway,
):
    """Test that render_inferences drops if the function does not exist at all."""
    rendered_inferences = embedded_sync_client.experimental_render_inferences(
        stored_inferences=[
            StoredInference(
                type="chat",
                function_name="non_existent_function",
                variant_name="default",
                input={
                    "system": {"assistant_name": "foo"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "value": "bar"}],
                        }
                    ],
                },
                output=[Text(text="Hello world")],
                episode_id=uuid7(),
                inference_id=uuid7(),
                tool_params=ToolParams(
                    tools_available=[],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
                output_schema=None,
            )
        ],
        variants={},
    )
    # TODO: test that the warning message is logged (we do this in Rust)
    assert len(rendered_inferences) == 0


def test_sync_render_inferences_unspecified_function(
    embedded_sync_client: TensorZeroGateway,
):
    """Test that render_inferences drops if the function is not specified in the variants map."""
    rendered_inferences = embedded_sync_client.experimental_render_inferences(
        stored_inferences=[
            StoredInference(
                type="chat",
                function_name="non_existent_function",
                variant_name="default",
                input={
                    "system": {"assistant_name": "foo"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "value": "bar"}],
                        }
                    ],
                },
                output=[Text(text="Hello world")],
                episode_id=uuid7(),
                inference_id=uuid7(),
                tool_params=ToolParams(
                    tools_available=[],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
                output_schema=None,
            )
        ],
        variants={},
    )
    assert len(rendered_inferences) == 0
    # TODO: test that the warning message is logged (we do this in Rust)


def test_sync_render_inferences_no_variant(embedded_sync_client: TensorZeroGateway):
    """Test that render_inferences drops an example if the variant is not found and logs a warning."""
    with pytest.raises(Exception) as excinfo:
        embedded_sync_client.experimental_render_inferences(
            stored_inferences=[
                StoredInference(
                    type="chat",
                    function_name="basic_test",  # This function exists in the config
                    variant_name="non_existent_variant",
                    input={
                        "system": {"assistant_name": "foo"},
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "value": "bar"}],
                            }
                        ],
                    },
                    output=[Text(text="Hello world")],
                    episode_id=uuid7(),
                    inference_id=uuid7(),
                    tool_params=ToolParams(
                        tools_available=[],
                        tool_choice="auto",
                        parallel_tool_calls=False,
                    ),
                    output_schema=None,
                )
            ],
            variants={"basic_test": "non_existent_variant"},
        )
    assert "Variant non_existent_variant for function basic_test not found" in str(
        excinfo.value
    )


def test_sync_render_inferences_missing_variable(
    embedded_sync_client: TensorZeroGateway,
):
    """Test that render_inferences drops an example if a template variable is missing."""
    rendered_inferences = embedded_sync_client.experimental_render_inferences(
        stored_inferences=[
            StoredInference(
                type="chat",
                function_name="basic_test",  # Uses assistant_name in system prompt
                variant_name="default",
                input={
                    "system": {"some_other_variable": "foo"},  # Missing assistant_name
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "value": "bar"}],
                        }
                    ],
                },
                output=[Text(text="Hello world")],
                episode_id=uuid7(),
                inference_id=uuid7(),
                tool_params=ToolParams(
                    tools_available=[],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
                output_schema=None,
            )
        ],
        variants={"basic_test": "test"},
    )
    assert len(rendered_inferences) == 0
    # TODO: test that the warning message is logged (we do this in Rust)


@pytest.mark.asyncio
async def test_async_render_inferences_success(
    embedded_async_client: AsyncTensorZeroGateway,
):
    rendered_inferences = await embedded_async_client.experimental_render_inferences(
        stored_inferences=[
            StoredInference(
                type="chat",
                function_name="basic_test",
                variant_name="default",
                input={
                    "system": {"assistant_name": "foo"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "thought", "text": "hmmm"},
                                {"type": "text", "value": "bar"},
                                {
                                    "type": "tool_call",
                                    "id": "123",
                                    "arguments": {"foo": "bar"},
                                    "name": "test_tool",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "value": "Hello world"},
                                {
                                    "type": "tool_result",
                                    "id": "123",
                                    "name": "test_tool",
                                    "result": "test",
                                },
                                {"type": "unknown", "data": [{"woo": "hoo"}]},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": {
                                        "mime_type": "image/png",
                                    },
                                    "storage_path": {
                                        "kind": {
                                            "type": "s3_compatible",
                                            "bucket_name": "tensorzero-e2e-test-images",
                                            "region": "us-east-1",
                                            "prefix": "",
                                        },
                                        "path": "observability/images/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png",
                                    },
                                }
                            ],
                        },
                    ],
                },
                output=[Text(text="Hello world")],
                episode_id=uuid7(),
                inference_id=uuid7(),
                tool_params=ToolParams(
                    tools_available=[
                        Tool(
                            name="test",
                            description="test",
                            parameters={"foo": "bar"},
                            strict=False,
                        )
                    ],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
                output_schema=None,
            ),
            StoredInference(
                type="json",
                function_name="json_success",
                variant_name="dummy",
                input={
                    "system": {"assistant_name": "Dr. Mehta"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "value": {"country": "Japan"}}
                            ],
                        }
                    ],
                },
                output=JsonInferenceOutput(
                    parsed={"answer": "Tokyo"},
                    raw="""{"answer": "Tokyo"}""",
                ),
                episode_id=uuid7(),
                inference_id=uuid7(),
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
                tool_params=None,
            ),
        ],
        variants={"basic_test": "test", "json_success": "test"},
    )
    assert len(rendered_inferences) == 2
    chat_inference = rendered_inferences[0]

    assert chat_inference.function_name == "basic_test"
    assert chat_inference.variant_name == "default"
    input = chat_inference.input
    # Test that templating actually happens here.
    assert input.system == "You are a helpful and friendly assistant named foo"
    messages = input.messages
    assert len(messages) == 3
    message = messages[0]
    assert message.role == "user"
    content = message.content
    assert len(content) == 3
    assert isinstance(content[0], Thought)
    assert content[0].type == "thought"
    assert content[0].text == "hmmm"
    assert isinstance(content[1], Text)
    assert content[1].type == "text"
    assert content[1].text == "bar"
    assert isinstance(content[2], ToolCall)
    assert content[2].type == "tool_call"
    assert content[2].id == "123"
    assert content[2].arguments == """{"foo":"bar"}"""
    assert content[2].name == "test_tool"
    message = messages[1]
    assert message.role == "assistant"
    content = message.content
    assert len(content) == 3
    assert isinstance(content[0], Text)
    assert content[0].type == "text"
    assert content[0].text == "Hello world"
    assert isinstance(content[1], ToolResult)
    assert content[1].type == "tool_result"
    assert content[1].id == "123"
    assert content[1].name == "test_tool"
    assert content[1].result == "test"
    assert isinstance(content[2], UnknownContentBlock)
    assert content[2].type == "unknown"
    assert content[2].data == [{"woo": "hoo"}]
    output = rendered_inferences[0].output

    message = messages[2]
    assert message.role == "user"
    content = message.content
    assert len(content) == 1
    assert isinstance(content[0], FileBase64)
    assert content[0].type == "file"
    assert content[0].mime_type == "image/png"
    assert len(content[0].data) > 1000

    assert isinstance(output, list)
    assert len(output) == 1
    assert isinstance(output[0], Text)
    assert output[0].type == "text"
    assert isinstance(output[0], Text)
    assert output[0].text == "Hello world"
    assert isinstance(rendered_inferences[0].episode_id, UUID)
    assert isinstance(rendered_inferences[0].inference_id, UUID)
    tool_params = rendered_inferences[0].tool_params
    assert tool_params is not None
    tools_available = tool_params.tools_available
    assert len(tools_available) == 1
    tool = tools_available[0]
    assert tool.name == "test"
    assert tool.description == "test"
    assert tool.parameters == {"foo": "bar"}
    assert not tool.strict
    # Not implemented yet
    # TODO: test this
    # assert tool_params.tool_choice == "auto"
    assert not tool_params.parallel_tool_calls
    assert rendered_inferences[0].output_schema is None

    json_inference = rendered_inferences[1]
    assert json_inference.function_name == "json_success"
    assert json_inference.variant_name == "dummy"
    input = json_inference.input
    # templating happens here
    assert (
        input.system
        == """You are a helpful and friendly assistant named Dr. Mehta.

Please answer the questions in a JSON with key "answer".

Do not include any other text than the JSON object. Do not include "```json" or "```" or anything else.

Example Response:

{
    "answer": "42"
}"""
    )
    messages = input.messages
    assert len(messages) == 1
    message = messages[0]
    assert message.role == "user"
    content = message.content
    assert len(content) == 1
    assert isinstance(content[0], Text)
    assert content[0].type == "text"
    # templating happens here
    assert content[0].text == "What is the name of the capital city of Japan?"
    output = json_inference.output
    assert json_inference.output_schema == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    assert isinstance(json_inference.episode_id, UUID)
    assert isinstance(json_inference.inference_id, UUID)
    assert json_inference.tool_params is None
    assert json_inference.output_schema == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }


@pytest.mark.asyncio
async def test_async_render_inferences_nonexistent_function(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test that render_inferences drops if the function does not exist at all."""
    rendered_inferences = await embedded_async_client.experimental_render_inferences(
        stored_inferences=[
            StoredInference(
                type="chat",
                function_name="non_existent_function",
                variant_name="default",
                input={
                    "system": {"assistant_name": "foo"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "value": "bar"}],
                        }
                    ],
                },
                output=[Text(text="Hello world")],
                episode_id=uuid7(),
                inference_id=uuid7(),
                tool_params=ToolParams(
                    tools_available=[],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
                output_schema=None,
            )
        ],
        variants={},
    )
    assert len(rendered_inferences) == 0
    # TODO: test that the warning message is logged (we do this in Rust)


@pytest.mark.asyncio
async def test_async_render_inferences_unspecified_function(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test that render_inferences drops if the function is not specified in the variants map."""
    rendered_inferences = await embedded_async_client.experimental_render_inferences(
        stored_inferences=[
            StoredInference(
                type="chat",
                function_name="non_existent_function",
                variant_name="default",
                input={
                    "system": {"assistant_name": "foo"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "value": "bar"}],
                        }
                    ],
                },
                output=[Text(text="Hello world")],
                episode_id=uuid7(),
                inference_id=uuid7(),
                tool_params=ToolParams(
                    tools_available=[],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
                output_schema=None,
            )
        ],
        variants={},
    )
    assert len(rendered_inferences) == 0
    # TODO: test that the warning message is logged (we do this in Rust)


@pytest.mark.asyncio
async def test_async_render_inferences_no_variant(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test that render_inferences drops an example if the variant is not found and logs a warning."""
    with pytest.raises(Exception) as excinfo:
        await embedded_async_client.experimental_render_inferences(
            stored_inferences=[
                StoredInference(
                    type="chat",
                    function_name="basic_test",  # This function exists in the config
                    variant_name="non_existent_variant",
                    input={
                        "system": {"assistant_name": "foo"},
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "value": "bar"}],
                            }
                        ],
                    },
                    output=[Text(text="Hello world")],
                    episode_id=uuid7(),
                    inference_id=uuid7(),
                    tool_params=ToolParams(
                        tools_available=[],
                        tool_choice="auto",
                        parallel_tool_calls=False,
                    ),
                    output_schema=None,
                )
            ],
            variants={"basic_test": "non_existent_variant"},
        )
    assert "Variant non_existent_variant for function basic_test not found" in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_async_render_inferences_missing_variable(
    embedded_async_client: AsyncTensorZeroGateway,
):
    """Test that render_inferences drops an example if a template variable is missing."""
    rendered_inferences = await embedded_async_client.experimental_render_inferences(
        stored_inferences=[
            StoredInference(
                type="chat",
                function_name="basic_test",  # Uses assistant_name in system prompt
                variant_name="default",
                input={
                    "system": {"some_other_variable": "foo"},  # Missing assistant_name
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "value": "bar"}],
                        }
                    ],
                },
                output=[Text(text="Hello world")],
                episode_id=uuid7(),
                inference_id=uuid7(),
                tool_params=ToolParams(
                    tools_available=[],
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
                output_schema=None,
            )
        ],
        variants={"basic_test": "test"},
    )
    assert len(rendered_inferences) == 0
    # TODO: test that the warning message is logged (we do this in Rust)
