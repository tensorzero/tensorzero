"""
Test round-trip serialization and reuse of tool calls.

Verifies that tool call data preserves all fields through:
- Inference response types (ToolCall)
- Storage types (ContentBlockChatOutputToolCall, StoredInputMessageContentToolCall)
- Serialization (asdict, JSON)
- Reuse in follow-up inferences
- Datapoint creation and retrieval (DatapointChat)

This catches type generation issues like PR #5803.
"""

import asyncio
import json
import time
from dataclasses import asdict

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    CreateDatapointsFromInferenceRequestParamsInferenceIds,
    TensorZeroGateway,
)
from tensorzero.generated_types import (
    ContentBlockChatOutputToolCall,
    DatapointChat,
    InputMessageContentToolCall,
    InputMessageContentToolResult,
    StoredInferenceChat,
    StoredInputMessageContentToolCall,
    StoredInputMessageContentToolResult,
)
from tensorzero.types import ToolCall
from uuid_utils import uuid7


@pytest.mark.asyncio
async def test_async_tool_call_roundtrip_complete_flow(
    async_client: AsyncTensorZeroGateway,
):
    """
    Comprehensive test verifying tool calls preserve all fields through:
    1. Inference response (ToolCall type)
    2. Storage retrieval (StoredInputMessageContentToolCall type)
    3. Serialization (asdict + JSON)
    4. Reuse in follow-up inference (complete tool use flow)
    5. Datapoint creation from inference
    6. Datapoint retrieval and validation (DatapointChat type)

    This test catches type generation issues like PR #5803 where fields were dropped.
    Tests 12 steps total (8 inference + 4 datapoint).
    """

    # ============================================================================
    # Step 1: Create initial inference with tool call
    # ============================================================================

    result = await async_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [{"role": "user", "content": "What's the weather in Brooklyn?"}],
        },
        stream=False,
    )

    # Basic result assertions
    assert isinstance(result, ChatInferenceResponse), "Result must be ChatInferenceResponse instance"
    assert result.content is not None, "Result content must not be None"
    assert result.inference_id is not None, "Result must have inference_id"
    assert len(result.content) == 1, "Result should have exactly 1 content block"
    assert result.content[0].type == "tool_call", "Content block must have type='tool_call'"
    assert isinstance(result.content[0], ToolCall), "Content block must be ToolCall instance"

    # ============================================================================
    # Step 2: Verify response ToolCall fields (types.py ToolCall)
    # ============================================================================

    tool_call_response = result.content[0]

    # Type discriminator
    assert tool_call_response.type == "tool_call", "Tool call must have type='tool_call'"

    # isinstance check
    assert isinstance(tool_call_response, ToolCall), "Response must be ToolCall instance"

    # Field existence (hasattr)
    assert hasattr(tool_call_response, "id"), "ToolCall must have 'id' field"
    assert hasattr(tool_call_response, "name"), "ToolCall must have 'name' field"
    assert hasattr(tool_call_response, "raw_name"), "ToolCall must have 'raw_name' field"
    assert hasattr(tool_call_response, "arguments"), "ToolCall must have 'arguments' field"
    assert hasattr(tool_call_response, "raw_arguments"), "ToolCall must have 'raw_arguments' field"

    # Field values not None
    assert tool_call_response.id is not None, "Tool call id must not be None"
    assert tool_call_response.name is not None, "Tool call name must not be None"
    assert tool_call_response.raw_name is not None, "Tool call raw_name must not be None"
    assert tool_call_response.arguments is not None, "Tool call arguments must not be None"
    assert tool_call_response.raw_arguments is not None, "Tool call raw_arguments must not be None"

    # Expected values (from weather_helper deterministic behavior)
    assert tool_call_response.id == "0", "Tool call id must be '0'"
    assert tool_call_response.name == "get_temperature", "Tool call name must be 'get_temperature'"
    assert tool_call_response.raw_name == "get_temperature", "Tool call raw_name must be 'get_temperature'"
    assert tool_call_response.arguments == {
        "location": "Brooklyn",
        "units": "celsius",
    }, "Tool call arguments must match expected dict"
    assert tool_call_response.raw_arguments == '{"location":"Brooklyn","units":"celsius"}', (
        "Tool call raw_arguments must match expected JSON string"
    )

    # Store inference_id for retrieval
    inference_id = str(result.inference_id)

    # Wait for results to be written to ClickHouse (required for batch writes)
    await asyncio.sleep(1)

    # ============================================================================
    # Step 3: Query inference back via get_inferences
    # ============================================================================

    get_response = await async_client.get_inferences(
        ids=[inference_id],
        function_name="weather_helper",  # Improves query performance
        output_source="inference",
    )

    assert get_response.inferences is not None, "get_inferences must return inferences"
    assert len(get_response.inferences) == 1, "Should retrieve exactly 1 inference"

    stored_inference = get_response.inferences[0]
    assert isinstance(stored_inference, StoredInferenceChat), "Must be StoredInferenceChat instance"
    assert stored_inference.output is not None, "Stored inference output must not be None"
    assert str(stored_inference.inference_id) == inference_id, "Retrieved inference ID must match original"

    # ============================================================================
    # Step 4: Verify stored output ToolCall (ContentBlockChatOutputToolCall)
    # ============================================================================

    stored_output = stored_inference.output
    assert len(stored_output) == 1, "Output should have exactly 1 content block"

    stored_tool_call = stored_output[0]

    # Type discriminator
    assert stored_tool_call.type == "tool_call", "Stored output must have type='tool_call'"

    # isinstance check
    assert isinstance(stored_tool_call, ContentBlockChatOutputToolCall), (
        "Stored output must be ContentBlockChatOutputToolCall instance"
    )

    # Field existence (hasattr)
    assert hasattr(stored_tool_call, "id"), "Stored tool call must have 'id' field"
    assert hasattr(stored_tool_call, "name"), "Stored tool call must have 'name' field"
    assert hasattr(stored_tool_call, "raw_name"), "Stored tool call must have 'raw_name' field"
    assert hasattr(stored_tool_call, "arguments"), "Stored tool call must have 'arguments' field"
    assert hasattr(stored_tool_call, "raw_arguments"), "Stored tool call must have 'raw_arguments' field"

    # Field values not None
    assert stored_tool_call.id is not None, "Stored tool call id must not be None"
    assert stored_tool_call.name is not None, "Stored tool call name must not be None"
    assert stored_tool_call.raw_name is not None, "Stored tool call raw_name must not be None"
    assert stored_tool_call.arguments is not None, "Stored tool call arguments must not be None"
    assert stored_tool_call.raw_arguments is not None, "Stored tool call raw_arguments must not be None"

    # Values match original
    assert stored_tool_call.id == "0", "Stored tool call id must be '0'"
    assert stored_tool_call.name == "get_temperature", "Stored tool call name must be 'get_temperature'"
    assert stored_tool_call.raw_name == "get_temperature", "Stored tool call raw_name must be 'get_temperature'"
    assert stored_tool_call.arguments == {
        "location": "Brooklyn",
        "units": "celsius",
    }, "Stored tool call arguments must match expected dict"
    assert stored_tool_call.raw_arguments == '{"location":"Brooklyn","units":"celsius"}', (
        "Stored tool call raw_arguments must match expected JSON string"
    )

    # ============================================================================
    # Step 5: Serialize tool call with asdict() and JSON
    # ============================================================================

    # Serialize to dict
    tool_call_dict = asdict(stored_tool_call)

    # Verify serialized dict has all required fields
    assert "type" in tool_call_dict and tool_call_dict["type"] == "tool_call", (
        "Serialized dict must include type='tool_call'"
    )
    assert "id" in tool_call_dict, "Serialized dict must include 'id' field"
    assert "name" in tool_call_dict, "Serialized dict must include 'name' field"
    assert "arguments" in tool_call_dict, "Serialized dict must include 'arguments' field"
    assert "raw_name" in tool_call_dict, "Serialized dict must include 'raw_name' field"
    assert "raw_arguments" in tool_call_dict, "Serialized dict must include 'raw_arguments' field"

    # Verify JSON serialization works (no encoding issues)
    tool_call_json = json.dumps(tool_call_dict)
    assert isinstance(tool_call_json, str), "Must serialize to JSON string"

    # Verify JSON deserialization works
    tool_call_from_json = json.loads(tool_call_json)
    assert tool_call_from_json["id"] == "0", "Deserialized JSON must preserve id field"
    assert tool_call_from_json["name"] == "get_temperature", "Deserialized JSON must preserve name field"
    assert tool_call_from_json["arguments"] == {
        "location": "Brooklyn",
        "units": "celsius",
    }, "Deserialized JSON must preserve arguments field"

    # ============================================================================
    # Step 6: Create mock tool result
    # ============================================================================

    tool_result = {
        "type": "tool_result",
        "id": "0",
        "name": "get_temperature",
        "result": '{"temperature": 72, "conditions": "sunny"}',
    }

    # ============================================================================
    # Step 7: Reuse in follow-up inference (complete tool use flow)
    # ============================================================================

    follow_up_result = await async_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                # Original user message
                {"role": "user", "content": "What's the weather in Brooklyn?"},
                # Assistant message with tool call (using serialized data)
                {
                    "role": "assistant",
                    "content": [tool_call_dict],  # Reuse serialized tool call
                },
                # Tool result
                {"role": "user", "content": [tool_result]},
                # Follow-up user message
                {"role": "user", "content": "How about tomorrow?"},
            ],
        },
        stream=False,
    )

    # The critical assertion: the inference succeeded
    assert isinstance(follow_up_result, ChatInferenceResponse), "Follow-up must return ChatInferenceResponse instance"
    assert follow_up_result.inference_id is not None, (
        "Follow-up inference must succeed when reusing serialized tool call data"
    )

    # Verify the inference ran (not just that it didn't crash)
    assert follow_up_result.content is not None, "Follow-up must have content"
    assert len(follow_up_result.content) > 0, "Follow-up must generate content"

    # Wait for follow-up results to be written to ClickHouse (required for batch writes)
    await asyncio.sleep(1)

    # ============================================================================
    # Step 8: Verify follow-up stored data
    # ============================================================================

    follow_up_id = str(follow_up_result.inference_id)
    follow_up_stored = await async_client.get_inferences(
        ids=[follow_up_id],
        function_name="weather_helper",
        output_source="inference",
    )

    assert len(follow_up_stored.inferences) == 1, "Should retrieve exactly 1 follow-up inference"
    follow_up_inference = follow_up_stored.inferences[0]

    # Verify input contains our tool call and tool result
    input_messages = follow_up_inference.input.messages
    assert input_messages is not None, "Input messages must not be None"
    assert len(input_messages) >= 3, "Should have user, assistant, and follow-up user messages"

    # Find assistant message with tool call
    assistant_msg = None
    for msg in input_messages:
        if msg.role == "assistant":
            assistant_msg = msg
            break

    assert assistant_msg is not None, "Should have assistant message in stored input"
    assert assistant_msg.content is not None, "Assistant message content must not be None"
    assert len(assistant_msg.content) > 0, "Assistant message should have content"

    # Verify the tool call was stored correctly (StoredInputMessageContentToolCall)
    tool_call_content = None
    for content in assistant_msg.content:
        if content.type == "tool_call":
            tool_call_content = content
            break

    assert tool_call_content is not None, "Should have tool call in assistant message"
    assert isinstance(tool_call_content, StoredInputMessageContentToolCall), (
        "Must be StoredInputMessageContentToolCall instance"
    )

    # Critical assertions: verify fields preserved through full round-trip
    assert hasattr(tool_call_content, "id"), "StoredInputMessageContentToolCall must have 'id' field"
    assert hasattr(tool_call_content, "name"), "StoredInputMessageContentToolCall must have 'name' field"
    assert hasattr(tool_call_content, "arguments"), "StoredInputMessageContentToolCall must have 'arguments' field"

    assert tool_call_content.id == "0", "Tool call id must be preserved through round-trip"
    assert tool_call_content.name == "get_temperature", "Tool call name must be preserved through round-trip"
    assert tool_call_content.arguments == '{"location":"Brooklyn","units":"celsius"}', (
        "Tool call arguments must be preserved through round-trip"
    )

    # Verify tool result was also stored correctly
    user_msg_with_result = None
    for msg in input_messages:
        if msg.role == "user":
            for content in msg.content:
                if content.type == "tool_result":
                    user_msg_with_result = content
                    break
            if user_msg_with_result:
                break

    assert user_msg_with_result is not None, "Should have tool result in stored input"
    assert isinstance(user_msg_with_result, StoredInputMessageContentToolResult), (
        "Must be StoredInputMessageContentToolResult instance"
    )
    assert user_msg_with_result.id == "0", "Tool result id must match tool call id"
    assert user_msg_with_result.name == "get_temperature", "Tool result name must match tool call name"
    assert user_msg_with_result.result == '{"temperature": 72, "conditions": "sunny"}', "Tool result must be preserved"

    # ============================================================================
    # Step 9: Create datapoint from follow-up inference (has complete tool flow)
    # ============================================================================

    dataset_name = f"test_tool_call_roundtrip_{uuid7()}"

    try:
        # Use follow_up_id instead of inference_id to get the complete conversation
        datapoint_response = await async_client.create_datapoints_from_inferences(
            dataset_name=dataset_name,
            params=CreateDatapointsFromInferenceRequestParamsInferenceIds(
                inference_ids=[follow_up_id]  # Use follow-up inference with full conversation
            ),
            output_source="inference",
        )

        assert datapoint_response.ids is not None, "create_datapoints_from_inferences must return IDs"
        assert len(datapoint_response.ids) == 1, "Should create exactly 1 datapoint"

        datapoint_id = datapoint_response.ids[0]

        # ============================================================================
        # Step 10: Retrieve datapoint via get_datapoints
        # ============================================================================
        datapoint_get_response = await async_client.get_datapoints(dataset_name=dataset_name, ids=[datapoint_id])

        assert datapoint_get_response.datapoints is not None, "get_datapoints must return datapoints"
        assert len(datapoint_get_response.datapoints) == 1, "Should retrieve exactly 1 datapoint"

        datapoint = datapoint_get_response.datapoints[0]
        assert isinstance(datapoint, DatapointChat), "Must be DatapointChat instance"
        assert datapoint.id == datapoint_id, "Datapoint ID must match"

        # ============================================================================
        # Step 11: Verify tool calls in datapoint input
        # ============================================================================

        datapoint_input_messages = datapoint.input.messages
        assert datapoint_input_messages is not None, "Datapoint input messages must not be None"
        assert len(datapoint_input_messages) >= 3, "Datapoint should have user, assistant, and follow-up messages"

        # Find assistant message with tool call in datapoint input
        datapoint_assistant_msg = None
        for msg in datapoint_input_messages:
            if msg.role == "assistant":
                datapoint_assistant_msg = msg
                break

        assert datapoint_assistant_msg is not None, "Datapoint should have assistant message"
        assert datapoint_assistant_msg.content is not None, "Datapoint assistant message content must not be None"
        assert len(datapoint_assistant_msg.content) > 0, "Datapoint assistant message should have content"

        # Verify tool call in datapoint input
        datapoint_tool_call = None
        for content in datapoint_assistant_msg.content:
            if content.type == "tool_call":
                datapoint_tool_call = content
                break

        assert datapoint_tool_call is not None, "Datapoint should have tool call in assistant message"
        assert isinstance(datapoint_tool_call, InputMessageContentToolCall), (
            "Datapoint tool call must be InputMessageContentToolCall instance"
        )

        # Verify tool call fields preserved in datapoint
        assert hasattr(datapoint_tool_call, "id"), "Datapoint tool call must have 'id' field"
        assert hasattr(datapoint_tool_call, "name"), "Datapoint tool call must have 'name' field"
        assert hasattr(datapoint_tool_call, "arguments"), "Datapoint tool call must have 'arguments' field"

        assert datapoint_tool_call.id == "0", "Datapoint tool call id must be preserved"
        assert datapoint_tool_call.name == "get_temperature", "Datapoint tool call name must be preserved"
        assert datapoint_tool_call.arguments == '{"location":"Brooklyn","units":"celsius"}', (
            "Datapoint tool call arguments must be preserved"
        )

        # Verify tool result in datapoint input
        datapoint_tool_result = None
        for msg in datapoint_input_messages:
            if msg.role == "user":
                for content in msg.content:
                    if content.type == "tool_result":
                        datapoint_tool_result = content
                        break
                if datapoint_tool_result:
                    break

        assert datapoint_tool_result is not None, "Datapoint should have tool result in input"
        assert isinstance(datapoint_tool_result, InputMessageContentToolResult), (
            "Datapoint tool result must be InputMessageContentToolResult instance"
        )

        # Verify tool result fields preserved in datapoint
        assert hasattr(datapoint_tool_result, "id"), "Datapoint tool result must have 'id' field"
        assert hasattr(datapoint_tool_result, "name"), "Datapoint tool result must have 'name' field"
        assert hasattr(datapoint_tool_result, "result"), "Datapoint tool result must have 'result' field"

        assert datapoint_tool_result.id == "0", "Datapoint tool result id must match tool call id"
        assert datapoint_tool_result.name == "get_temperature", "Datapoint tool result name must match tool call name"
        assert datapoint_tool_result.result == '{"temperature": 72, "conditions": "sunny"}', (
            "Datapoint tool result must be preserved"
        )

        # ============================================================================
        # Step 12: Verify tool calls in datapoint output
        # ============================================================================

        datapoint_output = datapoint.output
        assert datapoint_output is not None, "Datapoint must have output"
        assert len(datapoint_output) == 1, "Datapoint output should have exactly 1 content block"

        datapoint_output_tool_call = datapoint_output[0]
        assert datapoint_output_tool_call.type == "tool_call", "Datapoint output must have type='tool_call'"
        assert isinstance(datapoint_output_tool_call, ContentBlockChatOutputToolCall), (
            "Datapoint output tool call must be ContentBlockChatOutputToolCall instance"
        )

        # Verify output tool call fields
        assert hasattr(datapoint_output_tool_call, "id"), "Datapoint output tool call must have 'id' field"
        assert hasattr(datapoint_output_tool_call, "name"), "Datapoint output tool call must have 'name' field"
        assert hasattr(datapoint_output_tool_call, "arguments"), (
            "Datapoint output tool call must have 'arguments' field"
        )

        assert datapoint_output_tool_call.id == "0", "Datapoint output tool call id must be '0'"
        assert datapoint_output_tool_call.name == "get_temperature", (
            "Datapoint output tool call name must be 'get_temperature'"
        )
        assert datapoint_output_tool_call.arguments == {
            "location": "Brooklyn",
            "units": "celsius",
        }, "Datapoint output tool call arguments must match expected dict"

    finally:
        # Clean up: delete the dataset
        await async_client.delete_dataset(dataset_name=dataset_name)


def test_sync_tool_call_roundtrip_complete_flow(sync_client: TensorZeroGateway):
    """
    Comprehensive test verifying tool calls preserve all fields through:
    1. Inference response (ToolCall type)
    2. Storage retrieval (StoredInputMessageContentToolCall type)
    3. Serialization (asdict + JSON)
    4. Reuse in follow-up inference (complete tool use flow)
    5. Datapoint creation from inference
    6. Datapoint retrieval and validation (DatapointChat type)

    This test catches type generation issues like PR #5803 where fields were dropped.
    Tests 12 steps total (8 inference + 4 datapoint).
    """

    # ============================================================================
    # Step 1: Create initial inference with tool call
    # ============================================================================

    result = sync_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [{"role": "user", "content": "What's the weather in Brooklyn?"}],
        },
        stream=False,
    )

    # Basic result assertions
    assert isinstance(result, ChatInferenceResponse), "Result must be ChatInferenceResponse instance"
    assert result.content is not None, "Result content must not be None"
    assert result.inference_id is not None, "Result must have inference_id"
    assert len(result.content) == 1, "Result should have exactly 1 content block"
    assert result.content[0].type == "tool_call", "Content block must have type='tool_call'"
    assert isinstance(result.content[0], ToolCall), "Content block must be ToolCall instance"

    # ============================================================================
    # Step 2: Verify response ToolCall fields (types.py ToolCall)
    # ============================================================================

    tool_call_response = result.content[0]

    # Type discriminator
    assert tool_call_response.type == "tool_call", "Tool call must have type='tool_call'"

    # isinstance check
    assert isinstance(tool_call_response, ToolCall), "Response must be ToolCall instance"

    # Field existence (hasattr)
    assert hasattr(tool_call_response, "id"), "ToolCall must have 'id' field"
    assert hasattr(tool_call_response, "name"), "ToolCall must have 'name' field"
    assert hasattr(tool_call_response, "raw_name"), "ToolCall must have 'raw_name' field"
    assert hasattr(tool_call_response, "arguments"), "ToolCall must have 'arguments' field"
    assert hasattr(tool_call_response, "raw_arguments"), "ToolCall must have 'raw_arguments' field"

    # Field values not None
    assert tool_call_response.id is not None, "Tool call id must not be None"
    assert tool_call_response.name is not None, "Tool call name must not be None"
    assert tool_call_response.raw_name is not None, "Tool call raw_name must not be None"
    assert tool_call_response.arguments is not None, "Tool call arguments must not be None"
    assert tool_call_response.raw_arguments is not None, "Tool call raw_arguments must not be None"

    # Expected values (from weather_helper deterministic behavior)
    assert tool_call_response.id == "0", "Tool call id must be '0'"
    assert tool_call_response.name == "get_temperature", "Tool call name must be 'get_temperature'"
    assert tool_call_response.raw_name == "get_temperature", "Tool call raw_name must be 'get_temperature'"
    assert tool_call_response.arguments == {
        "location": "Brooklyn",
        "units": "celsius",
    }, "Tool call arguments must match expected dict"
    assert tool_call_response.raw_arguments == '{"location":"Brooklyn","units":"celsius"}', (
        "Tool call raw_arguments must match expected JSON string"
    )

    # Store inference_id for retrieval
    inference_id = str(result.inference_id)

    # Wait for results to be written to ClickHouse (required for batch writes)
    time.sleep(1)

    # ============================================================================
    # Step 3: Query inference back via get_inferences
    # ============================================================================

    get_response = sync_client.get_inferences(
        ids=[inference_id],
        function_name="weather_helper",  # Improves query performance
        output_source="inference",
    )

    assert get_response.inferences is not None, "get_inferences must return inferences"
    assert len(get_response.inferences) == 1, "Should retrieve exactly 1 inference"

    stored_inference = get_response.inferences[0]
    assert isinstance(stored_inference, StoredInferenceChat), "Must be StoredInferenceChat instance"
    assert stored_inference.output is not None, "Stored inference output must not be None"
    assert str(stored_inference.inference_id) == inference_id, "Retrieved inference ID must match original"

    # ============================================================================
    # Step 4: Verify stored output ToolCall (ContentBlockChatOutputToolCall)
    # ============================================================================

    stored_output = stored_inference.output
    assert len(stored_output) == 1, "Output should have exactly 1 content block"

    stored_tool_call = stored_output[0]

    # Type discriminator
    assert stored_tool_call.type == "tool_call", "Stored output must have type='tool_call'"

    # isinstance check
    assert isinstance(stored_tool_call, ContentBlockChatOutputToolCall), (
        "Stored output must be ContentBlockChatOutputToolCall instance"
    )

    # Field existence (hasattr)
    assert hasattr(stored_tool_call, "id"), "Stored tool call must have 'id' field"
    assert hasattr(stored_tool_call, "name"), "Stored tool call must have 'name' field"
    assert hasattr(stored_tool_call, "raw_name"), "Stored tool call must have 'raw_name' field"
    assert hasattr(stored_tool_call, "arguments"), "Stored tool call must have 'arguments' field"
    assert hasattr(stored_tool_call, "raw_arguments"), "Stored tool call must have 'raw_arguments' field"

    # Field values not None
    assert stored_tool_call.id is not None, "Stored tool call id must not be None"
    assert stored_tool_call.name is not None, "Stored tool call name must not be None"
    assert stored_tool_call.raw_name is not None, "Stored tool call raw_name must not be None"
    assert stored_tool_call.arguments is not None, "Stored tool call arguments must not be None"
    assert stored_tool_call.raw_arguments is not None, "Stored tool call raw_arguments must not be None"

    # Values match original
    assert stored_tool_call.id == "0", "Stored tool call id must be '0'"
    assert stored_tool_call.name == "get_temperature", "Stored tool call name must be 'get_temperature'"
    assert stored_tool_call.raw_name == "get_temperature", "Stored tool call raw_name must be 'get_temperature'"
    assert stored_tool_call.arguments == {
        "location": "Brooklyn",
        "units": "celsius",
    }, "Stored tool call arguments must match expected dict"
    assert stored_tool_call.raw_arguments == '{"location":"Brooklyn","units":"celsius"}', (
        "Stored tool call raw_arguments must match expected JSON string"
    )

    # ============================================================================
    # Step 5: Serialize tool call with asdict() and JSON
    # ============================================================================

    # Serialize to dict
    tool_call_dict = asdict(stored_tool_call)

    # Verify serialized dict has all required fields
    assert "type" in tool_call_dict and tool_call_dict["type"] == "tool_call", (
        "Serialized dict must include type='tool_call'"
    )
    assert "id" in tool_call_dict, "Serialized dict must include 'id' field"
    assert "name" in tool_call_dict, "Serialized dict must include 'name' field"
    assert "arguments" in tool_call_dict, "Serialized dict must include 'arguments' field"
    assert "raw_name" in tool_call_dict, "Serialized dict must include 'raw_name' field"
    assert "raw_arguments" in tool_call_dict, "Serialized dict must include 'raw_arguments' field"

    # Verify JSON serialization works (no encoding issues)
    tool_call_json = json.dumps(tool_call_dict)
    assert isinstance(tool_call_json, str), "Must serialize to JSON string"

    # Verify JSON deserialization works
    tool_call_from_json = json.loads(tool_call_json)
    assert tool_call_from_json["id"] == "0", "Deserialized JSON must preserve id field"
    assert tool_call_from_json["name"] == "get_temperature", "Deserialized JSON must preserve name field"
    assert tool_call_from_json["arguments"] == {
        "location": "Brooklyn",
        "units": "celsius",
    }, "Deserialized JSON must preserve arguments field"

    # ============================================================================
    # Step 6: Create mock tool result
    # ============================================================================

    tool_result = {
        "type": "tool_result",
        "id": "0",
        "name": "get_temperature",
        "result": '{"temperature": 72, "conditions": "sunny"}',
    }

    # ============================================================================
    # Step 7: Reuse in follow-up inference (complete tool use flow)
    # ============================================================================

    follow_up_result = sync_client.inference(
        function_name="weather_helper",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                # Original user message
                {"role": "user", "content": "What's the weather in Brooklyn?"},
                # Assistant message with tool call (using serialized data)
                {
                    "role": "assistant",
                    "content": [tool_call_dict],  # Reuse serialized tool call
                },
                # Tool result
                {"role": "user", "content": [tool_result]},
                # Follow-up user message
                {"role": "user", "content": "How about tomorrow?"},
            ],
        },
        stream=False,
    )

    # The critical assertion: the inference succeeded
    assert isinstance(follow_up_result, ChatInferenceResponse), "Follow-up must return ChatInferenceResponse instance"
    assert follow_up_result.inference_id is not None, (
        "Follow-up inference must succeed when reusing serialized tool call data"
    )

    # Verify the inference ran (not just that it didn't crash)
    assert follow_up_result.content is not None, "Follow-up must have content"
    assert len(follow_up_result.content) > 0, "Follow-up must generate content"

    # Wait for follow-up results to be written to ClickHouse (required for batch writes)
    time.sleep(1)

    # ============================================================================
    # Step 8: Verify follow-up stored data
    # ============================================================================

    follow_up_id = str(follow_up_result.inference_id)
    follow_up_stored = sync_client.get_inferences(
        ids=[follow_up_id],
        function_name="weather_helper",
        output_source="inference",
    )

    assert len(follow_up_stored.inferences) == 1, "Should retrieve exactly 1 follow-up inference"
    follow_up_inference = follow_up_stored.inferences[0]

    # Verify input contains our tool call and tool result
    input_messages = follow_up_inference.input.messages
    assert input_messages is not None, "Input messages must not be None"
    assert len(input_messages) >= 3, "Should have user, assistant, and follow-up user messages"

    # Find assistant message with tool call
    assistant_msg = None
    for msg in input_messages:
        if msg.role == "assistant":
            assistant_msg = msg
            break

    assert assistant_msg is not None, "Should have assistant message in stored input"
    assert assistant_msg.content is not None, "Assistant message content must not be None"
    assert len(assistant_msg.content) > 0, "Assistant message should have content"

    # Verify the tool call was stored correctly (StoredInputMessageContentToolCall)
    tool_call_content = None
    for content in assistant_msg.content:
        if content.type == "tool_call":
            tool_call_content = content
            break

    assert tool_call_content is not None, "Should have tool call in assistant message"
    assert isinstance(tool_call_content, StoredInputMessageContentToolCall), (
        "Must be StoredInputMessageContentToolCall instance"
    )

    # Critical assertions: verify fields preserved through full round-trip
    assert hasattr(tool_call_content, "id"), "StoredInputMessageContentToolCall must have 'id' field"
    assert hasattr(tool_call_content, "name"), "StoredInputMessageContentToolCall must have 'name' field"
    assert hasattr(tool_call_content, "arguments"), "StoredInputMessageContentToolCall must have 'arguments' field"

    assert tool_call_content.id == "0", "Tool call id must be preserved through round-trip"
    assert tool_call_content.name == "get_temperature", "Tool call name must be preserved through round-trip"
    assert tool_call_content.arguments == '{"location":"Brooklyn","units":"celsius"}', (
        "Tool call arguments must be preserved through round-trip"
    )

    # Verify tool result was also stored correctly
    user_msg_with_result = None
    for msg in input_messages:
        if msg.role == "user":
            for content in msg.content:
                if content.type == "tool_result":
                    user_msg_with_result = content
                    break
            if user_msg_with_result:
                break

    assert user_msg_with_result is not None, "Should have tool result in stored input"
    assert isinstance(user_msg_with_result, StoredInputMessageContentToolResult), (
        "Must be StoredInputMessageContentToolResult instance"
    )
    assert user_msg_with_result.id == "0", "Tool result id must match tool call id"
    assert user_msg_with_result.name == "get_temperature", "Tool result name must match tool call name"
    assert user_msg_with_result.result == '{"temperature": 72, "conditions": "sunny"}', "Tool result must be preserved"

    # ============================================================================
    # Step 9: Create datapoint from follow-up inference (has complete tool flow)
    # ============================================================================

    dataset_name = f"test_tool_call_roundtrip_{uuid7()}"

    try:
        # Use follow_up_id instead of inference_id to get the complete conversation
        datapoint_response = sync_client.create_datapoints_from_inferences(
            dataset_name=dataset_name,
            params=CreateDatapointsFromInferenceRequestParamsInferenceIds(
                inference_ids=[follow_up_id]  # Use follow-up inference with full conversation
            ),
            output_source="inference",
        )

        assert datapoint_response.ids is not None, "create_datapoints_from_inferences must return IDs"
        assert len(datapoint_response.ids) == 1, "Should create exactly 1 datapoint"

        datapoint_id = datapoint_response.ids[0]

        # ============================================================================
        # Step 10: Retrieve datapoint via get_datapoints
        # ============================================================================
        datapoint_get_response = sync_client.get_datapoints(dataset_name=dataset_name, ids=[datapoint_id])

        assert datapoint_get_response.datapoints is not None, "get_datapoints must return datapoints"
        assert len(datapoint_get_response.datapoints) == 1, "Should retrieve exactly 1 datapoint"

        datapoint = datapoint_get_response.datapoints[0]
        assert isinstance(datapoint, DatapointChat), "Must be DatapointChat instance"
        assert datapoint.id == datapoint_id, "Datapoint ID must match"

        # ============================================================================
        # Step 11: Verify tool calls in datapoint input
        # ============================================================================

        datapoint_input_messages = datapoint.input.messages
        assert datapoint_input_messages is not None, "Datapoint input messages must not be None"
        assert len(datapoint_input_messages) >= 3, "Datapoint should have user, assistant, and follow-up messages"

        # Find assistant message with tool call in datapoint input
        datapoint_assistant_msg = None
        for msg in datapoint_input_messages:
            if msg.role == "assistant":
                datapoint_assistant_msg = msg
                break

        assert datapoint_assistant_msg is not None, "Datapoint should have assistant message"
        assert datapoint_assistant_msg.content is not None, "Datapoint assistant message content must not be None"
        assert len(datapoint_assistant_msg.content) > 0, "Datapoint assistant message should have content"

        # Verify tool call in datapoint input
        datapoint_tool_call = None
        for content in datapoint_assistant_msg.content:
            if content.type == "tool_call":
                datapoint_tool_call = content
                break

        assert datapoint_tool_call is not None, "Datapoint should have tool call in assistant message"
        assert isinstance(datapoint_tool_call, InputMessageContentToolCall), (
            "Datapoint tool call must be InputMessageContentToolCall instance"
        )

        # Verify tool call fields preserved in datapoint
        assert hasattr(datapoint_tool_call, "id"), "Datapoint tool call must have 'id' field"
        assert hasattr(datapoint_tool_call, "name"), "Datapoint tool call must have 'name' field"
        assert hasattr(datapoint_tool_call, "arguments"), "Datapoint tool call must have 'arguments' field"

        assert datapoint_tool_call.id == "0", "Datapoint tool call id must be preserved"
        assert datapoint_tool_call.name == "get_temperature", "Datapoint tool call name must be preserved"
        assert datapoint_tool_call.arguments == '{"location":"Brooklyn","units":"celsius"}', (
            "Datapoint tool call arguments must be preserved"
        )

        # Verify tool result in datapoint input
        datapoint_tool_result = None
        for msg in datapoint_input_messages:
            if msg.role == "user":
                for content in msg.content:
                    if content.type == "tool_result":
                        datapoint_tool_result = content
                        break
                if datapoint_tool_result:
                    break

        assert datapoint_tool_result is not None, "Datapoint should have tool result in input"
        assert isinstance(datapoint_tool_result, InputMessageContentToolResult), (
            "Datapoint tool result must be InputMessageContentToolResult instance"
        )

        # Verify tool result fields preserved in datapoint
        assert hasattr(datapoint_tool_result, "id"), "Datapoint tool result must have 'id' field"
        assert hasattr(datapoint_tool_result, "name"), "Datapoint tool result must have 'name' field"
        assert hasattr(datapoint_tool_result, "result"), "Datapoint tool result must have 'result' field"

        assert datapoint_tool_result.id == "0", "Datapoint tool result id must match tool call id"
        assert datapoint_tool_result.name == "get_temperature", "Datapoint tool result name must match tool call name"
        assert datapoint_tool_result.result == '{"temperature": 72, "conditions": "sunny"}', (
            "Datapoint tool result must be preserved"
        )

        # ============================================================================
        # Step 12: Verify tool calls in datapoint output
        # ============================================================================

        datapoint_output = datapoint.output
        assert datapoint_output is not None, "Datapoint must have output"
        assert len(datapoint_output) == 1, "Datapoint output should have exactly 1 content block"

        datapoint_output_tool_call = datapoint_output[0]
        assert datapoint_output_tool_call.type == "tool_call", "Datapoint output must have type='tool_call'"
        assert isinstance(datapoint_output_tool_call, ContentBlockChatOutputToolCall), (
            "Datapoint output tool call must be ContentBlockChatOutputToolCall instance"
        )

        # Verify output tool call fields
        assert hasattr(datapoint_output_tool_call, "id"), "Datapoint output tool call must have 'id' field"
        assert hasattr(datapoint_output_tool_call, "name"), "Datapoint output tool call must have 'name' field"
        assert hasattr(datapoint_output_tool_call, "arguments"), (
            "Datapoint output tool call must have 'arguments' field"
        )

        assert datapoint_output_tool_call.id == "0", "Datapoint output tool call id must be '0'"
        assert datapoint_output_tool_call.name == "get_temperature", (
            "Datapoint output tool call name must be 'get_temperature'"
        )
        assert datapoint_output_tool_call.arguments == {
            "location": "Brooklyn",
            "units": "celsius",
        }, "Datapoint output tool call arguments must match expected dict"

    finally:
        # Clean up: delete the dataset
        sync_client.delete_dataset(dataset_name=dataset_name)
