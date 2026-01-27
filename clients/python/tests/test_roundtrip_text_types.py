"""
Test round-trip serialization and reuse of text content types.

Verifies that text and raw_text content preserves all fields through:
- Inference response types (Text, RawText)
- Storage types (ContentBlockChatOutputText, StoredInputMessageContentText, StoredInputMessageContentRawText)
- Serialization (asdict, JSON)
- Reuse in follow-up inferences
- Datapoint creation and retrieval

This catches type generation issues for text content types.
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
    Text,
)
from tensorzero.generated_types import (
    ContentBlockChatOutputText,
    DatapointChat,
    InputMessageContentRawText,
    InputMessageContentText,
    StoredInferenceChat,
    StoredInputMessageContentRawText,
    StoredInputMessageContentText,
)
from uuid_utils import uuid7


@pytest.mark.asyncio
async def test_async_text_content_roundtrip_complete_flow(
    async_client: AsyncTensorZeroGateway,
):
    """
    Comprehensive test verifying text content preserves all fields through:
    1. Inference response (Text type)
    2. Storage retrieval (ContentBlockChatOutputText, StoredInputMessageContentText types)
    3. Serialization (asdict + JSON)
    4. Reuse in follow-up inference
    5. Datapoint creation from inference
    6. Datapoint retrieval and validation

    This test validates 7 types across the complete lifecycle.
    Tests 12 steps total (8 inference + 4 datapoint).
    """

    # ============================================================================
    # Step 1: Create initial inference with text content
    # ============================================================================

    result = await async_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [{"role": "user", "content": "What's your name?"}],
        },
        stream=False,
    )

    # Basic result assertions
    assert isinstance(result, ChatInferenceResponse), "Result must be ChatInferenceResponse instance"
    assert result.content is not None, "Result content must not be None"
    assert len(result.content) >= 1, "Result should have at least 1 content block"
    assert result.inference_id is not None, "Result must have inference_id"

    # ============================================================================
    # Step 2: Verify response Text fields (types.py Text)
    # ============================================================================

    # Find text content in response
    text_response = None
    for content in result.content:
        if content.type == "text":
            text_response = content
            break

    assert text_response is not None, "Response must contain text content"
    assert isinstance(text_response, Text), "Response must be Text instance"
    assert hasattr(text_response, "text"), "Text must have 'text' field"
    assert text_response.text is not None, "Text field must not be None"

    original_text = text_response.text

    # Store inference_id for retrieval
    inference_id = str(result.inference_id)

    # Wait for results to be written to ClickHouse (required for batch writes)
    await asyncio.sleep(1)

    # ============================================================================
    # Step 3: Query inference back via get_inferences
    # ============================================================================

    get_response = await async_client.get_inferences(
        ids=[inference_id],
        function_name="basic_test",
        output_source="inference",
    )

    assert get_response.inferences is not None, "get_inferences must return inferences"
    assert len(get_response.inferences) == 1, "Should retrieve exactly 1 inference"

    stored_inference = get_response.inferences[0]
    assert isinstance(stored_inference, StoredInferenceChat), "Must be StoredInferenceChat instance"
    assert stored_inference.output is not None, "Stored inference output must not be None"
    assert str(stored_inference.inference_id) == inference_id, "Retrieved inference ID must match original"

    # ============================================================================
    # Step 4: Verify stored output Text (ContentBlockChatOutputText)
    # ============================================================================

    stored_output = stored_inference.output
    assert len(stored_output) >= 1, "Output should have at least 1 content block"

    # Find text in output
    stored_text = None
    for content in stored_output:
        if content.type == "text":
            stored_text = content
            break

    assert stored_text is not None, "Stored output must contain text"
    assert stored_text.type == "text", "Stored output must have type='text'"
    assert isinstance(stored_text, ContentBlockChatOutputText), (
        "Stored output must be ContentBlockChatOutputText instance"
    )
    assert hasattr(stored_text, "text"), "Stored text must have 'text' field"
    assert stored_text.text == original_text, "Stored text must match original"

    # ============================================================================
    # Step 5: Serialize text content
    # ============================================================================

    text_dict = asdict(stored_text)
    assert "type" in text_dict and text_dict["type"] == "text"
    assert "text" in text_dict and text_dict["text"] == original_text

    # Verify JSON round-trip
    text_json = json.dumps(text_dict)
    text_reloaded = json.loads(text_json)
    assert text_reloaded["text"] == original_text

    # ============================================================================
    # Step 6: Create raw_text content for follow-up
    # ============================================================================

    raw_text_content = {"type": "raw_text", "value": "This is raw text content"}

    # ============================================================================
    # Step 7: Reuse in follow-up inference
    # ============================================================================

    follow_up_result = await async_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                {"role": "user", "content": "What's your name?"},
                {"role": "assistant", "content": [text_dict]},  # Reuse serialized text
                {"role": "user", "content": [raw_text_content]},  # Add raw_text
            ],
        },
        stream=False,
    )

    assert isinstance(follow_up_result, ChatInferenceResponse), "Follow-up must return ChatInferenceResponse instance"
    assert follow_up_result.inference_id is not None, (
        "Follow-up inference must succeed when reusing serialized text data"
    )

    # Wait for follow-up results to be written to ClickHouse (required for batch writes)
    await asyncio.sleep(1)

    # ============================================================================
    # Step 8: Verify follow-up stored data
    # ============================================================================

    follow_up_id = str(follow_up_result.inference_id)
    follow_up_stored = await async_client.get_inferences(
        ids=[follow_up_id],
        function_name="basic_test",
        output_source="inference",
    )

    assert len(follow_up_stored.inferences) == 1, "Should retrieve exactly 1 follow-up inference"
    follow_up_inference = follow_up_stored.inferences[0]

    # Verify input contains our text and raw_text
    input_messages = follow_up_inference.input.messages
    assert input_messages is not None, "Input messages must not be None"
    assert len(input_messages) >= 3, "Should have user, assistant, and follow-up user messages"

    # Find assistant message with text
    assistant_msg = None
    for msg in input_messages:
        if msg.role == "assistant":
            assistant_msg = msg
            break

    assert assistant_msg is not None, "Should have assistant message in stored input"
    assert assistant_msg.content is not None, "Assistant message content must not be None"
    assert len(assistant_msg.content) > 0, "Assistant message should have content"

    # Verify the text was stored correctly (StoredInputMessageContentText)
    text_content = None
    for content in assistant_msg.content:
        if content.type == "text":
            text_content = content
            break

    assert text_content is not None, "Should have text in assistant message"
    assert isinstance(text_content, StoredInputMessageContentText), "Must be StoredInputMessageContentText instance"
    assert hasattr(text_content, "text"), "StoredInputMessageContentText must have 'text' field"
    assert text_content.text == original_text, "Text must be preserved through round-trip"

    # Verify raw_text was also stored correctly
    user_msg_with_raw_text = None
    for msg in input_messages:
        if msg.role == "user":
            for content in msg.content:
                if content.type == "raw_text":
                    user_msg_with_raw_text = content
                    break
            if user_msg_with_raw_text:
                break

    assert user_msg_with_raw_text is not None, "Should have raw_text in stored input"
    assert isinstance(user_msg_with_raw_text, StoredInputMessageContentRawText), (
        "Must be StoredInputMessageContentRawText instance"
    )
    assert user_msg_with_raw_text.value == "This is raw text content", "Raw text must be preserved"

    # ============================================================================
    # Step 9: Create datapoint from follow-up inference
    # ============================================================================

    dataset_name = f"test_text_roundtrip_{uuid7()}"

    try:
        datapoint_response = await async_client.create_datapoints_from_inferences(
            dataset_name=dataset_name,
            params=CreateDatapointsFromInferenceRequestParamsInferenceIds(inference_ids=[follow_up_id]),
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
        # Step 11: Verify text content in datapoint input
        # ============================================================================

        datapoint_input_messages = datapoint.input.messages
        assert datapoint_input_messages is not None, "Datapoint input messages must not be None"
        assert len(datapoint_input_messages) >= 3, "Datapoint should have user, assistant, and follow-up messages"

        # Find assistant message with text in datapoint input
        datapoint_assistant_msg = None
        for msg in datapoint_input_messages:
            if msg.role == "assistant":
                datapoint_assistant_msg = msg
                break

        assert datapoint_assistant_msg is not None, "Datapoint should have assistant message"
        assert datapoint_assistant_msg.content is not None, "Datapoint assistant message content must not be None"
        assert len(datapoint_assistant_msg.content) > 0, "Datapoint assistant message should have content"

        # Verify text in datapoint input
        datapoint_text = None
        for content in datapoint_assistant_msg.content:
            if content.type == "text":
                datapoint_text = content
                break

        assert datapoint_text is not None, "Datapoint should have text in assistant message"
        assert isinstance(datapoint_text, InputMessageContentText), (
            "Datapoint text must be InputMessageContentText instance"
        )
        assert hasattr(datapoint_text, "text"), "Datapoint text must have 'text' field"
        assert datapoint_text.text == original_text, "Datapoint text must be preserved"

        # Verify raw_text in datapoint input
        datapoint_raw_text = None
        for msg in datapoint_input_messages:
            if msg.role == "user":
                for content in msg.content:
                    if content.type == "raw_text":
                        datapoint_raw_text = content
                        break
                if datapoint_raw_text:
                    break

        assert datapoint_raw_text is not None, "Datapoint should have raw_text in input"
        assert isinstance(datapoint_raw_text, InputMessageContentRawText), (
            "Datapoint raw_text must be InputMessageContentRawText instance"
        )
        assert hasattr(datapoint_raw_text, "value"), "Datapoint raw_text must have 'value' field"
        assert datapoint_raw_text.value == "This is raw text content", "Datapoint raw_text must be preserved"

        # ============================================================================
        # Step 12: Verify text in datapoint output
        # ============================================================================

        datapoint_output = datapoint.output
        assert datapoint_output is not None, "Datapoint must have output"
        assert len(datapoint_output) >= 1, "Datapoint output should have at least 1 content block"

        # Find text in output
        datapoint_output_text = None
        for content in datapoint_output:
            if content.type == "text":
                datapoint_output_text = content
                break

        assert datapoint_output_text is not None, "Datapoint output must contain text"
        assert datapoint_output_text.type == "text", "Datapoint output must have type='text'"
        assert isinstance(datapoint_output_text, ContentBlockChatOutputText), (
            "Datapoint output text must be ContentBlockChatOutputText instance"
        )
        assert hasattr(datapoint_output_text, "text"), "Datapoint output text must have 'text' field"

    finally:
        await async_client.delete_dataset(dataset_name=dataset_name)


def test_sync_text_content_roundtrip_complete_flow(sync_client: TensorZeroGateway):
    """
    Sync version of test_async_text_content_roundtrip_complete_flow.
    Tests the same round-trip but with synchronous client.
    """

    # ============================================================================
    # Step 1: Create initial inference
    # ============================================================================

    result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [{"role": "user", "content": "What's your name?"}],
        },
        stream=False,
    )

    assert isinstance(result, ChatInferenceResponse)
    assert result.content is not None, "Result content must not be None"
    assert result.inference_id is not None, "Result must have inference_id"

    # Find text in response
    text_response = None
    for content in result.content:
        if content.type == "text":
            text_response = content
            break

    assert text_response is not None
    assert isinstance(text_response, Text)
    original_text = text_response.text

    inference_id = str(result.inference_id)

    # Wait for results to be written to ClickHouse (required for batch writes)
    time.sleep(1)

    # ============================================================================
    # Step 3: Query via get_inferences
    # ============================================================================

    get_response = sync_client.get_inferences(
        ids=[inference_id],
        function_name="basic_test",
        output_source="inference",
    )

    stored_inference = get_response.inferences[0]
    assert isinstance(stored_inference, StoredInferenceChat), "Must be StoredInferenceChat instance"
    assert stored_inference.output is not None, "Stored inference output must not be None"

    # Find text in stored output
    stored_text = None
    for content in stored_inference.output:
        if content.type == "text":
            stored_text = content
            break

    assert stored_text is not None
    assert isinstance(stored_text, ContentBlockChatOutputText)

    # ============================================================================
    # Step 5: Serialize and reuse
    # ============================================================================

    text_dict = asdict(stored_text)
    raw_text_content = {"type": "raw_text", "value": "This is raw text"}

    follow_up_result = sync_client.inference(
        function_name="basic_test",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": [text_dict]},
                {"role": "user", "content": [raw_text_content]},
            ],
        },
        stream=False,
    )

    assert isinstance(follow_up_result, ChatInferenceResponse), "Follow-up must return ChatInferenceResponse"
    assert follow_up_result.inference_id is not None, "Follow-up must have inference_id"

    follow_up_id = str(follow_up_result.inference_id)

    # Wait for follow-up results to be written to ClickHouse (required for batch writes)
    time.sleep(1)

    follow_up_stored = sync_client.get_inferences(
        ids=[follow_up_id],
        function_name="basic_test",
        output_source="inference",
    )

    follow_up_inference = follow_up_stored.inferences[0]

    # Verify StoredInputMessageContentText
    input_messages_sync = follow_up_inference.input.messages
    assert input_messages_sync is not None, "Input messages must not be None"

    assistant_msg = None
    for msg in input_messages_sync:
        if msg.role == "assistant":
            assistant_msg = msg
            break

    assert assistant_msg is not None, "Assistant message must not be None"
    assert assistant_msg.content is not None, "Assistant message content must not be None"

    text_content = None
    for content in assistant_msg.content:
        if content.type == "text":
            text_content = content
            break

    assert isinstance(text_content, StoredInputMessageContentText)
    assert text_content.text == original_text

    # Verify StoredInputMessageContentRawText
    raw_text_found = None
    for msg in input_messages_sync:
        if msg.role == "user":
            assert msg.content is not None, "User message content must not be None"
            for content in msg.content:
                if content.type == "raw_text":
                    raw_text_found = content
                    break
            if raw_text_found:
                break

    assert isinstance(raw_text_found, StoredInputMessageContentRawText)

    # ============================================================================
    # Step 9: Create datapoint
    # ============================================================================

    dataset_name = f"test_text_roundtrip_{uuid7()}"

    try:
        datapoint_response = sync_client.create_datapoints_from_inferences(
            dataset_name=dataset_name,
            params=CreateDatapointsFromInferenceRequestParamsInferenceIds(inference_ids=[follow_up_id]),
            output_source="inference",
        )

        datapoint_id = datapoint_response.ids[0]
        datapoint_get_response = sync_client.get_datapoints(dataset_name=dataset_name, ids=[datapoint_id])

        datapoint = datapoint_get_response.datapoints[0]
        assert isinstance(datapoint, DatapointChat)

        # Verify InputMessageContentText
        datapoint_input_messages_sync = datapoint.input.messages
        assert datapoint_input_messages_sync is not None, "Datapoint input messages must not be None"

        datapoint_assistant_msg = None
        for msg in datapoint_input_messages_sync:
            if msg.role == "assistant":
                datapoint_assistant_msg = msg
                break

        assert datapoint_assistant_msg is not None, "Datapoint assistant message must not be None"
        assert datapoint_assistant_msg.content is not None, "Datapoint assistant message content must not be None"

        datapoint_text = None
        for content in datapoint_assistant_msg.content:
            if content.type == "text":
                datapoint_text = content
                break

        assert isinstance(datapoint_text, InputMessageContentText)
        assert datapoint_text.text == original_text

        # Verify InputMessageContentRawText
        datapoint_raw_text = None
        for msg in datapoint_input_messages_sync:
            if msg.role == "user":
                assert msg.content is not None, "User message content must not be None"
                for content in msg.content:
                    if content.type == "raw_text":
                        datapoint_raw_text = content
                        break
                if datapoint_raw_text:
                    break

        assert isinstance(datapoint_raw_text, InputMessageContentRawText)

        # Verify ContentBlockChatOutputText in output
        assert datapoint.output is not None, "Datapoint output must not be None"
        output_text = None
        for content in datapoint.output:
            if content.type == "text":
                output_text = content
                break

        assert isinstance(output_text, ContentBlockChatOutputText)

    finally:
        sync_client.delete_dataset(dataset_name=dataset_name)
