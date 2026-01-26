"""
Test round-trip serialization and reuse of thought content types.

Verifies that thought content preserves all fields through:
- Inference response types (Thought)
- Storage types (ContentBlockChatOutputThought, StoredInputMessageContentThought)
- Serialization (asdict, JSON)
- Reuse in follow-up inferences
- Datapoint creation and retrieval

This catches type generation issues for thought content types.

Uses the dummy "reasoner" model which returns thought content blocks.
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
    Thought,
)
from tensorzero.generated_types import (
    ContentBlockChatOutputThought,
    DatapointChat,
    InputMessageContentThought,
    StoredInferenceChat,
    StoredInputMessageContentThought,
)
from uuid_utils import uuid7


@pytest.mark.asyncio
async def test_async_thought_content_roundtrip_complete_flow(
    async_client: AsyncTensorZeroGateway,
):
    """
    Comprehensive test verifying thought content preserves all fields through:
    1. Inference response (Thought type)
    2. Storage retrieval (ContentBlockChatOutputThought, StoredInputMessageContentThought types)
    3. Serialization (asdict + JSON)
    4. Reuse in follow-up inference
    5. Datapoint creation from inference
    6. Datapoint retrieval and validation

    This test validates 4 types across the complete lifecycle.
    Tests 12 steps total (8 inference + 4 datapoint).

    Uses the dummy "reasoner" model which returns thought content.
    """

    # ============================================================================
    # Step 1: Create initial inference with reasoner model
    # ============================================================================

    result = await async_client.inference(
        function_name="basic_test",
        variant_name="reasoner",  # Dummy model that returns thought content
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        stream=False,
    )

    # Basic result assertions
    assert isinstance(result, ChatInferenceResponse), "Result must be ChatInferenceResponse instance"
    assert result.content is not None, "Result content must not be None"
    assert result.inference_id is not None, "Result must have inference_id"
    assert len(result.content) >= 1, "Result should have at least 1 content block"

    # ============================================================================
    # Step 2: Verify response Thought fields (types.py Thought)
    # ============================================================================

    # Find thought content in response
    thought_response = None
    for content in result.content:
        if content.type == "thought":
            thought_response = content
            break

    assert thought_response is not None, "Response must contain thought content"
    assert isinstance(thought_response, Thought), "Response must be Thought instance"
    assert hasattr(thought_response, "text"), "Thought must have 'text' field"
    # Note: text can be None for thoughts with only signature

    original_thought_text = thought_response.text

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
    # Step 4: Verify stored output Thought (ContentBlockChatOutputThought)
    # ============================================================================

    stored_output = stored_inference.output
    assert len(stored_output) >= 1, "Output should have at least 1 content block"

    # Find thought in output
    stored_thought = None
    for content in stored_output:
        if content.type == "thought":
            stored_thought = content
            break

    assert stored_thought is not None, "Stored output must contain thought"
    assert stored_thought.type == "thought", "Stored output must have type='thought'"
    assert isinstance(stored_thought, ContentBlockChatOutputThought), (
        "Stored output must be ContentBlockChatOutputThought instance"
    )
    assert hasattr(stored_thought, "text"), "Stored thought must have 'text' field"
    if original_thought_text is not None:
        assert stored_thought.text == original_thought_text, "Stored thought text must match original"

    # ============================================================================
    # Step 5: Serialize thought content
    # ============================================================================

    thought_dict = asdict(stored_thought)
    assert "type" in thought_dict and thought_dict["type"] == "thought"
    assert "text" in thought_dict

    # Verify JSON round-trip
    thought_json = json.dumps(thought_dict)
    thought_reloaded = json.loads(thought_json)
    assert thought_reloaded["type"] == "thought"

    # ============================================================================
    # Step 6: Reuse in follow-up inference
    # ============================================================================

    follow_up_result = await async_client.inference(
        function_name="basic_test",
        variant_name="reasoner",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [thought_dict, {"type": "text", "text": "The answer is 4"}],
                },  # Reuse serialized thought + text
                {"role": "user", "content": "Now what is 3+3?"},
            ],
        },
        stream=False,
    )

    assert isinstance(follow_up_result, ChatInferenceResponse), "Follow-up must return ChatInferenceResponse instance"
    assert follow_up_result.inference_id is not None, (
        "Follow-up inference must succeed when reusing serialized thought data"
    )

    follow_up_id = str(follow_up_result.inference_id)

    # Wait for follow-up results to be written to ClickHouse (required for batch writes)
    await asyncio.sleep(1)

    # ============================================================================
    # Step 7: Verify follow-up stored data
    # ============================================================================

    follow_up_stored = await async_client.get_inferences(
        ids=[follow_up_id],
        function_name="basic_test",
        output_source="inference",
    )

    assert len(follow_up_stored.inferences) == 1, "Should retrieve exactly 1 follow-up inference"
    follow_up_inference = follow_up_stored.inferences[0]

    # Verify input contains our thought
    input_messages = follow_up_inference.input.messages
    assert input_messages is not None, "Input messages must not be None"
    assert len(input_messages) >= 3, "Should have user, assistant, and follow-up user messages"

    # Find assistant message with thought
    assistant_msg = None
    for msg in input_messages:
        if msg.role == "assistant":
            assistant_msg = msg
            break

    assert assistant_msg is not None, "Should have assistant message in stored input"
    assert assistant_msg.content is not None, "Assistant message content must not be None"
    assert len(assistant_msg.content) > 0, "Assistant message should have content"

    # Verify the thought was stored correctly (StoredInputMessageContentThought)
    thought_content = None
    for content in assistant_msg.content:
        if content.type == "thought":
            thought_content = content
            break

    assert thought_content is not None, "Should have thought in assistant message"
    assert isinstance(thought_content, StoredInputMessageContentThought), (
        "Must be StoredInputMessageContentThought instance"
    )
    assert hasattr(thought_content, "text"), "StoredInputMessageContentThought must have 'text' field"
    if original_thought_text is not None:
        assert thought_content.text == original_thought_text, "Thought must be preserved through round-trip"

    # ============================================================================
    # Step 8: Create datapoint from follow-up inference
    # ============================================================================

    dataset_name = f"test_thought_roundtrip_{uuid7()}"

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
        # Step 9: Retrieve datapoint via get_datapoints
        # ============================================================================
        datapoint_get_response = await async_client.get_datapoints(dataset_name=dataset_name, ids=[datapoint_id])

        assert datapoint_get_response.datapoints is not None, "get_datapoints must return datapoints"
        assert len(datapoint_get_response.datapoints) == 1, "Should retrieve exactly 1 datapoint"

        datapoint = datapoint_get_response.datapoints[0]
        assert isinstance(datapoint, DatapointChat), "Must be DatapointChat instance"
        assert datapoint.id == datapoint_id, "Datapoint ID must match"

        # ============================================================================
        # Step 10: Verify thought content in datapoint input
        # ============================================================================

        datapoint_input_messages = datapoint.input.messages
        assert datapoint_input_messages is not None, "Datapoint input messages must not be None"
        assert len(datapoint_input_messages) >= 3, "Datapoint should have user, assistant, and follow-up messages"

        # Find assistant message with thought in datapoint input
        datapoint_assistant_msg = None
        for msg in datapoint_input_messages:
            if msg.role == "assistant":
                datapoint_assistant_msg = msg
                break

        assert datapoint_assistant_msg is not None, "Datapoint should have assistant message"
        assert datapoint_assistant_msg.content is not None, "Datapoint assistant message content must not be None"
        assert len(datapoint_assistant_msg.content) > 0, "Datapoint assistant message should have content"

        # Verify thought in datapoint input
        datapoint_thought = None
        for content in datapoint_assistant_msg.content:
            if content.type == "thought":
                datapoint_thought = content
                break

        assert datapoint_thought is not None, "Datapoint should have thought in assistant message"
        assert isinstance(datapoint_thought, InputMessageContentThought), (
            "Datapoint thought must be InputMessageContentThought instance"
        )
        assert hasattr(datapoint_thought, "text"), "Datapoint thought must have 'text' field"
        if original_thought_text is not None:
            assert datapoint_thought.text == original_thought_text, "Datapoint thought must be preserved"

        # ============================================================================
        # Step 11: Verify thought in datapoint output
        # ============================================================================

        datapoint_output = datapoint.output
        assert datapoint_output is not None, "Datapoint must have output"
        assert len(datapoint_output) >= 1, "Datapoint output should have at least 1 content block"

        # Find thought in output
        datapoint_output_thought = None
        for content in datapoint_output:
            if content.type == "thought":
                datapoint_output_thought = content
                break

        assert datapoint_output_thought is not None, "Datapoint output must contain thought"
        assert datapoint_output_thought.type == "thought", "Datapoint output must have type='thought'"
        assert isinstance(datapoint_output_thought, ContentBlockChatOutputThought), (
            "Datapoint output thought must be ContentBlockChatOutputThought instance"
        )
        assert hasattr(datapoint_output_thought, "text"), "Datapoint output thought must have 'text' field"

    finally:
        await async_client.delete_dataset(dataset_name=dataset_name)


def test_sync_thought_content_roundtrip_complete_flow(sync_client: TensorZeroGateway):
    """
    Sync version of test_async_thought_content_roundtrip_complete_flow.
    Tests the same round-trip but with synchronous client.
    """

    # ============================================================================
    # Step 1: Create initial inference
    # ============================================================================

    result = sync_client.inference(
        function_name="basic_test",
        variant_name="reasoner",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        stream=False,
    )

    assert isinstance(result, ChatInferenceResponse)
    assert result.content is not None, "Result content must not be None"
    assert result.inference_id is not None, "Result must have inference_id"

    # Find thought in response
    thought_response = None
    for content in result.content:
        if content.type == "thought":
            thought_response = content
            break

    assert thought_response is not None
    assert isinstance(thought_response, Thought)
    original_thought_text = thought_response.text

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
    assert isinstance(stored_inference, StoredInferenceChat), "Must be StoredInferenceChat"
    assert stored_inference.output is not None, "Stored inference output must not be None"

    # Find thought in stored output
    stored_thought = None
    for content in stored_inference.output:
        if content.type == "thought":
            stored_thought = content
            break

    assert stored_thought is not None
    assert isinstance(stored_thought, ContentBlockChatOutputThought)

    # ============================================================================
    # Step 5: Serialize and reuse
    # ============================================================================

    thought_dict = asdict(stored_thought)

    follow_up_result = sync_client.inference(
        function_name="basic_test",
        variant_name="reasoner",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": [thought_dict, {"type": "text", "text": "Answer"}]},
                {"role": "user", "content": "Follow-up"},
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

    # Verify StoredInputMessageContentThought
    input_messages_sync = follow_up_inference.input.messages
    assert input_messages_sync is not None, "Input messages must not be None"

    assistant_msg = None
    for msg in input_messages_sync:
        if msg.role == "assistant":
            assistant_msg = msg
            break

    assert assistant_msg is not None, "Assistant message must not be None"
    assert assistant_msg.content is not None, "Assistant message content must not be None"

    thought_content = None
    for content in assistant_msg.content:
        if content.type == "thought":
            thought_content = content
            break

    assert isinstance(thought_content, StoredInputMessageContentThought)
    if original_thought_text is not None:
        assert thought_content.text == original_thought_text

    # ============================================================================
    # Step 9: Create datapoint
    # ============================================================================

    dataset_name = f"test_thought_roundtrip_{uuid7()}"

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

        # Verify InputMessageContentThought
        datapoint_input_messages_sync = datapoint.input.messages
        assert datapoint_input_messages_sync is not None, "Datapoint input messages must not be None"

        datapoint_assistant_msg = None
        for msg in datapoint_input_messages_sync:
            if msg.role == "assistant":
                datapoint_assistant_msg = msg
                break

        assert datapoint_assistant_msg is not None, "Datapoint assistant message must not be None"
        assert datapoint_assistant_msg.content is not None, "Datapoint assistant message content must not be None"

        datapoint_thought = None
        for content in datapoint_assistant_msg.content:
            if content.type == "thought":
                datapoint_thought = content
                break

        assert isinstance(datapoint_thought, InputMessageContentThought)
        if original_thought_text is not None:
            assert datapoint_thought.text == original_thought_text

        # Verify ContentBlockChatOutputThought in output
        assert datapoint.output is not None, "Datapoint output must not be None"
        output_thought = None
        for content in datapoint.output:
            if content.type == "thought":
                output_thought = content
                break

        assert isinstance(output_thought, ContentBlockChatOutputThought)

    finally:
        sync_client.delete_dataset(dataset_name=dataset_name)
