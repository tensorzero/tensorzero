"""
Test round-trip serialization and reuse of template content types.

Verifies that template content preserves all fields through:
- Datapoint creation with templates
- Storage types (StoredInputMessageContentTemplate)
- Serialization (asdict, JSON)
- Reuse in new datapoints

Template content only appears in inputs (not inference responses), so this test
focuses on the input/storage/datapoint lifecycle.

This catches type generation issues for template content types.
"""

import asyncio
import json
import time
from dataclasses import asdict

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    CreateDatapointsFromInferenceRequestParamsInferenceIds,
    JsonInferenceResponse,
    TensorZeroGateway,
)
from tensorzero.generated_types import (
    DatapointJson,
    InputMessageContentTemplate,
    JsonInferenceOutput,
    StoredInferenceJson,
    StoredInputMessageContentTemplate,
)
from uuid_utils import uuid7


@pytest.mark.asyncio
async def test_async_template_content_roundtrip_complete_flow(
    async_client: AsyncTensorZeroGateway,
):
    """
    Comprehensive test verifying template content preserves all fields through:
    1. Inference with template input
    2. Storage retrieval (StoredInputMessageContentTemplate)
    3. Serialization (asdict + JSON)
    4. Reuse in follow-up inference
    5. Datapoint creation from inference
    6. Datapoint retrieval and validation (InputMessageContentTemplate)

    This test validates 2 types across the storage/datapoint lifecycle.
    Templates only appear in inputs, not in inference responses.
    """

    # ============================================================================
    # Step 1: Create inference with template input
    # ============================================================================

    # json_success function has user_template configured
    result = await async_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "France"}}],
                }
            ],
        },
        stream=False,
    )

    # For JSON functions, we just need the inference_id
    assert isinstance(result, JsonInferenceResponse), "Result must be JsonInferenceResponse instance"
    assert result.inference_id is not None, "Inference must return ID"
    inference_id = str(result.inference_id)

    # Wait for results to be written to ClickHouse (required for batch writes)
    await asyncio.sleep(1)

    # ============================================================================
    # Step 2: Query inference back via get_inferences
    # ============================================================================

    get_response = await async_client.get_inferences(
        ids=[inference_id],
        function_name="json_success",
        output_source="inference",
    )

    assert get_response.inferences is not None, "get_inferences must return inferences"
    assert len(get_response.inferences) == 1, "Should retrieve exactly 1 inference"

    stored_inference = get_response.inferences[0]
    assert isinstance(stored_inference, StoredInferenceJson), "Must be StoredInferenceJson instance"
    assert str(stored_inference.inference_id) == inference_id, "Retrieved inference ID must match original"

    # ============================================================================
    # Step 3: Verify stored input Template (StoredInputMessageContentTemplate)
    # ============================================================================

    input_messages = stored_inference.input.messages
    assert input_messages is not None, "Input messages must not be None"
    assert len(input_messages) >= 1, "Should have at least 1 input message"

    # Find user message with template
    user_msg = None
    for msg in input_messages:
        if msg.role == "user":
            user_msg = msg
            break

    assert user_msg is not None, "Should have user message in stored input"
    assert user_msg.content is not None, "User message content must not be None"
    assert len(user_msg.content) > 0, "User message should have content"

    # Verify the template was stored correctly (StoredInputMessageContentTemplate)
    template_content = None
    for content in user_msg.content:
        if content.type == "template":
            template_content = content
            break

    assert template_content is not None, "Should have template in user message"
    assert isinstance(template_content, StoredInputMessageContentTemplate), (
        "Must be StoredInputMessageContentTemplate instance"
    )

    # Field existence
    assert hasattr(template_content, "name"), "StoredInputMessageContentTemplate must have 'name' field"
    assert hasattr(template_content, "arguments"), "StoredInputMessageContentTemplate must have 'arguments' field"

    # Field values
    assert template_content.name == "user", "Template name must be 'user'"
    assert template_content.arguments == {"country": "France"}, "Template arguments must be preserved"

    # ============================================================================
    # Step 4: Serialize template content
    # ============================================================================

    template_dict = asdict(template_content)
    assert "type" in template_dict and template_dict["type"] == "template"
    assert "name" in template_dict and template_dict["name"] == "user"
    assert "arguments" in template_dict and template_dict["arguments"] == {"country": "France"}

    # Verify JSON round-trip
    template_json = json.dumps(template_dict)
    template_reloaded = json.loads(template_json)
    assert template_reloaded["name"] == "user"
    assert template_reloaded["arguments"] == {"country": "France"}

    # ============================================================================
    # Step 5: Reuse serialized template in follow-up inference
    # ============================================================================

    follow_up_result = await async_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                {
                    "role": "user",
                    "content": [template_dict],  # Reuse serialized template
                }
            ],
        },
        stream=False,
    )

    assert isinstance(follow_up_result, JsonInferenceResponse), "Follow-up must return JsonInferenceResponse"
    assert follow_up_result.inference_id is not None, (
        "Follow-up inference must succeed when reusing serialized template data"
    )

    follow_up_id = str(follow_up_result.inference_id)

    # Wait for follow-up results to be written to ClickHouse (required for batch writes)
    await asyncio.sleep(1)

    # ============================================================================
    # Step 6: Verify follow-up stored data
    # ============================================================================

    follow_up_stored = await async_client.get_inferences(
        ids=[follow_up_id],
        function_name="json_success",
        output_source="inference",
    )

    assert len(follow_up_stored.inferences) == 1, "Should retrieve exactly 1 follow-up inference"
    follow_up_inference = follow_up_stored.inferences[0]

    # Verify template in follow-up input
    follow_up_messages = follow_up_inference.input.messages
    assert follow_up_messages is not None, "Follow-up messages must not be None"

    follow_up_user_msg = None
    for msg in follow_up_messages:
        if msg.role == "user":
            follow_up_user_msg = msg
            break

    assert follow_up_user_msg is not None, "Should have user message in follow-up"
    assert follow_up_user_msg.content is not None, "Follow-up user message content must not be None"

    follow_up_template = None
    for content in follow_up_user_msg.content:
        if content.type == "template":
            follow_up_template = content
            break

    assert follow_up_template is not None, "Should have template in follow-up"
    assert isinstance(follow_up_template, StoredInputMessageContentTemplate)
    assert follow_up_template.name == "user", "Template name must be preserved"
    assert follow_up_template.arguments == {"country": "France"}, "Template arguments must match reused serialized data"

    # ============================================================================
    # Step 7: Create datapoint from follow-up inference
    # ============================================================================

    dataset_name = f"test_template_roundtrip_{uuid7()}"

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
        # Step 8: Retrieve datapoint and verify InputMessageContentTemplate
        # ============================================================================
        datapoint_get_response = await async_client.get_datapoints(dataset_name=dataset_name, ids=[datapoint_id])

        assert datapoint_get_response.datapoints is not None, "get_datapoints must return datapoints"
        assert len(datapoint_get_response.datapoints) == 1, "Should retrieve exactly 1 datapoint"

        datapoint = datapoint_get_response.datapoints[0]
        assert isinstance(datapoint, DatapointJson), "Must be DatapointJson instance"
        assert datapoint.id == datapoint_id, "Datapoint ID must match"

        # Verify template in datapoint input
        datapoint_messages = datapoint.input.messages
        assert datapoint_messages is not None, "Datapoint messages must not be None"
        assert len(datapoint_messages) >= 1, "Datapoint should have at least 1 message"

        datapoint_user_msg = None
        for msg in datapoint_messages:
            if msg.role == "user":
                datapoint_user_msg = msg
                break

        assert datapoint_user_msg is not None, "Datapoint should have user message"
        assert datapoint_user_msg.content is not None, "Datapoint user message content must not be None"

        datapoint_template = None
        for content in datapoint_user_msg.content:
            if content.type == "template":
                datapoint_template = content
                break

        assert datapoint_template is not None, "Datapoint should have template in user message"
        assert isinstance(datapoint_template, InputMessageContentTemplate), (
            "Datapoint template must be InputMessageContentTemplate instance"
        )

        # Verify fields preserved in datapoint
        assert hasattr(datapoint_template, "name"), "Datapoint template must have 'name' field"
        assert hasattr(datapoint_template, "arguments"), "Datapoint template must have 'arguments' field"
        assert datapoint_template.name == "user", "Datapoint template name must be preserved"
        assert datapoint_template.arguments == {"country": "France"}, (
            "Datapoint template arguments must match reused serialized data"
        )

        # Verify output is JsonInferenceOutput
        assert isinstance(datapoint.output, JsonInferenceOutput), "Datapoint output must be JsonInferenceOutput"

    finally:
        await async_client.delete_dataset(dataset_name=dataset_name)


def test_sync_template_content_roundtrip_complete_flow(sync_client: TensorZeroGateway):
    """
    Sync version of test_async_template_content_roundtrip_complete_flow.
    Tests the same round-trip but with synchronous client.
    """

    # ============================================================================
    # Step 1: Create inference with template
    # ============================================================================

    result = sync_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Germany"}}],
                }
            ],
        },
        stream=False,
    )

    assert isinstance(result, JsonInferenceResponse), "Result must be JsonInferenceResponse"
    assert result.inference_id is not None, "Result must have inference_id"

    inference_id = str(result.inference_id)

    # Wait for results to be written to ClickHouse (required for batch writes)
    time.sleep(1)

    # ============================================================================
    # Step 2: Query via get_inferences
    # ============================================================================

    get_response = sync_client.get_inferences(
        ids=[inference_id],
        function_name="json_success",
        output_source="inference",
    )

    stored_inference = get_response.inferences[0]
    assert isinstance(stored_inference, StoredInferenceJson)

    # ============================================================================
    # Step 3: Verify StoredInputMessageContentTemplate
    # ============================================================================

    input_messages_sync = stored_inference.input.messages
    assert input_messages_sync is not None, "Input messages must not be None"

    user_msg = None
    for msg in input_messages_sync:
        if msg.role == "user":
            user_msg = msg
            break

    assert user_msg is not None, "User message must not be None"
    assert user_msg.content is not None, "User message content must not be None"

    template_content = None
    for content in user_msg.content:
        if content.type == "template":
            template_content = content
            break

    assert template_content is not None
    assert isinstance(template_content, StoredInputMessageContentTemplate)
    assert template_content.name == "user"
    assert template_content.arguments == {"country": "Germany"}

    # ============================================================================
    # Step 4: Serialize and create datapoint
    # ============================================================================

    template_dict = asdict(template_content)

    # Reuse in another inference
    follow_up_result = sync_client.inference(
        function_name="json_success",
        input={
            "system": {"assistant_name": "Test Assistant"},
            "messages": [
                {
                    "role": "user",
                    "content": [template_dict],
                }
            ],
        },
        stream=False,
    )

    assert isinstance(follow_up_result, JsonInferenceResponse), "Follow-up must return JsonInferenceResponse"
    assert follow_up_result.inference_id is not None, "Follow-up must have inference_id"

    follow_up_id = str(follow_up_result.inference_id)

    # Wait for follow-up results to be written to ClickHouse (required for batch writes)
    time.sleep(1)

    dataset_name = f"test_template_roundtrip_{uuid7()}"

    try:
        datapoint_response = sync_client.create_datapoints_from_inferences(
            dataset_name=dataset_name,
            params=CreateDatapointsFromInferenceRequestParamsInferenceIds(inference_ids=[follow_up_id]),
            output_source="inference",
        )

        datapoint_id = datapoint_response.ids[0]

        # ============================================================================
        # Step 5: Verify InputMessageContentTemplate in datapoint
        # ============================================================================
        datapoint_get_response = sync_client.get_datapoints(dataset_name=dataset_name, ids=[datapoint_id])

        datapoint = datapoint_get_response.datapoints[0]
        assert isinstance(datapoint, DatapointJson)

        # Find template in datapoint
        datapoint_messages_sync = datapoint.input.messages
        assert datapoint_messages_sync is not None, "Datapoint messages must not be None"

        datapoint_user_msg = None
        for msg in datapoint_messages_sync:
            if msg.role == "user":
                datapoint_user_msg = msg
                break

        assert datapoint_user_msg is not None, "Datapoint user message must not be None"
        assert datapoint_user_msg.content is not None, "Datapoint user message content must not be None"

        datapoint_template = None
        for content in datapoint_user_msg.content:
            if content.type == "template":
                datapoint_template = content
                break

        assert datapoint_template is not None
        assert isinstance(datapoint_template, InputMessageContentTemplate)
        assert datapoint_template.name == "user"
        assert datapoint_template.arguments == {"country": "Germany"}

    finally:
        sync_client.delete_dataset(dataset_name=dataset_name)
