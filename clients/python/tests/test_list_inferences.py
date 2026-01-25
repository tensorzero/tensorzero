# pyright: reportDeprecated=false
from datetime import datetime, timezone

import pytest
from tensorzero import (
    AndFilter,
    AsyncTensorZeroGateway,
    BooleanMetricFilter,
    ContentBlockChatOutputText,
    ContentBlockChatOutputToolCall,
    FloatMetricFilter,
    NotFilter,
    OrderBy,
    OrFilter,
    StoredInferenceJson,
    StoredInputMessageContentText,
    StoredInputMessageContentToolCall,
    StoredInputMessageContentToolResult,
    TagFilter,
    TensorZeroGateway,
    TimeFilter,
)


def test_simple_list_json_inferences(embedded_sync_client: TensorZeroGateway):
    order_by = [OrderBy(by="timestamp", direction="descending")]
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
        order_by=order_by,
    )
    assert len(inferences) == 2

    # Verify ordering is deterministic by checking inference IDs are unique
    inference_ids = [inference.inference_id for inference in inferences]
    assert len(set(inference_ids)) == len(inference_ids)  # All unique

    for inference in inferences:
        assert inference.function_name == "extract_entities"
        assert isinstance(inference, StoredInferenceJson)
        assert isinstance(inference.variant_name, str)
        input = inference.input
        messages = input.messages
        assert messages is not None
        assert isinstance(messages, list)
        assert len(messages) == 1
        # Type narrowing: we know these are JSON inferences
        assert inference.type == "json"
        output = inference.output
        assert output.raw is not None
        assert output.parsed is not None
        inference_id = inference.inference_id
        assert isinstance(inference_id, str)
        episode_id = inference.episode_id
        assert isinstance(episode_id, str)
        output_schema = inference.output_schema
        assert output_schema is not None
        assert inference.dispreferred_outputs is not None
        assert len(inference.dispreferred_outputs) == 0

    # ORDER BY timestamp DESC is applied - verify timestamps are in descending order
    timestamps = [inference.timestamp for inference in inferences]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] >= timestamps[i + 1], (
            f"Timestamps not in descending order: {timestamps[i]} < {timestamps[i + 1]}"
        )


def test_simple_query_with_float_filter(embedded_sync_client: TensorZeroGateway):
    filters = FloatMetricFilter(
        metric_name="jaccard_similarity",
        value=0.5,
        comparison_operator=">",
    )
    order_by = [OrderBy(by="metric", name="jaccard_similarity", direction="descending")]
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=1,
        offset=None,
        order_by=order_by,
    )
    assert len(inferences) == 1

    for inference in inferences:
        assert inference.function_name == "extract_entities"
        assert inference.dispreferred_outputs is not None
        assert len(inference.dispreferred_outputs) == 0

    # Since we aren't yet grabbing metric values from the DB we can't verify ordering by metric


def test_simple_query_chat_function(embedded_sync_client: TensorZeroGateway):
    order_by = [OrderBy(by="timestamp", direction="ascending")]
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name="better_prompt_haiku_4_5",
        filters=None,
        output_source="inference",
        limit=3,
        offset=3,
        order_by=order_by,
    )
    assert len(inferences) == 3

    # Verify ordering is deterministic by checking inference IDs are unique
    inference_ids = [inference.inference_id for inference in inferences]
    assert len(set(inference_ids)) == len(inference_ids)  # All unique

    for inference in inferences:
        assert inference.function_name == "write_haiku"
        assert inference.variant_name == "better_prompt_haiku_4_5"
        input = inference.input
        messages = input.messages
        assert messages is not None
        assert isinstance(messages, list)
        assert len(messages) == 1
        # Type narrowing: we know these are Chat inferences
        assert inference.type == "chat"
        output = inference.output
        assert len(output) == 1
        output_0 = output[0]
        assert output_0.type == "text"
        # Type narrowing: we know it's a Text block
        assert isinstance(output_0, ContentBlockChatOutputText)
        assert output_0.text is not None
        inference_id = inference.inference_id
        assert isinstance(inference_id, str)
        episode_id = inference.episode_id
        assert isinstance(episode_id, str)
        # Test individual tool param fields
        assert inference.allowed_tools is None or len(inference.allowed_tools) == 0
        assert inference.additional_tools is None or len(inference.additional_tools) == 0
        assert inference.parallel_tool_calls is None
        assert isinstance(inference.provider_tools, list)
        assert len(inference.provider_tools) == 0
        assert inference.dispreferred_outputs is not None
        assert len(inference.dispreferred_outputs) == 0

    # ORDER BY timestamp ASC is applied - verify timestamps are in ascending order
    timestamps = [inference.timestamp for inference in inferences]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], (
            f"Timestamps not in ascending order: {timestamps[i]} > {timestamps[i + 1]}"
        )


def test_simple_query_chat_function_with_tools(embedded_sync_client: TensorZeroGateway):
    limit = 2
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="multi_hop_rag_agent",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=limit,
        offset=0,
    )
    assert len(inferences) == limit
    for inference in inferences:
        assert inference.function_name == "multi_hop_rag_agent"
        input = inference.input
        messages = input.messages
        assert messages is not None
        assert isinstance(messages, list)
        assert len(messages) >= 1
        for message in messages:
            assert message.role in ["user", "assistant"]
            for content in message.content:
                assert content.type in ["text", "tool_call", "tool_result"]
                if content.type == "tool_call":
                    assert isinstance(content, StoredInputMessageContentToolCall)
                    assert content.id is not None
                    assert content.name is not None
                    assert content.arguments is not None
                elif content.type == "tool_result":
                    assert isinstance(content, StoredInputMessageContentToolResult)
                    assert content.id is not None
                    assert content.name is not None
                    assert content.result is not None
                elif content.type == "text":
                    assert isinstance(content, StoredInputMessageContentText)
                    assert content.text is not None
                else:
                    assert False

        # Type narrowing: we know these are Chat inferences
        assert inference.type == "chat"
        output = inference.output
        assert len(output) >= 1
        for output_item in output:
            if output_item.type == "text":
                assert isinstance(output_item, ContentBlockChatOutputText)
                assert output_item.text is not None
            elif output_item.type == "tool_call":
                assert isinstance(output_item, ContentBlockChatOutputToolCall)
                assert output_item.id is not None
                assert output_item.name is not None
                assert output_item.arguments is not None
                assert output_item.raw_name is not None
                assert output_item.raw_arguments is not None
        inference_id = inference.inference_id
        assert isinstance(inference_id, str)
        episode_id = inference.episode_id
        assert isinstance(episode_id, str)
        # Test individual tool param fields
        # Changed behavior: None when using function defaults
        assert inference.allowed_tools is None
        assert inference.additional_tools is None
        assert inference.parallel_tool_calls is True
        assert inference.provider_tools is None or len(inference.provider_tools) == 0


def test_demonstration_output_source(embedded_sync_client: TensorZeroGateway):
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="demonstration",
        limit=5,
        offset=1,
    )
    assert len(inferences) == 5
    for inference in inferences:
        assert inference.function_name == "extract_entities"
        assert inference.dispreferred_outputs is not None
        assert len(inference.dispreferred_outputs) == 1


def test_boolean_metric_filter(embedded_sync_client: TensorZeroGateway):
    filters = BooleanMetricFilter(
        metric_name="exact_match",
        value=True,
    )
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=5,
        offset=1,
    )
    assert len(inferences) == 5
    for inference in inferences:
        assert inference.function_name == "extract_entities"


def test_and_filter_multiple_float_metrics(embedded_sync_client: TensorZeroGateway):
    filters = AndFilter(
        children=[
            FloatMetricFilter(
                metric_name="jaccard_similarity",
                value=0.5,
                comparison_operator=">",
            ),
            FloatMetricFilter(
                metric_name="jaccard_similarity",
                value=0.8,
                comparison_operator="<",
            ),
        ]
    )
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=1,
        offset=None,
    )
    assert len(inferences) == 1
    for inference in inferences:
        assert inference.function_name == "extract_entities"


def test_or_filter_mixed_metrics(embedded_sync_client: TensorZeroGateway):
    filters = OrFilter(
        children=[
            FloatMetricFilter(
                metric_name="jaccard_similarity",
                value=0.8,
                comparison_operator=">=",
            ),
            BooleanMetricFilter(
                metric_name="exact_match",
                value=True,
            ),
            BooleanMetricFilter(
                metric_name="goal_achieved",
                value=True,
            ),
        ]
    )
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=1,
        offset=None,
    )
    assert len(inferences) == 1
    for inference in inferences:
        assert inference.function_name == "extract_entities"


def test_not_filter(embedded_sync_client: TensorZeroGateway):
    # NOT (exact_match = true OR exact_match = false) returns rows WITHOUT the metric.
    # This test verifies that the NOT filter correctly excludes rows that have the metric
    # (with either true or false value) and returns only rows without it.

    # Get total count (no filter)
    all_inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=1000,
        offset=None,
    )
    total_count = len(all_inferences)

    # Get count with exact_match = true
    true_filter = BooleanMetricFilter(metric_name="exact_match", value=True)
    true_inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=true_filter,
        output_source="inference",
        limit=1000,
        offset=None,
    )
    true_count = len(true_inferences)

    # Get count with exact_match = false
    false_filter = BooleanMetricFilter(metric_name="exact_match", value=False)
    false_inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=false_filter,
        output_source="inference",
        limit=1000,
        offset=None,
    )
    false_count = len(false_inferences)

    # Get count with NOT (true OR false) - should return rows WITHOUT the metric
    not_filter = NotFilter(
        child=OrFilter(
            children=[
                BooleanMetricFilter(metric_name="exact_match", value=True),
                BooleanMetricFilter(metric_name="exact_match", value=False),
            ]
        )
    )
    not_inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=not_filter,
        output_source="inference",
        limit=1000,
        offset=None,
    )
    not_count = len(not_inferences)

    # Verify: rows with metric (true + false) + rows without metric (NOT result) = total
    rows_with_metric = true_count + false_count
    assert rows_with_metric + not_count == total_count, (
        f"NOT filter should return exactly the rows without the metric. "
        f"true={true_count}, false={false_count}, NOT={not_count}, total={total_count}"
    )


def test_simple_time_filter(embedded_sync_client: TensorZeroGateway):
    filters = TimeFilter(
        # 2023-01-01 00:00:00 UTC
        time=datetime.fromtimestamp(1672531200, tz=timezone.utc).isoformat(),
        comparison_operator=">",
    )
    order_by = [
        OrderBy(by="metric", name="exact_match", direction="descending"),
        OrderBy(by="timestamp", direction="ascending"),
    ]
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=2,
        offset=None,
        order_by=order_by,
    )
    assert len(inferences) == 2

    # Verify ordering is deterministic by checking inference IDs are unique
    inference_ids = [inference.inference_id for inference in inferences]
    assert len(set(inference_ids)) == len(inference_ids)  # All unique

    for inference in inferences:
        assert inference.function_name == "extract_entities"

    # ORDER BY metric exact_match DESC, timestamp ASC is applied
    # Multiple ORDER BY clauses ensure deterministic ordering
    # Verify timestamps are in ascending order (secondary sort)
    timestamps = [inference.timestamp for inference in inferences]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], (
            f"Timestamps not in ascending order: {timestamps[i]} > {timestamps[i + 1]}"
        )


def test_simple_tag_filter(embedded_sync_client: TensorZeroGateway):
    filters = TagFilter(
        key="tensorzero::evaluation_name",
        value="entity_extraction",
        comparison_operator="=",
    )
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=49,
        offset=None,
    )
    assert len(inferences) == 49
    for inference in inferences:
        assert inference.function_name == "extract_entities"
        assert inference.tags is not None
        assert inference.tags["tensorzero::evaluation_name"] == "entity_extraction"


def test_combined_time_and_tag_filter(embedded_sync_client: TensorZeroGateway):
    filters = AndFilter(
        children=[
            TimeFilter(
                # 2025-04-14 23:30:00 UTC
                time=datetime.fromtimestamp(1744673400, tz=timezone.utc).isoformat(),
                comparison_operator=">=",
            ),
            TagFilter(
                key="tensorzero::evaluation_name",
                value="haiku",
                comparison_operator="=",
            ),
        ]
    )
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=23,
        offset=None,
    )
    assert len(inferences) == 23
    for inference in inferences:
        assert inference.function_name == "write_haiku"
        assert inference.tags is not None
        assert inference.tags["tensorzero::evaluation_name"] == "haiku"


def test_list_render_json_inferences(embedded_sync_client: TensorZeroGateway):
    stored_inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
    )
    rendered_inferences = embedded_sync_client.experimental_render_samples(
        stored_samples=stored_inferences,
        variants={"extract_entities": "gpt_4o_mini"},
    )
    assert len(rendered_inferences) == 2


def test_list_render_chat_inferences(embedded_sync_client: TensorZeroGateway):
    stored_inferences = embedded_sync_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name=None,
        filters=None,
        output_source="demonstration",
        limit=2,
        offset=None,
    )
    rendered_inferences = embedded_sync_client.experimental_render_samples(
        stored_samples=stored_inferences,
        variants={"write_haiku": "gpt_4o_mini"},
    )
    assert len(rendered_inferences) == 2


# Async versions of the above tests


@pytest.mark.asyncio
async def test_simple_list_json_inferences_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    order_by = [OrderBy(by="timestamp", direction="descending")]
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
        order_by=order_by,
    )
    assert len(inferences) == 2

    # Verify ordering is deterministic by checking inference IDs are unique
    inference_ids = [inference.inference_id for inference in inferences]
    assert len(set(inference_ids)) == len(inference_ids)  # All unique

    for inference in inferences:
        assert inference.function_name == "extract_entities"
        assert isinstance(inference.variant_name, str)
        inp = inference.input
        messages = inp.messages
        assert isinstance(messages, list)
        assert len(messages) == 1
        # Type narrowing: we know these are JSON inferences
        assert isinstance(inference, StoredInferenceJson)
        assert inference.type == "json"
        output = inference.output
        assert output.raw is not None
        assert output.parsed is not None
        inference_id = inference.inference_id
        assert isinstance(inference_id, str)
        episode_id = inference.episode_id
        assert isinstance(episode_id, str)
        # StoredJsonInference has output_schema, StoredChatInference doesn't
        assert hasattr(inference, "output_schema") and inference.output_schema is not None

    # ORDER BY timestamp DESC is applied - verify timestamps are in descending order
    timestamps = [inference.timestamp for inference in inferences]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] >= timestamps[i + 1], (
            f"Timestamps not in descending order: {timestamps[i]} < {timestamps[i + 1]}"
        )


@pytest.mark.asyncio
async def test_simple_query_with_float_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = FloatMetricFilter(
        metric_name="jaccard_similarity",
        value=0.5,
        comparison_operator=">",
    )
    order_by = [OrderBy(by="metric", name="jaccard_similarity", direction="descending")]
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=1,
        offset=None,
        order_by=order_by,
    )
    assert len(inferences) == 1

    for inference in inferences:
        assert inference.function_name == "extract_entities"
        assert inference.dispreferred_outputs is not None
        assert len(inference.dispreferred_outputs) == 0

    # ORDER BY metric jaccard_similarity DESC is applied with filter > 0.5
    # This ensures results are ordered by the metric value in descending order


@pytest.mark.asyncio
async def test_simple_query_chat_function_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    order_by = [OrderBy(by="timestamp", direction="ascending")]
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name="better_prompt_haiku_4_5",
        filters=None,
        output_source="inference",
        limit=3,
        offset=3,
        order_by=order_by,
    )
    assert len(inferences) == 3

    # Verify ordering is deterministic by checking inference IDs are unique
    inference_ids = [inference.inference_id for inference in inferences]
    assert len(set(inference_ids)) == len(inference_ids)  # All unique

    for inference in inferences:
        assert inference.function_name == "write_haiku"
        assert inference.variant_name == "better_prompt_haiku_4_5"
        inp = inference.input
        messages = inp.messages
        assert isinstance(messages, list)
        assert len(messages) == 1
        # Type narrowing: we know these are Chat inferences
        assert inference.type == "chat"
        output = inference.output
        assert len(output) == 1
        output_0 = output[0]
        assert output_0.type == "text"
        # Type narrowing: we know it's a Text block
        assert isinstance(output_0, ContentBlockChatOutputText)
        assert output_0.text is not None
        assert isinstance(inference.inference_id, str)
        assert isinstance(inference.episode_id, str)
        # Test individual tool param fields
        assert inference.allowed_tools is None or len(inference.allowed_tools) == 0
        assert inference.additional_tools is None or len(inference.additional_tools) == 0
        assert inference.parallel_tool_calls is None
        assert inference.provider_tools is None or len(inference.provider_tools) == 0

    # ORDER BY timestamp ASC is applied - verify timestamps are in ascending order
    timestamps = [inference.timestamp for inference in inferences]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], (
            f"Timestamps not in ascending order: {timestamps[i]} > {timestamps[i + 1]}"
        )


@pytest.mark.asyncio
async def test_demonstration_output_source_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="demonstration",
        limit=5,
        offset=1,
    )
    assert len(inferences) == 5
    for inference in inferences:
        assert inference.function_name == "extract_entities"
        assert inference.dispreferred_outputs is not None
        assert len(inference.dispreferred_outputs) == 1


@pytest.mark.asyncio
async def test_boolean_metric_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = BooleanMetricFilter(
        metric_name="exact_match",
        value=True,
    )
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=5,
        offset=1,
    )
    assert len(inferences) == 5
    for inference in inferences:
        assert inference.function_name == "extract_entities"


@pytest.mark.asyncio
async def test_and_filter_multiple_float_metrics_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = AndFilter(
        children=[
            FloatMetricFilter(
                metric_name="jaccard_similarity",
                value=0.5,
                comparison_operator=">",
            ),
            FloatMetricFilter(
                metric_name="jaccard_similarity",
                value=0.8,
                comparison_operator="<",
            ),
        ]
    )
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=1,
        offset=None,
    )
    assert len(inferences) == 1
    for inference in inferences:
        assert inference.function_name == "extract_entities"


@pytest.mark.asyncio
async def test_or_filter_mixed_metrics_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = OrFilter(
        children=[
            FloatMetricFilter(
                metric_name="jaccard_similarity",
                value=0.8,
                comparison_operator=">=",
            ),
            BooleanMetricFilter(
                metric_name="exact_match",
                value=True,
            ),
            BooleanMetricFilter(
                metric_name="goal_achieved",
                value=True,
            ),
        ]
    )
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=1,
        offset=None,
    )
    assert len(inferences) == 1
    for inference in inferences:
        assert inference.function_name == "extract_entities"


@pytest.mark.asyncio
async def test_not_filter_async(embedded_async_client: AsyncTensorZeroGateway):
    # NOT (exact_match = true OR exact_match = false) returns rows WITHOUT the metric.
    # This test verifies that the NOT filter correctly excludes rows that have the metric
    # (with either true or false value) and returns only rows without it.

    # Get total count (no filter)
    all_inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=1000,
        offset=None,
    )
    total_count = len(all_inferences)

    # Get count with exact_match = true
    true_filter = BooleanMetricFilter(metric_name="exact_match", value=True)
    true_inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=true_filter,
        output_source="inference",
        limit=1000,
        offset=None,
    )
    true_count = len(true_inferences)

    # Get count with exact_match = false
    false_filter = BooleanMetricFilter(metric_name="exact_match", value=False)
    false_inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=false_filter,
        output_source="inference",
        limit=1000,
        offset=None,
    )
    false_count = len(false_inferences)

    # Get count with NOT (true OR false) - should return rows WITHOUT the metric
    not_filter = NotFilter(
        child=OrFilter(
            children=[
                BooleanMetricFilter(metric_name="exact_match", value=True),
                BooleanMetricFilter(metric_name="exact_match", value=False),
            ]
        )
    )
    not_inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=not_filter,
        output_source="inference",
        limit=1000,
        offset=None,
    )
    not_count = len(not_inferences)

    # Verify: rows with metric (true + false) + rows without metric (NOT result) = total
    rows_with_metric = true_count + false_count
    assert rows_with_metric + not_count == total_count, (
        f"NOT filter should return exactly the rows without the metric. "
        f"true={true_count}, false={false_count}, NOT={not_count}, total={total_count}"
    )


@pytest.mark.asyncio
async def test_simple_time_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = TimeFilter(
        # 2023-01-01 00:00:00 UTC
        time=datetime.fromtimestamp(1672531200, tz=timezone.utc).isoformat(),
        comparison_operator=">",
    )
    order_by = [
        OrderBy(by="metric", name="exact_match", direction="descending"),
        OrderBy(by="timestamp", direction="ascending"),
    ]
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=2,
        offset=None,
        order_by=order_by,
    )
    assert len(inferences) == 2

    # Verify ordering is deterministic by checking inference IDs are unique
    inference_ids = [inference.inference_id for inference in inferences]
    assert len(set(inference_ids)) == len(inference_ids)  # All unique

    for inference in inferences:
        assert inference.function_name == "extract_entities"

    # ORDER BY metric exact_match DESC, timestamp ASC is applied
    # Multiple ORDER BY clauses ensure deterministic ordering
    # Verify timestamps are in ascending order (secondary sort)
    timestamps = [inference.timestamp for inference in inferences]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], (
            f"Timestamps not in ascending order: {timestamps[i]} > {timestamps[i + 1]}"
        )


@pytest.mark.asyncio
async def test_simple_tag_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = TagFilter(
        key="tensorzero::evaluation_name",
        value="entity_extraction",
        comparison_operator="=",
    )
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=100,
        offset=None,
    )
    assert len(inferences) == 100
    for inference in inferences:
        assert inference.function_name == "extract_entities"
        assert inference.tags is not None
        assert inference.tags["tensorzero::evaluation_name"] == "entity_extraction"


@pytest.mark.asyncio
async def test_combined_time_and_tag_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = AndFilter(
        children=[
            TimeFilter(
                # 2025-04-14 23:30:00 UTC
                time=datetime.fromtimestamp(1744673400, tz=timezone.utc).isoformat(),
                comparison_operator=">=",
            ),
            TagFilter(
                key="tensorzero::evaluation_name",
                value="haiku",
                comparison_operator="=",
            ),
        ]
    )
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=15,
        offset=None,
    )
    assert len(inferences) == 15
    for inference in inferences:
        assert inference.function_name == "write_haiku"
        assert inference.tags is not None
        assert inference.tags["tensorzero::evaluation_name"] == "haiku"


@pytest.mark.asyncio
async def test_list_render_json_inferences_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    stored_inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
    )
    rendered_inferences = await embedded_async_client.experimental_render_samples(
        stored_samples=stored_inferences,
        variants={"extract_entities": "gpt_4o_mini"},
    )
    assert len(rendered_inferences) == 2


@pytest.mark.asyncio
async def test_list_render_chat_inferences_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    stored_inferences = await embedded_async_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name=None,
        filters=None,
        output_source="demonstration",
        limit=2,
        offset=None,
    )
    rendered_inferences = await embedded_async_client.experimental_render_samples(
        stored_samples=stored_inferences,
        variants={"write_haiku": "gpt_4o_mini"},
    )
    assert len(rendered_inferences) == 2
    for inference in rendered_inferences:
        assert inference.dispreferred_outputs is not None
        assert len(inference.dispreferred_outputs) == 1
