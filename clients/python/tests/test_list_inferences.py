from datetime import datetime, timezone
from uuid import UUID

import pytest
from tensorzero import (
    AndFilter,
    AsyncTensorZeroGateway,
    BooleanMetricFilter,
    FloatMetricFilter,
    NotFilter,
    OrderBy,
    OrFilter,
    TagFilter,
    TensorZeroGateway,
    Text,
    TimeFilter,
    ToolCall,
    ToolResult,
)
from tensorzero.tensorzero import StoredInference


def test_simple_list_json_inferences(embedded_sync_client: TensorZeroGateway):
    order_by = [OrderBy(by="timestamp", direction="DESC")]
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
        assert isinstance(inference, StoredInference.Json)
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
        assert isinstance(inference_id, UUID)
        episode_id = inference.episode_id
        assert isinstance(episode_id, UUID)
        output_schema = inference.output_schema
        assert output_schema is not None
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
    order_by = [OrderBy(by="metric", name="jaccard_similarity", direction="DESC")]
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
        assert len(inference.dispreferred_outputs) == 0

    # Since we aren't yet grabbing metric values from the DB we can't verify ordering by metric


def test_simple_query_chat_function(embedded_sync_client: TensorZeroGateway):
    order_by = [OrderBy(by="timestamp", direction="ASC")]
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name="better_prompt_haiku_3_5",
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
        assert inference.variant_name == "better_prompt_haiku_3_5"
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
        assert isinstance(output_0, Text)
        assert output_0.text is not None
        inference_id = inference.inference_id
        assert isinstance(inference_id, UUID)
        episode_id = inference.episode_id
        assert isinstance(episode_id, UUID)
        tool_params = inference.tool_params
        assert tool_params is not None
        assert tool_params.tools_available == []
        assert tool_params.parallel_tool_calls is None
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
        assert inference.variant_name == "baseline"
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
                    assert isinstance(content, ToolCall)
                    assert content.id is not None
                    assert content.name is not None
                    assert content.arguments is not None
                    assert content.raw_name is not None
                    assert content.raw_arguments is not None
                elif content.type == "tool_result":
                    assert isinstance(content, ToolResult)
                    assert content.id is not None
                    assert content.name is not None
                    assert content.result is not None
                elif content.type == "text":
                    assert isinstance(content, Text)
                    assert (content.text is not None) ^ (content.arguments is not None)
                else:
                    assert False

        # Type narrowing: we know these are Chat inferences
        assert inference.type == "chat"
        output = inference.output
        assert len(output) >= 1
        for output_item in output:
            if output_item.type == "text":
                assert isinstance(output_item, Text)
                assert output_item.text is not None
            elif output_item.type == "tool_call":
                assert isinstance(output_item, ToolCall)
                assert output_item.id is not None
                assert output_item.name is not None
                assert output_item.arguments is not None
                assert output_item.raw_name is not None
                assert output_item.raw_arguments is not None
            elif output_item.type == "tool_result":
                assert isinstance(output_item, ToolResult)
                assert output_item.id is not None
                assert output_item.name is not None
                assert output_item.result is not None
                print(output_item)
                assert False
        inference_id = inference.inference_id
        assert isinstance(inference_id, UUID)
        episode_id = inference.episode_id
        assert isinstance(episode_id, UUID)
        tool_params = inference.tool_params
        assert tool_params is not None
        assert len(tool_params.tools_available) == 4
        for tool in tool_params.tools_available:
            assert tool.name in [
                "think",
                "search_wikipedia",
                "load_wikipedia_page",
                "answer_question",
            ]
            assert tool.description is not None
            assert tool.parameters is not None
            assert tool.strict is True
        assert tool_params.parallel_tool_calls


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
    filters = NotFilter(
        child=OrFilter(
            children=[
                BooleanMetricFilter(
                    metric_name="exact_match",
                    value=True,
                ),
                BooleanMetricFilter(
                    metric_name="exact_match",
                    value=False,
                ),
            ]
        )
    )
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=None,
        offset=None,
    )
    assert len(inferences) == 0


def test_simple_time_filter(embedded_sync_client: TensorZeroGateway):
    filters = TimeFilter(
        time=datetime.fromtimestamp(
            1672531200, tz=timezone.utc
        ).isoformat(),  # 2023-01-01 00:00:00 UTC
        comparison_operator=">",
    )
    order_by = [
        OrderBy(by="metric", name="exact_match", direction="DESC"),
        OrderBy(by="timestamp", direction="ASC"),
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
    order_by = [OrderBy(by="timestamp", direction="DESC")]
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
        assert inference.type == "json"
        output = inference.output
        assert output.raw is not None
        assert output.parsed is not None
        inference_id = inference.inference_id
        assert isinstance(inference_id, UUID)
        episode_id = inference.episode_id
        assert isinstance(episode_id, UUID)
        # StoredJsonInference has output_schema, StoredChatInference doesn't
        assert (
            hasattr(inference, "output_schema") and inference.output_schema is not None
        )

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
    order_by = [OrderBy(by="metric", name="jaccard_similarity", direction="DESC")]
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
        assert len(inference.dispreferred_outputs) == 0

    # ORDER BY metric jaccard_similarity DESC is applied with filter > 0.5
    # This ensures results are ordered by the metric value in descending order


@pytest.mark.asyncio
async def test_simple_query_chat_function_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    order_by = [OrderBy(by="timestamp", direction="ASC")]
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name="better_prompt_haiku_3_5",
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
        assert inference.variant_name == "better_prompt_haiku_3_5"
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
        assert isinstance(output_0, Text)
        assert output_0.text is not None
        assert isinstance(inference.inference_id, UUID)
        assert isinstance(inference.episode_id, UUID)
        tp = inference.tool_params
        assert tp is not None
        assert tp.tools_available == []
        assert tp.parallel_tool_calls is None

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
    filters = NotFilter(
        child=OrFilter(
            children=[
                BooleanMetricFilter(
                    metric_name="exact_match",
                    value=True,
                ),
                BooleanMetricFilter(
                    metric_name="exact_match",
                    value=False,
                ),
            ]
        )
    )
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=None,
        offset=None,
    )
    assert len(inferences) == 0


@pytest.mark.asyncio
async def test_simple_time_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = TimeFilter(
        time=datetime.fromtimestamp(
            1672531200, tz=timezone.utc
        ).isoformat(),  # 2023-01-01 00:00:00 UTC
        comparison_operator=">",
    )
    order_by = [
        OrderBy(by="metric", name="exact_match", direction="DESC"),
        OrderBy(by="timestamp", direction="ASC"),
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
        assert len(inference.dispreferred_outputs) == 1
