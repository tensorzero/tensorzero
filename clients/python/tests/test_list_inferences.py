from uuid import UUID

import pytest
from tensorzero import (
    AndNode,
    AsyncTensorZeroGateway,
    BooleanMetricNode,
    FloatMetricNode,
    NotNode,
    OrNode,
    TensorZeroGateway,
    Text,
    ToolCall,
    ToolResult,
)
from tensorzero.tensorzero import StoredInference


def test_simple_list_json_inferences(embedded_sync_client: TensorZeroGateway):
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
    )
    assert len(inferences) == 2
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


def test_simple_query_with_float_filter(embedded_sync_client: TensorZeroGateway):
    filters = FloatMetricNode(
        metric_name="jaccard_similarity",
        value=0.5,
        comparison_operator=">",
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


def test_simple_query_chat_function(embedded_sync_client: TensorZeroGateway):
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name="better_prompt_haiku_3_5",
        filters=None,
        output_source="inference",
        limit=3,
        offset=3,
    )
    assert len(inferences) == 3
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


def test_boolean_metric_filter(embedded_sync_client: TensorZeroGateway):
    filters = BooleanMetricNode(
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
    filters = AndNode(
        children=[
            FloatMetricNode(
                metric_name="jaccard_similarity",
                value=0.5,
                comparison_operator=">",
            ),
            FloatMetricNode(
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
    filters = OrNode(
        children=[
            FloatMetricNode(
                metric_name="jaccard_similarity",
                value=0.8,
                comparison_operator=">=",
            ),
            BooleanMetricNode(
                metric_name="exact_match",
                value=True,
            ),
            BooleanMetricNode(
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
    filters = NotNode(
        child=OrNode(
            children=[
                BooleanMetricNode(
                    metric_name="exact_match",
                    value=True,
                ),
                BooleanMetricNode(
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


def test_list_render_json_inferences(embedded_sync_client: TensorZeroGateway):
    stored_inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
    )
    rendered_inferences = embedded_sync_client.experimental_render_inferences(
        stored_inferences=stored_inferences,
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
    rendered_inferences = embedded_sync_client.experimental_render_inferences(
        stored_inferences=stored_inferences,
        variants={"write_haiku": "gpt_4o_mini"},
    )
    assert len(rendered_inferences) == 2


# Async versions of the above tests


@pytest.mark.asyncio
async def test_simple_list_json_inferences_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=None,
        output_source="inference",
        limit=2,
        offset=None,
    )
    assert len(inferences) == 2
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


@pytest.mark.asyncio
async def test_simple_query_with_float_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = FloatMetricNode(
        metric_name="jaccard_similarity",
        value=0.5,
        comparison_operator=">",
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
async def test_simple_query_chat_function_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    inferences = await embedded_async_client.experimental_list_inferences(
        function_name="write_haiku",
        variant_name="better_prompt_haiku_3_5",
        filters=None,
        output_source="inference",
        limit=3,
        offset=3,
    )
    assert len(inferences) == 3
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


@pytest.mark.asyncio
async def test_boolean_metric_filter_async(
    embedded_async_client: AsyncTensorZeroGateway,
):
    filters = BooleanMetricNode(
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
    filters = AndNode(
        children=[
            FloatMetricNode(
                metric_name="jaccard_similarity",
                value=0.5,
                comparison_operator=">",
            ),
            FloatMetricNode(
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
    filters = OrNode(
        children=[
            FloatMetricNode(
                metric_name="jaccard_similarity",
                value=0.8,
                comparison_operator=">=",
            ),
            BooleanMetricNode(
                metric_name="exact_match",
                value=True,
            ),
            BooleanMetricNode(
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
    filters = NotNode(
        child=OrNode(
            children=[
                BooleanMetricNode(
                    metric_name="exact_match",
                    value=True,
                ),
                BooleanMetricNode(
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
    rendered_inferences = await embedded_async_client.experimental_render_inferences(
        stored_inferences=stored_inferences,
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
    rendered_inferences = await embedded_async_client.experimental_render_inferences(
        stored_inferences=stored_inferences,
        variants={"write_haiku": "gpt_4o_mini"},
    )
    assert len(rendered_inferences) == 2
