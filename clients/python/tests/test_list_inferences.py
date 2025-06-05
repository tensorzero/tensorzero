from uuid import UUID

from tensorzero import TensorZeroGateway


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
        assert isinstance(inference.variant_name, str)
        input = inference.input
        assert "messages" in input
        messages = input["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 1
        output = inference.output
        assert output.raw is not None
        assert output.parsed is not None
        inference_id = inference.inference_id
        assert isinstance(inference_id, UUID)
        episode_id = inference.episode_id
        assert isinstance(episode_id, UUID)
        tool_params = inference.tool_params
        assert tool_params is None
        output_schema = inference.output_schema
        assert output_schema is not None


def test_simple_query_with_float_filter(embedded_sync_client: TensorZeroGateway):
    filters = {
        "type": "float_metric",
        "metric_name": "jaccard_similarity",
        "value": 0.5,
        "comparison_operator": ">",
    }
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
        assert "messages" in input
        messages = input["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 1
        output = inference.output
        assert isinstance(output, list)
        assert len(output) == 1
        output_0 = output[0]
        assert output_0["type"] == "text"
        assert output_0["text"] is not None
        inference_id = inference.inference_id
        assert isinstance(inference_id, UUID)
        episode_id = inference.episode_id
        assert isinstance(episode_id, UUID)
        tool_params = inference.tool_params
        assert tool_params is not None
        assert tool_params["tools_available"] == []
        assert tool_params["tool_choice"] == "auto"
        assert tool_params["parallel_tool_calls"] is None
        output_schema = inference.output_schema
        assert output_schema is None


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
    filters = {
        "type": "boolean_metric",
        "metric_name": "exact_match",
        "value": True,
    }
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
    filters = {
        "type": "and",
        "children": [
            {
                "type": "float_metric",
                "metric_name": "jaccard_similarity",
                "value": 0.5,
                "comparison_operator": ">",
            },
            {
                "type": "float_metric",
                "metric_name": "jaccard_similarity",
                "value": 0.8,
                "comparison_operator": "<",
            },
        ],
    }
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
    filters = {
        "type": "or",
        "children": [
            {
                "type": "float_metric",
                "metric_name": "jaccard_similarity",
                "value": 0.8,
                "comparison_operator": ">=",
            },
            {
                "type": "boolean_metric",
                "metric_name": "exact_match",
                "value": True,
            },
            {
                "type": "boolean_metric",
                "metric_name": "goal_achieved",
                "value": True,
            },
        ],
    }
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
    filters = {
        "type": "not",
        "child": {
            "type": "or",
            "children": [
                {
                    "type": "boolean_metric",
                    "metric_name": "exact_match",
                    "value": True,
                },
                {
                    "type": "boolean_metric",
                    "metric_name": "exact_match",
                    "value": False,
                },
            ],
        },
    }
    inferences = embedded_sync_client.experimental_list_inferences(
        function_name="extract_entities",
        variant_name=None,
        filters=filters,
        output_source="inference",
        limit=None,
        offset=None,
    )
    assert len(inferences) == 0
