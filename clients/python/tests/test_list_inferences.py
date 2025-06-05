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
        assert "raw" in output
        assert "parsed" in output
        assert output["raw"] is not None
        assert output["parsed"] is not None
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
        limit=2,
        offset=None,
    )
    assert len(inferences) == 2
