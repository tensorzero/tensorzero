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
        # assert isinstance(inference, StoredJsonInference)
        assert inference.function_name == "extract_entities"
        assert isinstance(inference.variant_name, str)
        input = inference.input
        print(input)
        assert False, "here"
