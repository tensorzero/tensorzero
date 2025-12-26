from tensorzero import InferenceFilterTag, InferenceResponse, ListInferencesRequest, TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")


# 1. Make an inference with a tag
response = t0.inference(
    model_name="openai::gpt-5-mini",
    input={"messages": [{"role": "user", "content": "Write a haiku about TensorZero."}]},
    tags={"my_tag": "my_value"},  # for filtering later
)
assert isinstance(response, InferenceResponse)
inference_id = str(response.inference_id)
print(f"Completed Inference: {inference_id}")


# 2. Query the inference by ID
get_response = t0.get_inferences(
    ids=[inference_id],
    output_source="inference",
)
print(f"Retrieved {len(get_response.inferences)} inference(s) by ID")


# 3. List inferences filtered by the tag
list_response = t0.list_inferences(
    request=ListInferencesRequest(
        output_source="inference",
        filters=InferenceFilterTag(
            key="my_tag",
            value="my_value",
            comparison_operator="=",
        ),
        limit=10,
    )
)
print(f"Found {len(list_response.inferences)} inference(s) with tag my_tag=my_value")
