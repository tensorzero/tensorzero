from tensorzero import (
    ChatCompletionConfig,
    FunctionConfigChat,
    FunctionConfigJson,
    TensorZeroGateway,
)


def test_config(
    embedded_sync_client: TensorZeroGateway,
):
    config = embedded_sync_client.experimental_get_config()
    basic_test = config.functions["basic_test"]
    assert isinstance(basic_test, FunctionConfigChat)
    assert basic_test.system_schema is not None
    assert basic_test.user_schema is None
    extract_entities = config.functions["extract_entities"]
    assert isinstance(extract_entities, FunctionConfigJson)
    assert extract_entities.output_schema is not None
    assert "person" in extract_entities.output_schema["properties"]
    variant = basic_test.variants["test"]
    assert isinstance(variant, ChatCompletionConfig)
    assert variant.system_template is not None
    system_template = variant.system_template
    assert (
        system_template
        == "You are a helpful and friendly {% include 'extra_templates/foo.minijinja' %} named {{ assistant_name }}\n"
    )
    assert variant.user_template is None
    assert variant.assistant_template is None
    assert variant.model == "test"
