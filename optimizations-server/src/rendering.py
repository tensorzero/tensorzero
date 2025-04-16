from minijinja import Environment
from tensorzero.tensorzero import AsyncTensorZeroGateway


def get_template_env(
    client: AsyncTensorZeroGateway, function_name: str, variant_name: str
) -> Environment:
    templates = client._internal_get_template_config(
        function_name=function_name, variant_name=variant_name
    )
    return Environment(templates=templates)
