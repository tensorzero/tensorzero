import asyncio
from typing import Any, Dict, List

from tensorzero import AsyncTensorZeroGateway, JsonInferenceResponse


async def generate_root_causes(
    gateway: AsyncTensorZeroGateway,
    row: Dict[str, Any],
    variant_name: str,
    metric_name: str,
    tools_available: List[Dict[str, str]],
    semaphore: asyncio.Semaphore,
    dryrun: bool = False,
) -> List[str]:
    try:
        arguments = {
            "messages": row["rendered_input"]["messages"],
            "feedback": [{"name": metric_name, "value": row["value"]}],
        }
        if row["rendered_input"].get("system"):
            arguments["system_message"] = row["rendered_input"]["system"]
        if len(tools_available) > 0:
            arguments["tools_available"] = tools_available
        gateway_input: Dict[str, Any] = {
            "system": {},
            "messages": [
                {"role": "user", "content": [{"type": "text", "arguments": arguments}]}
            ],
        }
        async with semaphore:
            response = await gateway.inference(
                input=gateway_input,
                function_name="root_cause_analysis",
                variant_name=variant_name,
                dryrun=dryrun,
            )
            assert isinstance(response, JsonInferenceResponse)
            assert response.output.parsed is not None
            return response.output.parsed["root_causes"]
    except Exception as e:
        print(f"Error generating root causes: {e}")
        return [""]
