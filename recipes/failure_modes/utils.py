import asyncio
import json
from typing import Any, Dict, List

from tensorzero import AsyncTensorZeroGateway, JsonInferenceResponse, RenderedSample


async def generate_root_causes(
    gateway: AsyncTensorZeroGateway,
    rendered_sample: RenderedSample,
    variant_name: str,
    semaphore: asyncio.Semaphore,
    dryrun: bool = False,
) -> List[str]:
    """
    Generate root causes for LLM failures using a RenderedSample.

    Args:
        gateway: TensorZero gateway instance
        rendered_sample: The rendered sample containing the complete inference data
        variant_name: Name of the variant to use for root cause analysis
        semaphore: Semaphore for rate limiting
        dryrun: Whether to run in dryrun mode

    Returns:
        List of root cause strings
    """
    try:
        # Use the built-in JSON serialization through __repr__
        # This leverages the Rust serde serialization which is guaranteed to be complete
        sample_json = str(rendered_sample)  # This calls __repr__ which uses serde_json
        sample_dict = json.loads(sample_json)

        gateway_input: Dict[str, Any] = {
            "system": {},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "arguments": sample_dict}],
                }
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
