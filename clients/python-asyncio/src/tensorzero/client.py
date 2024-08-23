import json
from uuid import UUID
import httpx
from typing import AsyncGenerator, Dict, Any, List, Optional, Union

class TensorZeroClient:
    def __init__(self, base_url: str):
        """
        Initialize the TensorZero client.

        :param base_url: The base URL of the TensorZero server. Should be something like "http://localhost:3000"
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def inference(
        self,
        function_name: str,
        episode_id: UUID,
        input: Dict[str, Any],
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request to the /inference endpoint.

        :param function_name: The name of the function to call
        :param episode_id: The episode ID to use for the request
                           (only use episode_ids that were returned by TensorZero previously)
        :param input: The input to the function {"system": str, "messages": List[{"role": "user" | "assistant", "content": Any}]}
                      this will be validated server side against the input schema of the function being called.
        :param stream: Whether to stream the response
        :param params: Override inference-time parameters for a particular variant type. Currently, we support
                        {"chat_completion": {"temperature": float, "max_tokens": int, "seed": int}}
        :param variant_name: The variant name to use for the request
        :param dryrun: If true, the request will be executed but nothing will be stored
        :param allowed_tools: A list of allowed tools to use for the request. Should be a subset of the tools configured for that function.
        :param additional_tools: A list of additional tools to use for the request. Each element should look like {"name": str, "parameters": valid JSON Schema, "description": str}
        :param tool_choice: Runtime override for what tool to use. Should be one of "none", "auto", "required", or {"tool": str}, where str is a valid tool name.
        :param parallel_tool_calls: If true, the request will be executed with parallel tool calls allowed.
        :return: If stream is false, returns TODO.
                 If stream is true, returns an async generator that yields TODO as they come in.
        """
        url = f"{self.base_url}/inference"
        data = {
            "function_name": function_name,
            "episode_id": episode_id,
            "input": input,
            "stream": stream,
            "params": params,
            "variant_name": variant_name,
            "dryrun": dryrun,
            "allowed_tools": allowed_tools,
            "additional_tools": additional_tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
        response = await self.client.post(url, json=data)
        response.raise_for_status()
        if not stream:
            return response.json()
        else:
            return self._stream_sse(response)

    async def feedback(
        self,
        metric_name: str,
        value: Any,
        episode_id: Optional[UUID] = None,
        inference_id: Optional[UUID] = None,
        dryrun: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Make a POST request to the /feedback endpoint.

        :param data: The feedback data to send in the request body
        :return: The response from the server
        """
        if episode_id is None and inference_id is None:
            raise ValueError("Either episode_id or inference_id must be provided")
        if episode_id is not None and inference_id is not None:
            raise ValueError("Only one of episode_id or inference_id can be provided, not both")
        data = {
            "metric_name": metric_name,
            "value": value,
            "episode_id": episode_id,
            "inference_id": inference_id,
            "dryrun": dryrun,
        }
        url = f"{self.base_url}/feedback"
        response = await self.client.post(url, json=data)
        response.raise_for_status()
        return response.json()

    async def close(self):
        """
        Close the httpx client session.
        """
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _stream_sse(self, response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Parse the SSE stream from the response.

        :param response: The httpx.Response object
        :yield: Parsed SSE events as dictionaries
        """
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                data = line[6:].strip()
                if data == '[DONE]':
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    print(f"Failed to parse SSE data: {data}")
