"""
TensorZero Client

This module provides synchronous and asynchronous clients for interacting with the TensorZero gateway.
It includes functionality for making inference requests and sending feedback.

The main classes, TensorZeroGateway and AsyncTensorZeroGateway, offer methods for:
- Initializing the client with a base URL
- Making inference requests (with optional streaming)
- Sending feedback on episodes or inferences
- Managing the client session using async context managers

Usage:
    with TensorZeroGateway(base_url) as client:
        response = client.inference(...)
        feedback = client.feedback(...)

    async with AsyncTensorZeroGateway(base_url) as client:
        response = await client.inference(...)
        feedback = await client.feedback(...)

"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generator, List, Literal, Optional, Union
from urllib.parse import urljoin
from uuid import UUID

import httpx

from .types import (
    FeedbackResponse,
    InferenceChunk,
    InferenceResponse,
    TensorZeroError,
    parse_inference_chunk,
    parse_inference_response,
)


class BaseTensorZeroGateway(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    def _prepare_inference_request(
        self,
        function_name: str,
        input: Dict[str, Any],
        episode_id: Optional[UUID] = None,
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[
            Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]
        ] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # Convert content blocks to dicts if necessary
        for message in input.get("messages", []):
            if isinstance(message["content"], list):
                message["content"] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in message["content"]
                ]

        data = {
            "function_name": function_name,
            "input": input,
        }
        if episode_id is not None:
            data["episode_id"] = str(episode_id)
        if stream is not None:
            data["stream"] = stream
        if params is not None:
            data["params"] = params
        if variant_name is not None:
            data["variant_name"] = variant_name
        if dryrun is not None:
            data["dryrun"] = dryrun
        if allowed_tools is not None:
            data["allowed_tools"] = allowed_tools
        if additional_tools is not None:
            data["additional_tools"] = additional_tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            data["parallel_tool_calls"] = parallel_tool_calls
        return data

    def _prepare_feedback_request(
        self,
        metric_name: str,
        value: Any,
        inference_id: Optional[UUID] = None,
        episode_id: Optional[UUID] = None,
        dryrun: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if episode_id is None and inference_id is None:
            raise ValueError("Either episode_id or inference_id must be provided")
        if episode_id is not None and inference_id is not None:
            raise ValueError(
                "Only one of episode_id or inference_id can be provided, not both"
            )
        data = {
            "metric_name": metric_name,
            "value": value,
        }
        if dryrun is not None:
            data["dryrun"] = dryrun
        if episode_id is not None:
            data["episode_id"] = str(episode_id)
        if inference_id is not None:
            data["inference_id"] = str(inference_id)
        return data

    @abstractmethod
    def inference(
        self,
        *,
        function_name: str,
        input: Dict[str, Any],
        episode_id: Optional[UUID] = None,
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[
            Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]
        ] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Union[InferenceResponse, Generator[InferenceChunk, None, None]]:
        pass

    @abstractmethod
    def feedback(
        self,
        *,
        metric_name: str,
        value: Any,
        inference_id: Optional[UUID] = None,
        episode_id: Optional[UUID] = None,
        dryrun: Optional[bool] = None,
    ) -> Dict[str, Any]:
        pass


class TensorZeroGateway(BaseTensorZeroGateway):
    def __init__(self, base_url: str, *, timeout: Optional[float] = None):
        """
        Initialize the TensorZero client.

        :param base_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
        """
        super().__init__(base_url)
        self.client = httpx.Client(timeout=timeout)

    def inference(
        self,
        *,
        function_name: str,
        input: Dict[str, Any],
        episode_id: Optional[UUID] = None,
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[
            Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]
        ] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Union[InferenceResponse, Generator[InferenceChunk, None, None]]:
        """
        Make a POST request to the /inference endpoint.

        :param function_name: The name of the function to call
        :param input: The input to the function
                      Structure: {"system": Optional[str], "messages": List[{"role": "user" | "assistant", "content": Any}]}
                      The input will be validated server side against the input schema of the function being called.
        :param episode_id: The episode ID to use for the inference.
                           If this is the first inference in an episode, leave this field blank. The TensorZero gateway will generate and return a new episode ID.
                           Note: Only use episode IDs generated by the TensorZero gateway. Don't generate them yourself.
        :param stream: If set, the TensorZero gateway will stream partial message deltas (e.g. generated tokens) as it receives them from model providers.
        :param params: Override inference-time parameters for a particular variant type. Currently, we support:
                        {"chat_completion": {"temperature": float, "max_tokens": int, "seed": int}}
        :param variant_name: If set, pins the inference request to a particular variant.
                             Note: You should generally not do this, and instead let the TensorZero gateway assign a
                             particular variant. This field is primarily used for testing or debugging purposes.
        :param dryrun: If true, the request will be executed but won't be stored to the database.
        :param allowed_tools: If set, restricts the tools available during this inference request.
                              The list of names should be a subset of the tools configured for the function.
                              Tools provided at inference time in `additional_tools` (if any) are always available.
        :param additional_tools: A list of additional tools to use for the request. Each element should look like {"name": str, "parameters": valid JSON Schema, "description": str}
        :param tool_choice: If set, overrides the tool choice strategy for the request.
                            It should be one of: "auto", "required", "off", or {"specific": str}. The last option pins the request to a specific tool name.
        :param parallel_tool_calls: If true, the request will allow for multiple tool calls in a single inference request.
        :return: If stream is false, returns an InferenceResponse.
                 If stream is true, returns an async generator that yields InferenceChunks as they come in.
        """
        url = urljoin(self.base_url, "inference")
        data = self._prepare_inference_request(
            function_name,
            input,
            episode_id,
            stream,
            params,
            variant_name,
            dryrun,
            allowed_tools,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
        )
        if stream:
            req = self.client.build_request("POST", url, json=data)
            response = self.client.send(req, stream=True)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise TensorZeroError(response) from e
            return self._stream_sse(response)
        else:
            response = self.client.post(url, json=data)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise TensorZeroError(response) from e
            return parse_inference_response(response.json())

    def feedback(
        self,
        *,
        metric_name: str,
        value: Any,
        inference_id: Optional[UUID] = None,
        episode_id: Optional[UUID] = None,
        dryrun: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request to the /feedback endpoint.

        :param metric_name: The name of the metric to provide feedback for
        :param value: The value of the feedback. It should correspond to the metric type.
        :param inference_id: The inference ID to assign the feedback to.
                             Only use inference IDs that were returned by the TensorZero gateway.
                             Note: You can assign feedback to either an episode or an inference, but not both.
        :param episode_id: The episode ID to use for the request
                           Only use episode IDs that were returned by the TensorZero gateway.
                           Note: You can assign feedback to either an episode or an inference, but not both.
        :param dryrun: If true, the feedback request will be executed but won't be stored to the database (i.e. no-op).
        :return: {"feedback_id": str}
        """
        url = urljoin(self.base_url, "feedback")
        data = self._prepare_feedback_request(
            metric_name, value, inference_id, episode_id, dryrun
        )
        response = self.client.post(url, json=data)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TensorZeroError(response) from e
        feedback_result = FeedbackResponse(**response.json())
        return feedback_result

    def close(self):
        """
        Close the connection to the TensorZero gateway.
        """
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _stream_sse(
        self, response: httpx.Response
    ) -> Generator[InferenceChunk, None, None]:
        """
        Parse the SSE stream from the response.

        :param response: The httpx.Response object
        :yield: Parsed SSE events as dictionaries
        """
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    data = json.loads(data)
                    yield parse_inference_chunk(data)
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse SSE data: {data}")
        response.close()


class AsyncTensorZeroGateway(BaseTensorZeroGateway):
    def __init__(self, base_url: str, *, timeout: Optional[float] = None):
        """
        Initialize the TensorZero client.

        :param base_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
        """
        super().__init__(base_url)
        self.client = httpx.AsyncClient(timeout=timeout)

    async def inference(
        self,
        *,
        function_name: str,
        input: Dict[str, Any],
        episode_id: Optional[UUID] = None,
        stream: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
        dryrun: Optional[bool] = None,
        allowed_tools: Optional[List[str]] = None,
        additional_tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[
            Union[Literal["auto", "required", "off"], Dict[Literal["specific"], str]]
        ] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Union[InferenceResponse, AsyncGenerator[InferenceChunk, None]]:
        """
        Make a POST request to the /inference endpoint.

        :param function_name: The name of the function to call
        :param input: The input to the function
                      Structure: {"system": Optional[str], "messages": List[{"role": "user" | "assistant", "content": Any}]}
                      The input will be validated server side against the input schema of the function being called.
        :param episode_id: The episode ID to use for the inference.
                           If this is the first inference in an episode, leave this field blank. The TensorZero gateway will generate and return a new episode ID.
                           Note: Only use episode IDs generated by the TensorZero gateway. Don't generate them yourself.
        :param stream: If set, the TensorZero gateway will stream partial message deltas (e.g. generated tokens) as it receives them from model providers.
        :param params: Override inference-time parameters for a particular variant type. Currently, we support:
                        {"chat_completion": {"temperature": float, "max_tokens": int, "seed": int}}
        :param variant_name: If set, pins the inference request to a particular variant.
                             Note: You should generally not do this, and instead let the TensorZero gateway assign a
                             particular variant. This field is primarily used for testing or debugging purposes.
        :param dryrun: If true, the request will be executed but won't be stored to the database.
        :param allowed_tools: If set, restricts the tools available during this inference request.
                              The list of names should be a subset of the tools configured for the function.
                              Tools provided at inference time in `additional_tools` (if any) are always available.
        :param additional_tools: A list of additional tools to use for the request. Each element should look like {"name": str, "parameters": valid JSON Schema, "description": str}
        :param tool_choice: If set, overrides the tool choice strategy for the request.
                            It should be one of: "auto", "required", "off", or {"specific": str}. The last option pins the request to a specific tool name.
        :param parallel_tool_calls: If true, the request will allow for multiple tool calls in a single inference request.
        :return: If stream is false, returns an InferenceResponse.
                 If stream is true, returns an async generator that yields InferenceChunks as they come in.
        """
        url = urljoin(self.base_url, "inference")
        data = self._prepare_inference_request(
            function_name,
            input,
            episode_id,
            stream,
            params,
            variant_name,
            dryrun,
            allowed_tools,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
        )
        if stream:
            req = self.client.build_request("POST", url, json=data)
            response = await self.client.send(req, stream=True)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise TensorZeroError(response) from e
            return self._stream_sse(response)
        else:
            response = await self.client.post(url, json=data)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise TensorZeroError(response) from e
            return parse_inference_response(response.json())

    async def feedback(
        self,
        *,
        metric_name: str,
        value: Any,
        inference_id: Optional[UUID] = None,
        episode_id: Optional[UUID] = None,
        dryrun: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request to the /feedback endpoint.

        :param metric_name: The name of the metric to provide feedback for
        :param value: The value of the feedback. It should correspond to the metric type.
        :param inference_id: The inference ID to assign the feedback to.
                             Only use inference IDs that were returned by the TensorZero gateway.
                             Note: You can assign feedback to either an episode or an inference, but not both.
        :param episode_id: The episode ID to use for the request
                           Only use episode IDs that were returned by the TensorZero gateway.
                           Note: You can assign feedback to either an episode or an inference, but not both.
        :param dryrun: If true, the feedback request will be executed but won't be stored to the database (i.e. no-op).
        :return: {"feedback_id": str}
        """
        url = urljoin(self.base_url, "feedback")
        data = self._prepare_feedback_request(
            metric_name, value, inference_id, episode_id, dryrun
        )
        response = await self.client.post(url, json=data)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TensorZeroError(response) from e
        feedback_result = FeedbackResponse(**response.json())
        return feedback_result

    async def close(self):
        """
        Close the connection to the TensorZero gateway.
        """
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _stream_sse(
        self, response: httpx.Response
    ) -> AsyncGenerator[InferenceChunk, None]:
        """
        Parse the SSE stream from the response.

        :param response: The httpx.Response object
        :yield: Parsed SSE events as dictionaries
        """
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    data = json.loads(data)
                    yield parse_inference_chunk(data)
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse SSE data: {data}")
        await response.aclose()
