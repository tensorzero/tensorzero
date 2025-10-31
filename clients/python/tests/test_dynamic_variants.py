"""
Tests for dynamic variant configuration

These tests verify that the `internal_dynamic_variant_config` parameter works correctly
for different variant types (chat_completion, mixture_of_n, best_of_n_sampling).

Based on the Rust e2e tests in tensorzero-core/tests/e2e/dynamic_variants.rs

Tests run in both HTTP gateway mode and embedded gateway mode via parameterized fixtures.
"""

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    TensorZeroError,
    TensorZeroGateway,
    Text,
)
from uuid_utils import uuid7


def test_dynamic_chat_variant(sync_client: TensorZeroGateway):
    """Test dynamic chat completion variant with custom system template"""
    episode_id = uuid7()
    input_data = {
        "system": {"assistant_name": "AskJeeves"},
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    # Test that it fails without dryrun
    with pytest.raises(TensorZeroError) as exc_info:
        sync_client.inference(
            function_name="basic_test",
            episode_id=episode_id,
            input=input_data,
            internal_dynamic_variant_config={
                "type": "chat_completion",
                "weight": 0.0,
                "model": "dummy::echo_request_messages",
                "system_template": {
                    "__tensorzero_remapped_path": "system",
                    "__data": "You are a cranky assistant named {{ assistant_name }}",
                },
            },
            stream=False,
        )

    assert exc_info.value.status_code == 400

    # Test that it succeeds with dryrun=True
    response = sync_client.inference(
        function_name="basic_test",
        episode_id=episode_id,
        input=input_data,
        internal_dynamic_variant_config={
            "type": "chat_completion",
            "weight": 0.0,
            "model": "dummy::echo_request_messages",
            "system_template": {
                "__tensorzero_remapped_path": "system",
                "__data": "You are a cranky assistant named {{ assistant_name }}",
            },
        },
        stream=False,
        dryrun=True,
    )

    assert isinstance(response, ChatInferenceResponse)
    assert response.inference_id is not None

    content = response.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    text = content[0].text
    assert text is not None
    assert "You are a cranky assistant named AskJeeves" in text


def test_dynamic_mixture_of_n(sync_client: TensorZeroGateway):
    """Test dynamic mixture_of_n variant with custom fuser"""
    episode_id = uuid7()
    input_data = {
        "system": {"assistant_name": "Alfred"},
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    # Test that it fails without dryrun
    with pytest.raises(TensorZeroError) as exc_info:
        sync_client.inference(
            function_name="basic_test",
            episode_id=episode_id,
            input=input_data,
            internal_dynamic_variant_config={
                "type": "experimental_mixture_of_n",
                "weight": 0.0,
                "candidates": ["test", "test2"],
                "fuser": {
                    "weight": 0.0,
                    "model": "dummy::echo_request_messages",
                    "system_template": {
                        "__tensorzero_remapped_path": "system",
                        "__data": "be mean {{ assistant_name }}",
                    },
                },
            },
            stream=False,
        )

    assert exc_info.value.status_code == 400

    # Test that it succeeds with dryrun=True
    response = sync_client.inference(
        function_name="basic_test",
        episode_id=episode_id,
        input=input_data,
        internal_dynamic_variant_config={
            "type": "experimental_mixture_of_n",
            "weight": 0.0,
            "candidates": ["test", "test2"],
            "fuser": {
                "weight": 0.0,
                "model": "dummy::echo_request_messages",
                "system_template": {
                    "__tensorzero_remapped_path": "system",
                    "__data": "be mean {{ assistant_name }}",
                },
            },
        },
        stream=False,
        dryrun=True,
    )

    assert isinstance(response, ChatInferenceResponse)
    assert response.inference_id is not None

    content = response.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    text = content[0].text
    assert text is not None
    assert "be mean Alfred" in text
    assert "You have been provided with a set of responses" in text
    assert "synthesize these responses into" in text
    assert "gleefully chanted" in text


def test_dynamic_best_of_n(sync_client: TensorZeroGateway):
    """Test dynamic best_of_n_sampling variant with custom evaluator"""
    episode_id = uuid7()
    input_data = {
        "system": {"assistant_name": "Watson"},
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    # Test that it fails without dryrun
    with pytest.raises(TensorZeroError) as exc_info:
        sync_client.inference(
            function_name="basic_test",
            episode_id=episode_id,
            input=input_data,
            internal_dynamic_variant_config={
                "type": "experimental_best_of_n_sampling",
                "weight": 0.0,
                "candidates": ["test", "test2"],
                "evaluator": {
                    "weight": 0.0,
                    "model": "dummy::echo_request_messages",
                    "system_template": {
                        "__tensorzero_remapped_path": "system",
                        "__data": "be mean {{ assistant_name }}",
                    },
                },
            },
            stream=False,
        )

    assert exc_info.value.status_code == 400

    # Test that it succeeds with dryrun=True
    response = sync_client.inference(
        function_name="basic_test",
        episode_id=episode_id,
        input=input_data,
        internal_dynamic_variant_config={
            "type": "experimental_best_of_n_sampling",
            "weight": 0.0,
            "candidates": ["test", "test2"],
            "evaluator": {
                "weight": 0.0,
                "model": "dummy::echo_request_messages",
                "system_template": {
                    "__tensorzero_remapped_path": "system",
                    "__data": "be mean {{ assistant_name }}",
                },
            },
        },
        stream=False,
        dryrun=True,
    )

    assert isinstance(response, ChatInferenceResponse)
    assert response.inference_id is not None

    content = response.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    text = content[0].text
    assert text is not None
    # The best_of_n always picks a candidate
    assert "gleefully chanted" in text


@pytest.mark.asyncio
async def test_async_dynamic_chat_variant(async_client: AsyncTensorZeroGateway):
    """Test dynamic chat completion variant with custom system template (async)"""
    episode_id = uuid7()
    input_data = {
        "system": {"assistant_name": "AskJeeves"},
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    # Test that it fails without dryrun
    with pytest.raises(TensorZeroError) as exc_info:
        await async_client.inference(
            function_name="basic_test",
            episode_id=episode_id,
            input=input_data,
            internal_dynamic_variant_config={
                "type": "chat_completion",
                "weight": 0.0,
                "model": "dummy::echo_request_messages",
                "system_template": {
                    "__tensorzero_remapped_path": "system",
                    "__data": "You are a cranky assistant named {{ assistant_name }}",
                },
            },
            stream=False,
        )

    assert exc_info.value.status_code == 400

    # Test that it succeeds with dryrun=True
    response = await async_client.inference(
        function_name="basic_test",
        episode_id=episode_id,
        input=input_data,
        internal_dynamic_variant_config={
            "type": "chat_completion",
            "weight": 0.0,
            "model": "dummy::echo_request_messages",
            "system_template": {
                "__tensorzero_remapped_path": "system",
                "__data": "You are a cranky assistant named {{ assistant_name }}",
            },
        },
        stream=False,
        dryrun=True,
    )

    assert isinstance(response, ChatInferenceResponse)
    assert response.inference_id is not None

    content = response.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    text = content[0].text
    assert text is not None
    assert "You are a cranky assistant named AskJeeves" in text


@pytest.mark.asyncio
async def test_async_dynamic_mixture_of_n(async_client: AsyncTensorZeroGateway):
    """Test dynamic mixture_of_n variant with custom fuser (async)"""
    episode_id = uuid7()
    input_data = {
        "system": {"assistant_name": "Alfred"},
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    # Test that it fails without dryrun
    with pytest.raises(TensorZeroError) as exc_info:
        await async_client.inference(
            function_name="basic_test",
            episode_id=episode_id,
            input=input_data,
            internal_dynamic_variant_config={
                "type": "experimental_mixture_of_n",
                "weight": 0.0,
                "candidates": ["test", "test2"],
                "fuser": {
                    "weight": 0.0,
                    "model": "dummy::echo_request_messages",
                    "system_template": {
                        "__tensorzero_remapped_path": "system",
                        "__data": "be mean {{ assistant_name }}",
                    },
                },
            },
            stream=False,
        )

    assert exc_info.value.status_code == 400

    # Test that it succeeds with dryrun=True
    response = await async_client.inference(
        function_name="basic_test",
        episode_id=episode_id,
        input=input_data,
        internal_dynamic_variant_config={
            "type": "experimental_mixture_of_n",
            "weight": 0.0,
            "candidates": ["test", "test2"],
            "fuser": {
                "weight": 0.0,
                "model": "dummy::echo_request_messages",
                "system_template": {
                    "__tensorzero_remapped_path": "system",
                    "__data": "be mean {{ assistant_name }}",
                },
            },
        },
        stream=False,
        dryrun=True,
    )

    assert isinstance(response, ChatInferenceResponse)
    assert response.inference_id is not None

    content = response.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    text = content[0].text
    assert text is not None
    assert "be mean Alfred" in text
    assert "You have been provided with a set of responses" in text
    assert "synthesize these responses into" in text
    assert "gleefully chanted" in text


@pytest.mark.asyncio
async def test_async_dynamic_best_of_n(async_client: AsyncTensorZeroGateway):
    """Test dynamic best_of_n_sampling variant with custom evaluator (async)"""
    episode_id = uuid7()
    input_data = {
        "system": {"assistant_name": "Watson"},
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    # Test that it fails without dryrun
    with pytest.raises(TensorZeroError) as exc_info:
        await async_client.inference(
            function_name="basic_test",
            episode_id=episode_id,
            input=input_data,
            internal_dynamic_variant_config={
                "type": "experimental_best_of_n_sampling",
                "weight": 0.0,
                "candidates": ["test", "test2"],
                "evaluator": {
                    "weight": 0.0,
                    "model": "dummy::echo_request_messages",
                    "system_template": {
                        "__tensorzero_remapped_path": "system",
                        "__data": "be mean {{ assistant_name }}",
                    },
                },
            },
            stream=False,
        )

    assert exc_info.value.status_code == 400

    # Test that it succeeds with dryrun=True
    response = await async_client.inference(
        function_name="basic_test",
        episode_id=episode_id,
        input=input_data,
        internal_dynamic_variant_config={
            "type": "experimental_best_of_n_sampling",
            "weight": 0.0,
            "candidates": ["test", "test2"],
            "evaluator": {
                "weight": 0.0,
                "model": "dummy::echo_request_messages",
                "system_template": {
                    "__tensorzero_remapped_path": "system",
                    "__data": "be mean {{ assistant_name }}",
                },
            },
        },
        stream=False,
        dryrun=True,
    )

    assert isinstance(response, ChatInferenceResponse)
    assert response.inference_id is not None

    content = response.content
    assert len(content) == 1
    assert content[0].type == "text"
    assert isinstance(content[0], Text)
    text = content[0].text
    assert text is not None
    # The best_of_n always picks a candidate
    assert "gleefully chanted" in text
