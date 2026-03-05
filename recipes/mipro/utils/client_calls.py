import asyncio
import json
from typing import Any, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema

_GENERATE_INSTRUCTION_SCHEMA: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {
        "name": "generate_instruction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"instructions": {"type": "string", "description": "The system prompt instructions."}},
            "required": ["instructions"],
            "additionalProperties": False,
        },
    },
}

_JUDGE_ANSWER_SCHEMA: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "thinking": {"type": "string"},
            },
            "required": ["score", "thinking"],
            "additionalProperties": False,
        },
    },
}


async def get_instructions(
    client: AsyncOpenAI,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Optional[dict[str, Any]]:
    """
    Generate candidate instructions using direct OpenAI API call.
    """
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model="o1",
                messages=[{"role": "user", "content": system_prompt}],
                response_format=_GENERATE_INSTRUCTION_SCHEMA,
            )
            content = response.choices[0].message.content
            return json.loads(content) if content else None
    except Exception as e:
        print(f"Error generating instructions: {e}")
        return None


async def candidate_inference(
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    system_prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
) -> Optional[Any]:
    """
    Run inference with a candidate prompt using direct OpenAI API call.
    """
    openai_messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
    openai_messages.extend(messages)

    try:
        async with semaphore:
            return await client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
            )
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None


async def judge_answer(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Optional[dict[str, Any]]:
    """
    Score a prediction using an LLM judge via direct OpenAI API call.
    """
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_JUDGE_ANSWER_SCHEMA,
            )
            content = response.choices[0].message.content
            return json.loads(content) if content else None
    except Exception as e:
        print(f"Error judging answer: {e}")
        return None
