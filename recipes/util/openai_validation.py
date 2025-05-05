from typing import Any, Dict, List

import tiktoken


def estimate_prompt_length(
    messages: List[Dict[str, Any]],
    tokens_per_message: int = 3,
    tokens_per_name: int = 1,
    encoding_name: str = "cl100k_base",
) -> int:
    """
    Estimates the token length of a prompt for OpenAI models.

    This function calculates the approximate number of tokens that will be used
    when sending a list of messages to OpenAI's chat models. It accounts for the
    base tokens per message, tokens for content, and additional tokens for names.

    Based on: https://cookbook.openai.com/examples/chat_finetuning_data_prep

    Args:
        messages: A list of message dictionaries. Each message should have keys like
                 'role' and 'content', and optionally 'name'.
        tokens_per_message: The number of tokens added for each message. Default is 3
                           as per OpenAI's tokenization scheme.
        tokens_per_name: The number of tokens added when a name is present. Default is 1
                        as per OpenAI's tokenization scheme.
        encoding: The name of the encoding to use for tokenization. Default is "cl100k_base",
                 which is used by models like GPT-4o.

    Returns:
        int: The estimated number of tokens that will be used for the provided messages.

    Example:
        ```python
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        token_count = estimate_prompt_length(messages)
        ```
    """
    encoding = tiktoken.get_encoding(encoding_name)

    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
            if key == "name":
                total_tokens += tokens_per_name
    total_tokens += 3  # add 3 tokens for the final assistant message marker

    return total_tokens
