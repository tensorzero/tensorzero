from util.openai_validation import estimate_prompt_length


def test_estimate_prompt_length() -> None:
    """
    Test the estimate_prompt_length function to ensure it correctly estimates the length of messages.
    """

    # Test case where the total tokens are within the limit
    messages_within_limit = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
    ]

    # Note: @GabrielBianconi confirmed that the OpenAI Playground returns 23 tokens for this prompt on 2025-02-27
    assert estimate_prompt_length(messages_within_limit) == 23

    # Test case where the total tokens exceed the limit
    messages_exceed_limit = [
        {"role": "user", "content": "Hello!" * 1000},
        {"role": "assistant", "content": "Hi there! How can I help you today?" * 1000},
    ]
    assert estimate_prompt_length(messages_exceed_limit) > 10_000
