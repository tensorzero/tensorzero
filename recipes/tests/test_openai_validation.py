from recipes.util.openai_validation import (
    check_example_length,
    check_system_user_messages,
)


def test_check_example_length() -> None:
    """
    Test the check_example_length function to ensure it correctly validates the length of messages.
    """

    # Test case where the total tokens are within the limit
    messages_within_limit = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
    ]
    assert check_example_length(messages_within_limit, max_token=50)

    # Test case where the total tokens exceed the limit
    messages_exceed_limit = [
        {"role": "user", "content": "Hello!" * 1000},
        {"role": "assistant", "content": "Hi there! How can I help you today?" * 1000},
    ]
    assert not check_example_length(messages_exceed_limit, max_token=50)


def test_check_system_user_messages() -> None:
    """
    Test the check_system_user_messages function to ensure it correctly identifies the presence of system and user messages.
    """

    # Test case where both system and user messages are present
    messages_with_system_and_user = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]
    assert check_system_user_messages(messages_with_system_and_user)

    # Test case where only user messages are present
    messages_with_only_user = [{"role": "user", "content": "User message"}]
    assert not check_system_user_messages(messages_with_only_user)

    # Test case where only system messages are present
    messages_with_only_system = [{"role": "system", "content": "System message"}]
    assert not check_system_user_messages(messages_with_only_system)

    # Test case where neither system nor user messages are present
    messages_with_neither = [{"role": "assistant", "content": "Assistant message"}]
    assert not check_system_user_messages(messages_with_neither)
