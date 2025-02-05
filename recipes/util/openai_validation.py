import tiktoken


def check_example_length(messages, max_token=2048):
    encoding = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    total_tokens = num_tokens_from_messages(messages)
    return total_tokens <= max_token


def check_system_user_messages(messages):
    has_system_message = any(message["role"] == "system" for message in messages)
    has_user_message = any(message["role"] == "user" for message in messages)
    return has_system_message and has_user_message
