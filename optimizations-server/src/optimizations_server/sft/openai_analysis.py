import logging
import typing as t

import tiktoken

logger = logging.getLogger(__name__)


def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """
    Gets encoding for a given model.
    Following OpenAI's implementation, we use cl100k_base encoding as a fallback for unknown models.
    https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken#6-counting-tokens-for-chat-completions-api-calls

    Args:
        model: Model name

    Returns:
        Tiktoken encoding
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Unknown model: {model}, using cl100k_base")
        return tiktoken.get_encoding("cl100k_base")


# Copied from https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
# This is best-effort, and should only be used for informational purposes in the ui.
def num_tokens_from_messages(
    messages: t.List[t.Dict[str, str]], encoding: tiktoken.Encoding, model: str
) -> int:
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(
            messages,
            encoding,
            model="gpt-3.5-turbo-0125",
        )
    elif "gpt-4o-mini" in model:
        return num_tokens_from_messages(
            messages, encoding, model="gpt-4o-mini-2024-07-18"
        )
    elif "gpt-4o" in model:
        return num_tokens_from_messages(messages, encoding, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, encoding, model="gpt-4-0613")
    else:
        logger.warning(
            "Unknown model %s for token counting, using default message/name counts"
            % model
        )
        return num_tokens_from_messages(messages, encoding, model="gpt-3.5-turbo-0125")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            try:
                num_tokens += len(encoding.encode(value))
            except Exception as e:
                logger.warning(f"Error encoding message: {e}. Skipping.")
                continue
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def analyze_dataset(dataset: t.Any, model: str, encoding: tiktoken.Encoding):
    """
    Analyzes dataset for warnings and token statistics

    Args:
        dataset: List of dataset entries containing messages
        model: Model name for token counting
        enc: Tokenizer instance

    Returns:
        Dict with warnings and token statistics
    """
    missing_system_count = 0
    missing_user_count = 0

    message_counts = []
    token_counts = []

    for entry in dataset:
        messages = entry["messages"]

        # Check for missing system/user messages
        if not any(m["role"] == "system" for m in messages):
            missing_system_count += 1

        if not any(m["role"] == "user" for m in messages):
            missing_user_count += 1

        message_counts.append(len(messages))
        token_counts.append(num_tokens_from_messages(messages, encoding, model))

    return {
        "missingSystemCount": missing_system_count,
        "missingUserCount": missing_user_count,
        "messageCounts": calculate_distribution(message_counts),
        "tokenCounts": calculate_distribution(token_counts),
        # TODO - decide whether or not we should remove this from the ui
        "assistantTokenCounts": calculate_distribution([]),
        "tooLongCount": 0,
    }


def calculate_distribution(values):
    """
    Calculates statistical distribution of numeric values

    Args:
        values: List of numeric values

    Returns:
        Dict with distribution statistics
    """
    if len(values) == 0:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p5": 0,
            "p95": 0,
        }

    # Sort values for percentile calculations
    sorted_values = sorted(values)

    # Calculate mean
    mean = sum(sorted_values) / len(sorted_values)

    # Calculate median
    mid_index = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        median = (sorted_values[mid_index - 1] + sorted_values[mid_index]) / 2
    else:
        median = sorted_values[mid_index]

    # Calculate percentiles
    p5_index = int(len(sorted_values) * 0.05)
    p95_index = int(len(sorted_values) * 0.95)

    return {
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean,
        "median": median,
        "p5": sorted_values[p5_index],
        "p95": sorted_values[p95_index],
    }
