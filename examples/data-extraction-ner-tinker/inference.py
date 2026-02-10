"""Inference using Tinker's sampling API."""

from typing import Optional

from tinker import types
from tqdm import tqdm

from prompts import build_chat_messages, parse_json_output


def run_inference(
    sampling_client,
    renderer,
    texts: list[str],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> list[Optional[dict]]:
    """Run NER inference on a list of texts.

    Returns a list of parsed NER dicts (or None for failures).
    """
    stop_sequences = renderer.get_stop_sequences()
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
    )

    predictions: list[Optional[dict]] = []
    for text in tqdm(texts, desc="Running inference"):
        messages = build_chat_messages(text)
        prompt = renderer.build_generation_prompt(messages)

        try:
            result = sampling_client.sample(
                prompt=prompt,
                sampling_params=sampling_params,
                num_samples=1,
            ).result()

            tokens = result.sequences[0].tokens
            parsed_message, success = renderer.parse_response(tokens)
            if success and parsed_message.get("content"):
                predictions.append(parse_json_output(parsed_message["content"]))
            else:
                # Fall back to raw token decoding
                tokenizer = sampling_client.get_tokenizer()
                raw_text = tokenizer.decode(tokens, skip_special_tokens=True)
                predictions.append(parse_json_output(raw_text))
        except Exception as e:
            print(f"Inference error: {e}")
            predictions.append(None)

    return predictions
