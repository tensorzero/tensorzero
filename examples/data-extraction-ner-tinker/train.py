"""LoRA fine-tuning via Tinker."""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tinker import types
from tqdm import tqdm

from prompts import build_chat_messages


@dataclass
class TrainConfig:
    rank: int = 32
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    checkpoint_name: str = "ner-lora"


def prepare_training_data(renderer, train_df: pd.DataFrame) -> list[types.Datum]:
    """Convert training examples into Tinker Datum objects for SFT."""
    data = []
    for _, row in tqdm(
        train_df.iterrows(), total=len(train_df), desc="Preparing training data"
    ):
        text = row["input"]
        ground_truth = row["output"]
        assistant_response = json.dumps(ground_truth)

        messages = build_chat_messages(text, assistant_response=assistant_response)
        model_input, weights = renderer.build_supervised_example(messages)

        # Build target tokens: shift input tokens left by 1
        input_tokens = model_input.tokens if hasattr(model_input, "tokens") else None
        if input_tokens is not None:
            target_tokens = list(input_tokens[1:]) + [0]
        else:
            # Fall back to extracting from chunks
            all_tokens = []
            for chunk in model_input.chunks:
                if hasattr(chunk, "tokens"):
                    all_tokens.extend(chunk.tokens)
            target_tokens = all_tokens[1:] + [0]

        datum = types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "weights": types.TensorData.from_numpy(np.array(weights)),
                "target_tokens": types.TensorData.from_numpy(
                    np.array(target_tokens)
                ),
            },
        )
        data.append(datum)

    return data


def train(
    service_client,
    model_name: str,
    renderer,
    train_df: pd.DataFrame,
    config: TrainConfig,
):
    """Run LoRA SFT and return (sampling_client, renderer).

    The renderer is unchanged (same tokenizer / chat template), but we return
    it for convenience so callers get a consistent interface.
    """
    print(f"\nCreating LoRA training client for {model_name} (rank={config.rank})...")
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=config.rank,
    )

    print("Preparing training data...")
    data = prepare_training_data(renderer, train_df)

    print(
        f"\nTraining for {config.epochs} epoch(s), "
        f"batch_size={config.batch_size}, lr={config.learning_rate}\n"
    )

    adam_params = types.AdamParams(learning_rate=config.learning_rate)
    n = len(data)

    for epoch in range(config.epochs):
        # Shuffle data each epoch
        indices = np.random.default_rng(seed=epoch).permutation(n)

        epoch_losses = []
        num_batches = (n + config.batch_size - 1) // config.batch_size

        for batch_idx in tqdm(
            range(num_batches), desc=f"Epoch {epoch + 1}/{config.epochs}"
        ):
            start = batch_idx * config.batch_size
            end = min(start + config.batch_size, n)
            batch = [data[i] for i in indices[start:end]]

            fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            # Compute batch loss
            logprobs = np.concatenate(
                [
                    output["logprobs"].tolist()
                    for output in fwd_bwd_result.loss_fn_outputs
                ]
            )
            weights = np.concatenate(
                [ex.loss_fn_inputs["weights"].tolist() for ex in batch]
            )
            weight_sum = weights.sum()
            if weight_sum > 0:
                loss = -np.dot(logprobs, weights) / weight_sum
                epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        print(f"  Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

    print(f"\nSaving checkpoint as '{config.checkpoint_name}'...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=config.checkpoint_name,
    )

    return sampling_client
