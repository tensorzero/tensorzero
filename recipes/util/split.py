from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tensorzero import RenderedSample


def train_val_split(
    rendered_samples: List[RenderedSample],
    val_size: float = 0.2,
    random_seed: int = 42,
    last_inference_only: bool = False,
    max_train_episodes: Optional[int] = None,
    max_val_episodes: Optional[int] = None,
) -> Tuple[List[Any], List[Any]]:
    """
    Split samples deterministically based on episode_id using NumPy's RNG.
    This ensures reproducible splits with the same random seed across runs.

    Parameters:
    -----------
    rendered_samples : List[RenderedSample]
        List of rendered samples to split
    val_size : float, default=0.2
        Proportion of episodes to include in val set
    random_seed : int, default=42
        Random seed for reproducibility
    max_train_episodes : int, optional
        Maximum number of episodes to include in training set.
        If specified, randomly samples this many episodes from the full training set.
    max_val_episodes : int, optional
        Maximum number of episodes to include in val set.
        If specified, randomly samples this many episodes from the full val set.
    last_inference_only : bool, default=False
        If True, only return the sample with the most recent inference_id (UUID7)
        for each episode. UUID7 contains timestamps, so sorting gives chronological order.

    Returns:
    --------
    Tuple[List[RenderedSample], List[RenderedSample]]
        (train_samples, val_samples)
    """
    # Filter to last inference only if requested
    if last_inference_only:
        # Group samples by episode_id
        episode_samples: Dict[str, List[RenderedSample]] = defaultdict(list)  # type: ignore
        for sample in rendered_samples:
            episode_samples[str(sample.episode_id)].append(sample)

        # For each episode, keep only the sample with the most recent inference_id
        # UUID7 sorts lexicographically in chronological order
        filtered_samples = []
        for _, samples in episode_samples.items():
            # Sort by inference_id (UUID7) and take the last one
            latest_sample = max(samples, key=lambda s: s.inference_id)  # type: ignore
            filtered_samples.append(latest_sample)  # type: ignore

        rendered_samples = filtered_samples

    # Create NumPy random generator for deterministic behavior
    rng = np.random.RandomState(random_seed)

    # Get unique episode IDs and sort them for determinism
    unique_episodes = sorted(list(set(str(sample.episode_id) for sample in rendered_samples)))

    # Create a shuffled index array
    indices = np.arange(len(unique_episodes))
    rng.shuffle(indices)

    # Determine cutoff for val set
    n_val_episodes = int(len(unique_episodes) * val_size)

    # Split indices into train and val
    val_indices = indices[:n_val_episodes]
    train_indices = indices[n_val_episodes:]

    # Get episode IDs based on shuffled indices
    val_episode_ids: List[str] = [unique_episodes[i] for i in val_indices]
    train_episode_ids: List[str] = [unique_episodes[i] for i in train_indices]

    # Apply episode limits if specified
    if max_train_episodes is not None and len(train_episode_ids) > max_train_episodes:
        # Use rng.choice for sampling without replacement
        sampled_indices = rng.choice(len(train_episode_ids), max_train_episodes, replace=False)
        train_episode_ids = [train_episode_ids[i] for i in sampled_indices]

    if max_val_episodes is not None and len(val_episode_ids) > max_val_episodes:
        # Use rng.choice for sampling without replacement
        sampled_indices = rng.choice(len(val_episode_ids), max_val_episodes, replace=False)
        val_episode_ids = [val_episode_ids[i] for i in sampled_indices]

    # Convert to sets for efficient lookup
    train_episode_set = set(train_episode_ids)
    val_episode_set = set(val_episode_ids)

    # Split samples based on episode_id
    train_samples: List[RenderedSample] = []
    val_samples: List[RenderedSample] = []

    for sample in rendered_samples:
        if str(sample.episode_id) in val_episode_set:
            val_samples.append(sample)
        elif str(sample.episode_id) in train_episode_set:
            train_samples.append(sample)
        # Note: samples with episodes not in either set are excluded

    print_split_summary(
        rendered_samples,
        train_samples,
        val_samples,
        max_train_episodes=max_train_episodes,
        max_val_episodes=max_val_episodes,
        last_inference_only=last_inference_only,
    )

    return train_samples, val_samples


def print_split_summary(
    rendered_samples: List[RenderedSample],
    train_samples: List[RenderedSample],
    val_samples: List[RenderedSample],
    max_train_episodes: Optional[int] = None,
    max_val_episodes: Optional[int] = None,
    last_inference_only: bool = False,
) -> None:
    """
    Print a summary of the train/val split including episode statistics.
    """
    # Get unique episode counts
    original_episodes = set(str(sample.episode_id) for sample in rendered_samples)
    train_episodes = set(str(sample.episode_id) for sample in train_samples)
    val_episodes = set(str(sample.episode_id) for sample in val_samples)

    print("=" * 60)
    print("TRAIN/VAL SPLIT SUMMARY")
    print("=" * 60)
    print(f"Original samples: {len(rendered_samples)}")
    print(f"Original episodes: {len(original_episodes)}")
    if last_inference_only:
        print("  (Filtered to last inference per episode)")
    print()
    print(f"Train samples: {len(train_samples)} ({len(train_samples) / len(rendered_samples) * 100:.1f}%)")
    print(f"Train episodes: {len(train_episodes)}")
    if max_train_episodes:
        print(f"  (limited to {max_train_episodes} episodes)")
    print(f"Avg samples per train episode: {len(train_samples) / len(train_episodes):.2f}")

    print()
    print(f"Val samples: {len(val_samples)} ({len(val_samples) / len(rendered_samples) * 100:.1f}%)")
    print(f"Val episodes: {len(val_episodes)}")
    if max_val_episodes:
        print(f"  (limited to {max_val_episodes} episodes)")
    print(f"Avg samples per val episode: {len(val_samples) / len(val_episodes):.2f}")

    print()
    print(f"Total samples used: {len(train_samples) + len(val_samples)}")
    print(f"Total episodes used: {len(train_episodes) + len(val_episodes)}")

    # Check if any samples were excluded
    used_episodes = train_episodes | val_episodes
    excluded_episodes = original_episodes - used_episodes
    if excluded_episodes:
        excluded_samples = sum(1 for s in rendered_samples if str(s.episode_id) in excluded_episodes)
        print()
        print(f"⚠️  Excluded {len(excluded_episodes)} episodes ({excluded_samples} samples) due to episode limits")

    print("=" * 60)
