"""Evaluation metrics for NER extraction."""

from collections import Counter
from typing import Optional

from tabulate import tabulate


def flatten_dict(d: dict[str, list[str]]) -> list[str]:
    """Flatten an NER dict into tagged strings for set comparison."""
    res = []
    for k, v in d.items():
        assert isinstance(v, list)
        for elt in v:
            res.append(f"__{k.upper()}__::{elt}")
    return res


def compute_exact_match(
    predicted: dict[str, list[str]], ground_truth: dict[str, list[str]]
) -> bool:
    """Check if predicted entities exactly match ground truth."""
    return set(flatten_dict(predicted)) == set(flatten_dict(ground_truth))


def compute_jaccard_similarity(
    predicted: dict[str, list[str]], ground_truth: dict[str, list[str]]
) -> float:
    """Compute Jaccard similarity between predicted and ground truth entities."""
    target_entities = flatten_dict(ground_truth)
    pred_entities = flatten_dict(predicted)
    target_count = Counter(target_entities)
    pred_count = Counter(pred_entities)
    num = 0
    den = 0
    all_keys = set(target_entities).union(set(pred_entities))
    for key in all_keys:
        num += min(target_count.get(key, 0), pred_count.get(key, 0))
        den += max(target_count.get(key, 0), pred_count.get(key, 0))
    if den == 0:
        return 1.0
    return num / den


def evaluate_single(
    predicted: Optional[dict], ground_truth: dict[str, list[str]]
) -> tuple[bool, bool, float]:
    """Evaluate a single prediction against ground truth.

    Returns (valid_output, exact_match, jaccard_similarity).
    """
    valid_output = predicted is not None
    exact_match = compute_exact_match(predicted, ground_truth) if predicted else False
    jaccard = (
        compute_jaccard_similarity(predicted, ground_truth) if predicted else 0.0
    )
    return valid_output, exact_match, jaccard


def evaluate_batch(
    predictions: list[Optional[dict]],
    ground_truths: list[dict[str, list[str]]],
) -> dict[str, float]:
    """Evaluate a batch and return aggregated metrics."""
    valid_outputs = []
    exact_matches = []
    jaccards = []

    for pred, gt in zip(predictions, ground_truths):
        vo, em, js = evaluate_single(pred, gt)
        valid_outputs.append(vo)
        exact_matches.append(em)
        jaccards.append(js)

    n = len(predictions)
    return {
        "valid_output": sum(valid_outputs) / n,
        "exact_match": sum(exact_matches) / n,
        "jaccard_similarity": sum(jaccards) / n,
    }


def print_comparison(results: dict[str, dict[str, float]]) -> None:
    """Print a comparison table of metrics across variants."""
    headers = ["Variant", "Valid Output", "Exact Match", "Jaccard Similarity"]
    rows = []
    for name, metrics in results.items():
        rows.append([
            name,
            f"{metrics['valid_output']:.1%}",
            f"{metrics['exact_match']:.1%}",
            f"{metrics['jaccard_similarity']:.1%}",
        ])
    print()
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()
