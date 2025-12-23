import json
from collections import Counter
from dataclasses import dataclass

import pandas as pd


@dataclass
class Row:
    input: str
    label: dict[str, list[str]]


def load_dataset():
    """
    Load the CoNLL++ dataset and yield individual rows.
    We treat this setting as a streaming RL environment, so splits are ignored.
    """
    df = pd.read_csv("data/conllpp.csv")
    for _, row in df.iterrows():
        yield Row(input=str(row["input"]), label=json.loads(str(row["output"])))


def flatten_dict(d: dict[str, list[str]]) -> list[str]:
    """
    Flatten a dictionary of lists into a list of strings.
    We use this function to compare the predicted dictionary with the label using the functions below.
    """
    res = []
    for k, v in d.items():
        assert isinstance(v, list)
        for elt in v:
            res.append(f"__{k.upper()}__::{elt}")
    return res


def compute_exact_match(predicted: dict[str, list[str]], label: dict[str, list[str]]) -> bool:
    """
    Check if the predicted entities exactly match the label.
    """
    return set(flatten_dict(predicted)) == set(flatten_dict(label))


def compute_jaccard_similarity(predicted: dict[str, list[str]], label: dict[str, list[str]]) -> float:
    """
    Compute the Jaccard similarity between the predicted and the label.

    This metric is more lenient than exact match.
    The implementation is different from the original code by Predibase, so the values won't be directly comparable.
    """
    target_entities = flatten_dict(label)
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
        return 1
    return num / den
