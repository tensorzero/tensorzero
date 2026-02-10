"""Dataset loading for the NER example."""

import json
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / ".." / "data-extraction-ner" / "data" / "conllpp.csv"


def load_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    num_train: int = 500,
    num_val: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the CoNLLpp dataset and return (train_df, val_df).

    The `output` column is parsed from JSON strings into dicts.
    """
    df = pd.read_csv(path)
    df["output"] = df["output"].apply(json.loads)

    train_df = df[df["split"] == 0]
    val_df = df[df["split"] == 1]

    # Shuffle with fixed seed for reproducibility
    train_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=0).reset_index(drop=True)

    train_df = train_df.iloc[:num_train]
    val_df = val_df.iloc[:num_val]

    return train_df, val_df
