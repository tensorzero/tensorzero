import os
import re
from typing import List

MODELS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "tokenizer_module": "torchtune.models.llama3.llama3_tokenizer",
        "tokenizer_path": "original/tokenizer.model",
        "module_lora": "torchtune.models.llama3_1.lora_llama3_1_8b",
        "module": "torchtune.models.llama3_1.llama3_1_8b",
        "model_type": "LLAMA3",
    }
}


def list_checkpoints(checkpoint_directory: str) -> List[str]:
    """
    Returns a sorted list of model checkpoint files in the given directory
    matching the pattern: model-xxxxx-of-yyyyy.safetensors.

    Args:
        checkpoint_directory (str): Path to the directory containing checkpoint files.

    Returns:
        List[str]: Sorted list of checkpoint filenames.
    """
    pattern = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")
    checkpoint_files = [
        fname for fname in os.listdir(checkpoint_directory) if pattern.fullmatch(fname)
    ]
    return sorted(checkpoint_files)
