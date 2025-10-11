import os
import re
from typing import List

MODELS = {
    # Llama 3.1
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": "original/tokenizer.model",
        },
        "modules": {
            "full": "torchtune.models.llama3_1.llama3_1_8b",
            "lora": "torchtune.models.llama3_1.lora_llama3_1_8b",
            "qlora": "torchtune.models.llama3_1.qlora_llama3_1_8b",
        },
        "checkpointer": {
            "model_type": "LLAMA3",
        },
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": "original/tokenizer.model",
        },
        "modules": {
            "full": "torchtune.models.llama3_1.llama3_1_70b",
            "lora": "torchtune.models.llama3_1.lora_llama3_1_70b",
            "qlora": "torchtune.models.llama3_1.qlora_llama3_1_70b",
        },
        "checkpointer": {
            "model_type": "LLAMA3",
        },
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": "original/tokenizer.model",
        },
        "modules": {
            "full": "torchtune.models.llama3_1.llama3_1_405b",
            "lora": "torchtune.models.llama3_1.lora_llama3_1_405b",
            "qlora": "torchtune.models.llama3_1.qlora_llama3_1_405b",
        },
        "checkpointer": {
            "model_type": "LLAMA3",
        },
    },
    # Llama 3.3
    "meta-llama/Meta-Llama-3.3-70B-Instruct": {
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": "original/tokenizer.model",
        },
        "modules": {
            "full": "torchtune.models.llama3_3.llama3_3_70b",
            "lora": "torchtune.models.llama3_3.lora_llama3_3_70b",
            "qlora": "torchtune.models.llama3_3.qlora_llama3_3_70b",
        },
        "checkpointer": {
            "model_type": "LLAMA3",
        },
    },
    # Qwen 2.5
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "tokenizer": {
            "_component_": "torchtune.models.qwen2_5.qwen2_5_tokenizer",
            "path": "vocab.json",
            "merges_file": "merges.txt",
        },
        "modules": {
            "full": "torchtune.models.qwen2_5.qwen2_5_0_5b",
            "lora": "torchtune.models.qwen2_5.lora_qwen2_5_0_5b",
        },
        "checkpointer": {
            "model_type": "QWEN2",
        },
    },
    # Qwen 3
    "Qwen/Qwen3-0.6B": {
        "tokenizer": {
            "_component_": "torchtune.models.qwen3.qwen3_tokenizer",
            "path": "vocab.json",
            "merges_file": "merges.txt",
        },
        "modules": {
            "full": "torchtune.models.qwen3.qwen3_0_6b_instruct",
            "lora": "torchtune.models.qwen3.lora_qwen3_0_6b_instruct",
        },
        "checkpointer": {
            "model_type": "QWEN3",
        },
    },
    "Qwen/Qwen3-1.7B": {
        "tokenizer": {
            "_component_": "torchtune.models.qwen3.qwen3_tokenizer",
            "path": "vocab.json",
            "merges_file": "merges.txt",
        },
        "modules": {
            "full": "torchtune.models.qwen3.qwen3_1_7b_instruct",
            "lora": "torchtune.models.qwen3.lora_qwen3_1_7b_instruct",
        },
        "checkpointer": {
            "model_type": "QWEN3",
        },
    },
    "Qwen/Qwen3-4B": {
        "tokenizer": {
            "_component_": "torchtune.models.qwen3.qwen3_tokenizer",
            "path": "vocab.json",
            "merges_file": "merges.txt",
        },
        "modules": {
            "full": "torchtune.models.qwen3.qwen3_4b_instruct",
            "lora": "torchtune.models.qwen3.lora_qwen3_4b_instruct",
        },
        "checkpointer": {
            "model_type": "QWEN3",
        },
    },
    "Qwen/Qwen3-8B": {
        "tokenizer": {
            "_component_": "torchtune.models.qwen3.qwen3_tokenizer",
            "path": "vocab.json",
            "merges_file": "merges.txt",
        },
        "modules": {
            "full": "torchtune.models.qwen3.qwen3_8b_instruct",
            "lora": "torchtune.models.qwen3.lora_qwen3_8b_instruct",
        },
        "checkpointer": {
            "model_type": "QWEN3",
        },
    },
    "Qwen/Qwen3-14B": {
        "tokenizer": {
            "_component_": "torchtune.models.qwen3.qwen3_tokenizer",
            "path": "vocab.json",
            "merges_file": "merges.txt",
        },
        "modules": {
            "full": "torchtune.models.qwen3.qwen3_14b_instruct",
            "lora": "torchtune.models.qwen3.lora_qwen3_14b_instruct",
        },
        "checkpointer": {
            "model_type": "QWEN3",
        },
    },
    "Qwen/Qwen3-32B": {
        "tokenizer": {
            "_component_": "torchtune.models.qwen3.qwen3_tokenizer",
            "path": "vocab.json",
            "merges_file": "merges.txt",
        },
        "modules": {
            "full": "torchtune.models.qwen3.qwen3_32b_instruct",
            "lora": "torchtune.models.qwen3.lora_qwen3_32b_instruct",
        },
        "checkpointer": {
            "model_type": "QWEN3",
        },
    },
    # Gemma 2
    "google/gemma-2-2b-it": {
        "tokenizer": {
            "_component_": "torchtune.models.gemma.gemma_tokenizer",
            "path": "tokenizer.model",
        },
        "modules": {
            "full": "torchtune.models.gemma2.gemma2_2b",
            "lora": "torchtune.models.gemma2.lora_gemma2_2b",
        },
        "checkpointer": {
            "model_type": "GEMMA2",
        },
    },
    "google/gemma-2-9b-it": {
        "tokenizer": {
            "_component_": "torchtune.models.gemma.gemma_tokenizer",
            "path": "tokenizer.model",
        },
        "modules": {
            "full": "torchtune.models.gemma2.gemma2_9b",
            "lora": "torchtune.models.gemma2.lora_gemma2_9b",
        },
        "checkpointer": {
            "model_type": "GEMMA2",
        },
    },
    "google/gemma-2-27b-it": {
        "tokenizer": {
            "_component_": "torchtune.models.gemma.gemma_tokenizer",
            "path": "tokenizer.model",
        },
        "modules": {
            "full": "torchtune.models.gemma2.gemma2_27b",
            "lora": "torchtune.models.gemma2.lora_gemma2_27b",
        },
        "checkpointer": {
            "model_type": "GEMMA2",
        },
    },
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
    pattern = re.compile(r"^model.*\.safetensors$")
    checkpoint_files = [fname for fname in os.listdir(checkpoint_directory) if pattern.fullmatch(fname)]
    return sorted(checkpoint_files)
