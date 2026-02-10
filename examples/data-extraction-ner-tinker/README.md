# Data Extraction (NER) with Tinker LoRA Fine-Tuning

This example demonstrates Named Entity Recognition (NER) optimization using [Tinker](https://thinkingmachines.ai/tinker/) for LoRA fine-tuning of an open-weight model. It is a standalone alternative to the [gateway-based NER example](../data-extraction-ner/) that replaces Docker, ClickHouse, the TensorZero gateway, and OpenAI with Tinker's managed training API.

## What It Does

1. **Loads** the CoNLLpp NER dataset (shared with the gateway example)
2. **Evaluates** a baseline open-weight model on the validation set
3. **Fine-tunes** the model with LoRA SFT via Tinker's API
4. **Evaluates** the fine-tuned model and prints a comparison table

## Prerequisites

- Python 3.11+
- A [Tinker API key](https://tinker-console.thinkingmachines.ai)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

```bash
cd examples/data-extraction-ner-tinker

# Install dependencies
uv sync

# Set your API key
export TINKER_API_KEY="your-key-here"
```

## Usage

Run the full pipeline (baseline eval → LoRA fine-tuning → fine-tuned eval):

```bash
uv run python main.py
```

Baseline evaluation only (no training):

```bash
uv run python main.py --skip-training
```

Custom settings:

```bash
uv run python main.py \
  --model Qwen/Qwen3-8B \
  --num-train 200 \
  --num-val 200 \
  --rank 32 \
  --epochs 3 \
  --batch-size 4 \
  --lr 1e-4
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-8B` | Base model to fine-tune |
| `--renderer` | auto-detect | Chat template renderer name |
| `--num-train` | 500 | Number of training examples |
| `--num-val` | 500 | Number of validation examples |
| `--rank` | 32 | LoRA rank |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--skip-baseline` | false | Skip baseline evaluation |
| `--skip-training` | false | Skip training (baseline only) |
| `--data-path` | auto | Path to `conllpp.csv` |

## Metrics

The example reports three metrics:

- **Valid Output** — fraction of model outputs that parse as valid NER JSON
- **Exact Match** — fraction of predictions that exactly match ground truth entities
- **Jaccard Similarity** — mean Jaccard similarity between predicted and ground truth entity sets (gives partial credit)
