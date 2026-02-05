# TensorZero Recipe: Automated Prompt Engineering with MIPRO

This recipe provides a step-by-step guide for performing automated prompt engineering on TensorZero functions using historical inference and feedback data.

## MIPRO (Multi-prompt Instruction PRoposal Optimizer)

MIPRO is an optimization framework designed for multi-stage LLM applications.
It enhances prompt effectiveness by systematically searching over instructions and few-shot demonstrations to maximize downstream task performance.
Unlike traditional prompt engineering, which relies on manual trial-and-error methods, MIPRO introduces algorithmic strategies for optimizing LM programs under constraints such as black-box model access.
For more details, see the [MIPRO paper](https://arxiv.org/abs/2406.11695v1).
This recipe implements the MIPROv2 variant of this algorithm from [DSPy](https://github.com/stanfordnlp/dspy).

## High-Level Structure

MIPRO operates within a structured optimization framework:

- Proposal Generation: Generates candidate instructions and demonstrations.
- Evaluation: Scores generated prompts based on their effectiveness.
- Optimization: Utilizes a surrogate model to refine prompt proposals based on observed performance.

In our implementation, we use an LLM judge to score the candidate prompts.
The judge is configurable to fit your problem by describing the task and metric you want to optimize. This assumes that the LLM judge will output scores that are correlated with the metric you want to optimize.
**We'll show that TensorZero can automatically optimize prompts for GPT-4o Mini using MIPRO.**

<p align="center">
  <img src="visualization.svg" alt="Metrics by Variant" />
</p>

Though &mdash; unsurprisingly &mdash; it doesn't outperform DICL and supervised fine-tuning, it materially outperforms a naive initial prompt.

> [!TIP]
>
> See our article [From NER to Agents: Does Automated Prompt Engineering Scale to Complex Tasks?](https://tensorzero.com/blog/from-ner-to-agents-does-automated-prompt-engineering-scale-to-complex-tasks) for more insights on automated prompt engineering with MIPRO.

## Getting Started

### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
uv pip sync requirements.txt  # Install the dependencies
```

### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
```

### Setup

1. Set the `OPENAI_API_KEY` and `TENSORZERO_CLICKHOUSE_URL` environment variables.
2. Run the `mipro.ipynb` Jupyter notebook.
