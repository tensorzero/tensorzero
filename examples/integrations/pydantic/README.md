# Example: Integrating Pydantic with TensorZero

## Overview

This example demonstrates how to integrate Pydantic with TensorZero to create a information extraction pipeline.

## Getting Started

### TensorZero

We provide a simple TensorZero configuration with a function `extract_email` that uses GPT-5 Nano.

### Prerequisites

1. Install Python 3.10+.
2. Install the Python dependencies with `pip install -r requirements.txt`.
3. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```
2. Run the script:
```bash
python main.py
```
