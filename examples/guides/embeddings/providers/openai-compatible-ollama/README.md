# Guide: How to use OpenAI-compatible embedding models (e.g. Ollama) with TensorZero

## Running the Example

1. Launch the Ollama server:

```bash
ollama serve
```

2. Download an embedding model:

```bash
ollama run nomic-embed-text
```

3. Run the example:

<details open>
<summary><b>Python (OpenAI SDK)</b></summary>

a. Install the dependencies:

```bash
# We recommend using Python 3.10+ and a virtual environment
pip install -r requirements.txt
```

b. Run the example:

```bash
python openai_sdk.py
```

</details>
