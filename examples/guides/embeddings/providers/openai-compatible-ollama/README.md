# Guide: How to use embedding models provided by Ollama with TensorZero

## Running the Example
1. Create the ollama container and pull the language model:
```bash
docker compose up -d
docker exec -it ollama ollama pull all-minilm
```

2. Run the example:

<details open>
<summary><b>Python (OpenAI)</b></summary>

a. Install the dependencies:

```bash
# We recommend using Python 3.9+ and a virtual environment
pip install -r requirements.txt
```

b. Run the example:

```bash
python openai_sdk_ollama.py
```

</details>
