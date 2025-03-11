# Guide: Prompt Templates & Schemas

This directory contains the code for the **[Prompt Templates & Schemas](https://www.tensorzero.com/docs/gateway/guides/prompt-templates-schemas)** guide.

## Running the Example

1. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..." # Replace with your OpenAI API key
```

2. Launch the TensorZero Gateway and a local ClickHouse database:

```bash
docker compose up
```

3. Run the example:

<details>
<summary><b>Python</b></summary>

a. Install the dependencies:

```bash
# We recommend using Python 3.10+ and a virtual environment
pip install -r requirements.txt
```

b. Run the example:

```bash
python main.py
```

</details>

<summary><b>Python (OpenAI)</b></summary>

a. Install the dependencies:

```bash
# We recommend using Python 3.10+ and a virtual environment
pip install -r requirements.txt
```

b. Run the example:

```bash
python main_openai.py
```

</details>

<details>
<summary><b>HTTP</b></summary>

Run the following commands to make a multimodal inference request to the TensorZero Gateway.
The first image is a remote image of Ferris the crab, and the second image is a one-pixel orange image encoded as a base64 string.

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "generate_haiku_with_topic",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "arguments": {
                "topic": "artificial intelligence"
              }
            }
          ]
        }
      ]
    }
  }'
```

</details>
