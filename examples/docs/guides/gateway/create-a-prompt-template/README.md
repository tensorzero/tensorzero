# Code Example: How to create a prompt template

This folder contains the code for the [Guides » Gateway » Create a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template) page in the documentation.

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
python tensorzero_sdk.py
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
python openai_sdk.py
```

</details>

<details>
<summary><b>HTTP</b></summary>

Run the following command to make an inference request to the TensorZero Gateway.

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "fun_fact",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "template",
              "name": "fun_fact_topic",
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
