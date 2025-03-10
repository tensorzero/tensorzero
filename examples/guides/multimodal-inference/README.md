# Guide: Multimodal Inference

This directory contains the code for the **[Multimodal Inference](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference)** guide.

## Running the Example

1. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..." # Replace with your OpenAI API key
```

2. Launch the TensorZero Gateway, ClickHouse, and MinIO (a local S3-compatible object storage service):

```bash
docker compose up
```

> [!TIP]
>
> You can use any S3-compatible object storage service (e.g. AWS S3, GCP Storage, Cloudflare R2).
> We use a local MinIO instance in this example for convenience.

3. Run the example:

<details>
<summary><b>Python</b></summary>

a. Install the dependencies:

```bash
# We recommend using Python 3.9+ and a virtual environment
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
# We recommend using Python 3.9+ and a virtual environment
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
    "model_name": "openai::gpt-4o-mini",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Do the images share any common features?"
            },
            {
              "type": "image",
              "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png"
            },
            {
              "type": "image",
              "mime_type": "image/png",
              "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII="
            }
          ]
        }
      ]
    }
  }'
```

</details>
