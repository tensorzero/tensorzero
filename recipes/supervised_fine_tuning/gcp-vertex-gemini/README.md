# TensorZero Recipe: Supervised Fine-Tuning with Google Vertex AI

The `google_vertex.ipynb` notebook provides a step-by-step recipe to perform supervised fine-tuning of Google Gemini models based on data collected by the TensorZero Gateway.
Set `TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero` in the shell your notebook will run in.

## Setup

### Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- [Google Cloud Local authentication credentials](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)
- [A Google Cloud Storage Bucket](https://cloud.google.com/storage/docs/creating-buckets).
- Set the environment variables (see `.env.example`)

We recommend using [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```
