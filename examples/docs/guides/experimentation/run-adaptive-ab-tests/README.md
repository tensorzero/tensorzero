# Code Example: How to run adaptive A/B tests

This folder contains the code for the [Guides » Experimentation » Run adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests/) page in the documentation.
[^1]

## Running the Experiment

### Prerequisites

Make sure you have the following environment variables set:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Setup

1. **Run Postgres migrations** (required on first run):

```bash
docker compose run --rm gateway --run-postgres-migrations
```

2. **Start all services**:

```bash
docker compose up
```

This will start:

- **ClickHouse**: Database for inference results and feedback (port 8123)
- **Postgres**: Database for TensorZero metadata (port 5432)
- **Gateway**: TensorZero Gateway (port 3000)
- **UI**: TensorZero observability UI (port 4000)

### Running the Experiment

Once the services are running, execute the experiment script:

```bash
uv run main.py
```

This will:

- Load NER (Named Entity Recognition) data from the CoNLL++ dataset
- Send inference requests to the TensorZero Gateway
- Submit feedback for each inference
- The Track-and-Stop algorithm will adaptively adjust sampling probabilities every 15 seconds

### Viewing Results

- **Real-time monitoring**: Open http://localhost:4000 to view the TensorZero UI

---

[^1]: We build off of the [CoNLL++ dataset](https://arxiv.org/abs/1909.01441v1) and [work](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora) from Predibase for the problem setting.
