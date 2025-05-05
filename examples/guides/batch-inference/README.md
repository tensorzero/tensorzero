# Guide: Batch Inference

This directory contains the code for the **[Batch Inference](https://www.tensorzero.com/docs/gateway/guides/batch-inference)** guide.

## Running the Example

1. Launch the TensorZero Gateway and ClickHouse database:

```bash
docker compose up
```

2. Start a batch inference job:

```bash
sh start_batch.sh
```

3. Poll the status of the batch inference job using the `batch_id` returned by the `start_batch.sh` script:

```bash
sh poll_batch.sh 00000000-0000-0000-0000-000000000000
```
