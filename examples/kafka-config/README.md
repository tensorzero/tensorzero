# Kafka Integration Example

This example demonstrates how to configure TensorZero to send inference data to Apache Kafka.

## Prerequisites

1. A running Kafka instance (see below for Docker setup)
2. TensorZero gateway with Kafka support
3. OpenAI API key (or another model provider)

## Running Kafka

Start Kafka using Docker:

```bash
docker run -d --name kafka -p 9092:9092 \
  -e KAFKA_NODE_ID=1 \
  -e KAFKA_PROCESS_ROLES=broker,controller \
  -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -e KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1 \
  -e KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1 \
  -e KAFKA_LOG_DIRS=/var/lib/kafka/data \
  -e KAFKA_CLUSTER_ID=MkU3OEVBNTcwNTJENDM2Qk \
  apache/kafka:3.7.0
```

## Running the Example

1. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export TENSORZERO_CLICKHOUSE_URL="http://user:password@localhost:8123/tensorzero"
   ```

2. Start the TensorZero gateway:
   ```bash
   cd examples/kafka-config
   tensorzero --config tensorzero.toml
   ```

3. Make an inference request:
   ```bash
   curl -X POST http://localhost:3000/inference \
     -H "Content-Type: application/json" \
     -d '{
       "function_name": "example_chat",
       "input": {
         "messages": [
           {"role": "user", "content": "Hello, how are you?"}
         ]
       }
     }'
   ```

4. Send feedback (this will only go to ClickHouse, not Kafka):
   ```bash
   curl -X POST http://localhost:3000/feedback \
     -H "Content-Type: application/json" \
     -d '{
       "inference_id": "<inference-id-from-above>",
       "metric_name": "accuracy",
       "value": 0.95
     }'
   ```

## Viewing Kafka Messages

You can consume messages from Kafka to verify they're being sent:

```bash
# List topics
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list

# Consume chat inference messages
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic tensorzero_chat_inference \
  --from-beginning

# Consume model inference messages
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic tensorzero_model_inference \
  --from-beginning
```

## Configuration Options

The `tensorzero.toml` file demonstrates various Kafka configuration options:

- **Basic settings**: broker addresses, topic prefix
- **Performance tuning**: compression, batching
- **Security**: SASL authentication (commented out)

## Monitoring

Check the metrics endpoint to see Kafka operation statistics:

```bash
curl http://localhost:3001/metrics | grep kafka
```

You should see metrics like:
- `kafka_writes_success`
- `kafka_writes_failed`
- `kafka_write_duration_ms`

## Troubleshooting

If messages aren't appearing in Kafka:

1. Check the gateway logs for Kafka connection errors
2. Verify Kafka is running: `docker ps`
3. Check if async_writes is enabled in the config
4. Ensure the ClickHouse URL is set (observability must be enabled)