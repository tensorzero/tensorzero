# Kafka Integration

TensorZero supports sending inference data to Apache Kafka in addition to ClickHouse. This enables real-time streaming of inference events for downstream processing and integration with existing data pipelines.

## Configuration

Kafka integration is configured in the `[gateway.observability]` section of your TensorZero configuration file:

```toml
[gateway.observability]
enabled = true
async_writes = true

[gateway.observability.kafka]
enabled = true
brokers = "localhost:9092"  # Comma-separated list of Kafka brokers
topic_prefix = "tensorzero"  # Prefix for all Kafka topics

# Optional settings
compression_type = "lz4"  # Options: none, gzip, snappy, lz4, zstd
batch_size = 1000  # Maximum number of messages to batch
linger_ms = 10  # Time to wait for batching messages
request_timeout_ms = 5000  # Request timeout in milliseconds

# Optional SASL authentication
[gateway.observability.kafka.sasl]
mechanism = "PLAIN"  # Options: PLAIN, SCRAM-SHA-256, SCRAM-SHA-512
username = "your-username"
password = "your-password"
```

## Topics

When enabled, TensorZero writes to the following Kafka topics (prefixed with your configured `topic_prefix`):

- `{topic_prefix}_model_inference` - Raw model provider responses
- `{topic_prefix}_chat_inference` - Chat completion inference results  
- `{topic_prefix}_json_inference` - JSON mode inference results

## Message Format

All messages are sent as JSON with the following structure:

### Model Inference Message
```json
{
  "id": "inference-uuid",
  "inference_id": "parent-inference-uuid",
  "raw_request": "...",
  "raw_response": "...",
  "input_messages": "...",
  "output": "...",
  "model_name": "gpt-4",
  "model_provider_name": "openai",
  "input_tokens": 100,
  "output_tokens": 200,
  "response_time_ms": 500,
  "cached": false
}
```

### Chat Inference Message
```json
{
  "id": "inference-uuid",
  "function_name": "chat_function",
  "variant_name": "variant_a",
  "episode_id": "episode-uuid",
  "input": {...},
  "output": [...],
  "processing_time_ms": 123,
  "tags": {"key": "value"}
}
```


## Message Keys

Messages use the following fields as Kafka message keys for partitioning:
- `id` (primary)
- `inference_id` (fallback)
- `episode_id` (fallback)
- `target_id` (fallback)

## Non-blocking Writes

Similar to ClickHouse, Kafka writes are non-blocking by default when `async_writes` is enabled. This ensures that:
- Client requests are not delayed by Kafka writes
- Failures in Kafka don't affect the main request flow
- Messages are sent asynchronously using Tokio tasks

Note: Feedback data is only written to ClickHouse and not sent to Kafka.

## Error Handling

- Connection failures during startup will prevent the gateway from starting if Kafka is enabled
- Write failures are logged but don't fail the request
- Serialization errors are tracked in metrics

## Metrics

The following metrics are exposed for monitoring Kafka operations:

- `kafka_connections_established` - Counter of successful Kafka connections
- `kafka_connection_errors` - Counter of connection failures
- `kafka_writes_success` - Counter of successful writes (labeled by topic and data_type)
- `kafka_writes_failed` - Counter of failed writes (labeled by topic and data_type)
- `kafka_serialization_errors` - Counter of serialization failures
- `kafka_write_duration_ms` - Histogram of write latencies (labeled by topic and data_type)

## Running Kafka for Development

For local development, you can run Kafka using Docker:

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

## Consuming Messages

You can consume messages using any Kafka client. Example using `kafka-console-consumer`:

```bash
# List all topics
kafka-topics --bootstrap-server localhost:9092 --list

# Consume from a specific topic
kafka-console-consumer --bootstrap-server localhost:9092 \
  --topic tensorzero_model_inference \
  --from-beginning
```

## Best Practices

1. **Topic Management**: Consider setting up topic auto-creation or pre-create topics with appropriate retention policies
2. **Compression**: Use compression (lz4 recommended) to reduce network bandwidth
3. **Batching**: Configure batch settings based on your throughput requirements
4. **Monitoring**: Set up alerts on Kafka metrics to detect issues early
5. **Security**: Use SASL authentication in production environments

## Troubleshooting

### Connection Issues
- Verify Kafka brokers are reachable from the TensorZero gateway
- Check firewall rules and network connectivity
- Ensure broker addresses are correctly formatted (host:port)

### Missing Messages
- Check if topics exist (auto-creation may be disabled)
- Verify the topic prefix configuration
- Check Kafka logs for any errors

### Performance Issues
- Adjust batch settings (batch_size, linger_ms)
- Consider using compression
- Monitor Kafka broker performance