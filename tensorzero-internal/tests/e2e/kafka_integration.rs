#![cfg(feature = "e2e_tests")]

use serde_json::json;
use tensorzero_internal::kafka::{KafkaConfig, KafkaConnectionInfo};
use uuid::Uuid;

#[tokio::test]
#[ignore] // This test requires a running Kafka instance
async fn test_kafka_integration_with_real_broker() {
    // This test assumes Kafka is running on localhost:9092
    // You can run it with: docker run -d --name kafka -p 9092:9092 apache/kafka:latest

    let config = KafkaConfig {
        enabled: true,
        brokers: "localhost:9092".to_string(),
        topic_prefix: "tensorzero_test".to_string(),
        compression_type: Some("lz4".to_string()),
        batch_size: Some(1000),
        linger_ms: Some(10),
        request_timeout_ms: Some(5000),
        sasl: None,
        security_protocol: None,
        buffer_max_size: 5000,
        flush_interval_seconds: 10,
        metrics_batch_size: 500,
        metrics_topic: "budMetricsMessages".to_string(),
    };

    let kafka_conn = KafkaConnectionInfo::new(Some(&config)).unwrap();

    // Test writing inference data
    let inference_id = Uuid::now_v7();
    let inference_payload = json!({
        "id": inference_id,
        "function_name": "test_function",
        "variant_name": "test_variant",
        "episode_id": Uuid::now_v7(),
        "output": {"content": "test response"},
        "processing_time_ms": 123
    });

    // This should succeed if Kafka is running
    let result = kafka_conn
        .write(&[inference_payload], "chat_inference")
        .await;
    assert!(result.is_ok());

    // Test batch writes
    let batch_payloads = vec![
        json!({"id": Uuid::now_v7(), "data": "batch1"}),
        json!({"id": Uuid::now_v7(), "data": "batch2"}),
        json!({"id": Uuid::now_v7(), "data": "batch3"}),
    ];

    let result = kafka_conn.write(&batch_payloads, "batch_test").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_kafka_connection_error_handling() {
    // Test with invalid broker address
    let config = KafkaConfig {
        enabled: true,
        brokers: "invalid:99999".to_string(),
        topic_prefix: "test".to_string(),
        compression_type: None,
        batch_size: None,
        linger_ms: None,
        request_timeout_ms: Some(1000), // Short timeout for test
        sasl: None,
        security_protocol: None,
        buffer_max_size: 5000,
        flush_interval_seconds: 10,
        metrics_batch_size: 500,
        metrics_topic: "budMetricsMessages".to_string(),
    };

    // This should fail to create the producer
    let kafka_conn = KafkaConnectionInfo::new(Some(&config));
    // Note: rdkafka might not fail immediately on creation, so we might need to test writes

    if let Ok(conn) = kafka_conn {
        let payload = json!({"id": "test", "data": "test"});
        let result = conn.write(&[payload], "test_topic").await;
        // This should fail with a timeout or connection error
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod docker_compose_tests {
    use super::*;

    // This test requires the docker-compose setup from tests/e2e/docker-compose.yml
    // to be running with a Kafka service added
    #[tokio::test]
    #[ignore]
    async fn test_kafka_with_docker_compose() {
        // Assuming docker-compose.yml includes:
        // kafka:
        //   image: apache/kafka:latest
        //   ports:
        //     - "9092:9092"
        //   environment:
        //     KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
        //     KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092

        let config = KafkaConfig {
            enabled: true,
            brokers: "localhost:9092".to_string(),
            topic_prefix: "e2e_test".to_string(),
            compression_type: Some("snappy".to_string()),
            batch_size: None,
            linger_ms: None,
            request_timeout_ms: None,
            sasl: None,
            security_protocol: None,
            buffer_max_size: 5000,
            flush_interval_seconds: 10,
            metrics_batch_size: 500,
            metrics_topic: "budMetricsMessages".to_string(),
        };

        let kafka_conn = KafkaConnectionInfo::new(Some(&config)).unwrap();

        // Test complete inference flow
        let inference_data = json!({
            "id": Uuid::now_v7(),
            "function_name": "chat_completion",
            "variant_name": "gpt-4",
            "episode_id": Uuid::now_v7(),
            "input": {"messages": [{"role": "user", "content": "Hello"}]},
            "output": {"content": "Hello! How can I help you?"},
            "model_name": "gpt-4",
            "model_provider_name": "openai",
            "input_tokens": 10,
            "output_tokens": 20,
            "response_time_ms": 500,
            "cached": false
        });

        let result = kafka_conn.write(&[inference_data], "model_inference").await;
        assert!(result.is_ok());
    }
}
