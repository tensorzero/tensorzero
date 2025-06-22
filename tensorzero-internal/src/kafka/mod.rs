pub mod buffer;
pub mod cloudevents;

use crate::error::{Error, ErrorDetails};
use metrics::{counter, histogram};
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info};

use self::buffer::{BufferConfig, MessageBuffer};
use self::cloudevents::{ObservabilityEntry, ObservabilityEvent};
use uuid::Uuid;

#[derive(Clone, Debug)]
pub enum KafkaConnectionInfo {
    Disabled,
    #[cfg(any(test, feature = "e2e_tests"))]
    Mock(MockKafkaConnectionInfo),
    Production(Arc<KafkaProducerInfo>),
}

pub struct KafkaProducerInfo {
    pub producer: FutureProducer,
    pub topic_prefix: String,
    pub buffer: Arc<MessageBuffer>,
    pub metrics_topic: String,
}

impl std::fmt::Debug for KafkaProducerInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KafkaProducerInfo")
            .field("topic_prefix", &self.topic_prefix)
            .field("buffer", &self.buffer)
            .field("metrics_topic", &self.metrics_topic)
            .finish_non_exhaustive()
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
#[derive(Clone, Debug)]
pub struct MockKafkaConnectionInfo {
    pub messages: Arc<tokio::sync::Mutex<Vec<(String, String, String)>>>, // (topic, key, value)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KafkaConfig {
    pub enabled: bool,
    pub brokers: String,
    #[serde(default = "default_topic_prefix")]
    pub topic_prefix: String,
    #[serde(default = "default_metrics_topic")]
    pub metrics_topic: String,
    #[serde(default)]
    pub compression_type: Option<String>,
    #[serde(default)]
    pub batch_size: Option<i32>,
    #[serde(default)]
    pub linger_ms: Option<i32>,
    #[serde(default)]
    pub request_timeout_ms: Option<i32>,
    #[serde(default)]
    pub sasl: Option<KafkaSaslConfig>,
    #[serde(default)]
    pub security_protocol: Option<String>,
    #[serde(default = "default_buffer_size")]
    pub buffer_max_size: usize,
    #[serde(default = "default_metrics_batch_size")]
    pub metrics_batch_size: usize,
    #[serde(default = "default_flush_interval")]
    pub flush_interval_seconds: u64,
}

fn default_topic_prefix() -> String {
    "tensorzero".to_string()
}

fn default_metrics_topic() -> String {
    "budMetricsMessages".to_string()
}

fn default_buffer_size() -> usize {
    5000
}

fn default_metrics_batch_size() -> usize {
    500
}

fn default_flush_interval() -> u64 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KafkaSaslConfig {
    pub mechanism: String,
    pub username: String,
    pub password: String,
}

impl KafkaConnectionInfo {
    pub fn new(config: Option<&KafkaConfig>) -> Result<Self, Error> {
        match config {
            None => Ok(KafkaConnectionInfo::Disabled),
            Some(config) if !config.enabled => Ok(KafkaConnectionInfo::Disabled),
            Some(config) => {
                let mut client_config = ClientConfig::new();

                // Basic configuration
                client_config
                    .set("bootstrap.servers", &config.brokers)
                    .set("message.timeout.ms", "5000")
                    .set("enable.idempotence", "false");

                // Optional compression
                if let Some(compression) = &config.compression_type {
                    client_config.set("compression.type", compression);
                }

                // Optional batch settings
                if let Some(batch_size) = config.batch_size {
                    client_config.set("batch.size", batch_size.to_string());
                }

                if let Some(linger_ms) = config.linger_ms {
                    client_config.set("linger.ms", linger_ms.to_string());
                }

                if let Some(timeout_ms) = config.request_timeout_ms {
                    client_config.set("request.timeout.ms", timeout_ms.to_string());
                }

                // Security protocol
                if let Some(protocol) = &config.security_protocol {
                    client_config.set("security.protocol", protocol);
                }

                // SASL configuration
                if let Some(sasl) = &config.sasl {
                    if config.security_protocol.is_none() {
                        client_config.set("security.protocol", "SASL_SSL");
                    }
                    client_config
                        .set("sasl.mechanism", &sasl.mechanism)
                        .set("sasl.username", &sasl.username)
                        .set("sasl.password", &sasl.password);
                }

                let producer: FutureProducer = client_config.create().map_err(|e| {
                    counter!("kafka_connection_errors").increment(1);
                    ErrorDetails::KafkaConnection {
                        message: format!("Failed to create Kafka producer: {e}"),
                    }
                })?;

                // Create message buffer
                let buffer_config = BufferConfig {
                    max_size: config.buffer_max_size,
                    batch_size: config.metrics_batch_size,
                    flush_interval: Duration::from_secs(config.flush_interval_seconds),
                };
                let buffer = Arc::new(MessageBuffer::new(buffer_config));

                counter!("kafka_connections_established").increment(1);
                info!(
                    "Kafka producer initialized with metrics topic: {}",
                    config.metrics_topic
                );

                Ok(KafkaConnectionInfo::Production(Arc::new(
                    KafkaProducerInfo {
                        producer,
                        topic_prefix: config.topic_prefix.clone(),
                        buffer,
                        metrics_topic: config.metrics_topic.clone(),
                    },
                )))
            }
        }
    }

    pub async fn write<T: Serialize>(&self, payloads: &[T], data_type: &str) -> Result<(), Error> {
        match self {
            KafkaConnectionInfo::Disabled => Ok(()),
            #[cfg(any(test, feature = "e2e_tests"))]
            KafkaConnectionInfo::Mock(mock) => {
                let mut messages = mock.messages.lock().await;
                for payload in payloads {
                    let value = serde_json::to_string(payload).map_err(|e| {
                        ErrorDetails::KafkaSerialization {
                            message: format!("Failed to serialize payload: {e}"),
                        }
                    })?;

                    // For mock, we'll use a simple key extraction
                    let key = extract_key_from_payload(&value);
                    let topic = format!("{}_{}", "test", data_type);

                    messages.push((topic, key, value));
                }
                Ok(())
            }
            KafkaConnectionInfo::Production(producer_info) => {
                let topic = format!("{}_{}", producer_info.topic_prefix, data_type);
                let labels = &[
                    ("topic", topic.clone()),
                    ("data_type", data_type.to_string()),
                ];

                for payload in payloads {
                    let start_time = std::time::Instant::now();

                    let value = serde_json::to_string(payload).map_err(|e| {
                        counter!("kafka_serialization_errors", labels).increment(1);
                        ErrorDetails::KafkaSerialization {
                            message: format!("Failed to serialize payload: {e}"),
                        }
                    })?;

                    let key = extract_key_from_payload(&value);

                    let record = FutureRecord::to(&topic).key(&key).payload(&value);

                    // Send with a timeout
                    let delivery_result = producer_info
                        .producer
                        .send(record, Timeout::After(Duration::from_secs(5)))
                        .await;

                    match delivery_result {
                        Ok((partition, offset)) => {
                            let duration = start_time.elapsed();
                            histogram!("kafka_write_duration_ms", labels)
                                .record(duration.as_millis() as f64);
                            counter!("kafka_writes_success", labels).increment(1);
                            tracing::debug!(
                                "Successfully sent message to Kafka topic {} partition {} offset {}",
                                topic, partition, offset
                            );
                        }
                        Err((e, _)) => {
                            counter!("kafka_writes_failed", labels).increment(1);
                            error!("Failed to send message to Kafka: {:?}", e);
                            return Err(ErrorDetails::KafkaProducer {
                                message: format!("Failed to send message to Kafka: {e:?}"),
                            }
                            .into());
                        }
                    }
                }

                Ok(())
            }
        }
    }

    /// Add an observability event to the buffer
    pub async fn add_observability_event(&self, event: ObservabilityEvent) -> Result<(), Error> {
        match self {
            KafkaConnectionInfo::Disabled => Ok(()),
            #[cfg(any(test, feature = "e2e_tests"))]
            KafkaConnectionInfo::Mock(mock) => {
                let project_id = event.project_id.clone();
                let entry = ObservabilityEntry::new(event);
                let cloud_event = serde_json::json!({
                    "specversion": "1.0",
                    "type": "add_observability_metrics",
                    "source": "gateway-service",
                    "id": Uuid::now_v7().to_string(),
                    "time": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                    "datacontenttype": "application/json",
                    "data": entry.event
                });

                let value = serde_json::to_string(&cloud_event).map_err(|e| {
                    ErrorDetails::KafkaSerialization {
                        message: format!("Failed to serialize CloudEvent: {e}"),
                    }
                })?;

                let mut messages = mock.messages.lock().await;
                messages.push(("budMetricsMessages".to_string(), project_id, value));
                Ok(())
            }
            KafkaConnectionInfo::Production(producer_info) => {
                // Validate the event
                if let Err(e) = cloudevents::validate_event(&event) {
                    counter!("kafka_event_validation_errors").increment(1);
                    return Err(ErrorDetails::KafkaSerialization {
                        message: format!("Invalid observability event: {e}"),
                    }
                    .into());
                }

                let entry = ObservabilityEntry::new(event);
                let producer_info_clone = producer_info.clone();

                // Add to buffer
                match producer_info.buffer.add(entry).await {
                    Ok(Some(batch)) => {
                        // Buffer returned a batch to send
                        tokio::spawn(async move {
                            if let Err(e) = send_metrics_batch(producer_info_clone, batch).await {
                                error!("Failed to send metrics batch: {}", e);
                            }
                        });
                    }
                    Ok(None) => {
                        // Entry added to buffer, no batch ready yet
                    }
                    Err(e) => {
                        counter!("kafka_buffer_errors").increment(1);
                        return Err(ErrorDetails::KafkaProducer {
                            message: format!("Buffer error: {e}"),
                        }
                        .into());
                    }
                }

                Ok(())
            }
        }
    }

    /// Start the background flush task for the buffer
    pub fn start_flush_task(&self) -> Option<tokio::task::JoinHandle<()>> {
        match self {
            KafkaConnectionInfo::Production(producer_info) => {
                let buffer = producer_info.buffer.clone();
                let producer_info_clone = producer_info.clone();

                Some(tokio::spawn(async move {
                    buffer::start_flush_task(buffer, move |entries| {
                        let producer_info = producer_info_clone.clone();
                        tokio::spawn(async move {
                            if let Err(e) = send_metrics_batch(producer_info, entries).await {
                                error!("Failed to send metrics batch during flush: {}", e);
                            }
                        });
                    })
                    .await;
                }))
            }
            _ => None,
        }
    }

    /// Flush any pending messages in the buffer
    pub async fn flush(&self) -> Result<(), Error> {
        match self {
            KafkaConnectionInfo::Production(producer_info) => {
                let entries = producer_info.buffer.flush().await;
                if !entries.is_empty() {
                    send_metrics_batch(producer_info.clone(), entries).await?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Send a batch of observability entries to Kafka
async fn send_metrics_batch(
    producer_info: Arc<KafkaProducerInfo>,
    entries: Vec<ObservabilityEntry>,
) -> Result<(), Error> {
    let num_entries = entries.len();

    // Send each observability event as a separate CloudEvent
    for entry in entries {
        let event_id = Uuid::now_v7().to_string();

        // Create a simple CloudEvent with the observability data directly
        let cloud_event = serde_json::json!({
            "specversion": "1.0",
            "type": "add_observability_metrics",
            "source": "gateway-service",
            "id": event_id,
            "time": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            "datacontenttype": "application/json",
            "data": entry.event
        });

        let key = entry.event.project_id.clone();
        let value = serde_json::to_string(&cloud_event).map_err(|e| {
            counter!("kafka_serialization_errors").increment(1);
            ErrorDetails::KafkaSerialization {
                message: format!("Failed to serialize CloudEvent: {e}"),
            }
        })?;

        let record = FutureRecord::to(&producer_info.metrics_topic)
            .key(&key)
            .payload(&value);

        let start_time = std::time::Instant::now();
        let delivery_result = producer_info
            .producer
            .send(record, Timeout::After(Duration::from_secs(30)))
            .await;

        match delivery_result {
            Ok((partition, offset)) => {
                let duration = start_time.elapsed();
                histogram!("gateway_kafka_publish_duration_seconds").record(duration.as_secs_f64());
                counter!("gateway_metrics_published_total").increment(1);
                debug!(
                    "Sent metric to Kafka topic {} partition {} offset {}",
                    producer_info.metrics_topic, partition, offset
                );
            }
            Err((e, _)) => {
                counter!("gateway_metrics_failed_total").increment(1);
                error!("Failed to send metric to Kafka: {:?}", e);
                return Err(ErrorDetails::KafkaProducer {
                    message: format!("Failed to send metric to Kafka: {e:?}"),
                }
                .into());
            }
        }
    }

    info!(
        "Sent {} metrics to Kafka topic {}",
        num_entries, producer_info.metrics_topic
    );
    Ok(())
}

// Helper function to extract ID from JSON payload for use as Kafka key
fn extract_key_from_payload(json_str: &str) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
        // Try to find an ID field
        if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
            return id.to_string();
        }
        if let Some(inference_id) = value.get("inference_id").and_then(|v| v.as_str()) {
            return inference_id.to_string();
        }
        if let Some(episode_id) = value.get("episode_id").and_then(|v| v.as_str()) {
            return episode_id.to_string();
        }
        if let Some(target_id) = value.get("target_id").and_then(|v| v.as_str()) {
            return target_id.to_string();
        }
    }

    // Fallback to empty key
    String::new()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod unit_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_key_from_payload() {
        let payload1 = json!({"id": "test-id-123", "data": "value"});
        assert_eq!(
            extract_key_from_payload(&payload1.to_string()),
            "test-id-123"
        );

        let payload2 = json!({"inference_id": "inf-456", "data": "value"});
        assert_eq!(extract_key_from_payload(&payload2.to_string()), "inf-456");

        let payload3 = json!({"episode_id": "ep-789", "data": "value"});
        assert_eq!(extract_key_from_payload(&payload3.to_string()), "ep-789");

        let payload4 = json!({"target_id": "tgt-000", "data": "value"});
        assert_eq!(extract_key_from_payload(&payload4.to_string()), "tgt-000");

        let payload5 = json!({"data": "value"});
        assert_eq!(extract_key_from_payload(&payload5.to_string()), "");
    }

    #[tokio::test]
    async fn test_kafka_connection_disabled() {
        let conn = KafkaConnectionInfo::Disabled;
        let payload = json!({"test": "data"});

        // Should succeed without doing anything
        assert!(conn.write(&[payload], "test_topic").await.is_ok());
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    #[tokio::test]
    async fn test_kafka_connection_mock() {
        let mock = MockKafkaConnectionInfo {
            messages: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        };
        let conn = KafkaConnectionInfo::Mock(mock.clone());

        let payload = json!({"id": "test-123", "data": "test-value"});

        conn.write(&[payload], "test_data").await.unwrap();

        let messages = mock.messages.lock().await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].0, "test_test_data");
        assert_eq!(messages[0].1, "test-123");
        assert!(messages[0].2.contains("test-value"));
    }

    #[test]
    fn test_kafka_config_creation() {
        // Test disabled config
        let config = None;
        let conn = KafkaConnectionInfo::new(config).unwrap();
        assert!(matches!(conn, KafkaConnectionInfo::Disabled));

        // Test explicitly disabled config
        let kafka_config = KafkaConfig {
            enabled: false,
            brokers: "localhost:9092".to_string(),
            topic_prefix: "test".to_string(),
            metrics_topic: "budMetricsMessages".to_string(),
            compression_type: None,
            batch_size: None,
            linger_ms: None,
            request_timeout_ms: None,
            sasl: None,
            security_protocol: None,
            buffer_max_size: 5000,
            metrics_batch_size: 500,
            flush_interval_seconds: 10,
        };
        let config = Some(&kafka_config);
        let conn = KafkaConnectionInfo::new(config).unwrap();
        assert!(matches!(conn, KafkaConnectionInfo::Disabled));
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    #[tokio::test]
    async fn test_kafka_mock_multiple_writes() {
        let mock = MockKafkaConnectionInfo {
            messages: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        };
        let conn = KafkaConnectionInfo::Mock(mock.clone());

        // Test multiple payloads at once
        let payloads = vec![
            json!({"id": "1", "type": "inference"}),
            json!({"inference_id": "2", "type": "model"}),
            json!({"episode_id": "3", "type": "chat"}),
        ];

        conn.write(&payloads, "bulk_test").await.unwrap();

        let messages = mock.messages.lock().await;
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].0, "test_bulk_test");
        assert_eq!(messages[0].1, "1");
        assert_eq!(messages[1].1, "2");
        assert_eq!(messages[2].1, "3");
    }

    #[test]
    fn test_extract_key_from_invalid_json() {
        let invalid_json = "not a json";
        assert_eq!(extract_key_from_payload(invalid_json), "");

        let json_without_id = json!({"data": "value", "other": "field"});
        assert_eq!(extract_key_from_payload(&json_without_id.to_string()), "");
    }
}
