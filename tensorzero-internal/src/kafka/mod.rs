use crate::error::{Error, ErrorDetails};
use metrics::{counter, histogram};
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, warn};

#[derive(Clone, Debug)]
pub enum KafkaConnectionInfo {
    Disabled,
    #[cfg(any(test, feature = "e2e_tests"))]
    Mock(MockKafkaConnectionInfo),
    Production(Arc<KafkaProducerInfo>),
}

#[derive(Debug)]
pub struct KafkaProducerInfo {
    pub producer: FutureProducer,
    pub topic_prefix: String,
}

#[cfg(any(test, feature = "e2e_tests"))]
#[derive(Clone, Debug)]
pub struct MockKafkaConnectionInfo {
    pub messages: Arc<tokio::sync::Mutex<Vec<(String, String, String)>>>, // (topic, key, value)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    pub enabled: bool,
    pub brokers: String,
    pub topic_prefix: String,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
                    .set("enable.idempotence", "true");
                
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
                
                // SASL configuration
                if let Some(sasl) = &config.sasl {
                    client_config
                        .set("security.protocol", "SASL_SSL")
                        .set("sasl.mechanism", &sasl.mechanism)
                        .set("sasl.username", &sasl.username)
                        .set("sasl.password", &sasl.password);
                }
                
                let producer: FutureProducer = client_config.create().map_err(|e| {
                    counter!("kafka_connection_errors").increment(1);
                    ErrorDetails::KafkaConnection {
                        message: format!("Failed to create Kafka producer: {}", e),
                    }
                })?;
                
                counter!("kafka_connections_established").increment(1);
                Ok(KafkaConnectionInfo::Production(Arc::new(
                    KafkaProducerInfo {
                        producer,
                        topic_prefix: config.topic_prefix.clone(),
                    },
                )))
            }
        }
    }
    
    pub async fn write<T: Serialize>(
        &self,
        payloads: &[T],
        data_type: &str,
    ) -> Result<(), Error> {
        match self {
            KafkaConnectionInfo::Disabled => Ok(()),
            #[cfg(any(test, feature = "e2e_tests"))]
            KafkaConnectionInfo::Mock(mock) => {
                let mut messages = mock.messages.lock().await;
                for payload in payloads {
                    let value = serde_json::to_string(payload).map_err(|e| {
                        ErrorDetails::KafkaSerialization {
                            message: format!("Failed to serialize payload: {}", e),
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
                let labels = &[("topic", topic.clone()), ("data_type", data_type.to_string())];
                
                for payload in payloads {
                    let start_time = std::time::Instant::now();
                    
                    let value = serde_json::to_string(payload).map_err(|e| {
                        counter!("kafka_serialization_errors", labels).increment(1);
                        ErrorDetails::KafkaSerialization {
                            message: format!("Failed to serialize payload: {}", e),
                        }
                    })?;
                    
                    let key = extract_key_from_payload(&value);
                    
                    let record = FutureRecord::to(&topic)
                        .key(&key)
                        .payload(&value);
                    
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
                                message: format!("Failed to send message to Kafka: {:?}", e),
                            }
                            .into());
                        }
                    }
                }
                
                Ok(())
            }
        }
    }
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
mod tests {
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
        assert_eq!(
            extract_key_from_payload(&payload2.to_string()),
            "inf-456"
        );
        
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
        let config = Some(&KafkaConfig {
            enabled: false,
            brokers: "localhost:9092".to_string(),
            topic_prefix: "test".to_string(),
            compression_type: None,
            batch_size: None,
            linger_ms: None,
            request_timeout_ms: None,
            sasl: None,
        });
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