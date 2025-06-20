use super::*;
use crate::kafka::cloudevents::ObservabilityEvent;
use std::sync::Arc;
use uuid::Uuid;

#[tokio::test]
async fn test_mock_kafka_observability_event() {
    // Create a mock Kafka connection
    let mock = MockKafkaConnectionInfo {
        messages: Arc::new(tokio::sync::Mutex::new(Vec::new())),
    };
    let kafka_conn = KafkaConnectionInfo::Mock(mock.clone());

    // Create an observability event
    let event = ObservabilityEvent {
        inference_id: Uuid::now_v7(),
        project_id: "test-project".to_string(),
        endpoint_id: "test-endpoint".to_string(),
        model_id: "test-model".to_string(),
        is_success: true,
        request_arrival_time: chrono::Utc::now(),
        request_forward_time: chrono::Utc::now() + chrono::Duration::milliseconds(10),
        request_ip: Some("192.168.1.1".to_string()),
        cost: Some(0.001),
        response_analysis: None,
    };

    // Send the event
    kafka_conn
        .add_observability_event(event.clone())
        .await
        .unwrap();

    // Verify the message was sent
    let messages = mock.messages.lock().await;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, "budMetricsMessages");
    assert_eq!(messages[0].1, event.project_id);

    // Verify the CloudEvent structure
    let cloud_event: serde_json::Value = serde_json::from_str(&messages[0].2).unwrap();
    assert_eq!(cloud_event["specversion"], "1.0");
    assert_eq!(cloud_event["type"], "add_observability_metrics");
    assert_eq!(cloud_event["source"], "gateway-service");
    assert_eq!(cloud_event["datacontenttype"], "application/json");
    assert_eq!(
        cloud_event["data"]["inference_id"],
        event.inference_id.to_string()
    );
    assert_eq!(cloud_event["data"]["project_id"], event.project_id);
}

#[tokio::test]
async fn test_mock_kafka_buffer_and_batch() {
    // Create a mock Kafka connection with buffer config
    let mock = MockKafkaConnectionInfo {
        messages: Arc::new(tokio::sync::Mutex::new(Vec::new())),
    };
    let kafka_conn = KafkaConnectionInfo::Mock(mock.clone());

    // Send multiple events
    let mut events = Vec::new();
    for i in 0..5 {
        let event = ObservabilityEvent {
            inference_id: Uuid::now_v7(),
            project_id: format!("project-{i}"),
            endpoint_id: "test-endpoint".to_string(),
            model_id: "test-model".to_string(),
            is_success: true,
            request_arrival_time: chrono::Utc::now(),
            request_forward_time: chrono::Utc::now() + chrono::Duration::milliseconds(10),
            request_ip: None,
            cost: Some(0.001 * i as f64),
            response_analysis: None,
        };
        events.push(event.clone());
        kafka_conn.add_observability_event(event).await.unwrap();
    }

    // Verify all messages were sent
    let messages = mock.messages.lock().await;
    assert_eq!(messages.len(), 5);

    // Verify each message
    for (i, (topic, key, value)) in messages.iter().enumerate() {
        assert_eq!(topic, "budMetricsMessages");
        assert_eq!(key, &format!("project-{i}"));

        let cloud_event: serde_json::Value = serde_json::from_str(value).unwrap();
        assert_eq!(cloud_event["data"]["project_id"], format!("project-{i}"));
        assert_eq!(cloud_event["data"]["cost"], 0.001 * i as f64);
    }
}

#[tokio::test]
async fn test_kafka_disabled() {
    let kafka_conn = KafkaConnectionInfo::Disabled;

    // Create an observability event
    let event = ObservabilityEvent {
        inference_id: Uuid::now_v7(),
        project_id: "test-project".to_string(),
        endpoint_id: "test-endpoint".to_string(),
        model_id: "test-model".to_string(),
        is_success: true,
        request_arrival_time: chrono::Utc::now(),
        request_forward_time: chrono::Utc::now() + chrono::Duration::milliseconds(10),
        request_ip: None,
        cost: None,
        response_analysis: None,
    };

    // This should succeed without doing anything
    let result = kafka_conn.add_observability_event(event).await;
    assert!(result.is_ok());

    // Test write as well
    let payload = serde_json::json!({"test": "data"});
    let result = kafka_conn.write(&[payload], "test_topic").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_kafka_event_validation() {
    // Test event with invalid time ordering
    let mut event = ObservabilityEvent {
        inference_id: Uuid::now_v7(),
        project_id: "test-project".to_string(),
        endpoint_id: "test-endpoint".to_string(),
        model_id: "test-model".to_string(),
        is_success: true,
        request_arrival_time: chrono::Utc::now(),
        request_forward_time: chrono::Utc::now() - chrono::Duration::seconds(1), // Invalid: before arrival
        request_ip: None,
        cost: None,
        response_analysis: None,
    };

    let validation_result = cloudevents::validate_event(&event);
    assert!(validation_result.is_err());
    assert!(validation_result
        .unwrap_err()
        .contains("request_forward_time must be >= request_arrival_time"));

    // Fix the time and add invalid cost
    event.request_forward_time = event.request_arrival_time + chrono::Duration::milliseconds(10);
    event.cost = Some(-1.0); // Invalid: negative cost

    let validation_result = cloudevents::validate_event(&event);
    assert!(validation_result.is_err());
    assert!(validation_result.unwrap_err().contains("cost must be >= 0"));

    // Fix cost and add invalid IP
    event.cost = Some(0.01);
    event.request_ip = Some("invalid-ip-address".to_string());

    let validation_result = cloudevents::validate_event(&event);
    assert!(validation_result.is_err());
    assert!(validation_result
        .unwrap_err()
        .contains("request_ip must be valid IPv4 format"));

    // Valid event should pass
    event.request_ip = Some("192.168.1.1".to_string());
    let validation_result = cloudevents::validate_event(&event);
    assert!(validation_result.is_ok());
}

#[tokio::test]
async fn test_kafka_write_multiple_payloads() {
    let mock = MockKafkaConnectionInfo {
        messages: Arc::new(tokio::sync::Mutex::new(Vec::new())),
    };
    let kafka_conn = KafkaConnectionInfo::Mock(mock.clone());

    // Test multiple payloads at once
    let payloads = vec![
        serde_json::json!({"id": "1", "type": "inference"}),
        serde_json::json!({"inference_id": "2", "type": "model"}),
        serde_json::json!({"episode_id": "3", "type": "chat"}),
    ];

    kafka_conn.write(&payloads, "bulk_test").await.unwrap();

    let messages = mock.messages.lock().await;
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].0, "test_bulk_test");
    assert_eq!(messages[0].1, "1");
    assert_eq!(messages[1].1, "2");
    assert_eq!(messages[2].1, "3");
}

#[test]
fn test_extract_key_from_payload() {
    let payload1 = serde_json::json!({"id": "test-id-123", "data": "value"});
    assert_eq!(
        extract_key_from_payload(&payload1.to_string()),
        "test-id-123"
    );

    let payload2 = serde_json::json!({"inference_id": "inf-456", "data": "value"});
    assert_eq!(extract_key_from_payload(&payload2.to_string()), "inf-456");

    let payload3 = serde_json::json!({"episode_id": "ep-789", "data": "value"});
    assert_eq!(extract_key_from_payload(&payload3.to_string()), "ep-789");

    let payload4 = serde_json::json!({"target_id": "tgt-000", "data": "value"});
    assert_eq!(extract_key_from_payload(&payload4.to_string()), "tgt-000");

    let payload5 = serde_json::json!({"data": "value"});
    assert_eq!(extract_key_from_payload(&payload5.to_string()), "");
}

#[tokio::test]
async fn test_kafka_connection_config() {
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
