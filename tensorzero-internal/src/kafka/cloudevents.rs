use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// CloudEvents v1.0 compliant message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudEvent {
    pub specversion: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub source: String,
    pub id: String,
    pub time: DateTime<Utc>,
    pub datacontenttype: String,
    pub data: CloudEventData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudEventData {
    pub entries: Vec<ObservabilityEntry>,
    pub id: String,
    pub metadata: HashMap<String, String>,
    pub pubsubname: String,
    pub topic: String,
    #[serde(rename = "type")]
    pub data_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityEntry {
    pub event: ObservabilityEvent,
    #[serde(rename = "entryId")]
    pub entry_id: String,
    pub metadata: HashMap<String, String>,
    #[serde(rename = "contentType")]
    pub content_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityEvent {
    pub inference_id: Uuid,
    pub project_id: String,
    pub endpoint_id: String,
    pub model_id: String,
    pub is_success: bool,
    pub request_arrival_time: DateTime<Utc>,
    pub request_forward_time: DateTime<Utc>,

    // Optional fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_ip: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_analysis: Option<Value>,
}

impl CloudEvent {
    /// Create a new CloudEvent for observability metrics
    pub fn new_observability_event(entries: Vec<ObservabilityEntry>) -> Self {
        let event_id = Uuid::now_v7().to_string();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "gateway-service".to_string());

        Self {
            specversion: "1.0".to_string(),
            event_type: "add_observability_metrics".to_string(),
            source: "gateway-service".to_string(),
            id: event_id.clone(),
            time: Utc::now(),
            datacontenttype: "application/json".to_string(),
            data: CloudEventData {
                entries,
                id: format!("bulk-request-{}", Uuid::now_v7()),
                metadata,
                pubsubname: "pubsub-kafka".to_string(),
                topic: "observability-metrics".to_string(),
                data_type: "com.bud.observability.bulk".to_string(),
            },
        }
    }
}

impl ObservabilityEntry {
    pub fn new(event: ObservabilityEvent) -> Self {
        let entry_id = Uuid::now_v7().to_string();
        let mut metadata = HashMap::new();
        metadata.insert("cloudevent.id".to_string(), Uuid::now_v7().to_string());
        metadata.insert(
            "cloudevent.type".to_string(),
            "com.bud.observability.inference".to_string(),
        );

        Self {
            event,
            entry_id,
            metadata,
            content_type: "application/json".to_string(),
        }
    }
}

/// Validates the observability event according to the requirements
pub fn validate_event(event: &ObservabilityEvent) -> Result<(), String> {
    // Validate request_forward_time >= request_arrival_time
    if event.request_forward_time < event.request_arrival_time {
        return Err("request_forward_time must be >= request_arrival_time".to_string());
    }

    // Validate cost >= 0 when provided
    if let Some(cost) = event.cost {
        if cost < 0.0 {
            return Err("cost must be >= 0".to_string());
        }
    }

    // Validate request_ip format when provided
    if let Some(ip) = &event.request_ip {
        if !is_valid_ipv4(ip) {
            return Err("request_ip must be valid IPv4 format".to_string());
        }
    }

    Ok(())
}

fn is_valid_ipv4(ip: &str) -> bool {
    ip.split('.').count() == 4 && ip.split('.').all(|octet| octet.parse::<u8>().is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloudevent_creation() {
        let event = ObservabilityEvent {
            inference_id: Uuid::now_v7(),
            project_id: "test-project".to_string(),
            endpoint_id: "test-endpoint".to_string(),
            model_id: "gpt-4".to_string(),
            is_success: true,
            request_arrival_time: Utc::now(),
            request_forward_time: Utc::now() + chrono::Duration::seconds(1),
            request_ip: Some("192.168.1.100".to_string()),
            cost: Some(0.002),
            response_analysis: None,
        };

        let entry = ObservabilityEntry::new(event);
        let cloud_event = CloudEvent::new_observability_event(vec![entry]);

        assert_eq!(cloud_event.specversion, "1.0");
        assert_eq!(cloud_event.event_type, "add_observability_metrics");
        assert_eq!(cloud_event.source, "gateway-service");
        assert_eq!(cloud_event.data.pubsubname, "pubsub-kafka");
        assert_eq!(cloud_event.data.topic, "observability-metrics");
    }

    #[test]
    fn test_event_validation() {
        let mut event = ObservabilityEvent {
            inference_id: Uuid::now_v7(),
            project_id: "test-project".to_string(),
            endpoint_id: "test-endpoint".to_string(),
            model_id: "gpt-4".to_string(),
            is_success: true,
            request_arrival_time: Utc::now(),
            request_forward_time: Utc::now() + chrono::Duration::seconds(1),
            request_ip: Some("192.168.1.100".to_string()),
            cost: Some(0.002),
            response_analysis: None,
        };

        // Valid event
        assert!(validate_event(&event).is_ok());

        // Invalid time ordering
        event.request_forward_time = event.request_arrival_time - chrono::Duration::seconds(1);
        assert!(validate_event(&event).is_err());

        // Invalid cost
        event.request_forward_time = event.request_arrival_time + chrono::Duration::seconds(1);
        event.cost = Some(-1.0);
        assert!(validate_event(&event).is_err());

        // Invalid IP
        event.cost = Some(0.002);
        event.request_ip = Some("invalid-ip".to_string());
        assert!(validate_event(&event).is_err());
    }
}
