use super::cloudevents::ObservabilityEntry;
use metrics::gauge;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::interval;

/// Configuration for the message buffer
#[derive(Debug, Clone)]
pub struct BufferConfig {
    pub max_size: usize,
    pub batch_size: usize,
    pub flush_interval: Duration,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_size: 5000,
            batch_size: 500,
            flush_interval: Duration::from_secs(10),
        }
    }
}

/// Buffer for batching observability entries before sending to Kafka
#[derive(Debug)]
pub struct MessageBuffer {
    entries: Arc<Mutex<VecDeque<ObservabilityEntry>>>,
    config: BufferConfig,
    last_flush: Arc<Mutex<Instant>>,
}

impl MessageBuffer {
    pub fn new(config: BufferConfig) -> Self {
        Self {
            entries: Arc::new(Mutex::new(VecDeque::new())),
            config,
            last_flush: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Add an entry to the buffer
    pub async fn add(
        &self,
        entry: ObservabilityEntry,
    ) -> Result<Option<Vec<ObservabilityEntry>>, String> {
        let mut entries = self.entries.lock().await;

        // Check if buffer is full
        if entries.len() >= self.config.max_size {
            gauge!("gateway_metrics_buffer_size").set(entries.len() as f64);
            return Err("Buffer is full".to_string());
        }

        entries.push_back(entry);
        gauge!("gateway_metrics_buffer_size").set(entries.len() as f64);

        // Check if we should flush
        if entries.len() >= self.config.batch_size {
            Ok(Some(self.drain_batch(&mut entries)))
        } else {
            Ok(None)
        }
    }

    /// Check if buffer should be flushed based on time
    pub async fn should_flush(&self) -> bool {
        let last_flush = *self.last_flush.lock().await;
        let entries = self.entries.lock().await;

        !entries.is_empty() && last_flush.elapsed() >= self.config.flush_interval
    }

    /// Flush the buffer, returning all entries
    pub async fn flush(&self) -> Vec<ObservabilityEntry> {
        let mut entries = self.entries.lock().await;
        let mut last_flush = self.last_flush.lock().await;

        *last_flush = Instant::now();
        gauge!("gateway_metrics_buffer_size").set(0.0);

        entries.drain(..).collect()
    }

    /// Flush a batch of entries
    pub async fn flush_batch(&self) -> Option<Vec<ObservabilityEntry>> {
        let mut entries = self.entries.lock().await;

        if entries.is_empty() {
            return None;
        }

        let batch = self.drain_batch(&mut entries);
        gauge!("gateway_metrics_buffer_size").set(entries.len() as f64);

        Some(batch)
    }

    /// Get current buffer size
    pub async fn size(&self) -> usize {
        self.entries.lock().await.len()
    }

    /// Drain a batch from the entries
    fn drain_batch(&self, entries: &mut VecDeque<ObservabilityEntry>) -> Vec<ObservabilityEntry> {
        let drain_count = std::cmp::min(self.config.batch_size, entries.len());
        entries.drain(..drain_count).collect()
    }
}

/// Background task for periodic flushing
pub async fn start_flush_task<F>(buffer: Arc<MessageBuffer>, mut flush_fn: F)
where
    F: FnMut(Vec<ObservabilityEntry>) + Send + 'static,
{
    let mut interval = interval(Duration::from_secs(1)); // Check every second

    loop {
        interval.tick().await;

        if buffer.should_flush().await {
            let entries = buffer.flush().await;
            if !entries.is_empty() {
                flush_fn(entries);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kafka::cloudevents::ObservabilityEvent;
    use chrono::Utc;
    use uuid::Uuid;

    fn create_test_entry() -> ObservabilityEntry {
        let event = ObservabilityEvent {
            inference_id: Uuid::now_v7(),
            project_id: "test-project".to_string(),
            endpoint_id: "test-endpoint".to_string(),
            model_id: "test-model".to_string(),
            is_success: true,
            request_arrival_time: Utc::now(),
            request_forward_time: Utc::now(),
            request_ip: None,
            cost: None,
            response_analysis: None,
        };
        ObservabilityEntry::new(event)
    }

    #[tokio::test]
    async fn test_buffer_batch_size() {
        let config = BufferConfig {
            max_size: 1000,
            batch_size: 3,
            flush_interval: Duration::from_secs(60),
        };
        let buffer = MessageBuffer::new(config);

        // Add entries up to batch size
        assert!(buffer.add(create_test_entry()).await.unwrap().is_none());
        assert!(buffer.add(create_test_entry()).await.unwrap().is_none());

        // Third entry should trigger batch
        let batch = buffer.add(create_test_entry()).await.unwrap();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 3);

        // Buffer should be empty
        assert_eq!(buffer.size().await, 0);
    }

    #[tokio::test]
    async fn test_buffer_max_size() {
        let config = BufferConfig {
            max_size: 2,
            batch_size: 10,
            flush_interval: Duration::from_secs(60),
        };
        let buffer = MessageBuffer::new(config);

        // Fill buffer
        buffer.add(create_test_entry()).await.unwrap();
        buffer.add(create_test_entry()).await.unwrap();

        // Should fail when full
        assert!(buffer.add(create_test_entry()).await.is_err());
    }

    #[tokio::test]
    async fn test_manual_flush() {
        let config = BufferConfig {
            max_size: 100,
            batch_size: 50,
            flush_interval: Duration::from_secs(60),
        };
        let buffer = MessageBuffer::new(config);

        // Add some entries
        buffer.add(create_test_entry()).await.unwrap();
        buffer.add(create_test_entry()).await.unwrap();

        // Manual flush
        let flushed = buffer.flush().await;
        assert_eq!(flushed.len(), 2);
        assert_eq!(buffer.size().await, 0);
    }
}
