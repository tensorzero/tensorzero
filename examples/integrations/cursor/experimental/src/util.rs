use chrono::{DateTime, Utc};
use uuid::{Builder, Uuid};

/// Generates the maximum UUIDv7 for a given timestamp.
pub fn get_max_uuidv7(timestamp: DateTime<Utc>) -> Uuid {
    // Create a byte array of 255s
    let bytes: [u8; 10] = [255; 10];
    // Get the unix timestamp in milliseconds as a u64
    let timestamp_ms = timestamp.timestamp_millis() as u64;
    // Create a builder
    let builder = Builder::from_unix_timestamp_millis(timestamp_ms, &bytes);
    // Build the UUID
    builder.into_uuid()
}

/// Generates the minimum UUIDv7 for a given timestamp.
pub fn get_min_uuidv7(timestamp: DateTime<Utc>) -> Uuid {
    // Create a byte array of 0s
    let bytes: [u8; 10] = [0; 10];
    // Get the unix timestamp in milliseconds as a u64
    let timestamp_ms = timestamp.timestamp_millis() as u64;
    // Create a builder
    let builder = Builder::from_unix_timestamp_millis(timestamp_ms, &bytes);
    // Build the UUID
    builder.into_uuid()
}
