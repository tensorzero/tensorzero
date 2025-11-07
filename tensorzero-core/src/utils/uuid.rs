use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::{Timestamp, Uuid};

use crate::error::DelayedError;
use crate::{
    error::{Error, ErrorDetails},
    inference::types::current_timestamp,
};

/// Timestamp when Scaling Laws for Neural Language Models was published.
/// No way anyone could use TensorZero prior to this.
const EARLIEST_TIMESTAMP: u64 = 1579751960;

pub fn validate_tensorzero_uuid(uuid: Uuid, kind: &str) -> Result<(), Error> {
    let version = uuid.get_version_num();
    if version != 7 {
        return Err(ErrorDetails::InvalidTensorzeroUuid {
            kind: kind.to_string(),
            message: format!("Version must be 7, got {version}"),
        }
        .into());
    }
    let (timestamp, _) = uuid
        .get_timestamp()
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidTensorzeroUuid {
                kind: kind.to_string(),
                message: "Timestamp is missing".to_string(),
            })
        })?
        .to_unix();
    if timestamp < EARLIEST_TIMESTAMP {
        return Err(Error::new(ErrorDetails::InvalidTensorzeroUuid {
            kind: kind.to_string(),
            message: "Timestamp is too early".to_string(),
        }));
    }
    let current_timestamp: u64 = current_timestamp();
    if timestamp > current_timestamp {
        return Err(ErrorDetails::InvalidTensorzeroUuid {
            kind: kind.to_string(),
            message: "Timestamp is in the future".to_string(),
        }
        .into());
    }
    Ok(())
}

pub fn uuid_elapsed(uuid: &Uuid) -> Result<Duration, DelayedError> {
    let version = uuid.get_version_num();
    if version != 7 {
        return Err(DelayedError::new(ErrorDetails::InvalidUuid {
            raw_uuid: uuid.to_string(),
        }));
    }

    let uuid_timestamp = uuid.get_timestamp().ok_or_else(|| {
        // This should never happen, since we already checked that the version was 7.
        // Dynamic evaluations should have a timestamp that exists but is far in the future.
        DelayedError::new(ErrorDetails::InvalidUuid {
            raw_uuid: uuid.to_string(),
        })
    })?;

    let (seconds, subsec_nanos) = uuid_timestamp.to_unix();

    let mut uuid_system_time =
        UNIX_EPOCH + Duration::from_secs(seconds) + Duration::from_nanos(subsec_nanos as u64);

    // If the UUID crosses the workflow evaluation threshold, we have to remove that offset
    if compare_timestamps(WORKFLOW_EVALUATION_THRESHOLD, uuid_timestamp) {
        uuid_system_time -= WORKFLOW_EVALUATION_OFFSET;
    }

    let elapsed = match SystemTime::now().duration_since(uuid_system_time) {
        Ok(duration) => duration,
        Err(e) => {
            let future_duration = e.duration();
            if future_duration > Duration::from_secs(1) {
                return Err(DelayedError::new(ErrorDetails::UuidInFuture {
                    raw_uuid: uuid.to_string(),
                }));
            }
            Duration::from_secs(0)
        }
    };
    Ok(elapsed)
}

/// The offset for generation of workflow evaluation run IDs.
const WORKFLOW_EVALUATION_OFFSET_S: u64 = 10_000_000_000;
/// It is ten billion seconds (~317 years)
pub const WORKFLOW_EVALUATION_OFFSET: Duration = Duration::from_secs(WORKFLOW_EVALUATION_OFFSET_S);

/// The threshold for generation of workflow evaluation run IDs.
/// This will seed the UUIDv7 with current time + 10 billion seconds.
/// We ignore nanoseconds, sequence number, and usable bits.
pub const WORKFLOW_EVALUATION_THRESHOLD: Timestamp = Timestamp::from_unix_time(
    WORKFLOW_EVALUATION_OFFSET_S,
    0, // ns
    0, // seq
    0, // bits
);

pub fn get_workflow_evaluation_cutoff_uuid() -> Uuid {
    Uuid::new_v7(WORKFLOW_EVALUATION_THRESHOLD)
}

#[expect(clippy::missing_panics_doc)]
pub fn generate_workflow_evaluation_run_episode_id() -> Uuid {
    #[expect(clippy::expect_used)]
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let now_plus_offset = now + WORKFLOW_EVALUATION_OFFSET;
    let timestamp = Timestamp::from_unix_time(
        now_plus_offset.as_secs(),
        now_plus_offset.subsec_nanos(),
        0, // counter
        0, // usable_counter_bits
    );
    Uuid::new_v7(timestamp)
}

/// Compares two UUID timestamps to determine if the first one is earlier than the second.
///
/// # Arguments
///
/// * `early` - The timestamp expected to be earlier
/// * `late` - The timestamp expected to be later
///
/// # Returns
///
/// * `true` if `early` is chronologically before `late`
/// * `false` otherwise
pub fn compare_timestamps(early: Timestamp, late: Timestamp) -> bool {
    let (early_s, early_ns) = early.to_unix();
    let (late_s, late_ns) = late.to_unix();
    early_s < late_s || (early_s == late_s && early_ns < late_ns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::timestamp::context::NoContext;
    use uuid::uuid;

    #[test]
    fn test_validate_episode_id() {
        let episode_id = Uuid::now_v7();
        assert!(validate_tensorzero_uuid(episode_id, "Episode").is_ok());

        let episode_id = uuid!("6790f6a1-3f8b-427e-ae24-f309329b9b0a");
        assert!(validate_tensorzero_uuid(episode_id, "Episode").is_err());

        let episode_id = uuid!("00000000-0000-0000-0000-000000000000");
        assert!(validate_tensorzero_uuid(episode_id, "Episode").is_err());

        let early_timestamp = 946684800; // 2000-01-01:00:00:00 UTC
        let early_uuid = Uuid::new_v7(Timestamp::from_unix(NoContext, early_timestamp, 0));
        assert!(validate_tensorzero_uuid(early_uuid, "Episode").is_err());

        let late_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs()
            + 10;
        let late_uuid = Uuid::new_v7(Timestamp::from_unix(NoContext, late_timestamp, 0));
        assert!(validate_tensorzero_uuid(late_uuid, "Episode").is_err());
    }

    #[test]
    fn test_uuid_elapsed() {
        // Get current time
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        // Subtract 5 seconds
        let five_seconds_ago = now
            .checked_sub(std::time::Duration::from_secs(5))
            .expect("Timestamp arithmetic overflow");

        // Extract seconds and subsec_nanos
        let seconds = five_seconds_ago.as_secs();
        let subsec_nanos = five_seconds_ago.subsec_nanos();

        // Create the timestamp
        let timestamp = Timestamp::from_unix_time(seconds, subsec_nanos, 0, 0);
        let uuid = Uuid::new_v7(timestamp);
        let elapsed = uuid_elapsed(&uuid).unwrap();
        assert!(elapsed > Duration::from_secs(4) && elapsed < Duration::from_secs(6));

        let uuid = Uuid::now_v7();
        let elapsed = uuid_elapsed(&uuid).unwrap();
        // It is not guaranteed that the elapsed time is exactly 0, so we allow a small margin of error
        assert!(elapsed > Duration::from_secs(0) && elapsed < Duration::from_millis(10));

        // Test UUID in future
        let future_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs()
            + 30;
        let future_uuid = Uuid::new_v7(Timestamp::from_unix(NoContext, future_timestamp, 0));
        assert_eq!(
            uuid_elapsed(&future_uuid).unwrap_err().get_details(),
            &ErrorDetails::UuidInFuture {
                raw_uuid: future_uuid.to_string(),
            }
        );

        // Test workflow evaluation threshold
        let now_for_workflow = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let five_seconds_ago_workflow_offset = now_for_workflow
            .checked_sub(std::time::Duration::from_secs(5))
            .expect("Timestamp arithmetic overflow");

        let timestamp_with_offset = Timestamp::from_unix_time(
            five_seconds_ago_workflow_offset.as_secs() + WORKFLOW_EVALUATION_OFFSET_S,
            five_seconds_ago_workflow_offset.subsec_nanos(),
            0,
            0,
        );
        let workflow_uuid = Uuid::new_v7(timestamp_with_offset);
        let elapsed_workflow = uuid_elapsed(&workflow_uuid).unwrap();
        assert!(
            elapsed_workflow > Duration::from_secs(0) && elapsed_workflow < Duration::from_secs(15),
            "Elapsed time for workflow UUID should be around 5s, got {elapsed_workflow:?}"
        );
    }

    #[test]
    fn test_generate_workflow_evaluation_run_episode_id() {
        let workflow_id = generate_workflow_evaluation_run_episode_id();

        // Verify it's a v7 UUID
        assert_eq!(workflow_id.get_version_num(), 7);

        // Extract the timestamp and verify it's in the expected range
        let timestamp_info = workflow_id.get_timestamp().expect("Should have timestamp");
        let (seconds, _) = timestamp_info.to_unix();

        // Current timestamp plus the offset
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let expected_approx_time = now + WORKFLOW_EVALUATION_OFFSET_S;

        // Allow small difference due to execution time
        let margin = 5; // 5 seconds margin
        assert!(
            seconds >= expected_approx_time - margin && seconds <= expected_approx_time + margin,
            "Expected timestamp around {expected_approx_time}, got {seconds}"
        );

        // Generate two UUIDs and ensure they're different
        let id1 = generate_workflow_evaluation_run_episode_id();
        let id2 = generate_workflow_evaluation_run_episode_id();
        assert_ne!(id1, id2, "Generated UUIDs should be unique");
    }

    #[test]
    fn test_compare_timestamps() {
        use uuid::NoContext;
        use uuid::Timestamp;

        // Case 1: First timestamp is before the second timestamp
        let timestamp1 = Timestamp::from_unix(NoContext, 1000, 0);
        let timestamp2 = Timestamp::from_unix(NoContext, 2000, 0);
        assert!(compare_timestamps(timestamp1, timestamp2));

        // Case 2: First timestamp is equal to the second timestamp
        let timestamp3 = Timestamp::from_unix(NoContext, 3000, 0);
        let timestamp4 = Timestamp::from_unix(NoContext, 3000, 0);
        assert!(!compare_timestamps(timestamp3, timestamp4));

        // Case 3: First timestamp is after the second timestamp
        let timestamp5 = Timestamp::from_unix(NoContext, 5000, 0);
        let timestamp6 = Timestamp::from_unix(NoContext, 4000, 0);
        assert!(!compare_timestamps(timestamp5, timestamp6));

        // Case 4: Subsecond precision comparison
        let timestamp7 = Timestamp::from_unix(NoContext, 6000, 499);
        let timestamp8 = Timestamp::from_unix(NoContext, 6000, 500);
        assert!(compare_timestamps(timestamp7, timestamp8));

        // Case 5: Subsecond precision equal
        let timestamp9 = Timestamp::from_unix(NoContext, 7000, 500);
        let timestamp10 = Timestamp::from_unix(NoContext, 7000, 500);
        assert!(!compare_timestamps(timestamp9, timestamp10));
    }
}
