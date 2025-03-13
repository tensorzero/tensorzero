use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

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
            message: format!("Version must be 7, got {}", version),
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

pub fn uuid_elapsed(uuid: &Uuid) -> Result<Duration, Error> {
    let version = uuid.get_version_num();
    if version != 7 {
        return Err(ErrorDetails::InvalidUuid {
            raw_uuid: uuid.to_string(),
        }
        .into());
    }
    let (seconds, subsec_nanos) = uuid
        .get_timestamp()
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidUuid {
                raw_uuid: uuid.to_string(),
            })
        })?
        .to_unix();
    let uuid_system_time =
        UNIX_EPOCH + Duration::from_secs(seconds) + Duration::from_nanos(subsec_nanos as u64);
    let elapsed = match SystemTime::now().duration_since(uuid_system_time) {
        Ok(duration) => duration,
        Err(e) => {
            let future_duration = e.duration();
            if future_duration > Duration::from_secs(1) {
                return Err(ErrorDetails::UuidInFuture {
                    raw_uuid: uuid.to_string(),
                }
                .into());
            }
            Duration::from_secs(0)
        }
    };
    Ok(elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::timestamp::context::NoContext;
    use uuid::{uuid, Timestamp};

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
    }
}
