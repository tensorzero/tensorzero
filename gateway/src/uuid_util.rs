use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::error::Error;

/// Timestamp when Scaling Laws for Neural Language Models was published.
/// No way anyone could use TensorZero prior to this.
const EARLIEST_TIMESTAMP: u64 = 1579751960;

pub fn validate_episode_id(episode_id: Uuid) -> Result<(), Error> {
    let version = episode_id.get_version_num();
    if version != 7 {
        return Err(Error::InvalidEpisodeId {
            message: format!("Version must be 7, got {}", version),
        });
    }
    let (timestamp, _) = episode_id
        .get_timestamp()
        .ok_or(Error::InvalidEpisodeId {
            message: "Timestamp is missing".to_string(),
        })?
        .to_unix();
    if timestamp < EARLIEST_TIMESTAMP {
        return Err(Error::InvalidEpisodeId {
            message: "Timestamp is too early".to_string(),
        });
    }
    #[allow(clippy::expect_used)]
    let current_timestamp: u64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();
    if timestamp > current_timestamp {
        return Err(Error::InvalidEpisodeId {
            message: "Timestamp is in the future".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::timestamp::context::NoContext;
    use uuid::{uuid, Timestamp};

    #[test]
    fn test_validate_episode_id() {
        let episode_id = Uuid::now_v7();
        assert!(validate_episode_id(episode_id).is_ok());

        let episode_id = uuid!("6790f6a1-3f8b-427e-ae24-f309329b9b0a");
        assert!(validate_episode_id(episode_id).is_err());

        let episode_id = uuid!("00000000-0000-0000-0000-000000000000");
        assert!(validate_episode_id(episode_id).is_err());

        let early_timestamp = EARLIEST_TIMESTAMP - 1;
        let early_uuid = Uuid::new_v7(Timestamp::from_unix(NoContext, early_timestamp, 0));
        assert!(validate_episode_id(early_uuid).is_err());

        let late_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs()
            + 10;
        let late_uuid = Uuid::new_v7(Timestamp::from_unix(NoContext, late_timestamp, 0));
        assert!(validate_episode_id(late_uuid).is_err());
    }
}
