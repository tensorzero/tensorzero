use secrecy::SecretString;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// A parsed TensorZero API key, with the long key hashed.
/// This does not contain the original long key
#[derive(Debug)]
#[expect(dead_code)]
pub struct TensorZeroApiKey {
    short_id: String,
    hashed_long_key: SecretString,
}

const SK_PREFIX: &str = "sk";
const T0_PREFIX: &str = "t0";
const SHORT_ID_LENGTH: usize = 12;
const LONG_KEY_LENGTH: usize = 48;

impl TensorZeroApiKey {
    /// Hashes the long key using SHA-256, producing a hex string that we'll use for a database lookup
    fn hash_long_key(long_key: &str) -> String {
        hex::encode(Sha256::digest(long_key.as_bytes()))
    }

    /// Validates that the provided key is of the format `sk-t0-<short_id>-<long_key>`,
    /// where <short_id> is 12 alphanumeric characters and <long_key> is 48 alphanumeric characters.
    /// Returns a `TensorZeroApiKey` containing the extracted short ID and long key.
    pub fn parse(key: &str) -> Result<Self, TensorZeroAuthError> {
        let parts = key.split('-').collect::<Vec<&str>>();
        let [sk, t0, short_id, long_key] = parts.as_slice() else {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "API key must be of the form `sk-t0-<short_id>-<long_key>`",
            ));
        };
        if sk != &SK_PREFIX {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "API key must start with `sk-t0-`",
            ));
        }
        if t0 != &T0_PREFIX {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "API key must start with `sk-t0-`",
            ));
        }
        if short_id.len() != SHORT_ID_LENGTH {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "Short ID must be 12 characters",
            ));
        }
        if long_key.len() != LONG_KEY_LENGTH {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "Long key must be 48 characters",
            ));
        }
        if !short_id.chars().all(char::is_alphanumeric) {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "Short ID must be alphanumeric",
            ));
        }
        if !long_key.chars().all(char::is_alphanumeric) {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "Long key must be alphanumeric",
            ));
        }
        Ok(Self {
            short_id: short_id.to_string(),
            hashed_long_key: Self::hash_long_key(long_key).into(),
        })
    }
}

#[derive(Debug, Error)]
pub enum TensorZeroAuthError {
    #[error("Invalid format for TensorZero API key: {0}")]
    InvalidKeyFormat(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_invalid_key() {
        assert_eq!(
            TensorZeroApiKey::parse("invalid-key")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: API key must be of the form `sk-t0-<short_id>-<long_key>`"
        );
        assert_eq!(
            TensorZeroApiKey::parse("too-many-dashes-in-my-api-key")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: API key must be of the form `sk-t0-<short_id>-<long_key>`"
        );
        assert_eq!(
            TensorZeroApiKey::parse("bad-with-four-components")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: API key must start with `sk-t0-`"
        );
        assert_eq!(
            TensorZeroApiKey::parse("sk-other-123456789012-12345678901234567890123456789012")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: API key must start with `sk-t0-`"
        );
        assert_eq!(
            TensorZeroApiKey::parse("sk-t0-12-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: Short ID must be 12 characters"
        );
        assert_eq!(
            TensorZeroApiKey::parse("sk-t0-aaaaaaaaaaaa-bb")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: Long key must be 48 characters"
        );
        assert_eq!(
            TensorZeroApiKey::parse(
                "sk-t0-!!!!!!!!!!!!-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )
            .unwrap_err()
            .to_string(),
            "Invalid format for TensorZero API key: Short ID must be alphanumeric"
        );
        assert_eq!(
            TensorZeroApiKey::parse(
                "sk-t0-aaaaaaaaaaaa-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!"
            )
            .unwrap_err()
            .to_string(),
            "Invalid format for TensorZero API key: Long key must be alphanumeric"
        );
    }

    #[test]
    fn test_parse_valid_key() {
        assert_eq!(
            format!(
                "{:?}",
                TensorZeroApiKey::parse(
                    "sk-t0-123456789012-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                )
                .unwrap()
            ),
            "TensorZeroApiKey { short_id: \"123456789012\", hashed_long_key: SecretBox<str>([REDACTED]) }"
        );
    }
}
