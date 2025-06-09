use mime::MediaType;
use scoped_tls::scoped_thread_local;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use super::{resolved_input::FileWithPath, ContentBlock, RequestMessage};
use crate::error::{Error, ErrorDetails};
use aws_smithy_types::base64;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

scoped_thread_local!(static SERIALIZE_FILE_DATA: ());

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FileEncoding {
    Base64,
    Url,
}

pub fn require_image(mime_type: &MediaType, provider_type: &str) -> Result<(), Error> {
    if mime_type.type_() != mime::IMAGE {
        return Err(Error::new(ErrorDetails::UnsupportedContentBlockType {
            content_block_type: format!("file: {mime_type}"),
            provider_type: provider_type.to_string(),
        }));
    }
    Ok(())
}

fn skip_serialize_file_data(_: &Option<String>) -> bool {
    !SERIALIZE_FILE_DATA.is_set()
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Base64File {
    // The original url we used to download the file
    pub url: Option<Url>,
    pub mime_type: MediaType,
    // TODO - should we add a wrapper type to enforce base64?
    #[serde(skip_serializing_if = "skip_serialize_file_data")]
    #[serde(default)]
    // This is normally `Some`, unless it was deserialized from ClickHouse
    // (with the image data stripped out).
    pub data: Option<String>,
}

impl std::fmt::Display for Base64File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl Base64File {
    pub fn data(&self) -> Result<&String, Error> {
        self.data.as_ref().ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: "Tried to get image data from deserialized Base64File".to_string(),
            })
        })
    }
}
#[cfg(feature = "pyo3")]
#[pymethods]
impl Base64File {
    #[getter(url)]
    pub fn url_string(&self) -> Option<String> {
        self.url.as_ref().map(|u| u.to_string())
    }

    #[getter(mime_type)]
    pub fn mime_type_string(&self) -> String {
        self.mime_type.to_string()
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

pub fn serialize_with_file_data<T: Serialize>(value: &T) -> Result<Value, Error> {
    SERIALIZE_FILE_DATA.set(&(), || {
        serde_json::to_value(value).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing value: {e}"),
            })
        })
    })
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum File {
    Url {
        url: Url,
        #[serde(default)]
        mime_type: Option<MediaType>,
    },
    Base64 {
        mime_type: MediaType,
        data: String,
    },
}

impl File {
    pub async fn take_or_fetch(self, client: &reqwest::Client) -> Result<Base64File, Error> {
        match self {
            File::Url { url, mime_type } => {
                let response = client.get(url.clone()).send().await.map_err(|e| {
                    Error::new(ErrorDetails::BadImageFetch {
                        url: url.clone(),
                        message: format!("Error fetching image: {e:?}"),
                    })
                })?;

                let mime_type = if let Some(mime_type) = mime_type {
                    mime_type
                } else if let Some(content_type) =
                    response.headers().get(http::header::CONTENT_TYPE)
                {
                    content_type
                        .to_str()
                        .map_err(|e| {
                            Error::new(ErrorDetails::BadImageFetch {
                                url: url.clone(),
                                message: format!("Content-Type header is not a valid string: {e}"),
                            })
                        })?
                        .parse::<MediaType>()
                        .map_err(|e| {
                            Error::new(ErrorDetails::BadImageFetch {
                                url: url.clone(),
                                message: format!(
                                    "Content-Type header is not a valid mime type: {e}"
                                ),
                            })
                        })?
                } else {
                    return Err(Error::new(ErrorDetails::BadImageFetch {
                        url: url.clone(),
                        message:
                            "`mime_type` not provided, and no Content-Type response header found"
                                .to_string(),
                    }));
                };

                let bytes = response.bytes().await.map_err(|e| {
                    Error::new(ErrorDetails::BadImageFetch {
                        url: url.clone(),
                        message: format!("Error reading image bytes: {e}"),
                    })
                })?;

                let data = base64::encode(bytes);
                Ok(Base64File {
                    url: Some(url.clone()),
                    mime_type,
                    data: Some(data),
                })
            }
            File::Base64 { mime_type, data } => Ok(Base64File {
                url: None,
                mime_type,
                data: Some(data),
            }),
        }
    }
}

/// Strips out image data from the raw request, replacing it with a placeholder.
/// This is a best-effort attempt to avoid filling up ClickHouse with image data.
pub fn sanitize_raw_request(input_messages: &[RequestMessage], mut raw_request: String) -> String {
    let mut i = 0;
    for message in input_messages {
        for content in &message.content {
            if let ContentBlock::File(file) = content {
                let FileWithPath {
                    file,
                    storage_path: _,
                } = &**file;
                if let Some(data) = &file.data {
                    raw_request = raw_request.replace(data, &format!("<TENSORZERO_FILE_{i}>"));
                }
                i += 1;
            }
        }
    }
    raw_request
}

/// Tries to convert a mime type to a file extension, picking an arbitrary extension if there are multiple
/// extensions for the mime type.
/// This is used when writing a file input to object storage, and when determining the file name
/// to provide to OpenAI (which doesn't accept mime types for file input)
pub fn mime_type_to_ext(mime_type: &MediaType) -> Result<Option<&'static str>, Error> {
    Ok(match mime_type {
        _ if mime_type == &mime::IMAGE_JPEG => Some("jpg"),
        _ if mime_type == &mime::IMAGE_PNG => Some("png"),
        _ if mime_type == &mime::IMAGE_GIF => Some("gif"),
        _ if mime_type == &mime::APPLICATION_PDF => Some("pdf"),
        _ if mime_type == "image/webp" => Some("webp"),
        _ => {
            let guess = mime_guess::get_mime_extensions_str(mime_type.as_ref())
                .and_then(|types| types.last());
            if guess.is_some() {
                tracing::warn!("Guessed file extension {guess:?} for mime-type {mime_type} - this may not be correct");
            }
            guess.copied()
        }
    })
}

/// Tries to convert a filename to a mime type, based on the file extension.
/// This picks an arbitrary mime type if there are multiple mime types for the extension.
///
/// This is used by the openai-compatible endpoint to determine the mime type for
/// a file input (the OpenAI chat-completions endpoint doesn't provide a mime type for
/// file inputs)
pub fn filename_to_mime_type(filename: &str) -> Result<MediaType, Error> {
    let ext = filename.split('.').next_back().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "File name must contain a file extension".to_string(),
        })
    })?;

    Ok(match ext {
        "jpeg" | "jpg" => mime::IMAGE_JPEG,
        "png" => mime::IMAGE_PNG,
        "gif" => mime::IMAGE_GIF,
        "pdf" => mime::APPLICATION_PDF,
        "webp" => "image/webp".parse::<MediaType>().map_err(|_| {
            Error::new(ErrorDetails::InternalError {
                message: "Unknown mime-type `image/webp`. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports.".to_string(),
            })
        })?,
        _ => {
            let mime_type = mime_guess::from_ext(ext).first().ok_or_else(|| {
                Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: format!("Unknown file extension `{ext}`"),
                })
            })?;
            tracing::warn!("Guessed mime-type `{mime_type}` for file with extension `{ext}` - this may not be correct");
            // Reparse to handle different `mime` crate versions
            mime_type.to_string().parse::<MediaType>().map_err(|_| {
                Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: format!("Unknown mime-type `{mime_type}`"),
                })
            })?
        }
    })
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use crate::inference::types::{
        file::{filename_to_mime_type, sanitize_raw_request},
        resolved_input::FileWithPath,
        storage::{StorageKind, StoragePath},
        Base64File, ContentBlock, RequestMessage, Role,
    };

    #[test]
    fn test_sanitize_input() {
        assert_eq!(
            sanitize_raw_request(&[], "my-fake-input".to_string()),
            "my-fake-input"
        );

        assert_eq!(
            sanitize_raw_request(
                &[
                    RequestMessage {
                        role: Role::User,
                        content: vec![
                            ContentBlock::File(Box::new(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: Some("my-image-1-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            })),
                            ContentBlock::File(Box::new(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: Some("my-image-2-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-2-path")
                                        .unwrap(),
                                },
                            })),
                            ContentBlock::File(Box::new(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: Some("my-image-1-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            })),
                        ],
                    },
                    RequestMessage {
                        role: Role::User,
                        content: vec![
                            ContentBlock::File(Box::new(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: Some("my-image-3-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-3-path")
                                        .unwrap(),
                                },
                            })),
                            ContentBlock::File(Box::new(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: Some("my-image-1-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            }))
                        ],
                    }
                ],
                "First my-image-1-data then my-image-2-data then my-image-3-data".to_string()
            ),
            // Each occurrence of the file data should be replaced with the first matching file content block
            "First <TENSORZERO_FILE_0> then <TENSORZERO_FILE_1> then <TENSORZERO_FILE_3>"
                .to_string()
        );
    }

    #[test]
    #[traced_test]
    fn test_filename_to_mime_type() {
        assert_eq!(filename_to_mime_type("test.png").unwrap(), mime::IMAGE_PNG);
        assert_eq!(filename_to_mime_type("test.jpg").unwrap(), mime::IMAGE_JPEG);
        assert_eq!(
            filename_to_mime_type("test.jpeg").unwrap(),
            mime::IMAGE_JPEG
        );
        assert_eq!(filename_to_mime_type("test.gif").unwrap(), mime::IMAGE_GIF);
        assert_eq!(filename_to_mime_type("test.webp").unwrap(), "image/webp");
        assert_eq!(
            filename_to_mime_type("test.pdf").unwrap(),
            mime::APPLICATION_PDF
        );
        assert!(!logs_contain("Guessed"))
    }

    #[test]
    #[traced_test]
    fn test_guessed_mime_type_warning() {
        assert_eq!(
            filename_to_mime_type("my_file.txt").unwrap(),
            mime::TEXT_PLAIN
        );
        assert!(logs_contain("Guessed"))
    }
}
