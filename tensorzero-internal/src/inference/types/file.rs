use std::fmt::{self, Display};

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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum FileKind {
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/webp")]
    WebP,
    #[serde(rename = "application/pdf")]
    Pdf,
}

impl FileKind {
    /// Produces an error if this is not an image
    pub fn require_image(&self, provider_type: &str) -> Result<(), Error> {
        if !self.is_image() {
            return Err(Error::new(ErrorDetails::UnsupportedContentBlockType {
                content_block_type: format!("file: {self}"),
                provider_type: provider_type.to_string(),
            }));
        }
        Ok(())
    }

    pub fn is_image(&self) -> bool {
        match self {
            FileKind::Jpeg | FileKind::Png | FileKind::WebP => true,
            FileKind::Pdf => false,
        }
    }
}

impl TryFrom<&str> for FileKind {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let extension = value.split('.').last().ok_or_else(|| {
            Error::new(ErrorDetails::MissingFileExtension {
                file_name: value.to_string(),
            })
        })?;
        match extension {
            "jpg" => Ok(FileKind::Jpeg),
            "jpeg" => Ok(FileKind::Jpeg),
            "png" => Ok(FileKind::Png),
            "webp" => Ok(FileKind::WebP),
            "pdf" => Ok(FileKind::Pdf),
            _ => Err(Error::new(ErrorDetails::UnsupportedFileExtension {
                extension: extension.to_string(),
            })),
        }
    }
}

impl Display for FileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileKind::Jpeg => write!(f, "image/jpeg"),
            FileKind::Png => write!(f, "image/png"),
            FileKind::WebP => write!(f, "image/webp"),
            FileKind::Pdf => write!(f, "application/pdf"),
        }
    }
}

fn skip_serialize_file_data(_: &Option<String>) -> bool {
    !SERIALIZE_FILE_DATA.is_set()
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Base64File {
    // The original url we used to download the file
    pub url: Option<Url>,
    pub mime_type: FileKind,
    // TODO - should we add a wrapper type to enforce base64?
    #[serde(skip_serializing_if = "skip_serialize_file_data")]
    #[serde(default)]
    // This is normally `Some`, unless it was deserialized from ClickHouse
    // (with the image data stripped out).
    pub data: Option<String>,
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
    Url { url: Url },
    Base64 { mime_type: FileKind, data: String },
}

impl File {
    pub async fn take_or_fetch(self, client: &reqwest::Client) -> Result<Base64File, Error> {
        match self {
            File::Url { url } => {
                let response = client.get(url.clone()).send().await.map_err(|e| {
                    Error::new(ErrorDetails::BadImageFetch {
                        url: url.clone(),
                        message: format!("Error fetching image: {e:?}"),
                    })
                })?;
                let bytes = response.bytes().await.map_err(|e| {
                    Error::new(ErrorDetails::BadImageFetch {
                        url: url.clone(),
                        message: format!("Error reading image bytes: {e}"),
                    })
                })?;
                let kind = match image::guess_format(&bytes) {
                    Ok(image::ImageFormat::Jpeg) => FileKind::Jpeg,
                    Ok(image::ImageFormat::Png) => FileKind::Png,
                    Ok(image::ImageFormat::WebP) => FileKind::WebP,
                    Ok(format) => {
                        return Err(Error::new(ErrorDetails::BadImageFetch {
                            url: url.clone(),
                            message: format!("Unsupported image format: {format:?}"),
                        }))
                    }
                    Err(e) => {
                        return Err(Error::new(ErrorDetails::BadImageFetch {
                            url,
                            message: format!("Error guessing image format: {e}"),
                        }))
                    }
                };
                let data = base64::encode(bytes);
                Ok(Base64File {
                    url: Some(url.clone()),
                    mime_type: kind,
                    data: Some(data),
                })
            }
            File::Base64 {
                mime_type: kind,
                data,
            } => Ok(Base64File {
                url: None,
                mime_type: kind,
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
            if let ContentBlock::File(FileWithPath {
                file,
                storage_path: _,
            }) = content
            {
                if let Some(data) = &file.data {
                    raw_request = raw_request.replace(data, &format!("<TENSORZERO_FILE_{i}>"));
                }
                i += 1;
            }
        }
    }
    raw_request
}

#[cfg(test)]
mod tests {
    use crate::inference::types::{
        file::sanitize_raw_request,
        resolved_input::FileWithPath,
        storage::{StorageKind, StoragePath},
        Base64File, ContentBlock, FileKind, RequestMessage, Role,
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
                            ContentBlock::File(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: FileKind::Jpeg,
                                    data: Some("my-image-1-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            }),
                            ContentBlock::File(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: FileKind::Jpeg,
                                    data: Some("my-image-2-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-2-path")
                                        .unwrap(),
                                },
                            }),
                            ContentBlock::File(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: FileKind::Jpeg,
                                    data: Some("my-image-1-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            }),
                        ],
                    },
                    RequestMessage {
                        role: Role::User,
                        content: vec![
                            ContentBlock::File(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: FileKind::Jpeg,
                                    data: Some("my-image-3-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-3-path")
                                        .unwrap(),
                                },
                            }),
                            ContentBlock::File(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: FileKind::Jpeg,
                                    data: Some("my-image-1-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            })
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
}
