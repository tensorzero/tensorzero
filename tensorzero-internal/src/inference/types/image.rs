use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};
use url::Url;

use crate::error::{Error, ErrorDetails};
use aws_smithy_types::base64;

use super::{resolved_input::ImageWithPath, ContentBlock, RequestMessage};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEncoding {
    Base64,
    Url,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum ImageKind {
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/webp")]
    WebP,
}

impl Display for ImageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageKind::Jpeg => write!(f, "image/jpeg"),
            ImageKind::Png => write!(f, "image/png"),
            ImageKind::WebP => write!(f, "image/webp"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Base64Image {
    // The original url we used to download the image
    pub url: Option<Url>,
    pub mime_type: ImageKind,
    // TODO - should we add a wrapper type to enforce base64?
    #[serde(skip)]
    #[serde(default)]
    // This is normally `Some`, unless it was deserialized from ClickHouse
    // (with the image data stripped out).
    pub data: Option<String>,
}

impl Base64Image {
    pub fn data(&self) -> Result<&String, Error> {
        self.data.as_ref().ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: "Tried to get image data from deserialized Base64Image".to_string(),
            })
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Image {
    Url { url: Url },
    Base64 { mime_type: ImageKind, data: String },
}

impl Image {
    pub async fn take_or_fetch(self, client: &reqwest::Client) -> Result<Base64Image, Error> {
        match self {
            Image::Url { url } => {
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
                    Ok(image::ImageFormat::Jpeg) => ImageKind::Jpeg,
                    Ok(image::ImageFormat::Png) => ImageKind::Png,
                    Ok(image::ImageFormat::WebP) => ImageKind::WebP,
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
                Ok(Base64Image {
                    url: Some(url.clone()),
                    mime_type: kind,
                    data: Some(data),
                })
            }
            Image::Base64 {
                mime_type: kind,
                data,
            } => Ok(Base64Image {
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
            if let ContentBlock::Image(ImageWithPath {
                image,
                storage_path: _,
            }) = content
            {
                if let Some(data) = &image.data {
                    raw_request = raw_request.replace(data, &format!("<TENSORZERO_IMAGE_{i}>"));
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
        image::sanitize_raw_request,
        resolved_input::ImageWithPath,
        storage::{StorageKind, StoragePath},
        Base64Image, ContentBlock, ImageKind, RequestMessage, Role,
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
                            ContentBlock::Image(ImageWithPath {
                                image: Base64Image {
                                    url: None,
                                    mime_type: ImageKind::Jpeg,
                                    data: Some("my-image-1-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            }),
                            ContentBlock::Image(ImageWithPath {
                                image: Base64Image {
                                    url: None,
                                    mime_type: ImageKind::Jpeg,
                                    data: Some("my-image-2-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-2-path")
                                        .unwrap(),
                                },
                            }),
                            ContentBlock::Image(ImageWithPath {
                                image: Base64Image {
                                    url: None,
                                    mime_type: ImageKind::Jpeg,
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
                            ContentBlock::Image(ImageWithPath {
                                image: Base64Image {
                                    url: None,
                                    mime_type: ImageKind::Jpeg,
                                    data: Some("my-image-3-data".to_string()),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-3-path")
                                        .unwrap(),
                                },
                            }),
                            ContentBlock::Image(ImageWithPath {
                                image: Base64Image {
                                    url: None,
                                    mime_type: ImageKind::Jpeg,
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
            // Each occurrence of the image data should be replaced with the first matching image content block
            "First <TENSORZERO_IMAGE_0> then <TENSORZERO_IMAGE_1> then <TENSORZERO_IMAGE_3>"
                .to_string()
        );
    }
}
