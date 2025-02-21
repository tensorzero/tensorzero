use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};
use url::Url;

use crate::error::{Error, ErrorDetails};
use aws_smithy_types::base64;

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
                        message: format!("Error fetching image: {e}"),
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
