use std::borrow::Cow;

use futures::FutureExt;
use mime::MediaType;
use scoped_tls::scoped_thread_local;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use super::{ContentBlock, RequestMessage};
use crate::{
    error::{Error, ErrorDetails},
    http::TensorzeroHttpClient,
    inference::types::resolved_input::LazyFile,
};
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

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Base64File {
    // The original url we used to download the file
    pub url: Option<Url>,
    #[cfg_attr(test, ts(type = "string"))]
    pub mime_type: MediaType,
    // TODO - should we add a wrapper type to enforce base64?
    pub data: String,
}

/// Like `Base64File`, but without the data field.
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Base64FileMetadata {
    // The original url we used to download the file
    pub url: Option<Url>,
    #[cfg_attr(test, ts(type = "string"))]
    pub mime_type: MediaType,
}

impl std::fmt::Display for Base64File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl Base64File {
    pub fn data(&self) -> Result<&String, Error> {
        Ok(&self.data)
    }
}
#[cfg(feature = "pyo3")]
#[pymethods]
impl Base64File {
    #[getter(url)]
    pub fn url_string(&self) -> Option<String> {
        self.url.as_ref().map(Url::to_string)
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(untagged, deny_unknown_fields)]
pub enum File {
    Url {
        url: Url,
        #[serde(default)]
        #[ts(type = "string | null")]
        mime_type: Option<MediaType>,
    },
    Base64 {
        #[ts(type = "string")]
        mime_type: MediaType,
        data: String,
    },
}

impl File {
    pub async fn take_or_fetch(self, client: &TensorzeroHttpClient) -> Result<Base64File, Error> {
        match self {
            File::Url { url, mime_type } => {
                let response = client.get(url.clone()).send().await.map_err(|e| {
                    Error::new(ErrorDetails::BadFileFetch {
                        url: url.clone(),
                        message: format!("Error fetching image: {e:?}"),
                    })
                })?;

                // Extract headers before consuming response
                let content_type_header =
                    response.headers().get(http::header::CONTENT_TYPE).cloned();

                let bytes = response.bytes().await.map_err(|e| {
                    Error::new(ErrorDetails::BadFileFetch {
                        url: url.clone(),
                        message: format!("Error reading image bytes: {e}"),
                    })
                })?;

                let mime_type = if let Some(mime_type) = mime_type {
                    // Priority 1: Explicitly provided mime_type
                    mime_type
                } else {
                    // Priority 2: Infer from file content
                    let inferred = infer::get(&bytes);

                    if let Some(inferred_type) = inferred {
                        let inferred_mime = inferred_type
                            .mime_type()
                            .parse::<MediaType>()
                            .map_err(|e| {
                                Error::new(ErrorDetails::BadFileFetch {
                                    url: url.clone(),
                                    message: format!("Inferred mime type is not valid: {e}"),
                                })
                            })?;

                        // Check if Content-Type header differs and log warning
                        if let Some(content_type) = &content_type_header {
                            if let Ok(content_type_str) = content_type.to_str() {
                                if let Ok(header_mime) = content_type_str.parse::<MediaType>() {
                                    if header_mime != inferred_mime {
                                        tracing::warn!(
                                            "Inferred MIME type `{}` differs from Content-Type header `{}` for URL {}. The gateway will send `{}` to the model provider",
                                            inferred_mime,
                                            header_mime,
                                            url,
                                            inferred_mime,
                                        );
                                    }
                                } else {
                                    tracing::warn!("Content-Type header is not a valid mime type: `{content_type_str}`");
                                }
                            }
                        }

                        inferred_mime
                    } else if let Some(content_type) = &content_type_header {
                        // Priority 3: Content-Type header
                        content_type
                            .to_str()
                            .map_err(|e| {
                                Error::new(ErrorDetails::BadFileFetch {
                                    url: url.clone(),
                                    message: format!(
                                        "Content-Type header is not a valid string: {e}"
                                    ),
                                })
                            })?
                            .parse::<MediaType>()
                            .map_err(|e| {
                                Error::new(ErrorDetails::BadFileFetch {
                                    url: url.clone(),
                                    message: format!(
                                        "Content-Type header is not a valid mime type: {e}"
                                    ),
                                })
                            })?
                    } else {
                        return Err(Error::new(ErrorDetails::BadFileFetch {
                            url: url.clone(),
                            message:
                                "`mime_type` not provided, and unable to infer from file content or Content-Type header"
                                    .to_string(),
                        }));
                    }
                };

                let data = base64::encode(bytes);
                Ok(Base64File {
                    url: Some(url.clone()),
                    mime_type,
                    data,
                })
            }
            File::Base64 { mime_type, data } => Ok(Base64File {
                url: None,
                mime_type,
                data,
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
                let file_with_path = match &**file {
                    LazyFile::Url {
                        future,
                        file_url: _,
                    } => {
                        // If we actually sent the file bytes to some model provider, then the
                        // Shared future must be ready, so we'll get a file from `now_or_never`.
                        // Otherwise, the file cannot have been sent to a model provider (since the
                        // future was never `.await`ed before we constructed `raw_request`), so
                        // there's nothing to strip from the message.
                        // We ignore errors here, since an error during file resolution means that
                        // we cannot have included the file bytes in `raw_request`.
                        if let Some(Ok(file)) = future.clone().now_or_never() {
                            Some(Cow::Owned(file))
                        } else {
                            None
                        }
                    }
                    LazyFile::FileWithPath(file) => Some(Cow::Borrowed(file)),
                };
                if let Some(file) = file_with_path {
                    raw_request =
                        raw_request.replace(&file.file.data, &format!("<TENSORZERO_FILE_{i}>"));
                    i += 1;
                }
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
        _ if mime_type == "text/plain" => Some("txt"),
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
        resolved_input::{FileWithPath, LazyFile},
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
                            ContentBlock::File(Box::new(LazyFile::FileWithPath(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: "my-image-1-data".to_string(),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            }))),
                            ContentBlock::File(Box::new(LazyFile::FileWithPath(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: "my-image-2-data".to_string(),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-2-path")
                                        .unwrap(),
                                },
                            }))),
                            ContentBlock::File(Box::new(LazyFile::FileWithPath(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: "my-image-1-data".to_string(),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            }))),
                        ],
                    },
                    RequestMessage {
                        role: Role::User,
                        content: vec![
                            ContentBlock::File(Box::new(LazyFile::FileWithPath(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: "my-image-3-data".to_string(),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-3-path")
                                        .unwrap(),
                                },
                            }))),
                            ContentBlock::File(Box::new(LazyFile::FileWithPath(FileWithPath {
                                file: Base64File {
                                    url: None,
                                    mime_type: mime::IMAGE_JPEG,
                                    data: "my-image-1-data".to_string(),
                                },
                                storage_path: StoragePath {
                                    kind: StorageKind::Disabled,
                                    path: object_store::path::Path::parse("my-image-1-path")
                                        .unwrap(),
                                },
                            })))
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
        assert!(!logs_contain("Guessed"));
    }

    #[test]
    #[traced_test]
    fn test_guessed_mime_type_warning() {
        assert_eq!(
            filename_to_mime_type("my_file.txt").unwrap(),
            mime::TEXT_PLAIN
        );
        assert!(logs_contain("Guessed"));
    }

    #[test]
    fn test_infer_mime_type_from_bytes() {
        // Test JPEG detection (magic bytes: FF D8 FF)
        let jpeg_bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let inferred = infer::get(&jpeg_bytes).expect("Should detect JPEG");
        assert_eq!(inferred.mime_type(), "image/jpeg");
        assert_eq!(inferred.extension(), "jpg");

        // Test PNG detection (magic bytes: 89 50 4E 47 0D 0A 1A 0A)
        let png_bytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let inferred = infer::get(&png_bytes).expect("Should detect PNG");
        assert_eq!(inferred.mime_type(), "image/png");
        assert_eq!(inferred.extension(), "png");

        // Test GIF detection (magic bytes: 47 49 46 38 39 61)
        let gif_bytes = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61];
        let inferred = infer::get(&gif_bytes).expect("Should detect GIF");
        assert_eq!(inferred.mime_type(), "image/gif");
        assert_eq!(inferred.extension(), "gif");

        // Test PDF detection (magic bytes: %PDF)
        let pdf_bytes = b"%PDF-1.4\n";
        let inferred = infer::get(pdf_bytes).expect("Should detect PDF");
        assert_eq!(inferred.mime_type(), "application/pdf");
        assert_eq!(inferred.extension(), "pdf");

        // Test WebP detection (magic bytes: RIFF....WEBP)
        let webp_bytes = b"RIFF\x00\x00\x00\x00WEBPVP8 ";
        let inferred = infer::get(webp_bytes).expect("Should detect WebP");
        assert_eq!(inferred.mime_type(), "image/webp");
        assert_eq!(inferred.extension(), "webp");

        // Test that unknown bytes return None
        let unknown_bytes = [0x00, 0x01, 0x02, 0x03];
        assert!(infer::get(&unknown_bytes).is_none());
    }
}
