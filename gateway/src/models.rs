#[derive(Deserialize, Serialize)]
struct PDFInput {
    base64_content: Option<String>,
    file_path: Option<String>,
}

#[derive(clickhouse::Row, Serialize)]
struct PdfMetadata {
    file_name: String,
    text: String,
}
