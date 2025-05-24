use clickhouse::Client;

async fn handle_tensorzero_inference(
    Json(payload): Json<TensorZeroInferenceRequest>,
    clickhouse: Extension<Client>,
) -> Result<Json<TensorZeroResponse>, StatusCode> {
    if let Some(pdf) = payload.pdf {
        let document = Pdfium::new(Pdfium::bind_to_system_library()?).load_pdf_from_file(pdf.file_path.unwrap_or_default())?;
        let text = document.pages().iter().map(|page| page.text().unwrap_or_default()).collect::<Vec<_>>().join("\n");
        clickhouse.insert("pdf_metadata", PdfMetadata { file_name: pdf.file_path.unwrap_or_default(), text })?;
    }
    Ok(Json(TensorZeroResponse { /* ... */ }))
}
