use crate::common::start_gateway_on_random_port;

mod common;

#[tokio::test]
async fn test_no_error_json() {
    let child_data = start_gateway_on_random_port("", None).await;
    let inference_response = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        inference_response,
        r#"{"error":"Failed to parse the request body as JSON: EOF while parsing a value at line 1 column 0 (400 Bad Request)"}"#
    );
}

#[tokio::test]
async fn test_error_json() {
    let child_data = start_gateway_on_random_port("unstable_error_json = true", None).await;
    let inference_response = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        inference_response,
        r#"{"error":"Failed to parse the request body as JSON: EOF while parsing a value at line 1 column 0 (400 Bad Request)","error_json":{"JsonRequest":{"message":"Failed to parse the request body as JSON: EOF while parsing a value at line 1 column 0 (400 Bad Request)"}}}"#
    );
}
