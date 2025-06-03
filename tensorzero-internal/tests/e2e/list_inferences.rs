use tensorzero::{InferenceOutputSource, ListInferencesParams, StoredInference};
use tensorzero_internal::clickhouse::ClickhouseFormat;

use crate::providers::common::make_embedded_gateway;

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_json_function() {
    let client = make_embedded_gateway().await;
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        limit: Some(1),
        offset: None,
        format: ClickhouseFormat::JsonEachRow,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    assert_eq!(res.len(), 1);
    let inference = res.first().unwrap();
    let StoredInference::Json(json_inference) = inference else {
        panic!("Expected a JSON inference");
    };
    assert_eq!(json_inference.function_name, "extract_entities");
}
