#![expect(clippy::print_stdout)]
use tensorzero::{InferenceOutputSource, ListInferencesParams};

use crate::providers::common::make_embedded_gateway;

#[tokio::test(flavor = "multi_thread")]
pub async fn test_simple_query_json_function() {
    let client = make_embedded_gateway().await;
    let opts = ListInferencesParams {
        function_name: "extract_entities",
        variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        limit: None,
        offset: None,
    };
    let res = client.experimental_list_inferences(opts).await.unwrap();
    println!("{:?}", res);
    panic!();
}
