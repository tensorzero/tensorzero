use common::dicl::{test_dicl_workflow_with_embedded_client, test_dicl_workflow_with_http_client};

mod common;

// SFT workflow tests
embedded_workflow_test_case!(fireworks_sft, common::fireworks_sft::FireworksSFTTestCase());
http_workflow_test_case!(fireworks_sft, common::fireworks_sft::FireworksSFTTestCase());

embedded_workflow_test_case!(
    gcp_vertex_gemini_sft,
    common::gcp_vertex_gemini_sft::GCPVertexGeminiSFTTestCase()
);
http_workflow_test_case!(
    gcp_vertex_gemini_sft,
    common::gcp_vertex_gemini_sft::GCPVertexGeminiSFTTestCase()
);

embedded_workflow_test_case!(openai_rft, common::openai_rft::OpenAIRFTTestCase());
http_workflow_test_case!(openai_rft, common::openai_rft::OpenAIRFTTestCase());

embedded_workflow_test_case!(openai_sft, common::openai_sft::OpenAISFTTestCase());
http_workflow_test_case!(openai_sft, common::openai_sft::OpenAISFTTestCase());

embedded_workflow_test_case!(together_sft, common::together_sft::TogetherSFTTestCase());
http_workflow_test_case!(together_sft, common::together_sft::TogetherSFTTestCase());

// DICL workflow tests
#[tokio::test(flavor = "multi_thread")]
async fn test_embedded_slow_optimization_dicl() {
    test_dicl_workflow_with_embedded_client().await;
}

#[tokio::test]
async fn test_http_slow_optimization_dicl() {
    test_dicl_workflow_with_http_client().await;
}
