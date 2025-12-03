use common::dicl::{test_dicl_optimization_chat, test_dicl_optimization_json};
use common::gepa::{test_gepa_optimization_chat, test_gepa_optimization_json};

mod common;

// Fine Tuning Tests
optimization_test_case!(openai_rft, common::openai_rft::OpenAIRFTTestCase());
optimization_test_case!(openai_sft, common::openai_sft::OpenAISFTTestCase());
optimization_test_case!(fireworks_sft, common::fireworks_sft::FireworksSFTTestCase());
optimization_test_case!(
    gcp_vertex_gemini_sft,
    common::gcp_vertex_gemini_sft::GCPVertexGeminiSFTTestCase()
);
optimization_test_case!(together_sft, common::together_sft::TogetherSFTTestCase());

// DICL Tests
#[tokio::test(flavor = "multi_thread")]
async fn test_slow_optimization_dicl() {
    Box::pin(test_dicl_optimization_chat()).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_slow_optimization_dicl_json() {
    Box::pin(test_dicl_optimization_json()).await;
}

// GEPA Tests
#[tokio::test(flavor = "multi_thread")]
async fn test_slow_optimization_gepa_chat() {
    test_gepa_optimization_chat().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_slow_optimization_gepa_json() {
    test_gepa_optimization_json().await;
}
