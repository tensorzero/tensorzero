mod common;

optimization_test_case!(openai_sft, common::openai_sft::OpenAISFTTestCase());
optimization_test_case!(fireworks_sft, common::fireworks_sft::FireworksSFTTestCase());
optimization_test_case!(gcp_vertex_gemini_sft, GCPVertexGeminiSFTTestCase());
optimization_test_case!(together_sft, TogetherSFTTestCase());
