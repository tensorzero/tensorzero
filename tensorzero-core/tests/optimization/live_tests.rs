mod common;

optimization_test_case!(openai_sft, common::openai_sft::OpenAISFTTestCase());
optimization_test_case!(fireworks_sft, common::fireworks_sft::FireworksSFTTestCase());
