mod common;

embedded_workflow_test_case!(fireworks_sft, common::fireworks_sft::FireworksSFTTestCase());
http_workflow_test_case!(fireworks_sft, common::fireworks_sft::FireworksSFTTestCase());

embedded_workflow_test_case!(openai_sft, common::openai_sft::OpenAISFTTestCase());
http_workflow_test_case!(openai_sft, common::openai_sft::OpenAISFTTestCase());
