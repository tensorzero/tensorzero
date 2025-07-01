import { describe, it, expect } from "vitest";
import { get_fine_tuned_provider_config, dump_provider_config } from "./models";

describe("get_fine_tuned_model_config", () => {
  it("should create correct config for anthropic model", async () => {
    const result = get_fine_tuned_provider_config("claude-2", "anthropic");
    expect(result).toEqual({
      models: {
        "claude-2": {
          routing: ["claude-2"],
          providers: {
            "claude-2": {
              type: "anthropic",
              model_name: "claude-2",
            },
          },
        },
      },
    });
  });

  it("should create correct config for aws bedrock model", async () => {
    const result = get_fine_tuned_provider_config(
      "anthropic.claude-v2",
      "aws_bedrock",
    );
    expect(result).toEqual({
      models: {
        "anthropic.claude-v2": {
          routing: ["anthropic.claude-v2"],
          providers: {
            "anthropic.claude-v2": {
              type: "aws_bedrock",
              model_id: "anthropic.claude-v2",
            },
          },
        },
      },
    });
  });
});

describe("dump_model_config", () => {
  it("should correctly stringify model config", () => {
    const fullyQualifiedModelConfig = get_fine_tuned_provider_config(
      "test",
      "dummy",
    );

    const result = dump_provider_config("test", fullyQualifiedModelConfig);
    const expected_result =
      '[models.test]\nrouting = [ "test" ]\n\n[models.test.providers.test]\ntype = "dummy"\nmodel_name = "test"';
    expect(result).toBe(expected_result);
  });
});
