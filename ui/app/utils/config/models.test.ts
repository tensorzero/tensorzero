import { describe, it, expect } from "vitest";
import {
  get_fine_tuned_provider_config,
  dump_optimizer_output,
} from "./models";

describe("get_fine_tuned_model_config", () => {
  it("should create correct config for fireworks model", async () => {
    const result = get_fine_tuned_provider_config("claude-2", "fireworks");
    expect(result).toEqual({
      type: "fireworks",
      model_name: "claude-2",
      parse_think_blocks: false,
    });
    const result_string = dump_provider_config("claude-2", result);
    expect(result_string).toBe(
      '[models.claude-2]\nrouting = [ "claude-2" ]\n\n[models.claude-2.providers.claude-2]\ntype = "fireworks"\nmodel_name = "claude-2"\nparse_think_blocks = false',
    );
  });

  it("should create correct config for openai model", async () => {
    const result = get_fine_tuned_provider_config("gpt-4o", "openai");
    expect(result).toEqual({
      type: "openai",
      model_name: "gpt-4o",
      api_base: null,
    });
    const result_string = dump_provider_config("gpt-4o", result);
    expect(result_string).toBe(
      '[models.gpt-4o]\nrouting = [ "gpt-4o" ]\n\n[models.gpt-4o.providers.gpt-4o]\ntype = "openai"\nmodel_name = "gpt-4o"',
    );
  });
});
