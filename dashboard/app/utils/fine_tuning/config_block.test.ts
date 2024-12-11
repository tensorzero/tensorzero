import {
  get_fine_tuned_model_config,
  dump_model_config,
  create_dump_variant_config,
} from "./config_block";
import { describe, it, expect } from "vitest";

describe("config_block functions", () => {
  describe("get_fine_tuned_model_config", () => {
    it("should create correct model config for OpenAI model", async () => {
      const result = await get_fine_tuned_model_config("ft:gpt-3.5", "openai");

      expect(result).toEqual({
        routing: ["ft:gpt-3.5"],
        providers: {
          "ft:gpt-3.5": {
            type: "openai",
            model_name: "ft:gpt-3.5",
          },
        },
      });
    });
  });

  describe("dump_model_config", () => {
    it("should convert model config to TOML string", () => {
      const modelConfig = {
        routing: ["test-model"],
        providers: {
          "test-model": {
            type: "openai" as const,
            model_name: "test-model",
          },
        },
      };

      const result = dump_model_config(modelConfig);
      expect(result).toContain('routing = [ "test-model" ]');
      expect(result).toContain("[providers.test-model]");
      expect(result).toContain('type = "openai"');
      expect(result).toContain('model_name = "test-model"');
    });
  });

  describe("create_dump_variant_config", () => {
    it("should create correct variant config TOML", () => {
      const oldVariant = {
        type: "chat_completion" as const,
        weight: 1,
        model: "old-model",
        json_mode: "strict" as const,
        system_template: "test-template",
      };

      const result = create_dump_variant_config(
        oldVariant,
        "new-model",
        "test_function",
      );

      console.log(result);
      expect(result).toContain("[functions.test_function.variants.new-model]");
      expect(result).toContain("weight = 0");
      expect(result).toContain('model_name = "new-model"');
    });
  });
});
