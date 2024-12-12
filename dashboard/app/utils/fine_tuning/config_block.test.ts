import {
  get_fine_tuned_model_config,
  dump_model_config,
  FullyQualifiedModelConfig,
} from "./config_block";
import { describe, it, expect } from "vitest";

describe("config_block functions", () => {
  describe("get_fine_tuned_model_config", () => {
    it("should create correct model config for OpenAI model", async () => {
      const result = await get_fine_tuned_model_config("ft:gpt-3.5", "openai");

      expect(result).toEqual({
        models: {
          "ft:gpt-3.5": {
            routing: ["ft:gpt-3.5"],
            providers: {
              "ft:gpt-3.5": {
                type: "openai",
                model_name: "ft:gpt-3.5",
              },
            },
          },
        },
      });
    });
  });

  describe("dump_model_config", () => {
    it("should convert model config to TOML string", () => {
      const modelConfig: FullyQualifiedModelConfig = {
        models: {
          "test-model": {
            routing: ["test-model"],
            providers: {
              "test-model": {
                type: "openai" as const,
                model_name: "test-model",
              },
            },
          },
        },
      };

      const result = dump_model_config(modelConfig);
      expect(result).toContain('routing = [ "test-model" ]');
      expect(result).toContain("[models.test-model]");
      expect(result).toContain('type = "openai"');
      expect(result).toContain('model_name = "test-model"');
    });
  });
});
