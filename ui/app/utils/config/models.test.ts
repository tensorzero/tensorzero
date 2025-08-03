import { describe, it, expect } from "vitest";
import { dump_optimizer_output } from "./models";
import type { OptimizerOutput } from "tensorzero-node";

describe("dump_optimizer_output", () => {
  it("should create correct config for fireworks model", async () => {
    const optimizerOutput: OptimizerOutput = {
      type: "model",
      routing: ["claude-2"],
      providers: {
        "claude-2": {
          type: "fireworks",
          model_name: "claude-2",
          parse_think_blocks: false,
          api_key_location: null,
          discard_unknown_chunks: false,
          extra_body: null,
          extra_headers: null,
          timeouts: null,
        },
      },
      timeouts: {
        non_streaming: {
          total_ms: null,
        },
        streaming: {
          ttft_ms: null,
        },
      },
    };
    const result_string = dump_optimizer_output(optimizerOutput);
    expect(result_string).toBe(
      '[models.claude-2]\nrouting = [ "claude-2" ]\n\n[models.claude-2.providers.claude-2]\ntype = "fireworks"\nmodel_name = "claude-2"\nparse_think_blocks = false\ndiscard_unknown_chunks = false',
    );
  });

  it("should create correct config for openai model", async () => {
    const optimizerOutput: OptimizerOutput = {
      type: "model",
      routing: ["gpt-4o"],
      providers: {
        "gpt-4o": {
          type: "openai",
          model_name: "gpt-4o",
          api_base: null,
          api_key_location: null,
          discard_unknown_chunks: false,
          extra_body: null,
          extra_headers: null,
          timeouts: null,
        },
      },
      timeouts: {
        non_streaming: {
          total_ms: null,
        },
        streaming: {
          ttft_ms: null,
        },
      },
    };
    const result_string = dump_optimizer_output(optimizerOutput);
    expect(result_string).toBe(
      '[models.gpt-4o]\nrouting = [ "gpt-4o" ]\n\n[models.gpt-4o.providers.gpt-4o]\ntype = "openai"\nmodel_name = "gpt-4o"\ndiscard_unknown_chunks = false',
    );
  });
});
