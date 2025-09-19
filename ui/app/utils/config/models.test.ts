import { describe, it, expect } from "vitest";
import { dump_optimizer_output } from "./models";
import type { OptimizerOutput, TimeoutsConfig } from "tensorzero-node";

describe("dump_optimizer_output", () => {
  it("should create correct config for fireworks model", async () => {
    const optimizerOutput: OptimizerOutput = {
      type: "model",
      content: {
        routing: ["claude-2"],
        providers: {
          "claude-2": {
            type: "fireworks",
            model_name: "claude-2",
            parse_think_blocks: false,
            api_key_location: null,
            discard_unknown_chunks: false,
            timeouts: {
              non_streaming: {
                total_ms: null,
              },
              streaming: {
                ttft_ms: null,
              },
            },
            extra_body: null,
            extra_headers: null,
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
      content: {
        routing: ["gpt-4o"],
        providers: {
          "gpt-4o": {
            type: "openai",
            model_name: "gpt-4o",
            api_base: null,
            api_key_location: null,
            discard_unknown_chunks: false,
            timeouts: {} as TimeoutsConfig,
            extra_body: [
              { pointer: "/temperature", value: 0.123 },
              { pointer: "/my-delete", delete: true },
            ],
            extra_headers: [
              { name: "my-first-header", value: "my-value" },
              { name: "my-second-header", delete: true },
            ],
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
      },
    };
    const result_string = dump_optimizer_output(optimizerOutput);
    expect(result_string).toBe(`[models.gpt-4o]
routing = [ "gpt-4o" ]

[models.gpt-4o.providers.gpt-4o]
type = "openai"
model_name = "gpt-4o"
discard_unknown_chunks = false

[[models.gpt-4o.providers.gpt-4o.extra_body]]
pointer = "/temperature"
value = 0.123

[[models.gpt-4o.providers.gpt-4o.extra_body]]
pointer = "/my-delete"
delete = true

[[models.gpt-4o.providers.gpt-4o.extra_headers]]
name = "my-first-header"
value = "my-value"

[[models.gpt-4o.providers.gpt-4o.extra_headers]]
name = "my-second-header"
delete = true`);
  });
});
