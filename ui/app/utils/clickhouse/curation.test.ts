import { handle_llm_judge_output } from "./curation.server";

import { describe, expect, it } from "vitest";

describe("handle_llm_judge_output", () => {
  it("should remove the thinking field from the output", () => {
    const output = handle_llm_judge_output(
      `{"parsed":{"thinking":"This is a test","answer": "test"},"raw":"{\\"thinking\\":\\"This is a test\\",\\"answer\\":\\"test\\"}"}`,
    );
    expect(output).toBe(
      '{"parsed":{"answer":"test"},"raw":"{\\"answer\\":\\"test\\"}"}',
    );
  });
  it("The correct output is unmodified", () => {
    const output = handle_llm_judge_output(
      '{"parsed": {"answer": "This is a test"}, "raw": "{\\"answer\\": \\"This is a test\\"}"}',
    );
    expect(output).toBe(
      '{"parsed":{"answer":"This is a test"},"raw":"{\\"answer\\": \\"This is a test\\"}"}',
    );
  });
  it("should not modify the output if the parsed field is not present", () => {
    const output = handle_llm_judge_output('{"raw": "This is a test"}');
    expect(output).toBe('{"raw": "This is a test"}');
  });
  it("should not modify the output if the parsed field is not present", () => {
    const output = handle_llm_judge_output('{"raw": "This is a test"}');
    expect(output).toBe('{"raw": "This is a test"}');
  });
});
