import { describe, it, expect } from "vitest";
import { getTotalInferenceUsage, uuidv7ToTimestamp } from "./helpers";
import type { ParsedModelInferenceRow } from "./inference";

function makeModelInference(
  overrides: Partial<ParsedModelInferenceRow>,
): ParsedModelInferenceRow {
  return {
    id: "00000000-0000-7000-0000-000000000000",
    inference_id: "00000000-0000-7000-0000-000000000000",
    raw_request: undefined,
    raw_response: undefined,
    model_name: "test",
    model_provider_name: "test",
    response_time_ms: undefined,
    ttft_ms: undefined,
    timestamp: "2024-01-01T00:00:00Z",
    system: undefined,
    input_messages: [],
    output: [],
    cached: false,
    cost: undefined,
    ...overrides,
  };
}

describe("getTotalInferenceUsage", () => {
  it("sums tokens and cost when all values are present", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ input_tokens: 10, output_tokens: 20, cost: 0.001 }),
      makeModelInference({ input_tokens: 30, output_tokens: 40, cost: 0.002 }),
    ]);
    expect(result.input_tokens).toBe(40);
    expect(result.output_tokens).toBe(60);
    expect(result.cost).toBeCloseTo(0.003);
  });

  it("returns null cost when any inference has undefined cost", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ input_tokens: 10, output_tokens: 20, cost: 0.001 }),
      makeModelInference({ input_tokens: 30, output_tokens: 40 }),
    ]);
    expect(result.input_tokens).toBe(40);
    expect(result.output_tokens).toBe(60);
    expect(result.cost).toBeNull();
  });

  it("returns null cost when all inferences have undefined cost", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ input_tokens: 10, output_tokens: 20 }),
      makeModelInference({ input_tokens: 30, output_tokens: 40 }),
    ]);
    expect(result.cost).toBeNull();
  });

  it("returns zeros for an empty array", () => {
    const result = getTotalInferenceUsage([]);
    expect(result.input_tokens).toBe(0);
    expect(result.output_tokens).toBe(0);
    expect(result.cost).toBeNull();
  });

  it("treats missing tokens as 0", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ input_tokens: 10, cost: 0.001 }),
      makeModelInference({ output_tokens: 40, cost: 0.002 }),
    ]);
    expect(result.input_tokens).toBe(10);
    expect(result.output_tokens).toBe(40);
    expect(result.cost).toBeCloseTo(0.003);
  });
});

describe("uuidv7ToTimestamp", () => {
  it("converts a valid UUIDv7 to the correct Date", () => {
    // Example UUIDv7 with timestamp portion "017f22b779a8"
    const uuid = "017f22b7-79a8-7c44-b5c2-11e9390c6c3c";
    const expectedTimestamp = parseInt("017f22b779a8", 16);

    const result = uuidv7ToTimestamp(uuid);
    expect(result.getTime()).toBe(expectedTimestamp);
  });

  it("converts an valid UUIDv7 to another correct Date", () => {
    const uuid = "0195b582-463d-7b40-ab7b-db2a522acc1d";
    const expectedTimestamp = 1742506968637;

    const result = uuidv7ToTimestamp(uuid);
    expect(result.getTime()).toBe(expectedTimestamp);
  });

  it("throws an error for an invalid UUID format", () => {
    const invalidUuid = "invalid-uuid";
    expect(() => uuidv7ToTimestamp(invalidUuid)).toThrow("Invalid UUID format");
  });

  it("throws an error if the UUID version is not 7", () => {
    // Change the version nibble to "4" instead of "7"
    const uuidNotV7 = "017f22b7-79a8-4c44-b5c2-11e9390c6c3c";
    expect(() => uuidv7ToTimestamp(uuidNotV7)).toThrow(
      "Invalid UUID version. Expected version 7.",
    );
  });
});
