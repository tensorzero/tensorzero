import { describe, it, expect } from "vitest";
import {
  uuidv7ToTimestamp,
  formatCost,
  getTotalInferenceUsage,
} from "./helpers";
import type { ParsedModelInferenceRow } from "./inference";

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

// ─── formatCost ──────────────────────────────────────────────────────────────

describe("formatCost", () => {
  // Note: formatCost does NOT include a "$" prefix — the CostIcon in the
  // Chip component provides the currency indicator.

  // --- Exact zero ---
  it("formats zero as 0.00", () => {
    expect(formatCost(0)).toBe("0.00");
  });

  // --- Large costs (>= $1) — always 2 decimal places ---
  it("formats $1.00 with 2 decimal places", () => {
    expect(formatCost(1)).toBe("1.00");
  });

  it("formats $1.50 with 2 decimal places", () => {
    expect(formatCost(1.5)).toBe("1.50");
  });

  it("formats $123.456 rounded to 2 decimal places", () => {
    expect(formatCost(123.456)).toBe("123.46");
  });

  it("formats $10.00 with 2 decimal places", () => {
    expect(formatCost(10)).toBe("10.00");
  });

  it("formats $999999.99 correctly", () => {
    expect(formatCost(999999.99)).toBe("999999.99");
  });

  // --- Medium costs ($0.01 to $1) — up to 4 decimal places ---
  it("formats $0.50 without trailing zeros", () => {
    expect(formatCost(0.5)).toBe("0.5");
  });

  it("formats $0.1234 with 4 decimal places", () => {
    expect(formatCost(0.1234)).toBe("0.1234");
  });

  it("formats $0.01 without trailing zeros", () => {
    expect(formatCost(0.01)).toBe("0.01");
  });

  it("formats $0.0150 without trailing zeros", () => {
    expect(formatCost(0.015)).toBe("0.015");
  });

  it("formats $0.99 correctly", () => {
    expect(formatCost(0.99)).toBe("0.99");
  });

  it("formats $0.12345 rounded to 4 decimal places", () => {
    expect(formatCost(0.12345)).toBe("0.1235");
  });

  // --- Small costs ($0.001 to $0.01) — up to 6 decimal places ---
  it("formats $0.003 (typical inference cost)", () => {
    expect(formatCost(0.003)).toBe("0.003");
  });

  it("formats $0.001500 without trailing zeros", () => {
    expect(formatCost(0.0015)).toBe("0.0015");
  });

  it("formats $0.009999 correctly", () => {
    expect(formatCost(0.009999)).toBe("0.009999");
  });

  // --- Very small costs (< $0.001) — up to 8 decimal places ---
  it("formats $0.00001875 (real-world gpt-4o-mini cost) without rounding", () => {
    expect(formatCost(0.00001875)).toBe("0.00001875");
  });

  it("formats $0.000123 correctly", () => {
    expect(formatCost(0.000123)).toBe("0.000123");
  });

  it("formats $0.000001 (1 micro-dollar) correctly", () => {
    expect(formatCost(0.000001)).toBe("0.000001");
  });

  it("formats $0.00000025 correctly with 8 decimal places", () => {
    expect(formatCost(0.00000025)).toBe("0.00000025");
  });

  it("formats $0.0000001 correctly", () => {
    expect(formatCost(0.0000001)).toBe("0.0000001");
  });

  it("formats $0.00000001 (precision floor) correctly", () => {
    expect(formatCost(0.00000001)).toBe("0.00000001");
  });

  // --- Below precision floor (DB stores Decimal(18,9), we display 8 digits) ---
  it("formats costs below $0.00000001 as <0.00000001", () => {
    expect(formatCost(0.000000001)).toBe("<0.00000001");
  });

  it("formats extremely small positive cost as <0.00000001", () => {
    expect(formatCost(1e-10)).toBe("<0.00000001");
  });

  // --- Negative costs (caching discounts can make total negative) ---
  it("formats -$0.003 correctly", () => {
    expect(formatCost(-0.003)).toBe("-0.003");
  });

  it("formats -$1.50 correctly", () => {
    expect(formatCost(-1.5)).toBe("-1.50");
  });

  it("formats -$0.00001875 correctly", () => {
    expect(formatCost(-0.00001875)).toBe("-0.00001875");
  });

  it("formats very small negative cost correctly", () => {
    expect(formatCost(-0.000000001)).toBe("-<0.00000001");
  });

  // --- Pathological values ---
  it("handles NaN gracefully", () => {
    expect(formatCost(NaN)).toBe("—");
  });

  it("handles Infinity gracefully", () => {
    expect(formatCost(Infinity)).toBe("—");
  });

  it("handles -Infinity gracefully", () => {
    expect(formatCost(-Infinity)).toBe("—");
  });

  // --- Boundary values ---
  it("formats exactly $0.001 (boundary between very small and small)", () => {
    expect(formatCost(0.001)).toBe("0.001");
  });

  it("formats $0.0009999 (just below $0.001 boundary)", () => {
    expect(formatCost(0.0009999)).toBe("0.0009999");
  });

  it("formats exactly $0.01 (boundary between small and medium)", () => {
    expect(formatCost(0.01)).toBe("0.01");
  });

  it("formats $0.0099 (just below $0.01 boundary)", () => {
    expect(formatCost(0.0099)).toBe("0.0099");
  });

  it("formats exactly $1.00 (boundary between medium and large)", () => {
    expect(formatCost(1.0)).toBe("1.00");
  });

  it("formats $0.9999 (just below $1 boundary)", () => {
    expect(formatCost(0.9999)).toBe("0.9999");
  });
});

// ─── getTotalInferenceUsage ──────────────────────────────────────────────────

/**
 * Helper to create a minimal ParsedModelInferenceRow for testing.
 * Only the fields used by getTotalInferenceUsage matter.
 */
function makeModelInference(
  overrides: Partial<ParsedModelInferenceRow> = {},
): ParsedModelInferenceRow {
  return {
    id: "00000000-0000-0000-0000-000000000000",
    inference_id: "00000000-0000-0000-0000-000000000000",
    raw_request: "",
    raw_response: "",
    model_name: "test-model",
    model_provider_name: "test-provider",
    response_time_ms: null,
    ttft_ms: null,
    timestamp: "2025-01-01T00:00:00Z",
    system: null,
    input_messages: [],
    output: [],
    cached: false,
    ...overrides,
  };
}

describe("getTotalInferenceUsage", () => {
  // --- Empty input ---
  it("returns zeros and null cost for empty array", () => {
    const result = getTotalInferenceUsage([]);
    expect(result.input_tokens).toBe(0);
    expect(result.output_tokens).toBe(0);
    expect(result.cost).toBeNull();
  });

  // --- Single inference ---
  it("passes through a single inference with all fields", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({
        input_tokens: 100,
        output_tokens: 50,
        cost: 0.003,
      }),
    ]);
    expect(result.input_tokens).toBe(100);
    expect(result.output_tokens).toBe(50);
    expect(result.cost).toBe(0.003);
  });

  it("passes through a single inference with no cost", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({
        input_tokens: 100,
        output_tokens: 50,
      }),
    ]);
    expect(result.input_tokens).toBe(100);
    expect(result.output_tokens).toBe(50);
    expect(result.cost).toBeNull();
  });

  it("passes through a single inference with no tokens", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({
        cost: 0.005,
      }),
    ]);
    expect(result.input_tokens).toBe(0);
    expect(result.output_tokens).toBe(0);
    expect(result.cost).toBe(0.005);
  });

  // --- Multiple inferences, all with cost ---
  it("sums costs across multiple inferences", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({
        input_tokens: 100,
        output_tokens: 50,
        cost: 0.001,
      }),
      makeModelInference({
        input_tokens: 200,
        output_tokens: 100,
        cost: 0.002,
      }),
    ]);
    expect(result.input_tokens).toBe(300);
    expect(result.output_tokens).toBe(150);
    expect(result.cost).toBeCloseTo(0.003, 10);
  });

  // --- Multiple inferences, all without cost ---
  it("returns null cost when no inferences have cost", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ input_tokens: 100, output_tokens: 50 }),
      makeModelInference({ input_tokens: 200, output_tokens: 100 }),
    ]);
    expect(result.input_tokens).toBe(300);
    expect(result.output_tokens).toBe(150);
    expect(result.cost).toBeNull();
  });

  // --- Mixed: some with cost, some without (poison semantics) ---
  it("returns null cost when any inference is missing cost (poison semantics)", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({
        input_tokens: 100,
        output_tokens: 50,
        cost: 0.003,
      }),
      makeModelInference({
        input_tokens: 200,
        output_tokens: 100,
        // no cost — provider doesn't have cost tracking configured
      }),
    ]);
    expect(result.input_tokens).toBe(300);
    expect(result.output_tokens).toBe(150);
    // Poison semantics: any null makes the total null
    expect(result.cost).toBeNull();
  });

  it("returns null cost when missing inference comes first in the array", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({
        input_tokens: 200,
        output_tokens: 100,
        // no cost
      }),
      makeModelInference({
        input_tokens: 100,
        output_tokens: 50,
        cost: 0.005,
      }),
    ]);
    expect(result.input_tokens).toBe(300);
    expect(result.output_tokens).toBe(150);
    expect(result.cost).toBeNull();
  });

  // --- Zero cost is distinct from missing cost ---
  it("treats cost=0 as present (not the same as missing)", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ cost: 0 }),
      makeModelInference({ cost: 0.002 }),
    ]);
    expect(result.cost).toBe(0.002);
  });

  it("returns 0 when all inferences have cost=0", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ cost: 0 }),
      makeModelInference({ cost: 0 }),
    ]);
    expect(result.cost).toBe(0);
  });

  // --- Token aggregation (pre-existing behavior, documenting for completeness) ---
  it("sums tokens, treating missing as 0", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ input_tokens: 100 }),
      makeModelInference({ output_tokens: 50 }),
    ]);
    expect(result.input_tokens).toBe(100);
    expect(result.output_tokens).toBe(50);
  });

  // --- Many inferences (floating point accumulation) ---
  it("sums many small costs without gross floating-point error", () => {
    const inferences = Array.from({ length: 100 }, () =>
      makeModelInference({ cost: 0.003 }),
    );
    const result = getTotalInferenceUsage(inferences);
    // 100 * 0.003 = 0.3
    // Allow small floating-point drift but should be very close
    expect(result.cost).not.toBeNull();
    expect(result.cost!).toBeCloseTo(0.3, 6);
  });

  // --- Three-way mix: some with cost, some without, some with zero ---
  it("returns null for three-way mix of present, zero, and missing costs", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ cost: 0.01 }),
      makeModelInference({ cost: 0 }),
      makeModelInference({}), // missing cost — poisons the total
      makeModelInference({ cost: 0.02 }),
    ]);
    expect(result.cost).toBeNull();
  });

  // --- Negative costs (caching discounts) ---
  it("sums costs when some are negative (caching discount scenario)", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ cost: 0.01 }), // base cost
      makeModelInference({ cost: -0.003 }), // caching discount
    ]);
    expect(result.cost).toBeCloseTo(0.007, 10);
  });

  it("handles all-negative costs (unusual but possible)", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ cost: -0.001 }),
      makeModelInference({ cost: -0.002 }),
    ]);
    expect(result.cost).toBeCloseTo(-0.003, 10);
  });

  it("returns null when mixing negative costs with missing costs (poison semantics)", () => {
    const result = getTotalInferenceUsage([
      makeModelInference({ cost: -0.001 }),
      makeModelInference({}), // missing — poisons the total
    ]);
    expect(result.cost).toBeNull();
  });
});
