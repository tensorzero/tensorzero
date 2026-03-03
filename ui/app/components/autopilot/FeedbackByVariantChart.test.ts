import { expect, test, describe } from "vitest";
import { parseFeedbackByVariant } from "./FeedbackByVariantChart";

describe("parseFeedbackByVariant", () => {
  test("parses valid feedback array", () => {
    const input = [
      { variant_name: "v1", mean: 0.8, variance: 0.04, count: 100 },
      { variant_name: "v2", mean: 0.6, variance: null, count: 50 },
    ];
    const result = parseFeedbackByVariant(input);
    expect(result).toEqual(input);
  });

  test("returns null for non-array input", () => {
    expect(parseFeedbackByVariant("not an array")).toBeNull();
    expect(parseFeedbackByVariant(42)).toBeNull();
    expect(parseFeedbackByVariant(null)).toBeNull();
    expect(parseFeedbackByVariant({})).toBeNull();
  });

  test("returns empty array for empty input", () => {
    expect(parseFeedbackByVariant([])).toEqual([]);
  });

  test("returns null when entry is missing variant_name", () => {
    const input = [{ mean: 0.8, count: 100 }];
    expect(parseFeedbackByVariant(input)).toBeNull();
  });

  test("returns null when entry has wrong type for mean", () => {
    const input = [{ variant_name: "v1", mean: "not a number", count: 100 }];
    expect(parseFeedbackByVariant(input)).toBeNull();
  });

  test("returns null when entry has wrong type for count", () => {
    const input = [{ variant_name: "v1", mean: 0.8, count: "100" }];
    expect(parseFeedbackByVariant(input)).toBeNull();
  });

  test("returns null if any entry in array is invalid", () => {
    const input = [
      { variant_name: "v1", mean: 0.8, count: 100 },
      { variant_name: "v2", mean: "bad", count: 50 },
    ];
    expect(parseFeedbackByVariant(input)).toBeNull();
  });

  test("accepts entries with null variance", () => {
    const input = [{ variant_name: "v1", mean: 0.8, variance: null, count: 1 }];
    const result = parseFeedbackByVariant(input);
    expect(result).toEqual(input);
  });

  test("accepts entries without variance field", () => {
    const input = [{ variant_name: "v1", mean: 0.8, count: 1 }];
    const result = parseFeedbackByVariant(input);
    expect(result).toEqual(input);
  });
});
