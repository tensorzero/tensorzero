import { expect, test, describe } from "vitest";
import { parseFeedbackByVariant } from "./FeedbackByVariantChart";

describe("parseFeedbackByVariant", () => {
  test("parses valid feedback array with null variance", () => {
    const input = [
      { variant_name: "v1", mean: 0.8, variance: 0.04, count: 100 },
      { variant_name: "v2", mean: 0.6, variance: null, count: 50 },
    ];
    const result = parseFeedbackByVariant(input);
    expect(result).toEqual(input);
  });

  test("returns null for non-array input", () => {
    expect(parseFeedbackByVariant(null)).toBeNull();
    expect(parseFeedbackByVariant({})).toBeNull();
  });

  test("returns null for empty array", () => {
    expect(parseFeedbackByVariant([])).toBeNull();
  });

  test("returns null when required fields are missing or wrong type", () => {
    expect(parseFeedbackByVariant([{ mean: 0.8, count: 100 }])).toBeNull();
    expect(
      parseFeedbackByVariant([
        { variant_name: "v1", mean: "not a number", count: 100 },
      ]),
    ).toBeNull();
  });

  test("rejects entire array if any entry is invalid", () => {
    const input = [
      { variant_name: "v1", mean: 0.8, count: 100 },
      { variant_name: "v2", mean: "bad", count: 50 },
    ];
    expect(parseFeedbackByVariant(input)).toBeNull();
  });

  test("returns null when variance is wrong type", () => {
    const input = [
      { variant_name: "v1", mean: 0.8, count: 100, variance: "0.04" },
    ];
    expect(parseFeedbackByVariant(input)).toBeNull();
  });
});
