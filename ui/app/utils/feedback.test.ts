import { describe, expect, test } from "vitest";
import type { FeedbackRow, FeedbackBounds } from "~/types/tensorzero";
import { filterToLatestFeedback, groupFeedbackByType } from "./feedback";

const booleanRow = (id: string, metricName = "exact_match"): FeedbackRow => ({
  type: "boolean",
  id,
  target_id: "inf-1",
  metric_name: metricName,
  value: true,
  tags: {},
  timestamp: "2024-01-01T00:00:00Z",
});

const floatRow = (id: string, metricName = "score"): FeedbackRow => ({
  type: "float",
  id,
  target_id: "inf-1",
  metric_name: metricName,
  value: 0.95,
  tags: {},
  timestamp: "2024-01-01T00:00:00Z",
});

const commentRow = (id: string): FeedbackRow => ({
  type: "comment",
  id,
  target_id: "inf-1",
  target_type: "inference",
  value: "test comment",
  tags: {},
  timestamp: "2024-01-01T00:00:00Z",
});

const demonstrationRow = (id: string): FeedbackRow => ({
  type: "demonstration",
  id,
  inference_id: "inf-1",
  value: '{"output": "test"}',
  tags: {},
  timestamp: "2024-01-01T00:00:00Z",
});

const makeBounds = (
  overrides: Partial<FeedbackBounds> = {},
): FeedbackBounds => ({
  first_id: "first",
  last_id: "last",
  by_type: {
    boolean: {},
    float: {},
    comment: {},
    demonstration: {},
  },
  ...overrides,
});

describe("groupFeedbackByType", () => {
  test("groups feedback by type", () => {
    const feedback = [
      booleanRow("b1"),
      floatRow("f1"),
      commentRow("c1"),
      demonstrationRow("d1"),
      booleanRow("b2"),
    ];

    const result = groupFeedbackByType(feedback);

    expect(result.metrics).toHaveLength(3);
    expect(result.comments).toHaveLength(1);
    expect(result.demonstrations).toHaveLength(1);
    expect(result.metrics.map((m) => m.id)).toEqual(["b1", "f1", "b2"]);
    expect(result.comments[0].id).toBe("c1");
    expect(result.demonstrations[0].id).toBe("d1");
  });

  test("returns empty arrays for empty input", () => {
    const result = groupFeedbackByType([]);
    expect(result.metrics).toHaveLength(0);
    expect(result.comments).toHaveLength(0);
    expect(result.demonstrations).toHaveLength(0);
  });

  test("handles all-metrics feedback", () => {
    const feedback = [booleanRow("b1"), floatRow("f1")];
    const result = groupFeedbackByType(feedback);
    expect(result.metrics).toHaveLength(2);
    expect(result.comments).toHaveLength(0);
    expect(result.demonstrations).toHaveLength(0);
  });
});

describe("filterToLatestFeedback", () => {
  test("returns all feedback when bounds are missing", () => {
    const feedback = [booleanRow("b1"), commentRow("c1")];
    expect(filterToLatestFeedback(feedback)).toEqual(feedback);
    expect(filterToLatestFeedback(feedback, undefined, {})).toEqual(feedback);
    expect(filterToLatestFeedback(feedback, makeBounds())).toEqual(feedback);
  });

  test("keeps only latest comment by bounds", () => {
    const feedback = [commentRow("c1"), commentRow("c2")];
    const bounds = makeBounds({
      by_type: {
        boolean: {},
        float: {},
        comment: { last_id: "c2" },
        demonstration: {},
      },
    });

    const result = filterToLatestFeedback(feedback, bounds, {});
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("c2");
  });

  test("keeps only latest demonstration by bounds", () => {
    const feedback = [demonstrationRow("d1"), demonstrationRow("d2")];
    const bounds = makeBounds({
      by_type: {
        boolean: {},
        float: {},
        comment: {},
        demonstration: { last_id: "d1" },
      },
    });

    const result = filterToLatestFeedback(feedback, bounds, {});
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("d1");
  });

  test("keeps only latest metric per metric name", () => {
    const feedback = [
      booleanRow("b1", "exact_match"),
      booleanRow("b2", "exact_match"),
      floatRow("f1", "score"),
      floatRow("f2", "score"),
    ];
    const latestByMetric = { exact_match: "b2", score: "f1" };

    const result = filterToLatestFeedback(
      feedback,
      makeBounds(),
      latestByMetric,
    );
    expect(result.map((r) => r.id)).toEqual(["b2", "f1"]);
  });

  test("keeps comments when bounds have no last_id", () => {
    const feedback = [commentRow("c1"), commentRow("c2")];
    const bounds = makeBounds(); // no last_id for comment

    const result = filterToLatestFeedback(feedback, bounds, {});
    expect(result).toHaveLength(2);
  });

  test("filters mixed feedback correctly", () => {
    const feedback = [
      booleanRow("b1", "exact_match"),
      booleanRow("b2", "exact_match"),
      commentRow("c1"),
      commentRow("c2"),
      demonstrationRow("d1"),
    ];
    const bounds = makeBounds({
      by_type: {
        boolean: {},
        float: {},
        comment: { last_id: "c2" },
        demonstration: { last_id: "d1" },
      },
    });
    const latestByMetric = { exact_match: "b1" };

    const result = filterToLatestFeedback(feedback, bounds, latestByMetric);
    expect(result.map((r) => r.id)).toEqual(["b1", "c2", "d1"]);
  });
});
