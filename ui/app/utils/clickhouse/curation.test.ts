import { countFeedbacksForMetric } from "./curation.server";

import { expect, test } from "vitest";
import {
  countCuratedInferences,
  getCuratedInferences,
} from "./curation.server";

// Test boolean metrics
test("countCuratedInferences for boolean metrics", async () => {
  // JSON Inference level
  const jsonInferenceResult = await countCuratedInferences(
    "extract_entities",
    { type: "json", variants: {} },
    "exact_match",
    { type: "boolean", optimize: "max", level: "inference" },
    0, // threshold not used for boolean
  );
  expect(jsonInferenceResult).toBe(41);

  // JSON Episode level
  const jsonEpisodeResult = await countCuratedInferences(
    "extract_entities",
    { type: "json", variants: {} },
    "exact_match_episode",
    { type: "boolean", optimize: "max", level: "episode" },
    0,
  );
  expect(jsonEpisodeResult).toBe(29);

  // Chat Inference level
  const chatInferenceResult = await countCuratedInferences(
    "write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "haiku_score",
    { type: "boolean", optimize: "max", level: "inference" },
    0,
  );
  expect(chatInferenceResult).toBe(80);

  // Chat Episode level
  const chatEpisodeResult = await countCuratedInferences(
    "write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "haiku_score_episode",
    { type: "boolean", optimize: "max", level: "episode" },
    0,
  );
  expect(chatEpisodeResult).toBe(9);
});

// Test float metrics
test("countCuratedInferences for float metrics", async () => {
  // JSON Inference level
  const jsonInferenceResult = await countCuratedInferences(
    "extract_entities",
    { type: "json", variants: {} },
    "jaccard_similarity",
    { type: "float", optimize: "max", level: "inference" },
    0.8,
  );
  expect(jsonInferenceResult).toBe(54);

  // JSON Episode level
  const jsonEpisodeResult = await countCuratedInferences(
    "extract_entities",
    { type: "json", variants: {} },
    "jaccard_similarity_episode",
    { type: "float", optimize: "max", level: "episode" },
    0.8,
  );
  expect(jsonEpisodeResult).toBe(35);

  // Chat Inference level
  const chatInferenceResult = await countCuratedInferences(
    "write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "haiku_rating",
    { type: "float", optimize: "max", level: "inference" },
    0.8,
  );
  expect(chatInferenceResult).toBe(67);

  // Chat Episode level
  const chatEpisodeResult = await countCuratedInferences(
    "write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "haiku_rating_episode",
    { type: "float", optimize: "max", level: "episode" },
    0.8,
  );
  expect(chatEpisodeResult).toBe(11);
});

// Test demonstration metrics
test("countCuratedInferences for demonstration metrics", async () => {
  const jsonResult = await countCuratedInferences(
    "extract_entities",
    { type: "json", variants: {} },
    "unused_metric_name",
    { type: "demonstration", level: "inference" },
    0,
  );
  expect(jsonResult).toBe(100);

  const chatResult = await countCuratedInferences(
    "write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "unused_metric_name",
    { type: "demonstration", level: "inference" },
    0,
  );
  expect(chatResult).toBe(493);
});

// Test getCuratedInferences
test("getCuratedInferences retrieves correct data", async () => {
  // Test with boolean metric
  const booleanResults = await getCuratedInferences(
    "extract_entities",
    { type: "json", variants: {} },
    "exact_match",
    { type: "boolean", optimize: "max", level: "inference" },
    0,
    undefined,
  );
  expect(booleanResults.length).toBe(41);

  // Test with float metric
  const floatResults = await getCuratedInferences(
    "write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "haiku_rating",
    { type: "float", optimize: "max", level: "inference" },
    0.8,
    undefined,
  );
  expect(floatResults.length).toBe(67);

  // Test with demonstration
  const demoResults = await getCuratedInferences(
    "extract_entities",
    { type: "json", variants: {} },
    "unused_metric_name",
    { type: "demonstration", level: "inference" },
    0,
    20,
  );
  expect(demoResults.length).toBe(20);

  // Test without metric (should return all inferences)
  const allResults = await getCuratedInferences(
    "extract_entities",
    { type: "json", variants: {} },
    null,
    null,
    0,
    undefined,
  );
  expect(allResults.length).toBe(400);
});

// Test countFeedbacksForMetric
test("countFeedbacksForMetric returns correct counts", async () => {
  // Test boolean metrics
  const booleanInferenceCount = await countFeedbacksForMetric(
    "extract_entities",
    { type: "json", variants: {} },
    "exact_match",
    { type: "boolean", optimize: "max", level: "inference" },
  );
  expect(booleanInferenceCount).toBe(99);

  // Test float metrics
  const floatInferenceCount = await countFeedbacksForMetric(
    "write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "haiku_rating",
    { type: "float", optimize: "max", level: "inference" },
  );
  expect(floatInferenceCount).toBe(491);

  // Test demonstration
  const demoCount = await countFeedbacksForMetric(
    "extract_entities",
    { type: "json", variants: {} },
    "unused_metric_name",
    { type: "demonstration", level: "inference" },
  );
  expect(demoCount).toBe(100);
});
