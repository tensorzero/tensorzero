import { expect, test } from "vitest";
import {
  checkClickhouseConnection,
  countCuratedInferences,
  countFeedbacksForMetric,
  countInferencesForFunction,
  getCuratedInferences,
  queryInferenceTable,
} from "./clickhouse";

test("checkClickhouseConnection", async () => {
  const result = await checkClickhouseConnection();
  expect(result).toBe(true);
});

// Test boolean metrics
test("countCuratedInferences for boolean metrics", async () => {
  // JSON Inference level
  const jsonInferenceResult = await countCuratedInferences(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "dashboard_fixture_exact_match",
    { type: "boolean", optimize: "max", level: "inference" },
    0, // threshold not used for boolean
  );
  expect(jsonInferenceResult).toBe(41);

  // JSON Episode level
  const jsonEpisodeResult = await countCuratedInferences(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "dashboard_fixture_exact_match_episode",
    { type: "boolean", optimize: "max", level: "episode" },
    0,
  );
  expect(jsonEpisodeResult).toBe(29);

  // Chat Inference level
  const chatInferenceResult = await countCuratedInferences(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "dashboard_fixture_haiku_score",
    { type: "boolean", optimize: "max", level: "inference" },
    0,
  );
  expect(chatInferenceResult).toBe(80);

  // Chat Episode level
  const chatEpisodeResult = await countCuratedInferences(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "dashboard_fixture_haiku_score_episode",
    { type: "boolean", optimize: "max", level: "episode" },
    0,
  );
  expect(chatEpisodeResult).toBe(9);
});

// Test float metrics
test("countCuratedInferences for float metrics", async () => {
  // JSON Inference level
  const jsonInferenceResult = await countCuratedInferences(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "dashboard_fixture_jaccard_similarity",
    { type: "float", optimize: "max", level: "inference" },
    0.8,
  );
  expect(jsonInferenceResult).toBe(54);

  // JSON Episode level
  const jsonEpisodeResult = await countCuratedInferences(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "dashboard_fixture_jaccard_similarity_episode",
    { type: "float", optimize: "max", level: "episode" },
    0.8,
  );
  expect(jsonEpisodeResult).toBe(35);

  // Chat Inference level
  const chatInferenceResult = await countCuratedInferences(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "dashboard_fixture_haiku_rating",
    { type: "float", optimize: "max", level: "inference" },
    0.8,
  );
  expect(chatInferenceResult).toBe(67);

  // Chat Episode level
  const chatEpisodeResult = await countCuratedInferences(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "dashboard_fixture_haiku_rating_episode",
    { type: "float", optimize: "max", level: "episode" },
    0.8,
  );
  expect(chatEpisodeResult).toBe(11);
});

// Test demonstration metrics
test("countCuratedInferences for demonstration metrics", async () => {
  const jsonResult = await countCuratedInferences(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "unused_metric_name",
    { type: "demonstration", level: "inference" },
    0,
  );
  expect(jsonResult).toBe(100);

  const chatResult = await countCuratedInferences(
    "dashboard_fixture_write_haiku",
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
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "dashboard_fixture_exact_match",
    { type: "boolean", optimize: "max", level: "inference" },
    0,
    undefined,
  );
  expect(booleanResults.length).toBe(41);

  // Test with float metric
  const floatResults = await getCuratedInferences(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "dashboard_fixture_haiku_rating",
    { type: "float", optimize: "max", level: "inference" },
    0.8,
    undefined,
  );
  expect(floatResults.length).toBe(67);

  // Test with demonstration
  const demoResults = await getCuratedInferences(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "unused_metric_name",
    { type: "demonstration", level: "inference" },
    0,
    20,
  );
  expect(demoResults.length).toBe(20);

  // Test without metric (should return all inferences)
  const allResults = await getCuratedInferences(
    "dashboard_fixture_extract_entities",
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
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "dashboard_fixture_exact_match",
    { type: "boolean", optimize: "max", level: "inference" },
  );
  expect(booleanInferenceCount).toBe(99);

  // Test float metrics
  const floatInferenceCount = await countFeedbacksForMetric(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "dashboard_fixture_haiku_rating",
    { type: "float", optimize: "max", level: "inference" },
  );
  expect(floatInferenceCount).toBe(491);

  // Test demonstration
  const demoCount = await countFeedbacksForMetric(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
    "unused_metric_name",
    { type: "demonstration", level: "inference" },
  );
  expect(demoCount).toBe(100);
});

// Test countInferencesForFunction
test("countInferencesForFunction returns correct counts", async () => {
  const jsonCount = await countInferencesForFunction(
    "dashboard_fixture_extract_entities",
    { type: "json", variants: {} },
  );
  expect(jsonCount).toBe(400);

  const chatCount = await countInferencesForFunction(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
  );
  expect(chatCount).toBe(494);
});

test("queryInferenceTable", async () => {
  const inferences = await queryInferenceTable({
    offset: 0,
    page_size: 10,
  });
  console.log(inferences);
  expect(inferences.length).toBe(10);
});
