import { expect, test } from "vitest";
import {
  checkClickhouseConnection,
  queryDemonstrationDataForFunction,
  countDemonstrationDataForFunction,
  countFeedbacksForMetric,
  queryAllInferencesForFunction,
  queryMetricData,
  countMetricData,
} from "./clickhouse";

test("checkClickhouseConnection", async () => {
  const result = await checkClickhouseConnection();
  expect(result).toBe(true);
});
test("queryBooleanMetricDataJsonInference", async () => {
  const result = await queryMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_exact_match",
    "JsonInference",
    "id",
    "boolean",
    { filterGood: true, maximize: true },
  );
  // The fixture was written to have 41 rows with good boolean metric data that should be returned
  expect(result.length).toBe(41);
});

test("queryBooleanMetricDataJsonEpisode", async () => {
  const result = await queryMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_exact_match_episode",
    "JsonInference",
    "episode_id",
    "boolean",
    { filterGood: true, maximize: true },
  );
  // The fixture was written to have 29 rows with good boolean metric data that should be returned
  expect(result.length).toBe(29);
});

test("queryBooleanMetricDataChatInference", async () => {
  const result = await queryMetricData(
    "dashboard_fixture_write_haiku",
    "dashboard_fixture_haiku_score",
    "ChatInference",
    "id",
    "boolean",
    { filterGood: true, maximize: true },
  );
  // The fixture was written to have 80 rows with good boolean metric data
  expect(result.length).toBe(80);
});

test("queryBooleanMetricDataChatEpisode", async () => {
  const result = await queryMetricData(
    "dashboard_fixture_write_haiku",
    "dashboard_fixture_haiku_score_episode",
    "ChatInference",
    "episode_id",
    "boolean",
    { filterGood: true, maximize: true },
  );
  // The fixture was written to have 9 rows with good boolean metric data
  expect(result.length).toBe(9);
});

test("countGoodBooleanMetricDataJsonInference", async () => {
  const result = await countMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_exact_match",
    "JsonInference",
    "id",
    "boolean",
    { filterGood: true, maximize: true },
  );
  // The fixture should have 41 rows with good boolean metric data
  expect(result).toBe(41);
});

test("countGoodBooleanMetricDataJsonEpisode", async () => {
  const result = await countMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_exact_match_episode",
    "JsonInference",
    "episode_id",
    "boolean",
    { filterGood: true, maximize: true },
  );
  // The fixture should have 29 rows with good boolean metric data
  expect(result).toBe(29);
});

test("countGoodBooleanMetricDataChatInference", async () => {
  const result = await countMetricData(
    "dashboard_fixture_write_haiku",
    "dashboard_fixture_haiku_score",
    "ChatInference",
    "id",
    "boolean",
    { filterGood: true, maximize: true },
  );
  // The fixture should have 80 rows with good boolean metric data
  expect(result).toBe(80);
});

test("countGoodBooleanMetricDataChatEpisode", async () => {
  const result = await countMetricData(
    "dashboard_fixture_write_haiku",
    "dashboard_fixture_haiku_score_episode",
    "ChatInference",
    "episode_id",
    "boolean",
    { filterGood: true, maximize: true },
  );
  // The fixture should have 9 rows with good boolean metric data
  expect(result).toBe(9);
});

test("queryGoodFloatMetricDataJsonInference", async () => {
  const result = await queryMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_jaccard_similarity",
    "JsonInference",
    "id",
    "float",
    { filterGood: true, maximize: true, threshold: 0.8 },
  );
  // The fixture should have 54 rows with float metric data above 0.8
  expect(result.length).toBe(54);
});

test("queryGoodFloatMetricDataJsonEpisode", async () => {
  const result = await queryMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_jaccard_similarity_episode",
    "JsonInference",
    "episode_id",
    "float",
    { filterGood: true, maximize: true, threshold: 0.8 },
  );
  // The fixture should have 35 rows with float metric data above 0.8
  expect(result.length).toBe(35);
});

test("queryGoodFloatMetricDataChatInference", async () => {
  const result = await queryMetricData(
    "dashboard_fixture_write_haiku",
    "dashboard_fixture_haiku_rating",
    "ChatInference",
    "id",
    "float",
    { filterGood: true, maximize: true, threshold: 0.8 },
  );
  // The fixture should have 67 rows with float metric data above 0.8
  expect(result.length).toBe(67);
});

test("queryGoodFloatMetricDataChatEpisode", async () => {
  const result = await queryMetricData(
    "dashboard_fixture_write_haiku",
    "dashboard_fixture_haiku_rating_episode",
    "ChatInference",
    "episode_id",
    "float",
    { filterGood: true, maximize: true, threshold: 0.8 },
  );
  // The fixture should have 11 rows with float metric data above 0.8
  expect(result.length).toBe(11);
});

test("countGoodFloatMetricDataJsonInference", async () => {
  const result = await countMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_jaccard_similarity",
    "JsonInference",
    "id",
    "float",
    { filterGood: true, maximize: true, threshold: 0.8 },
  );
  // The fixture should have 54 rows with float metric data above 0.8
  expect(result).toBe(54);
});

test("countGoodFloatMetricDataJsonEpisode", async () => {
  const result = await countMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_jaccard_similarity_episode",
    "JsonInference",
    "episode_id",
    "float",
    { filterGood: true, maximize: true, threshold: 0.8 },
  );
  // The fixture should have 35 rows with float metric data above 0.8
  expect(result).toBe(35);
});

test("countGoodFloatMetricDataChatInference", async () => {
  const result = await countMetricData(
    "dashboard_fixture_write_haiku",
    "dashboard_fixture_haiku_rating",
    "ChatInference",
    "id",
    "float",
    { filterGood: true, maximize: true, threshold: 0.8 },
  );
  // The fixture should have 67 rows with float metric data above 0.8
  expect(result).toBe(67);
});

test("countGoodFloatMetricDataChatEpisode", async () => {
  const result = await countMetricData(
    "dashboard_fixture_write_haiku",
    "dashboard_fixture_haiku_rating_episode",
    "ChatInference",
    "episode_id",
    "float",
    { filterGood: true, maximize: true, threshold: 0.8 },
  );
  // The fixture should have 11 rows with float metric data above 0.8
  expect(result).toBe(11);
});

test("queryDemonstrationDataJson", async () => {
  const result = await queryDemonstrationDataForFunction(
    "dashboard_fixture_extract_entities",
    "JsonInference",
    undefined,
  );
  // The fixture should have 100 rows with demonstration data
  expect(result.length).toBe(100);

  const tooBigLimitedResult = await queryDemonstrationDataForFunction(
    "dashboard_fixture_extract_entities",
    "JsonInference",
    120,
  );
  // The fixture should have 100 rows with demonstration data since 120 is too big
  expect(tooBigLimitedResult.length).toBe(100);

  const limitedResult = await queryDemonstrationDataForFunction(
    "dashboard_fixture_extract_entities",
    "JsonInference",
    20,
  );
  // The fixture should have 100 rows with demonstration data since 120 is too big
  expect(limitedResult.length).toBe(20);

  // Check a selected inference retrieved to make sure it's a demonstration and not the inference itself
  const selectedResults = result.filter((element) => {
    if (element.episode_id === "0193da94-17e7-7933-9256-2cec500b9515")
      return element;
    return undefined;
  });
  expect(selectedResults.length).toBe(1);
  const selectedResult = selectedResults[0];
  expect(selectedResult.output).toStrictEqual({
    raw: '{"person":[],"organization":[],"location":[],"miscellaneous":["Doetinchem-Doetinchem"]}',
    parsed: {
      person: [],
      organization: [],
      location: [],
      miscellaneous: ["Doetinchem-Doetinchem"],
    },
  });
});

test("queryDemonstrationDataChat", async () => {
  const result = await queryDemonstrationDataForFunction(
    "dashboard_fixture_write_haiku",
    "ChatInference",
    undefined,
  );
  // The fixture should have 493 rows with demonstration data
  expect(result.length).toBe(493);

  // Check a selected inference retrieved to make sure it's a demonstration and not the inference itself
  const selectedResults = result.filter((element) => {
    if (element.episode_id === "0193fb9d-7a21-7c41-a428-4ca775426ab4")
      return element;
    return undefined;
  });
  expect(selectedResults.length).toBe(1);
  const selectedResult = selectedResults[0];
  expect(selectedResult.output).toStrictEqual([
    {
      type: "text",
      text: "Alright, let's dive into the concept of virtue. Virtue often suggests moral excellence, goodness, and righteousness. Hyperbole could be utilized to emphasize the grandness or the celestial quality of virtue.\n\nNow, to construct a haiku:\n- I'll start with a 5-syllable line capturing a core element of virtue.\n- Then, I'll think of a 7-syllable line that incorporates hyperbole.\n- Lastly, I'll wrap it up with another 5-syllable line relating to virtue.\n\nHere is the haiku:\n\nHeart of purest gold,  \nMountains bow to its great light,  \nGuiding stars above.",
    },
  ]);
});

test("queryDemonstrationDataChat", async () => {
  const result = await queryDemonstrationDataForFunction(
    "dashboard_fixture_write_haiku",
    "ChatInference",
    undefined,
  );
  // The fixture should have 493 rows with demonstration data
  expect(result.length).toBe(493);

  // Check a selected inference retrieved to make sure it's a demonstration and not the inference itself
  const selectedResults = result.filter((element) => {
    if (element.episode_id === "0193fb9d-7a21-7c41-a428-4ca775426ab4")
      return element;
    return undefined;
  });
  expect(selectedResults.length).toBe(1);
  const selectedResult = selectedResults[0];
  expect(selectedResult.output).toStrictEqual([
    {
      type: "text",
      text: "Alright, let's dive into the concept of virtue. Virtue often suggests moral excellence, goodness, and righteousness. Hyperbole could be utilized to emphasize the grandness or the celestial quality of virtue.\n\nNow, to construct a haiku:\n- I'll start with a 5-syllable line capturing a core element of virtue.\n- Then, I'll think of a 7-syllable line that incorporates hyperbole.\n- Lastly, I'll wrap it up with another 5-syllable line relating to virtue.\n\nHere is the haiku:\n\nHeart of purest gold,  \nMountains bow to its great light,  \nGuiding stars above.",
    },
  ]);
});

test("countDemonstrationDataJson", async () => {
  const result = await countDemonstrationDataForFunction(
    "dashboard_fixture_extract_entities",
    "JsonInference",
  );
  // The fixture should have 100 rows with demonstration data
  expect(result).toBe(100);
});

test("countDemonstrationDataChat", async () => {
  const result = await countDemonstrationDataForFunction(
    "dashboard_fixture_write_haiku",
    "ChatInference",
  );
  // The fixture should have 493 rows with demonstration data
  expect(result).toBe(493);
});

test("countFeedbacksForMetric for demonstration type", async () => {
  const result = await countFeedbacksForMetric(
    "dashboard_fixture_extract_entities",
    {
      type: "json",
      variants: {},
    },
    "unused_metric_name",
    {
      type: "demonstration",
      level: "inference",
    },
  );

  // This should return null for demonstration
  expect(result).toBe(100);
});

test("countFeedbacksForMetric for float type and inference level", async () => {
  const result = await countFeedbacksForMetric(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "dashboard_fixture_haiku_rating",
    {
      type: "float",
      optimize: "max",
      level: "inference",
    },
  );

  // The fixture should have 491 rows for haiku rating
  expect(result).toBe(491);
});

test("countFeedbacksForMetric for float type and episode level", async () => {
  const result = await countFeedbacksForMetric(
    "dashboard_fixture_write_haiku",
    {
      type: "chat",
      variants: {},
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    "dashboard_fixture_haiku_rating_episode",
    {
      type: "float",
      optimize: "max",
      level: "episode",
    },
  );

  // The fixture should have 85 rows for haiku rating
  expect(result).toBe(85);
});

test("countFeedbacksForMetric for boolean type and inference level", async () => {
  const result = await countFeedbacksForMetric(
    "dashboard_fixture_extract_entities",
    {
      type: "json",
      variants: {},
    },
    "dashboard_fixture_exact_match",
    {
      type: "boolean",
      optimize: "max",
      level: "inference",
    },
  );

  // The fixture should have 99 rows for exact match
  expect(result).toBe(99);
});

test("countFeedbacksForMetric for boolean type and episode level", async () => {
  const result = await countFeedbacksForMetric(
    "dashboard_fixture_extract_entities",
    {
      type: "json",
      variants: {},
    },
    "dashboard_fixture_exact_match_episode",
    {
      type: "boolean",
      optimize: "max",
      level: "episode",
    },
  );

  // The fixture should have 69 rows for exact match
  expect(result).toBe(69);
});

test("queryAllInferencesForFunctionJson", async () => {
  const result = await queryAllInferencesForFunction(
    "dashboard_fixture_extract_entities",
    "JsonInference",
    undefined,
  );
  // The fixture should have 400 rows for this function
  expect(result.length).toBe(400);
});

test("queryAllInferencesForFunctionChat", async () => {
  const result = await queryAllInferencesForFunction(
    "dashboard_fixture_write_haiku",
    "ChatInference",
    undefined,
  );
  // The fixture should have 400 rows for this function
  expect(result.length).toBe(494);

  const limitedResult = await queryAllInferencesForFunction(
    "dashboard_fixture_write_haiku",
    "ChatInference",
    150,
  );
  // The fixture should have 400 rows for this function
  expect(limitedResult.length).toBe(150);
});
