import { expect, test } from "vitest";
import {
  checkClickhouseConnection,
  queryGoodBooleanMetricData,
  countGoodBooleanMetricData,
  queryGoodFloatMetricData,
  countGoodFloatMetricData,
  queryDemonstrationData,
  countDemonstrationData,
} from "./clickhouse";

test("checkClickhouseConnection", async () => {
  const result = await checkClickhouseConnection();
  expect(result).toBe(true);
});

test("queryBooleanMetricData", async () => {
  const result = await queryGoodBooleanMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_exact_match",
    "JsonInference",
    "id",
    true,
    undefined,
  );
  // The fixture was written to have 41 rows with good boolean metric data that should be returned
  expect(result.length).toBe(41);
});

test("countGoodBooleanMetricData", async () => {
  const result = await countGoodBooleanMetricData(
    "dashboard_fixture_extract_entities",
    "dashboard_fixture_exact_match",
    "JsonInference",
    "id",
    true,
  );
  // The fixture should have 41 rows with good boolean metric data
  expect(result).toBe(41);
});

test("queryGoodFloatMetricData", async () => {
  const result = await queryGoodFloatMetricData(
    "dashboard_fixture_extract_entities",
    "jaccard_similarity",
    "JsonInference",
    "id",
    true,
    0.8,
    undefined,
  );
  // The fixture should have 54 rows with float metric data above 0.8
  expect(result.length).toBe(54);
});

test("countGoodFloatMetricData", async () => {
  const result = await countGoodFloatMetricData(
    "dashboard_fixture_extract_entities",
    "jaccard_similarity",
    "JsonInference",
    "id",
    true,
    0.8,
  );
  // The fixture should have 54 rows with float metric data above 0.8
  expect(result).toBe(54);
});

test("queryDemonstrationData", async () => {
  const result = await queryDemonstrationData(
    "dashboard_fixture_extract_entities",
    "JsonInference",
    undefined,
  );
  // The fixture should have 100 rows with demonstration data
  expect(result.length).toBe(100);
});

test("countDemonstrationData", async () => {
  const result = await countDemonstrationData(
    "dashboard_fixture_extract_entities",
    "JsonInference",
  );
  // The fixture should have 100 rows with demonstration data
  expect(result).toBe(100);
});
