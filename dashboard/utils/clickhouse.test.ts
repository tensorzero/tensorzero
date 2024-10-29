import { expect, test } from "vitest";
import {
  checkClickhouseConnection,
  queryGoodBooleanMetricData,
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
  console.log(result.length);
});
