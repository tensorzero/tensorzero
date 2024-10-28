import { expect, test } from "vitest";
import { checkClickhouseConnection } from "./clickhouse";

test("checkClickhouseConnection", async () => {
  const result = await checkClickhouseConnection();
  expect(result).toBe(true);
});
