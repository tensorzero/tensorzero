import { expect, test } from "vitest";
import { checkClickHouseConnection } from "./client.server";

test("checkClickHouseConnection", async () => {
  const result = await checkClickHouseConnection();
  expect(result).toBe(true);
});
