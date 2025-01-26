import { expect, test } from "vitest";
import { checkClickHouseConnection } from "./common";

test("checkClickHouseConnection", async () => {
  const result = await checkClickHouseConnection();
  expect(result).toBe(true);
});
