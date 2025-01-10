import { expect, test } from "vitest";
import { checkClickhouseConnection } from "./common";

test("checkClickhouseConnection", async () => {
  const result = await checkClickhouseConnection();
  expect(result).toBe(true);
});
