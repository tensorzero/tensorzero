import { expect, describe, test } from "vitest";
import { checkClickHouseConnection } from "./client.server";
import { storageKindSchema } from "./common";

test("checkClickHouseConnection", async () => {
  const result = await checkClickHouseConnection();
  expect(result).toBe(true);
});

describe("parsing storageKind", () => {
  test("storagekind with undefined region", () => {
    const storageKind = {
      type: "s3_compatible",
      bucket_name: "tensorzero",
      region: undefined,
      endpoint: "http://minio:9000",
      allow_http: true,
    };
    storageKindSchema.parse(storageKind);
  });
});
