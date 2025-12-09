import { describe, test } from "vitest";
import { storageKindSchema } from "./common";

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
