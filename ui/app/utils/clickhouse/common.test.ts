import { describe, expect, test } from "vitest";
import { resolvedBase64FileSchema, storageKindSchema } from "./common";

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

describe("resolvedBase64FileSchema", () => {
  test("accepts raw base64 data", () => {
    const resolvedFile = {
      data: "GkXfo59ChoEBQveBAULygQ==",
      mime_type: "audio/webm;codecs=opus",
    };

    expect(resolvedBase64FileSchema.parse(resolvedFile)).toEqual(resolvedFile);
  });

  test("rejects data URLs", () => {
    expect(() =>
      resolvedBase64FileSchema.parse({
        data: "data:audio/webm;codecs=opus;base64,GkXfo59ChoEBQveBAULygQ==",
        mime_type: "audio/webm;codecs=opus",
      }),
    ).toThrow("Resolved file data must be raw base64, not a data URL");
  });
});
