import { describe, expect, it } from "vitest";

import { toDisplayUrl } from "./use-blob-url";

describe("toDisplayUrl", () => {
  it("wraps raw base64 in a data URL for immediate render safety", () => {
    expect(
      toDisplayUrl("GkXfo59ChoEBQveBAULygQ==", "audio/webm;codecs=opus"),
    ).toBe("data:audio/webm;codecs=opus;base64,GkXfo59ChoEBQveBAULygQ==");
  });

  it("preserves existing data URLs", () => {
    const dataUrl = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==";
    expect(toDisplayUrl(dataUrl, "image/png")).toBe(dataUrl);
  });

  it("preserves existing blob URLs", () => {
    const blobUrl = "blob:https://app.example/1234";
    expect(toDisplayUrl(blobUrl, "image/png")).toBe(blobUrl);
  });
});
