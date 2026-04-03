import { describe, expect, it } from "vitest";

import { createNewConfigPath } from "./editor-utils";

describe("createNewConfigPath", () => {
  it("returns the first unused template path", () => {
    expect(createNewConfigPath([])).toBe("new-template-1");
    expect(createNewConfigPath(["new-template-1", "new-template-2"])).toBe(
      "new-template-3",
    );
  });
});
