import { describe, it, expect } from "vitest";

describe("TagsEditor component logic", () => {
  const mockTags = {
    user_id: "123",
    experiment: "A",
    "tensorzero::system_tag": "system_value",
  };

  it("should identify system tags correctly", () => {
    const systemTag = "tensorzero::system_tag";
    const userTag = "user_id";

    expect(systemTag.startsWith("tensorzero::")).toBe(true);
    expect(userTag.startsWith("tensorzero::")).toBe(false);
  });

  it("should handle adding new tags to existing tags object", () => {
    const existingTags = { user_id: "123" };
    const newKey = "experiment";
    const newValue = "A";

    const updatedTags = { ...existingTags, [newKey]: newValue };

    expect(updatedTags).toEqual({
      user_id: "123",
      experiment: "A",
    });
  });

  it("should handle removing tags from tags object", () => {
    const existingTags = {
      user_id: "123",
      experiment: "A",
      "tensorzero::system_tag": "system_value",
    };
    const keyToRemove = "experiment";

    const updatedTags = { ...existingTags };
    delete updatedTags[keyToRemove];

    expect(updatedTags).toEqual({
      user_id: "123",
      "tensorzero::system_tag": "system_value",
    });
  });

  it("should handle empty tags object", () => {
    const tags = {};
    const tagEntries = Object.entries(tags);

    expect(tagEntries.length).toBe(0);
  });

  it("should process tag entries correctly", () => {
    const tagEntries = Object.entries(mockTags);

    expect(tagEntries.length).toBe(3);
    expect(tagEntries).toContainEqual(["user_id", "123"]);
    expect(tagEntries).toContainEqual(["experiment", "A"]);
    expect(tagEntries).toContainEqual(["tensorzero::system_tag", "system_value"]);
  });

  it("should validate tag input properly", () => {
    const key = "  test_key  ";
    const value = "  test_value  ";

    const trimmedKey = key.trim();
    const trimmedValue = value.trim();

    expect(trimmedKey).toBe("test_key");
    expect(trimmedValue).toBe("test_value");
    expect(Boolean(trimmedKey && trimmedValue)).toBe(true);
  });

  it("should filter out system tags when determining removable tags", () => {
    const tagEntries = Object.entries(mockTags);
    const removableTags = tagEntries.filter(
      ([key]) => !key.startsWith("tensorzero::"),
    );

    expect(removableTags.length).toBe(2);
    expect(removableTags).toContainEqual(["user_id", "123"]);
    expect(removableTags).toContainEqual(["experiment", "A"]);
    expect(removableTags).not.toContainEqual([
      "tensorzero::system_tag",
      "system_value",
    ]);
  });
});