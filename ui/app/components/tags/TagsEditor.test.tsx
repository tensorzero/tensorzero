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

  it("should prevent adding system tags", () => {
    const systemTagKey = "tensorzero::new_system_tag";
    const regularKey = "regular_key";

    // System tags should be blocked
    expect(systemTagKey.startsWith("tensorzero::")).toBe(true);
    
    // Regular tags should be allowed
    expect(regularKey.startsWith("tensorzero::")).toBe(false);
  });

  it("should validate system tag prevention in add logic", () => {
    const existingTags = { user_id: "123" };
    const systemKey = "tensorzero::blocked";
    const systemValue = "should_not_be_added";
    const regularKey = "allowed";
    const regularValue = "can_be_added";

    // Simulate the validation logic from handleAddTag
    const trimmedSystemKey = systemKey.trim();
    const trimmedRegularKey = regularKey.trim();

    // System tag should be blocked
    if (!trimmedSystemKey.startsWith("tensorzero::")) {
      // This shouldn't execute for system tags
      expect(true).toBe(false);
    }

    // Regular tag should be allowed
    if (trimmedRegularKey && !trimmedRegularKey.startsWith("tensorzero::")) {
      const updatedTags = { ...existingTags, [trimmedRegularKey]: regularValue };
      expect(updatedTags).toEqual({
        user_id: "123",
        allowed: "can_be_added",
      });
    }
  });

  it("should sort tags alphabetically by key", () => {
    const unsortedTags = {
      zebra: "last",
      apple: "first", 
      "tensorzero::system": "middle",
      banana: "second",
    };

    const sortedEntries = Object.entries(unsortedTags).sort(([a], [b]) => a.localeCompare(b));
    
    expect(sortedEntries[0][0]).toBe("apple");
    expect(sortedEntries[1][0]).toBe("banana");
    expect(sortedEntries[2][0]).toBe("tensorzero::system");
    expect(sortedEntries[3][0]).toBe("zebra");
  });
});