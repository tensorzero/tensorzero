import { describe, it, expect } from "vitest";

describe("TagsBadges component logic", () => {
  it("should truncate display text longer than 20 characters", () => {
    const key = "very_long_tag_key_that_might_overflow";
    const value =
      "very_long_tag_value_that_will_definitely_need_truncation_in_the_ui";
    const displayText = `${key}=${value}`;
    const truncatedText =
      displayText.length > 20
        ? `${displayText.substring(0, 17)}...`
        : displayText;

    expect(truncatedText).toBe("very_long_tag_key...");
  });

  it("should identify system tags correctly", () => {
    const systemTag = "tensorzero::human_feedback";
    const userTag = "user_id";

    expect(systemTag.startsWith("tensorzero::")).toBe(true);
    expect(userTag.startsWith("tensorzero::")).toBe(false);
  });

  it("should handle empty tags object", () => {
    const tags = {};
    const tagEntries = Object.entries(tags);

    expect(tagEntries.length).toBe(0);
  });

  it("should process multiple tags correctly", () => {
    const tags = {
      user_id: "123",
      experiment: "A",
      "tensorzero::human_feedback": "true",
    };
    const tagEntries = Object.entries(tags);

    expect(tagEntries.length).toBe(3);
    expect(tags["user_id"]).toBe("123");
    expect(tags["experiment"]).toBe("A");
    expect(tags["tensorzero::human_feedback"]).toBe("true");
  });

  it("should ignore tags with undefined values", () => {
    const tags = {
      defined: "value",
      optional: undefined,
    } as Record<string, string | undefined>;

    const tagEntries = Object.entries(tags).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    );

    expect(tagEntries.length).toBe(1);
    expect(tagEntries[0][0]).toBe("defined");
    expect(tagEntries[0][1]).toBe("value");
  });
});
