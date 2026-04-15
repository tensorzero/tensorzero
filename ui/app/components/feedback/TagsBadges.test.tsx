import { describe, it, expect } from "vitest";

const MAX_VISIBLE_TAGS = 3;

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

describe("TagsBadges collapse logic", () => {
  it("should show all tags when count is at or below threshold", () => {
    const tags = { a: "1", b: "2", c: "3" };
    const entries = Object.entries(tags).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    );
    const visibleTags = entries.slice(0, MAX_VISIBLE_TAGS);
    const hiddenTags = entries.slice(MAX_VISIBLE_TAGS);

    expect(visibleTags.length).toBe(3);
    expect(hiddenTags.length).toBe(0);
  });

  it("should collapse tags when count exceeds threshold", () => {
    const tags = { a: "1", b: "2", c: "3", d: "4", e: "5" };
    const entries = Object.entries(tags).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    );
    const visibleTags = entries.slice(0, MAX_VISIBLE_TAGS);
    const hiddenTags = entries.slice(MAX_VISIBLE_TAGS);

    expect(visibleTags.length).toBe(3);
    expect(hiddenTags.length).toBe(2);
  });

  it("should produce correct '+N more' text", () => {
    const tags = {
      a: "1",
      b: "2",
      c: "3",
      d: "4",
      e: "5",
      f: "6",
      g: "7",
      h: "8",
      i: "9",
      j: "10",
    };
    const entries = Object.entries(tags).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    );
    const hiddenCount = entries.length - MAX_VISIBLE_TAGS;

    expect(`+${hiddenCount} more`).toBe("+7 more");
  });

  it("should include all hidden tags in tooltip content", () => {
    const tags = {
      visible1: "a",
      visible2: "b",
      visible3: "c",
      hidden1: "d",
      hidden2: "e",
    };
    const entries = Object.entries(tags).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    );
    const hiddenEntries = entries.slice(MAX_VISIBLE_TAGS);
    const tooltipLines = hiddenEntries.map(([k, v]) => `${k}=${v}`);

    expect(tooltipLines).toEqual(["hidden1=d", "hidden2=e"]);
  });

  it("should handle exactly threshold+1 tags", () => {
    const tags = { a: "1", b: "2", c: "3", d: "4" };
    const entries = Object.entries(tags).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    );
    const hiddenTags = entries.slice(MAX_VISIBLE_TAGS);

    expect(hiddenTags.length).toBe(1);
  });

  it("should filter undefined values before counting for collapse", () => {
    const tags = {
      a: "1",
      b: "2",
      c: "3",
      d: "4",
      e: undefined,
    } as Record<string, string | undefined>;

    const validEntries = Object.entries(tags).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    );
    const hiddenTags = validEntries.slice(MAX_VISIBLE_TAGS);

    expect(validEntries.length).toBe(4);
    expect(hiddenTags.length).toBe(1);
  });
});
