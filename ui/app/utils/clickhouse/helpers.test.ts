import { describe, it, expect } from "vitest";
import { uuidv7ToTimestamp } from "./helpers";

describe("uuidv7ToTimestamp", () => {
  it("converts a valid UUIDv7 to the correct Date", () => {
    // Example UUIDv7 with timestamp portion "017f22b779a8"
    const uuid = "017f22b7-79a8-7c44-b5c2-11e9390c6c3c";
    const expectedTimestamp = parseInt("017f22b779a8", 16);

    const result = uuidv7ToTimestamp(uuid);
    expect(result.getTime()).toBe(expectedTimestamp);
  });

  it("converts an valid UUIDv7 to another correct Date", () => {
    const uuid = "0195b582-463d-7b40-ab7b-db2a522acc1d";
    const expectedTimestamp = 1742506968637;

    const result = uuidv7ToTimestamp(uuid);
    expect(result.getTime()).toBe(expectedTimestamp);
  });

  it("throws an error for an invalid UUID format", () => {
    const invalidUuid = "invalid-uuid";
    expect(() => uuidv7ToTimestamp(invalidUuid)).toThrow("Invalid UUID format");
  });

  it("throws an error if the UUID version is not 7", () => {
    // Change the version nibble to "4" instead of "7"
    const uuidNotV7 = "017f22b7-79a8-4c44-b5c2-11e9390c6c3c";
    expect(() => uuidv7ToTimestamp(uuidNotV7)).toThrow(
      "Invalid UUID version. Expected version 7.",
    );
  });
});
