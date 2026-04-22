import { describe, expect, it } from "vitest";
import {
  computeExpiresAt,
  CUSTOM_EXPIRATION_INVALID_ERROR,
  CUSTOM_EXPIRATION_PAST_ERROR,
  CUSTOM_EXPIRATION_REQUIRED_ERROR,
  getCustomExpirationError,
  getExpirationDateError,
  isExpirationPreset,
} from "./expiration";

describe("api key expiration helpers", () => {
  it("validates known expiration presets", () => {
    expect(isExpirationPreset("custom")).toBe(true);
    expect(isExpirationPreset("7d")).toBe(true);
    expect(isExpirationPreset("tomorrow")).toBe(false);
  });

  it("computes preset expiration dates from the provided time", () => {
    const now = new Date("2026-03-17T15:00:00.000Z");

    expect(computeExpiresAt("7d", now)?.toISOString()).toBe(
      "2026-03-24T15:00:00.000Z",
    );
    expect(computeExpiresAt("none", now)).toBeUndefined();
    expect(computeExpiresAt("custom", now)).toBeUndefined();
  });

  it("requires a custom expiration when requested", () => {
    expect(
      getCustomExpirationError(undefined, {
        requireValue: true,
      }),
    ).toBe(CUSTOM_EXPIRATION_REQUIRED_ERROR);
  });

  it("rejects custom expirations that are not in the future", () => {
    const now = new Date("2026-03-17T15:00:00.000Z");

    expect(
      getCustomExpirationError(new Date("2026-03-17T15:00:00.000Z"), {
        now,
      }),
    ).toBe(CUSTOM_EXPIRATION_PAST_ERROR);
    expect(
      getCustomExpirationError(new Date("2026-03-17T15:01:00.000Z"), {
        now,
      }),
    ).toBeNull();
  });

  it("validates expiration date strings", () => {
    const now = new Date("2026-03-17T15:00:00.000Z");

    expect(getExpirationDateError("not-a-date", now)).toBe(
      CUSTOM_EXPIRATION_INVALID_ERROR,
    );
    expect(getExpirationDateError("2026-03-17T14:59:00.000Z", now)).toBe(
      CUSTOM_EXPIRATION_PAST_ERROR,
    );
    expect(getExpirationDateError("2026-03-17T15:01:00.000Z", now)).toBeNull();
  });
});
