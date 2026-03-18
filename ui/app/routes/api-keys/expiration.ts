export const EXPIRATION_PRESET_OPTIONS = [
  { value: "none", label: "No expiration" },
  { value: "7d", label: "7 days" },
  { value: "30d", label: "30 days" },
  { value: "90d", label: "90 days" },
  { value: "1y", label: "1 year" },
  { value: "custom", label: "Custom" },
] as const;

const EXPIRATION_PRESET_SET: ReadonlySet<string> = new Set(
  EXPIRATION_PRESET_OPTIONS.map(({ value }) => value),
);

export type ExpirationPreset =
  (typeof EXPIRATION_PRESET_OPTIONS)[number]["value"];

export const CUSTOM_EXPIRATION_REQUIRED_ERROR =
  "Choose an expiration date and time.";
export const CUSTOM_EXPIRATION_INVALID_ERROR =
  "Expiration must be a valid date and time.";
export const CUSTOM_EXPIRATION_PAST_ERROR = "Expiration must be in the future.";

export function isExpirationPreset(value: string): value is ExpirationPreset {
  return EXPIRATION_PRESET_SET.has(value);
}

export function computeExpiresAt(
  preset: ExpirationPreset,
  now: Date = new Date(),
): Date | undefined {
  switch (preset) {
    case "7d":
      return new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);
    case "30d":
      return new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);
    case "90d":
      return new Date(now.getTime() + 90 * 24 * 60 * 60 * 1000);
    case "1y":
      return new Date(now.getTime() + 365 * 24 * 60 * 60 * 1000);
    case "none":
    case "custom":
      return undefined;
  }
}

interface GetCustomExpirationErrorOptions {
  now?: Date;
  requireValue?: boolean;
}

export function getCustomExpirationError(
  expiresAt: Date | undefined,
  options: GetCustomExpirationErrorOptions = {},
): string | null {
  const { now = new Date(), requireValue = false } = options;

  if (!expiresAt) {
    return requireValue ? CUSTOM_EXPIRATION_REQUIRED_ERROR : null;
  }

  return expiresAt > now ? null : CUSTOM_EXPIRATION_PAST_ERROR;
}

export function getExpirationDateError(
  expiresAt: string,
  now: Date = new Date(),
): string | null {
  const parsedExpiresAt = new Date(expiresAt);

  if (Number.isNaN(parsedExpiresAt.getTime())) {
    return CUSTOM_EXPIRATION_INVALID_ERROR;
  }

  return parsedExpiresAt > now ? null : CUSTOM_EXPIRATION_PAST_ERROR;
}
