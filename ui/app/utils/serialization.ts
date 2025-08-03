/**
 * Safely stringifies objects that may contain BigInt values.
 * BigInt values are converted to strings to ensure JSON compatibility.
 */
export function safeStringify(obj: unknown): string {
  return JSON.stringify(obj, (_key, value) =>
    typeof value === "bigint" ? value.toString() : value,
  );
}
