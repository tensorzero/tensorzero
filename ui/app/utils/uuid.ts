/**
 * Regex matching UUID-like strings (v1-v7 format).
 * Uses the global flag so it can be used with `String.prototype.matchAll`.
 *
 * IMPORTANT: Always reset `lastIndex` before use since the `g` flag is stateful.
 */
export const UUID_REGEX =
  /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;

/** Regex that matches a string that is exactly one UUID (with optional surrounding whitespace). */
export const EXACT_UUID_RE =
  /^\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\s*$/i;

/**
 * Split a string into segments, separating UUID matches from surrounding text.
 * Returns an array of `{ text: string, isUuid: boolean }` segments in order.
 */
export function splitTextOnUuids(
  text: string,
): { text: string; isUuid: boolean }[] {
  UUID_REGEX.lastIndex = 0;

  const segments: { text: string; isUuid: boolean }[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(UUID_REGEX)) {
    const matchStart = match.index;
    const matchEnd = matchStart + match[0].length;

    if (matchStart > lastIndex) {
      segments.push({ text: text.slice(lastIndex, matchStart), isUuid: false });
    }

    segments.push({ text: match[0], isUuid: true });
    lastIndex = matchEnd;
  }

  if (lastIndex < text.length) {
    segments.push({ text: text.slice(lastIndex), isUuid: false });
  }

  return segments;
}
