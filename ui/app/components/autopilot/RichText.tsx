import { UuidLink } from "~/components/autopilot/UuidLink";

/**
 * Regex matching UUID-like strings (v1-v7 format).
 * Uses the global flag so it can be used with matchAll.
 */
export const UUID_REGEX =
  /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;

/**
 * Processes a plain text string and enriches it with interactive elements.
 *
 * Currently enriches:
 * - UUIDs → UuidLink components (resolved to entity links)
 *
 * This is the single extension point for adding new text enrichments
 * (e.g. function names, metric references) in the future.
 */
export function renderRichText(text: string): React.ReactNode {
  UUID_REGEX.lastIndex = 0;

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(UUID_REGEX)) {
    const matchStart = match.index;
    const matchEnd = matchStart + match[0].length;

    if (matchStart > lastIndex) {
      parts.push(text.slice(lastIndex, matchStart));
    }

    parts.push(<UuidLink key={matchStart} uuid={match[0]} />);
    lastIndex = matchEnd;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  if (parts.length === 0) {
    return text;
  }

  return <>{parts}</>;
}
