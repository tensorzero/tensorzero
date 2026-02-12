import { UuidLink } from "~/components/autopilot/UuidLink";
import { splitTextOnUuids } from "~/utils/uuid";

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
  const segments = splitTextOnUuids(text);

  if (segments.length === 0) {
    return text;
  }

  const hasUuid = segments.some((s) => s.isUuid);
  if (!hasUuid) {
    return text;
  }

  return (
    <>
      {segments.map((segment, i) =>
        segment.isUuid ? (
          <UuidLink key={i} uuid={segment.text} />
        ) : (
          segment.text
        ),
      )}
    </>
  );
}
