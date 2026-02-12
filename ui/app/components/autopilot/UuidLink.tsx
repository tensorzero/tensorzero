import { Link } from "react-router";
import { useResolveUuid } from "~/hooks/useResolveUuid";
import type { ResolvedObject } from "~/types/tensorzero";
import { toDatapointUrl, toEpisodeUrl, toInferenceUrl } from "~/utils/urls";

/**
 * Given a resolved object, return the URL to navigate to.
 * Returns null for object types that don't have a detail page (e.g. feedback).
 */
function getUrlForResolvedObject(
  uuid: string,
  obj: ResolvedObject,
): string | null {
  switch (obj.type) {
    case "inference":
      return toInferenceUrl(uuid);
    case "episode":
      return toEpisodeUrl(uuid);
    case "chat_datapoint":
      return toDatapointUrl(obj.dataset_name, uuid);
    case "json_datapoint":
      return toDatapointUrl(obj.dataset_name, uuid);
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      // No dedicated feedback detail pages
      return null;
    default:
      return null;
  }
}

/**
 * Renders a UUID as an orange link if it resolves to exactly one known entity,
 * otherwise renders it as plain monospace text.
 */
export function UuidLink({ uuid }: { uuid: string }) {
  const { data } = useResolveUuid(uuid);

  // Not yet resolved, multiple types, or no types → plain text
  if (!data || data.object_types.length !== 1) {
    return <span className="font-mono">{uuid}</span>;
  }

  const url = getUrlForResolvedObject(uuid, data.object_types[0]);
  if (!url) {
    return <span className="font-mono">{uuid}</span>;
  }

  return (
    <Link
      to={url}
      className="font-mono text-orange-500 no-underline hover:underline"
    >
      {uuid}
    </Link>
  );
}
