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

function UuidCode({ children }: { children: string }) {
  return (
    <code className="bg-muted rounded px-1.5 py-0.5 font-mono text-xs font-medium">
      {children}
    </code>
  );
}

/**
 * Renders a UUID as an orange link if it resolves to exactly one known entity,
 * otherwise renders it as plain monospace text.
 */
export function UuidLink({ uuid }: { uuid: string }) {
  const { data } = useResolveUuid(uuid);

  // Not yet resolved, multiple types, or no types → plain text
  if (!data || data.object_types.length !== 1) {
    return <UuidCode>{uuid}</UuidCode>;
  }

  const url = getUrlForResolvedObject(uuid, data.object_types[0]);
  if (!url) {
    return <UuidCode>{uuid}</UuidCode>;
  }

  return (
    <Link
      to={url}
      className="rounded bg-orange-50 px-1 py-0.5 font-mono text-xs text-orange-500 no-underline hover:underline"
    >
      {uuid}
    </Link>
  );
}
