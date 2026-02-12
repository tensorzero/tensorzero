import { Link } from "react-router";
import { useResolveUuid } from "~/hooks/useResolveUuid";
import type { ResolvedObject } from "~/types/tensorzero";
import { toDatapointUrl, toEpisodeUrl, toInferenceUrl } from "~/utils/urls";

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
    case "json_datapoint":
      return toDatapointUrl(obj.dataset_name, uuid);
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      return null;
    default:
      return null;
  }
}

export function UuidLink({ uuid }: { uuid: string }) {
  const { data } = useResolveUuid(uuid);

  const url =
    data?.object_types.length === 1
      ? getUrlForResolvedObject(uuid, data.object_types[0])
      : null;

  if (!url) {
    return (
      <code className="bg-muted rounded px-1.5 py-0.5 font-mono text-xs font-medium">
        {uuid}
      </code>
    );
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
