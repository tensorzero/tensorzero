import { Link } from "react-router";
import { useResolveUuid } from "~/hooks/useResolveUuid";
import type { ResolvedObject } from "~/types/tensorzero";
import { cn } from "~/utils/common";
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
    case "model_inference":
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      return null;
    default: {
      const _exhaustiveCheck: never = obj;
      return _exhaustiveCheck;
    }
  }
}

export function UuidLink({ uuid }: { uuid: string }) {
  const { data } = useResolveUuid(uuid);

  const url =
    data?.object_types.length === 1
      ? getUrlForResolvedObject(uuid, data.object_types[0])
      : null;

  return (
    <code
      className={cn(
        "relative rounded px-1.5 py-0.5 font-mono text-xs font-medium transition-colors duration-300",
        url ? "bg-orange-50 text-orange-400" : "bg-muted",
      )}
    >
      {url ? (
        <Link
          to={url}
          className="text-inherit no-underline after:absolute after:inset-0 hover:underline"
        >
          {uuid}
        </Link>
      ) : (
        uuid
      )}
    </code>
  );
}
