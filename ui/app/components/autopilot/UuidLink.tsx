import { Link } from "react-router";
import { useResolveUuid } from "~/hooks/useResolveUuid";
import type { ResolvedObject } from "~/types/tensorzero";
import { cn } from "~/utils/common";
import { toDatapointUrl, toEpisodeUrl, toInferenceUrl } from "~/utils/urls";
import { Inferences, Episodes, Dataset } from "~/components/icons/Icons";

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

function getEntityIcon(obj: ResolvedObject) {
  switch (obj.type) {
    case "inference":
      return <Inferences className="h-2.5 w-2.5 shrink-0" />;
    case "episode":
      return <Episodes className="h-2.5 w-2.5 shrink-0" />;
    case "chat_datapoint":
    case "json_datapoint":
      return <Dataset className="h-2.5 w-2.5 shrink-0" />;
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

  const obj = data?.object_types.length === 1 ? data.object_types[0] : null;
  const url = obj ? getUrlForResolvedObject(uuid, obj) : null;

  return (
    <code
      className={cn(
        "relative rounded px-1.5 py-0.5 font-mono text-xs font-medium transition-colors duration-300",
        url ? "bg-orange-50 text-orange-500" : "bg-muted",
      )}
    >
      {url && obj ? (
        <Link
          to={url}
          className="inline-flex items-center gap-1 text-inherit no-underline after:absolute after:inset-0 hover:underline"
        >
          {getEntityIcon(obj)}
          {uuid}
        </Link>
      ) : (
        uuid
      )}
    </code>
  );
}
