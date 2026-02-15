import { Link } from "react-router";
import { useResolveUuid } from "~/hooks/useResolveUuid";
import type { ResolvedObject } from "~/types/tensorzero";
import { cn } from "~/utils/common";
import { toDatapointUrl, toEpisodeUrl, toInferenceUrl } from "~/utils/urls";
import { Inferences, Episodes, Dataset } from "~/components/icons/Icons";

const ICON_SIZE = 12;
const ICON_CLASS = "shrink-0";

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

function EntityIcon({ type }: { type: ResolvedObject["type"] }) {
  switch (type) {
    case "inference":
      return <Inferences className={ICON_CLASS} size={ICON_SIZE} />;
    case "episode":
      return <Episodes className={ICON_CLASS} size={ICON_SIZE} />;
    case "chat_datapoint":
    case "json_datapoint":
      return <Dataset className={ICON_CLASS} size={ICON_SIZE} />;
    case "model_inference":
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      return null;
    default: {
      const _exhaustiveCheck: never = type;
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
        "relative inline-flex items-center gap-1 align-middle rounded px-1.5 py-0.5 font-mono text-xs font-medium transition-colors duration-300",
        url ? "bg-orange-50 text-orange-500" : "bg-muted",
      )}
    >
      {obj && <EntityIcon type={obj.type} />}
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
