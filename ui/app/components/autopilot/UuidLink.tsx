import type { ReactNode } from "react";
import { Link } from "react-router";
import { useResolveUuid } from "~/hooks/useResolveUuid";
import type { ResolvedObject } from "~/types/tensorzero";
import { cn } from "~/utils/common";
import { toDatapointUrl, toEpisodeUrl, toInferenceUrl } from "~/utils/urls";
import { CircleDot, CircleHelp } from "lucide-react";
import { Inferences, Episodes, Dataset } from "~/components/icons/Icons";
import { Skeleton } from "~/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";

const ICON_SIZE = 12;
const ICON_CLASS = "mr-1 inline align-middle -translate-y-px";

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

function getEntityLabel(type: ResolvedObject["type"]): string {
  switch (type) {
    case "inference":
      return "Inference";
    case "episode":
      return "Episode";
    case "chat_datapoint":
    case "json_datapoint":
      return "Datapoint";
    case "model_inference":
      return "Model Inference";
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      return "Feedback";
    default: {
      const _exhaustiveCheck: never = type;
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
      return <CircleDot className={ICON_CLASS} size={ICON_SIZE} />;
    default: {
      const _exhaustiveCheck: never = type;
      return _exhaustiveCheck;
    }
  }
}

function IconWithTooltip({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="relative z-10 cursor-help">{children}</span>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
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
      {url ? (
        <Link
          to={url}
          className="text-inherit no-underline after:absolute after:inset-0 hover:underline"
        >
          {obj && (
            <IconWithTooltip label={getEntityLabel(obj.type)}>
              <EntityIcon type={obj.type} />
            </IconWithTooltip>
          )}
          {uuid}
        </Link>
      ) : (
        <>
          {obj ? (
            <IconWithTooltip label={getEntityLabel(obj.type)}>
              <EntityIcon type={obj.type} />
            </IconWithTooltip>
          ) : data ? (
            <IconWithTooltip label="Unknown">
              <CircleHelp className={ICON_CLASS} size={ICON_SIZE} />
            </IconWithTooltip>
          ) : (
            <IconWithTooltip label="Loading">
              <Skeleton
                className={cn(ICON_CLASS, "inline-block h-3 w-3 rounded-full")}
              />
            </IconWithTooltip>
          )}
          {uuid}
        </>
      )}
    </code>
  );
}
