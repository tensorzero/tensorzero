import { Link } from "react-router";
import type { ResolvedObject } from "~/types/tensorzero";
import { useEntityPreview } from "~/hooks/useEntityPreview";
import { getRelativeTimeString } from "~/utils/date";
import { TimestampTooltip } from "~/components/ui/TimestampTooltip";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import { DotSeparator } from "~/components/ui/DotSeparator";

interface InferencePreview {
  timestamp: string;
}

interface EpisodePreview {
  inference_count: number;
}

interface UuidHoverCardContentProps {
  uuid: string;
  obj: ResolvedObject;
  url: string;
  isOpen: boolean;
}

export function UuidHoverCardContent({
  uuid,
  obj,
  url,
  isOpen,
}: UuidHoverCardContentProps) {
  switch (obj.type) {
    case "inference":
      return (
        <InferenceHoverContent
          uuid={uuid}
          obj={obj}
          url={url}
          isOpen={isOpen}
        />
      );
    case "episode":
      return <EpisodeHoverContent uuid={uuid} url={url} isOpen={isOpen} />;
    case "chat_datapoint":
    case "json_datapoint":
      return <DatapointHoverContent obj={obj} url={url} />;
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

function InferenceHoverContent({
  uuid,
  obj,
  url,
  isOpen,
}: {
  uuid: string;
  obj: Extract<ResolvedObject, { type: "inference" }>;
  url: string;
  isOpen: boolean;
}) {
  const { data, isLoading } = useEntityPreview<InferencePreview>(
    `/api/tensorzero/inference_preview/${encodeURIComponent(uuid)}`,
    isOpen,
  );

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <TypeBadge>
          Inference <DotSeparator /> {obj.function_type}
        </TypeBadge>
        <LazyTimestamp data={data} isLoading={isLoading} />
      </div>
      <div className="flex flex-col gap-1">
        <InfoRow label="Function" value={obj.function_name} mono />
        <InfoRow label="Variant" value={obj.variant_name} mono />
      </div>
      <ViewDetailsLink url={url} />
    </div>
  );
}

function EpisodeHoverContent({
  uuid,
  url,
  isOpen,
}: {
  uuid: string;
  url: string;
  isOpen: boolean;
}) {
  const { data, isLoading } = useEntityPreview<EpisodePreview>(
    `/api/tensorzero/episode_preview/${encodeURIComponent(uuid)}`,
    isOpen,
  );

  return (
    <div className="flex flex-col gap-2">
      <TypeBadge>Episode</TypeBadge>
      <div className="flex flex-col gap-1">
        <LazyInfoRow
          label="Inferences"
          data={data}
          isLoading={isLoading}
          render={(d) =>
            `${d.inference_count} inference${d.inference_count !== 1 ? "s" : ""}`
          }
        />
      </div>
      <ViewDetailsLink url={url} />
    </div>
  );
}

function DatapointHoverContent({
  obj,
  url,
}: {
  obj: Extract<ResolvedObject, { type: "chat_datapoint" | "json_datapoint" }>;
  url: string;
}) {
  const typeLabel =
    obj.type === "chat_datapoint" ? "Chat Datapoint" : "JSON Datapoint";

  return (
    <div className="flex flex-col gap-2">
      <TypeBadge>{typeLabel}</TypeBadge>
      <div className="flex flex-col gap-1">
        <InfoRow label="Dataset" value={obj.dataset_name} mono />
        <InfoRow label="Function" value={obj.function_name} mono />
      </div>
      <ViewDetailsLink url={url} />
    </div>
  );
}

function TypeBadge({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-muted-foreground inline-flex items-center gap-1 text-xs font-medium tracking-wide uppercase">
      {children}
    </div>
  );
}

function InfoRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-muted-foreground text-xs">{label}</span>
      <span className={cn("text-foreground text-xs", mono && "font-mono")}>
        {value}
      </span>
    </div>
  );
}

function LazyInfoRow<T>({
  label,
  data,
  isLoading,
  render,
}: {
  label: string;
  data: T | null;
  isLoading: boolean;
  render: (data: T) => string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-muted-foreground text-xs">{label}</span>
      {data ? (
        <span className="text-foreground text-xs">{render(data)}</span>
      ) : isLoading ? (
        <span className="bg-muted h-3 w-20 animate-pulse rounded" />
      ) : null}
    </div>
  );
}

function LazyTimestamp({
  data,
  isLoading,
}: {
  data: InferencePreview | null;
  isLoading: boolean;
}) {
  if (data) {
    const date = new Date(data.timestamp);
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="text-muted-foreground cursor-default text-xs">
            {getRelativeTimeString(date)}
          </span>
        </TooltipTrigger>
        <TooltipContent className="border-border bg-bg-secondary text-fg-primary border shadow-lg">
          <TimestampTooltip timestamp={data.timestamp} />
        </TooltipContent>
      </Tooltip>
    );
  }
  if (isLoading) {
    return <span className="bg-muted h-3 w-16 animate-pulse rounded" />;
  }
  return null;
}

function ViewDetailsLink({ url }: { url: string }) {
  return (
    <Link
      to={url}
      className="text-muted-foreground hover:text-foreground -mx-3 -mb-3 mt-1 border-t px-3 pb-3 pt-3 text-xs transition-colors"
    >
      View details &rarr;
    </Link>
  );
}
