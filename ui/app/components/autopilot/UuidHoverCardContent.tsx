import { Link } from "react-router";
import type { ResolvedObject } from "~/types/tensorzero";
import { useInferencePreview } from "~/hooks/useInferencePreview";
import { useEpisodePreview } from "~/hooks/useEpisodePreview";
import { formatDate, getRelativeTimeString } from "~/utils/date";

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
  const { data, isLoading } = useInferencePreview(uuid, isOpen);

  return (
    <div className="flex flex-col gap-2">
      <div className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
        Inference · {obj.function_type}
      </div>
      <div className="flex flex-col gap-1">
        <InfoRow label="Function" value={obj.function_name} mono />
        <InfoRow label="Variant" value={obj.variant_name} mono />
        <LazyInfoRow
          label="Timestamp"
          data={data}
          isLoading={isLoading}
          render={(d) => formatTimestamp(d.timestamp)}
        />
        <LazyInfoRow
          label="Latency"
          data={data}
          isLoading={isLoading}
          render={(d) =>
            d.processing_time_ms !== null ? `${d.processing_time_ms} ms` : "—"
          }
        />
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
  const { data, isLoading } = useEpisodePreview(uuid, isOpen);

  return (
    <div className="flex flex-col gap-2">
      <div className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
        Episode
      </div>
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
      <div className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
        {typeLabel}
      </div>
      <div className="flex flex-col gap-1">
        <InfoRow label="Dataset" value={obj.dataset_name} mono />
        <InfoRow label="Function" value={obj.function_name} mono />
      </div>
      <ViewDetailsLink url={url} />
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
      <span className={`text-foreground text-xs ${mono ? "font-mono" : ""}`}>
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

function ViewDetailsLink({ url }: { url: string }) {
  return (
    <Link
      to={url}
      className="text-muted-foreground hover:text-foreground mt-1 border-t pt-2 text-xs transition-colors"
    >
      View details &rarr;
    </Link>
  );
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return `${formatDate(date)} · ${getRelativeTimeString(date)}`;
}
