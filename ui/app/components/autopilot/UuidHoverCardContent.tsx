import { ChevronRight } from "lucide-react";
import { Link } from "react-router";
import type { ResolvedObject } from "~/types/tensorzero";
import { useEntityPreview } from "~/hooks/useEntityPreview";
import { getRelativeTimeString, getTimestampTooltipData } from "~/utils/date";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import { getFunctionTypeIcon } from "~/utils/icon";
import { useFunctionConfig } from "~/context/config";

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
  const functionConfig = useFunctionConfig(obj.function_name);
  const variantType =
    functionConfig?.variants[obj.variant_name]?.inner.type ?? null;

  return (
    <div className="flex flex-col gap-2">
      <TypeBadgeLink url={url}>Inference</TypeBadgeLink>
      <FunctionItem
        functionName={obj.function_name}
        functionType={obj.function_type}
      />
      <InfoItem
        label="Variant"
        value={obj.variant_name}
        secondaryValue={variantType}
        mono
      />
      <LazyTimestamp data={data} isLoading={isLoading} />
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
      <TypeBadgeLink url={url}>Episode</TypeBadgeLink>
      <LazyInfoItem
        label="Inferences"
        data={data}
        isLoading={isLoading}
        render={(d) =>
          `${d.inference_count} inference${d.inference_count !== 1 ? "s" : ""}`
        }
      />
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
      <TypeBadgeLink url={url}>{typeLabel}</TypeBadgeLink>
      <InfoItem label="Dataset" value={obj.dataset_name} mono />
      <InfoItem label="Function" value={obj.function_name} mono />
    </div>
  );
}

function TypeBadgeLink({
  children,
  url,
}: {
  children: React.ReactNode;
  url: string;
}) {
  return (
    <Link
      to={url}
      className="text-muted-foreground hover:text-foreground inline-flex items-center text-xs transition-colors"
    >
      {children}
      <ChevronRight className="ml-0.5 h-3 w-3" />
    </Link>
  );
}

function Item({
  label,
  align = "baseline",
  children,
}: {
  label: string;
  align?: "baseline" | "center";
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-[4rem_1fr] gap-2",
        align === "center" ? "items-center" : "items-baseline",
      )}
    >
      <span className="text-muted-foreground text-xs">{label}</span>
      {children}
    </div>
  );
}

function FunctionItem({
  functionName,
  functionType,
}: {
  functionName: string;
  functionType: string;
}) {
  const iconConfig = getFunctionTypeIcon(functionType);
  return (
    <Item label="Function" align="center">
      <span
        className="text-foreground inline-flex min-w-0 items-center gap-1 font-mono text-xs"
        title={`${functionName} · ${functionType}`}
      >
        <span
          className={cn(
            "inline-flex shrink-0 items-center justify-center rounded p-0.5",
            iconConfig.iconBg,
          )}
        >
          {iconConfig.icon}
        </span>
        <span className="truncate">{functionName}</span>
        <span className="text-muted-foreground shrink-0">· {functionType}</span>
      </span>
    </Item>
  );
}

function InfoItem({
  label,
  value,
  secondaryValue,
  mono,
}: {
  label: string;
  value: string;
  secondaryValue?: string | null;
  mono?: boolean;
}) {
  return (
    <Item label={label}>
      <span
        className={cn(
          "text-foreground min-w-0 truncate text-xs",
          mono && "font-mono",
        )}
        title={secondaryValue ? `${value} · ${secondaryValue}` : value}
      >
        {value}
        {secondaryValue && (
          <span className="text-muted-foreground"> · {secondaryValue}</span>
        )}
      </span>
    </Item>
  );
}

function LazyInfoItem<T>({
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
    <Item label={label}>
      {data ? (
        <span className="text-foreground min-w-0 truncate text-xs">
          {render(data)}
        </span>
      ) : isLoading ? (
        <span className="bg-muted h-4 w-20 animate-pulse rounded" />
      ) : null}
    </Item>
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
    const { formattedDate, formattedTime } = getTimestampTooltipData(
      data.timestamp,
    );
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="text-muted-foreground w-fit cursor-default text-xs">
            {getRelativeTimeString(date)}
          </span>
        </TooltipTrigger>
        <TooltipContent className="border-border bg-bg-secondary text-fg-primary border shadow-lg">
          <div className="flex flex-col gap-1">
            <div>{formattedDate}</div>
            <div>{formattedTime}</div>
          </div>
        </TooltipContent>
      </Tooltip>
    );
  }
  if (isLoading) {
    return <span className="bg-muted h-4 w-16 animate-pulse rounded" />;
  }
  return null;
}
