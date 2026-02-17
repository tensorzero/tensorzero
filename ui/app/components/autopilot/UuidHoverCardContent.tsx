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

interface InferenceHoverContentProps {
  uuid: string;
  obj: Extract<ResolvedObject, { type: "inference" }>;
  url: string;
  isOpen: boolean;
}

function InferenceHoverContent({
  uuid,
  obj,
  url,
  isOpen,
}: InferenceHoverContentProps) {
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
      <TimestampItem data={data} isLoading={isLoading} />
    </div>
  );
}

interface EpisodeHoverContentProps {
  uuid: string;
  url: string;
  isOpen: boolean;
}

function EpisodeHoverContent({
  uuid,
  url,
  isOpen,
}: EpisodeHoverContentProps) {
  const { data, isLoading } = useEntityPreview<EpisodePreview>(
    `/api/tensorzero/episode_preview/${encodeURIComponent(uuid)}`,
    isOpen,
  );

  const inferenceCountText = data
    ? `${data.inference_count} inference${data.inference_count !== 1 ? "s" : ""}`
    : null;

  return (
    <div className="flex flex-col gap-2">
      <TypeBadgeLink url={url}>Episode</TypeBadgeLink>
      <InfoItem
        label="Inferences"
        value={inferenceCountText}
        isLoading={isLoading}
      />
    </div>
  );
}

interface DatapointHoverContentProps {
  obj: Extract<ResolvedObject, { type: "chat_datapoint" | "json_datapoint" }>;
  url: string;
}

function DatapointHoverContent({ obj, url }: DatapointHoverContentProps) {
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

interface TypeBadgeLinkProps {
  children: React.ReactNode;
  url: string;
}

function TypeBadgeLink({ children, url }: TypeBadgeLinkProps) {
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

interface ItemProps {
  label: string;
  align?: "baseline" | "center";
  children: React.ReactNode;
}

function Item({ label, align = "baseline", children }: ItemProps) {
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

interface FunctionItemProps {
  functionName: string;
  functionType: string;
}

function FunctionItem({ functionName, functionType }: FunctionItemProps) {
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

interface InfoItemProps {
  label: string;
  value?: string | null;
  secondaryValue?: string | null;
  mono?: boolean;
  isLoading?: boolean;
}

function InfoItem({
  label,
  value,
  secondaryValue,
  mono,
  isLoading,
}: InfoItemProps) {
  return (
    <Item label={label}>
      {value ? (
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
      ) : isLoading ? (
        <span className="bg-muted h-4 w-20 animate-pulse rounded" />
      ) : null}
    </Item>
  );
}

interface TimestampItemProps {
  data: InferencePreview | null;
  isLoading: boolean;
}

function TimestampItem({ data, isLoading }: TimestampItemProps) {
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
