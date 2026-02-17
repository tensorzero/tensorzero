import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { Link } from "react-router";
import { HoverCard } from "radix-ui";
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

interface UuidHoverCardProps {
  uuid: string;
  obj: ResolvedObject;
  url: string;
  children: React.ReactNode;
}

export function UuidHoverCard({
  uuid,
  obj,
  url,
  children,
}: UuidHoverCardProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <HoverCard.Root openDelay={300} closeDelay={200} onOpenChange={setIsOpen}>
      <HoverCard.Trigger asChild>{children}</HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side="top"
          sideOffset={4}
          className={cn(
            "bg-popover text-popover-foreground z-50 rounded-md border p-3 shadow-md",
            obj.type === "inference" ? "w-80" : "w-56",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
            "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
          )}
        >
          <HoverCardContent
            uuid={uuid}
            obj={obj}
            url={url}
            isOpen={isOpen}
          />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}

interface HoverCardContentProps {
  uuid: string;
  obj: ResolvedObject;
  url: string;
  isOpen: boolean;
}

function HoverCardContent({
  uuid,
  obj,
  url,
  isOpen,
}: HoverCardContentProps) {
  switch (obj.type) {
    case "inference":
      return (
        <InferenceContent uuid={uuid} obj={obj} url={url} isOpen={isOpen} />
      );
    case "episode":
      return <EpisodeContent uuid={uuid} url={url} isOpen={isOpen} />;
    case "chat_datapoint":
    case "json_datapoint":
      return <DatapointContent obj={obj} url={url} />;
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

interface InferenceContentProps {
  uuid: string;
  obj: Extract<ResolvedObject, { type: "inference" }>;
  url: string;
  isOpen: boolean;
}

function InferenceContent({
  uuid,
  obj,
  url,
  isOpen,
}: InferenceContentProps) {
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

interface EpisodeContentProps {
  uuid: string;
  url: string;
  isOpen: boolean;
}

function EpisodeContent({ uuid, url, isOpen }: EpisodeContentProps) {
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

interface DatapointContentProps {
  obj: Extract<ResolvedObject, { type: "chat_datapoint" | "json_datapoint" }>;
  url: string;
}

function DatapointContent({ obj, url }: DatapointContentProps) {
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
