import { useState, useCallback } from "react";
import { ChevronRight } from "lucide-react";
import { Link } from "react-router";
import { HoverCard } from "radix-ui";
import type { ResolvedObject } from "~/types/tensorzero";
import { EntityPreviewType, useEntityPreview } from "~/hooks/useEntityPreview";
import { getRelativeTimeString, getTimestampTooltipData } from "~/utils/date";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import { Skeleton } from "~/components/ui/skeleton";
import { getFunctionTypeIcon } from "~/utils/icon";
import { useFunctionConfig } from "~/context/config";
import { toFunctionUrl, toResolvedObjectUrl, toVariantUrl } from "~/utils/urls";
import { useEntitySheet } from "~/context/entity-sheet";

interface UuidHoverCardProps {
  uuid: string;
  obj: ResolvedObject;
  children: React.ReactNode;
}

export function getHoverCardWidth(type: ResolvedObject["type"]): string {
  switch (type) {
    case "inference":
      return "w-80";
    case "episode":
      return "w-44";
    case "chat_datapoint":
    case "json_datapoint":
      return "w-56";
    case "model_inference":
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      return "w-56";
    default: {
      const _exhaustiveCheck: never = type;
      return _exhaustiveCheck;
    }
  }
}

export function UuidHoverCard({ uuid, obj, children }: UuidHoverCardProps) {
  const [isOpen, setIsOpen] = useState(false);
  const url = toResolvedObjectUrl(uuid, obj);

  if (!url) {
    return <>{children}</>;
  }

  return (
    <HoverCard.Root openDelay={300} closeDelay={200} onOpenChange={setIsOpen}>
      <HoverCard.Trigger asChild>{children}</HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side="top"
          sideOffset={4}
          className={cn(
            "bg-popover text-popover-foreground z-50 rounded-md border p-3 shadow-md",
            getHoverCardWidth(obj.type),
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
            "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
          )}
        >
          <HoverCardContent uuid={uuid} obj={obj} isOpen={isOpen} />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}

interface HoverCardContentProps {
  uuid: string;
  obj: ResolvedObject;
  isOpen: boolean;
}

function HoverCardContent({ uuid, obj, isOpen }: HoverCardContentProps) {
  switch (obj.type) {
    case "inference":
      return <InferenceContent uuid={uuid} obj={obj} isOpen={isOpen} />;
    case "episode":
      return <EpisodeContent uuid={uuid} obj={obj} isOpen={isOpen} />;
    case "chat_datapoint":
    case "json_datapoint":
      return <DatapointContent uuid={uuid} obj={obj} />;
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

export interface InferencePreview {
  timestamp: string;
}

interface InferenceContentProps {
  uuid: string;
  obj: Extract<ResolvedObject, { type: "inference" }>;
  isOpen: boolean;
}

function InferenceContent({ uuid, obj, isOpen }: InferenceContentProps) {
  const { data, isLoading } = useEntityPreview<InferencePreview>({
    type: EntityPreviewType.Inference,
    id: uuid,
    enabled: isOpen,
  });
  const functionConfig = useFunctionConfig(obj.function_name);
  const variantType =
    functionConfig?.variants[obj.variant_name]?.inner.type ?? null;

  return (
    <div className="flex flex-col gap-4">
      <TypeHeaderLink uuid={uuid} obj={obj}>
        Inference
      </TypeHeaderLink>
      <FunctionItem
        functionName={obj.function_name}
        functionType={obj.function_type}
      />
      <VariantItem
        functionName={obj.function_name}
        variantName={obj.variant_name}
        variantType={variantType}
      />
      <Timestamp data={data} isLoading={isLoading} />
    </div>
  );
}

interface EpisodePreview {
  inference_count: number;
}

interface EpisodeContentProps {
  uuid: string;
  obj: Extract<ResolvedObject, { type: "episode" }>;
  isOpen: boolean;
}

function EpisodeContent({ uuid, obj, isOpen }: EpisodeContentProps) {
  const { data, isLoading } = useEntityPreview<EpisodePreview>({
    type: EntityPreviewType.Episode,
    id: uuid,
    enabled: isOpen,
  });

  return (
    <div className="flex flex-col gap-4">
      <TypeHeaderLink uuid={uuid} obj={obj}>
        Episode
      </TypeHeaderLink>
      <InfoItem
        label="Inferences"
        value={data ? String(data.inference_count) : null}
        isLoading={isLoading}
      />
    </div>
  );
}

interface DatapointContentProps {
  uuid: string;
  obj: Extract<ResolvedObject, { type: "chat_datapoint" | "json_datapoint" }>;
}

function getDatapointTypeLabel(
  type: "chat_datapoint" | "json_datapoint",
): string {
  switch (type) {
    case "chat_datapoint":
      return "Chat Datapoint";
    case "json_datapoint":
      return "JSON Datapoint";
  }
}

function DatapointContent({ uuid, obj }: DatapointContentProps) {
  const typeLabel = getDatapointTypeLabel(obj.type);

  return (
    <div className="flex flex-col gap-4">
      <TypeHeaderLink uuid={uuid} obj={obj}>
        {typeLabel}
      </TypeHeaderLink>
      <InfoItem label="Dataset" value={obj.dataset_name} />
      <InfoItem label="Function" value={obj.function_name} />
    </div>
  );
}

interface TypeHeaderLinkProps {
  uuid: string;
  obj: ResolvedObject;
  children: React.ReactNode;
}

export function TypeHeaderLink({ uuid, obj, children }: TypeHeaderLinkProps) {
  const url = toResolvedObjectUrl(uuid, obj);
  const { openInferenceSheet, openEpisodeSheet } = useEntitySheet();

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      if (obj.type === "inference") {
        e.preventDefault();
        openInferenceSheet(uuid);
      } else if (obj.type === "episode") {
        e.preventDefault();
        openEpisodeSheet(uuid);
      }
    },
    [obj.type, uuid, openInferenceSheet, openEpisodeSheet],
  );

  if (!url) return null;

  return (
    <Link
      to={url}
      onClick={handleClick}
      className="text-muted-foreground hover:text-foreground inline-flex items-center text-xs transition-colors"
    >
      {children}
      <ChevronRight className="ml-0.5 h-3 w-3" />
    </Link>
  );
}

interface ItemProps {
  label: string;
  children: React.ReactNode;
}

function Item({ label, children }: ItemProps) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-muted-foreground text-xs">{label}</span>
      {children}
    </div>
  );
}

interface FunctionItemProps {
  functionName: string;
  functionType: string;
}

export function FunctionItem({
  functionName,
  functionType,
}: FunctionItemProps) {
  const iconConfig = getFunctionTypeIcon(functionType);
  return (
    <div className="flex flex-col gap-1">
      <span className="text-muted-foreground text-xs">Function</span>
      <span className="inline-flex min-w-0 items-center gap-1 font-mono text-xs">
        <span
          className={cn(
            "inline-flex shrink-0 items-center justify-center rounded p-0.5",
            iconConfig.iconBg,
          )}
        >
          {iconConfig.icon}
        </span>
        <Link
          to={toFunctionUrl(functionName)}
          className="text-foreground hover:text-foreground/80 truncate transition-colors hover:underline"
          title={functionName}
        >
          {functionName}
        </Link>
        <span className="text-muted-foreground shrink-0">· {functionType}</span>
      </span>
    </div>
  );
}

interface VariantItemProps {
  functionName: string;
  variantName: string;
  variantType: string | null;
}

export function VariantItem({
  functionName,
  variantName,
  variantType,
}: VariantItemProps) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-muted-foreground text-xs">Variant</span>
      <span className="inline-flex min-w-0 items-center font-mono text-xs">
        <Link
          to={toVariantUrl(functionName, variantName)}
          className="text-foreground hover:text-foreground/80 truncate transition-colors hover:underline"
          title={variantName}
        >
          {variantName}
        </Link>
        {variantType && (
          <span className="text-muted-foreground shrink-0">
            {" "}
            · {variantType}
          </span>
        )}
      </span>
    </div>
  );
}

interface InfoItemProps {
  label: string;
  value?: string | null;
  secondaryValue?: string | null;
  isLoading?: boolean;
}

export function InfoItem({
  label,
  value,
  secondaryValue,
  isLoading,
}: InfoItemProps) {
  return (
    <Item label={label}>
      {value ? (
        <span
          className="text-foreground min-w-0 truncate font-mono text-xs"
          title={secondaryValue ? `${value} · ${secondaryValue}` : value}
        >
          {value}
          {secondaryValue && (
            <span className="text-muted-foreground"> · {secondaryValue}</span>
          )}
        </span>
      ) : isLoading ? (
        <Skeleton className="h-4 w-8" />
      ) : null}
    </Item>
  );
}

interface TimestampProps {
  data: InferencePreview | null;
  isLoading: boolean;
}

export function Timestamp({ data, isLoading }: TimestampProps) {
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
        <TooltipContent className="bg-popover text-popover-foreground border shadow-md">
          <div className="flex flex-col gap-1">
            <div>{formattedDate}</div>
            <div>{formattedTime}</div>
          </div>
        </TooltipContent>
      </Tooltip>
    );
  }
  if (isLoading) {
    return <Skeleton className="h-4 w-24" />;
  }
  return null;
}
